"""
RIM weighting.

Upload an SPSS .sav, add weighting variables, and optionally add further
variables under any response of a variable already in the tree. Targets are
entered against the node you select and are always relative to that node.

The tree is flattened into a single set of raking margins before it reaches
weightipy: a variable sitting under a response becomes a derived margin
covering the whole sample, carrying that variable inside the response and a
single lump outside it. That keeps conditional and unconditional targets in
one rim run.

Run with:  streamlit run app.py
"""

import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pyreadstat
import streamlit as st
import weightipy as wp

st.set_page_config(page_title="Weighting", layout="wide")

MAX_CATS = 50          # variables with more categories aren't offered
MAX_DEPTH = 3          # variables allowed on one branch;
                       # measured as len(path) everywhere
WARN_NODE_N = 100      # warn when weighting inside a node this small
BLOCK_NODE_N = 30      # refuse to weight inside a node smaller than this
PCT_TOLERANCE = 0.5    # allowed drift from 100 when entering percentages
CACHE_TTL = "2h"       # how long a parsed file may stay in the cache
CACHE_ENTRIES = 3      # how many parsed files the process may hold at once

VERSION = "v28"   # recorded in project files, not displayed

ROOT = "root"
OUTSIDE = "\u00b7outside"   # lump category for cases outside a node
STEP = "\u203a"             # separates path steps in keys and labels
SEP_X = " \u00d7 "          # joins the parts of an interlocked category


# ---------------------------------------------------------------- data loading

# The cache is shared by the whole process, not by one session. Entries are
# keyed on the file's own bytes, so nobody can reach another person's data,
# but every parsed file sits in memory until it is evicted. A short ttl and a
# small cap bound how long uploaded data can linger on a shared host.
@st.cache_data(
    show_spinner="Reading SPSS file...",
    ttl=CACHE_TTL,
    max_entries=CACHE_ENTRIES,
)
def load_sav(file_bytes: bytes):
    """Read a .sav with value labels applied.

    Returns the data, the variable labels, and the declared categories of
    each variable in code order. Codes SPSS declares as user-missing are left
    out: they arrive as NaN in the data and are not responses to weight to.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        df, meta = pyreadstat.read_sav(
            tmp_path,
            apply_value_formats=True,   # show "Female" rather than 2
            # categoricals store small codes plus one copy of each label,
            # which is the difference between a large tracker fitting in
            # memory and not
            formats_as_category=True,
        )
        # missing_ranges only comes through on a user_missing read, and this
        # one is metadata only so it costs almost nothing
        _, miss_meta = pyreadstat.read_sav(
            tmp_path, metadataonly=True, user_missing=True
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # An unanswered text question comes back as an empty string rather than
    # missing, which would otherwise show up as a category of its own.
    for column in df.columns:
        if df[column].dtype == object or pd.api.types.is_string_dtype(
            df[column]
        ):
            df[column] = df[column].replace(r"^\s*$", None, regex=True)

    var_labels = {
        name: (label or "")
        for name, label in zip(meta.column_names, meta.column_labels or [])
    }

    ranges = getattr(miss_meta, "missing_ranges", None) or {}
    all_labels = miss_meta.variable_value_labels or {}

    def declared_missing(var, code) -> bool:
        for span in ranges.get(var, []):
            try:
                if span["lo"] <= code <= span["hi"]:
                    return True
            except TypeError:      # string codes can't be compared to a range
                if code in (span.get("lo"), span.get("hi")):
                    return True
        return False

    # Declared categories in code order. Responses nobody picked stay in, so
    # they are visible in the grid; user-missing codes drop out.
    value_labels = {
        var: [
            labels[code] for code in sorted(labels)
            if not declared_missing(var, code)
        ]
        for var, labels in all_labels.items()
    }
    return df, var_labels, value_labels


def numeric_vars(frame: pd.DataFrame) -> list:
    """Columns that look continuous enough to band."""
    out = []
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]) \
                and frame[col].nunique(dropna=True) > 8:
            out.append(col)
    return out


def apply_recipes(base_df, base_var_labels, base_value_labels, recipes):
    """Rebuild derived variables from their recipes.

    Recipes are stored rather than the resulting columns, so the same
    definitions can be replayed on next wave's file and travel with the
    scheme. They are applied in order, so one recipe can build on another.
    """
    # a shallow copy shares the existing columns; only the new ones cost
    frame = base_df if not recipes else base_df.copy(deep=False)
    var_labels = dict(base_var_labels)
    value_labels = dict(base_value_labels)
    notes = []

    for recipe in recipes:
        name, kind = recipe["name"], recipe["kind"]
        try:
            if kind == "combine":
                source = recipe["source"]
                if source not in frame.columns:
                    notes.append(f"{name}: {source} is not in this file")
                    continue
                mapping = {
                    value: label
                    for label, values in recipe["groups"]
                    for value in values
                }
                frame[name] = frame[source].astype("object").map(mapping)
                value_labels[name] = [label for label, _ in recipe["groups"]]
                var_labels[name] = f"Grouped from {source}"

            elif kind == "interlock":
                sources = recipe["sources"]
                missing = [s for s in sources if s not in frame.columns]
                if missing:
                    notes.append(f"{name}: missing {', '.join(missing)}")
                    continue
                complete = frame[sources].notna().all(axis=1)
                joined = frame[sources[0]].astype(str)
                for source in sources[1:]:
                    joined = joined + SEP_X + frame[source].astype(str)
                frame[name] = joined.where(complete)

                # declared cells = the full cross, so empty cells stay visible
                declared = [[]]
                for source in sources:
                    cats = value_labels.get(source)
                    if not cats:
                        cats = sorted(frame[source].dropna().unique(), key=str)
                    declared = [c + [str(x)] for c in declared for x in cats]
                    if len(declared) > 200:   # too many cells to enumerate
                        declared = []
                        break
                if declared:
                    value_labels[name] = [SEP_X.join(c) for c in declared]
                var_labels[name] = "Interlock of " + " x ".join(sources)

            elif kind == "band":
                source = recipe["source"]
                if source not in frame.columns:
                    notes.append(f"{name}: {source} is not in this file")
                    continue
                values = pd.to_numeric(frame[source], errors="coerce")
                # Bands are independent intervals rather than one continuous
                # cut, so a deliberate gap between them leaves those cases
                # missing instead of folding them into a neighbour.
                banded = pd.Series(None, index=frame.index, dtype="object")
                for (lower, upper), label in zip(recipe["intervals"],
                                                 recipe["labels"]):
                    top = float("inf") if upper is None else float(upper)
                    inside = (values >= float(lower)) & (values < top)
                    banded = banded.mask(inside, label)
                frame[name] = banded
                value_labels[name] = list(recipe["labels"])
                var_labels[name] = f"Banded from {source}"

        except Exception as exc:  # noqa: BLE001 - a bad recipe must not stop the app
            notes.append(f"{name}: could not be built ({exc})")

    return frame, var_labels, value_labels, notes


def categorical_vars(frame: pd.DataFrame) -> list:
    return [
        col for col in frame.columns
        if 1 < frame[col].nunique(dropna=True) <= MAX_CATS
    ]


def categories_of(frame: pd.DataFrame, var: str) -> list:
    """Categories with at least one case here - the ones that can be weighted."""
    return sorted(frame[var].dropna().unique(), key=str)


def describe_variable(frame: pd.DataFrame, col: str) -> str:
    """A useful one-liner for the sidebar, according to what the column is.

    Counting distinct values and calling them categories is misleading for
    anything continuous or free text, so each kind gets what is actually
    worth knowing about it.
    """
    column = frame[col]
    distinct = int(column.nunique(dropna=True))
    missing = int(column.isna().sum())

    if pd.api.types.is_numeric_dtype(column) and distinct > MAX_CATS:
        low, high = column.min(), column.max()
        whole = float(low).is_integer() and float(high).is_integer()
        span = (f"{low:,.0f} to {high:,.0f}" if whole
                else f"{low:,.2f} to {high:,.2f}")
        detail = f"range {span}"
    elif distinct <= MAX_CATS:
        detail = f"{distinct} categories"
    else:
        detail = f"text, {distinct:,} distinct values"

    if missing:
        detail += f" | {missing:,} missing"
    return detail


def declared_categories(labels_map: dict, frame: pd.DataFrame, var: str):
    """Declared categories first, then any unlabelled values that appear."""
    seen = sorted(frame[var].dropna().unique(), key=str)
    declared = labels_map.get(var) or []
    cats = list(declared) + [c for c in seen if c not in declared]
    return cats or seen


def cats_and_counts(frame: pd.DataFrame, var: str):
    """Every declared category plus any unlabelled values, with their counts.

    Categories with no cases are kept so they are visible, but they cannot
    take a target: raking multiplies existing cases, so no factor turns zero
    respondents into a non-zero share.
    """
    counts = frame[var].value_counts(dropna=True)
    cats = declared_categories(value_labels, frame, var)
    return cats, [int(counts.get(c, 0)) for c in cats]


def scope_mask(frame: pd.DataFrame, path: list) -> pd.Series:
    """Boolean mask for the cases sitting under a path of (variable, response)."""
    mask = pd.Series(True, index=frame.index)
    for var, value in path:
        mask &= frame[var] == value
    return mask


def path_key(path: list) -> str:
    """Stable key for a container node: the root, or a response of a variable."""
    if not path:
        return ROOT
    return ROOT + "".join(f"{STEP}{var}={value}" for var, value in path)


def path_from_key(container_key: str) -> list:
    """Rebuild a node path from its container key."""
    if container_key == ROOT:
        return []
    path = []
    for step in container_key.split(STEP)[1:]:
        variable, _, value = step.partition("=")
        path.append((variable, value))
    return path


def id_candidates(frame: pd.DataFrame) -> list:
    """Variables that could identify a respondent: complete and unique."""
    total = len(frame)
    out = []
    for col in frame.columns:
        column = frame[col]
        if column.isna().any():
            continue
        if column.nunique(dropna=False) == total:
            out.append(col)
    return out


def as_id_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Ids as they should be written out - whole numbers without a .0."""
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values) and (values % 1 == 0).all():
        return values.astype("int64")
    return values


# ------------------------------------------------------------------ app state

DEFAULTS = {
    "vars": {},        # {container_key: [variable, ...]}
    "targets": {},     # {variable_node_key: [value, ...]}
    "units": {},       # {variable_node_key: "Percentages" | "Counts"}
    "versions": {},    # {variable_node_key: int} to force grid redraws
    "creator_gen": 0,   # bumped to remount the creator with a clean slate
    "recipes": [],      # derived variable definitions, applied in order
    "id_var": None,     # respondent identifier, used when exporting weights
    "loaded_project": None,   # fingerprint of the project file already applied
    "project_notes": [],
    "clipboard": None,  # a copied variable branch
    "paste_report": [],
    "file_bytes": None,
    "file_name": None,
    "expanded": set(),  # variable nodes whose responses are shown
    "sel": ROOT,
    "result": None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def vars_at(container_key: str) -> list:
    return list(st.session_state.vars.get(container_key, []))


def node_key(container_key: str, var: str) -> str:
    return f"{container_key}::{var}"


def snapshot(container_key: str, var: str, path: list) -> dict:
    """Copy a variable node, its targets, and everything beneath it."""
    key = node_key(container_key, var)
    scope = df[scope_mask(df, path)]
    cats, _ = cats_and_counts(scope, var)
    children = {}
    for cat in cats:
        child_path = path + [(var, cat)]
        child_key = path_key(child_path)
        subs = [
            snapshot(child_key, sub, child_path)
            for sub in vars_at(child_key)
        ]
        if subs:
            children[str(cat)] = subs
    return {
        "var": var,
        "categories": [str(c) for c in cats],
        "targets": list(st.session_state.targets.get(key) or []),
        "unit": st.session_state.units.get(key, "Percentages"),
        "children": children,
    }


def restore(container_key: str, path: list, snap: dict, with_targets: bool,
            with_children: bool, report: list):
    """Paste a copied branch under a container, skipping what cannot apply."""
    var = snap["var"]
    if var in [v for v, _ in path]:
        report.append(f"{var} is already used higher up this branch")
        return
    if var not in df.columns:
        report.append(f"{var} is not in this file")
        return

    current = vars_at(container_key)
    if var not in current:
        st.session_state.vars[container_key] = current + [var]

    key = node_key(container_key, var)
    scope = df[scope_mask(df, path)]
    cats = [str(c) for c in cats_and_counts(scope, var)[0]]

    if with_targets and snap["targets"]:
        if cats == snap["categories"]:
            st.session_state.units[key] = snap["unit"]
            set_targets(key, list(snap["targets"]))
            if snap["unit"] == "Counts":
                report.append(
                    f"{var}: counts pasted but this node has a different "
                    "base - check the total"
                )
        else:
            report.append(
                f"{var}: categories differ here, so targets were not pasted"
            )

    if with_children:
        for cat, subs in snap["children"].items():
            if cat not in cats:
                report.append(f"{var}: response {cat} does not exist here")
                continue
            child_path = path + [(var, cat)]
            for sub in subs:
                restore(path_key(child_path), child_path, sub,
                        with_targets, True, report)


def forget_pickers():
    """Drop multiselect widget state so it re-reads the tree on next run."""
    for widget_key in [k for k in st.session_state if str(k).startswith("pick_")]:
        del st.session_state[widget_key]


def stored(key: str, size: int) -> list:
    values = st.session_state.targets.get(key)
    if not isinstance(values, list) or len(values) != size:
        values = [None] * size
        st.session_state.targets[key] = values
    return values


def set_targets(key: str, values: list):
    st.session_state.targets[key] = values
    st.session_state.versions[key] = st.session_state.versions.get(key, 0) + 1


def resolve(key: str, observed: list):
    """Turn entered targets into percentages of their own node.

    Only categories with cases can carry a target. Counts must add to the
    node's base; percentages must add to 100. Returns (percentages aligned to
    the rows, with None for empty categories | None, problem | None).
    """
    values = st.session_state.targets.get(key)
    size = len(observed)
    if not isinstance(values, list) or len(values) != size:
        return None, "no targets entered"

    weightable = [i for i, n in enumerate(observed) if n > 0]
    if not weightable:
        return None, "no cases here"

    missing = [
        i for i in weightable
        if values[i] is None or pd.isna(values[i])
    ]
    if missing:
        return None, (
            f"{len(missing)} of {len(weightable)} rows still need a target"
        )

    numbers = [float(values[i]) for i in weightable]
    total = sum(numbers)
    if total <= 0:
        return None, "targets must be greater than zero"

    base = sum(observed)
    if st.session_state.units.get(key, "Percentages") == "Counts":
        tolerance = max(1.0, base * 0.005)
        if abs(total - base) > tolerance:
            return None, (
                f"counts add to {total:,.0f} but the base here is {base:,} "
                f"({total - base:+,.0f})"
            )
    elif abs(total - 100) > PCT_TOLERANCE:
        return None, f"percentages add to {total:,.2f}, not 100"

    percentages = [None] * size
    for i, value in zip(weightable, numbers):
        percentages[i] = value / total * 100
    return percentages, None


# ---------------------------------------------------------------------- upload

st.title("Weighting")

# The uploader is only shown until a file is loaded. Leaving it on screen
# invites an accidental clear of a tree that took real work to build.
if st.session_state.file_bytes is None:
    uploaded = st.file_uploader("SPSS data file", type=["sav"])
    if uploaded is None:
        st.info(
            "Upload a .sav file to begin. The file is held for this session "
            "only and is not written to permanent storage."
        )
        st.stop()
    st.session_state.file_bytes = uploaded.getvalue()
    st.session_state.file_name = uploaded.name
    st.rerun()

base_df, base_var_labels, base_value_labels = load_sav(
    st.session_state.file_bytes
)
df, var_labels, value_labels, recipe_notes = apply_recipes(
    base_df, base_var_labels, base_value_labels, st.session_state.recipes
)
usable = categorical_vars(df)
derived_names = {r["name"] for r in st.session_state.recipes}

head, swap = st.columns([5, 1], vertical_alignment="center")
head.caption(
    f"{st.session_state.file_name}  \u00b7  {len(df):,} cases  \u00b7  "
    f"{len(df.columns):,} variables"
)
if swap.button("Change file", icon=":material/swap_horiz:",
               help="Clears the tree and starts again with another file"):
    for state_key, default in DEFAULTS.items():
        st.session_state[state_key] = (
            set() if isinstance(default, set)
            else dict(default) if isinstance(default, dict)
            else default
        )
    st.rerun()


# ------------------------------------------------------------- sidebar: vars

with st.sidebar:
    st.subheader(st.session_state.file_name)
    st.caption(f"{len(df):,} cases | {len(df.columns):,} variables")

    search = st.text_input("Filter variables", "")
    st.divider()

    shown = 0
    for col in df.columns:
        label = var_labels.get(col, "")
        if search and search.lower() not in f"{col} {label}".lower():
            continue
        shown += 1
        if shown > 200:
            st.caption("...list truncated, refine the filter")
            break
        ok = col in usable
        tag = "  \u00b7 derived" if col in derived_names else ""
        st.markdown(f"**{col}**{tag}" if ok else f"{col}{tag}")
        detail = f"{label} | " if label else ""
        detail += describe_variable(df, col)
        st.caption(detail)


# --------------------------------------------------------------- project file

PROJECT_SCHEMA = 2


def build_project() -> dict:
    """Everything needed to rebuild this session, and no respondent data.

    Definitions only: recipes, the tree, its targets and units. The point of
    the file is to carry a method onto another file, so nothing here is tied
    to the data it was built on. The source name and date are recorded as
    provenance and are never checked against the file being loaded.

    Weights are not stored either - they are reproduced by pressing Run,
    which is what makes this a record of the method rather than of one run.
    """
    return {
        "schema": PROJECT_SCHEMA,
        "app_version": VERSION,
        "saved": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "saved_from": st.session_state.file_name,
        "recipes": st.session_state.recipes,
        "tree": st.session_state.vars,
        "targets": st.session_state.targets,
        "units": st.session_state.units,
    }


def check_recipe(recipe: dict, known: set, frame: pd.DataFrame):
    """Can this derived variable still be built? (status, detail)."""
    kind = recipe.get("kind")
    sources = recipe.get("sources") or [recipe.get("source")]
    absent = [s for s in sources if s and s not in known]
    if absent:
        return "dropped", f"needs {', '.join(absent)}"

    if kind == "combine":
        source = recipe["source"]
        if source not in frame.columns:
            return "restored", "source is itself derived"
        present = set(frame[source].dropna().unique())
        wanted = [v for _, values in recipe["groups"] for v in values]
        gone = [str(v) for v in wanted if v not in present]
        if len(gone) == len(wanted):
            return "dropped", "none of its responses are in this file"
        if gone:
            return "restored", f"no cases now for {', '.join(gone[:4])}"

    elif kind == "band":
        source = recipe["source"]
        if source in frame.columns and not pd.api.types.is_numeric_dtype(
            frame[source]
        ):
            return "dropped", f"{source} is not numeric here"

    return "restored", ""


def apply_project(project: dict) -> list:
    """Restore a saved project onto whatever file is loaded now.

    Returns a row per stored item saying whether it still applies, so a
    method can be carried to a new wave and the differences are stated
    rather than discovered later.
    """
    report = []
    schema = project.get("schema")
    if schema not in (1, PROJECT_SCHEMA):
        return [{
            "Item": "project file", "Status": "dropped",
            "Detail": f"schema {schema}, this app expects {PROJECT_SCHEMA}",
        }]

    # recipes first, in order: later ones may build on earlier ones
    recipes, known = [], set(base_df.columns)
    working = base_df
    for recipe in project.get("recipes", []):
        status, detail = check_recipe(recipe, known, working)
        report.append({
            "Item": f"variable: {recipe.get('name', '?')}",
            "Status": status, "Detail": detail,
        })
        if status == "restored":
            recipes.append(recipe)
            known.add(recipe["name"])
            working, _, _, _ = apply_recipes(
                base_df, base_var_labels, base_value_labels, recipes
            )
    st.session_state.recipes = recipes

    rebuilt, _, rebuilt_labels, problems = apply_recipes(
        base_df, base_var_labels, base_value_labels, recipes
    )
    for problem in problems:
        report.append({"Item": "variable", "Status": "dropped",
                       "Detail": problem})

    # the tree
    tree = {}
    for container, variables in (project.get("tree") or {}).items():
        where = "root" if container == ROOT else container.replace(STEP, " / ")
        kept = []
        for variable in variables:
            if variable in rebuilt.columns:
                kept.append(variable)
            else:
                report.append({
                    "Item": f"tree: {variable} in {where}",
                    "Status": "dropped", "Detail": "not in this file",
                })
        if kept:
            tree[container] = kept
    st.session_state.vars = tree

    # targets, only where the categories still line up
    targets, units = {}, {}
    saved_units = project.get("units") or {}
    for key, values in (project.get("targets") or {}).items():
        container, _, variable = key.partition("::")
        if variable not in tree.get(container, []):
            continue
        where = "root" if container == ROOT else container.replace(STEP, " / ")
        path = path_from_key(container)
        try:
            scope = rebuilt[scope_mask(rebuilt, path)]
            cats = declared_categories(rebuilt_labels, scope, variable)
        except Exception as exc:  # noqa: BLE001
            report.append({
                "Item": f"targets: {variable} in {where}",
                "Status": "dropped", "Detail": f"cannot be placed ({exc})",
            })
            continue

        if len(cats) == len(values):
            targets[key] = values
            units[key] = saved_units.get(key, "Percentages")
            report.append({
                "Item": f"targets: {variable} in {where}",
                "Status": "restored", "Detail": "",
            })
        else:
            report.append({
                "Item": f"targets: {variable} in {where}",
                "Status": "dropped",
                "Detail": f"{len(values)} targets saved, "
                          f"{len(cats)} categories here",
            })
    st.session_state.targets = targets
    st.session_state.units = units

    st.session_state.expanded = set()
    forget_pickers()
    return report


# ------------------------------------------------------- variable creator

def drop_variable_everywhere(name: str):
    """Remove a derived variable from the tree when its recipe goes."""
    st.session_state.vars = {
        container: [v for v in variables if v != name]
        for container, variables in st.session_state.vars.items()
    }
    forget_pickers()


def reset_creator():
    """Put the creator back to a blank panel after building a variable.

    Streamlit widgets remember their own value under their key, so the only
    dependable reset is to give them new keys. Bumping the generation does
    that for every control in the panel, including the expander itself,
    which therefore comes back closed.
    """
    for key in list(st.session_state):
        name = str(key)
        if name.startswith(("cmb_", "bnd_", "ilk_", "creator_kind")):
            del st.session_state[key]
    st.session_state.creator_gen += 1


def combine_state(source: str):
    """Working set of groups for one source variable, one group per response."""
    state = st.session_state.get("cmb_state")
    if not state or state.get("source") != source:
        cats, _ = cats_and_counts(df, source)
        state = {
            "source": source,
            "groups": [{"label": str(c), "values": [c]} for c in cats],
            "version": 0,
        }
        st.session_state.cmb_state = state
    return state


def creator_combine():
    """Group the responses of one variable by selecting rows and merging."""
    source = st.selectbox(
        "Variable to group", options=usable, index=None,
        placeholder="Choose a variable", key=f"cmb_src_{GEN}",
    )
    if not source:
        return

    state = combine_state(source)
    groups = state["groups"]

    st.caption(
        "Tick the responses that belong together and press Group selected. "
        "Rename a group by typing over its name."
    )

    edited = st.data_editor(
        pd.DataFrame({
            "": [False] * len(groups),
            "Group": [g["label"] for g in groups],
            "n": [int(df[source].isin(g["values"]).sum()) for g in groups],
            "Built from": [
                ", ".join(str(v) for v in g["values"]) for g in groups
            ],
        }),
        key=f"cmb_grid_{GEN}_{source}_{state['version']}",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["n", "Built from"],
        column_config={
            "": st.column_config.CheckboxColumn(width="small"),
            "Group": st.column_config.TextColumn(width="medium"),
            "n": st.column_config.NumberColumn(format="%d", width="small"),
            "Built from": st.column_config.TextColumn(width="large"),
        },
    )

    # keep any renaming the user typed
    for group, label in zip(groups, edited["Group"]):
        group["label"] = (label or "").strip() or group["label"]

    picked = [i for i, tick in enumerate(edited[""]) if tick]

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Group selected", key=f"cmb_merge_{GEN}",
                 disabled=len(picked) < 2):
        merged = {
            "label": groups[picked[0]]["label"],
            "values": [v for i in picked for v in groups[i]["values"]],
        }
        state["groups"] = [
            merged if i == picked[0] else g
            for i, g in enumerate(groups) if i == picked[0] or i not in picked
        ]
        state["version"] += 1
        st.rerun()

    if b2.button("Ungroup selected", key=f"cmb_split_{GEN}", disabled=not picked):
        rebuilt = []
        for i, group in enumerate(groups):
            if i in picked and len(group["values"]) > 1:
                rebuilt += [{"label": str(v), "values": [v]}
                            for v in group["values"]]
            else:
                rebuilt.append(group)
        state["groups"] = rebuilt
        state["version"] += 1
        st.rerun()

    if b3.button("Drop selected", key=f"cmb_drop_{GEN}", disabled=not picked,
                 help="Those cases become missing on the new variable"):
        state["groups"] = [g for i, g in enumerate(groups) if i not in picked]
        state["version"] += 1
        st.rerun()

    if b4.button("Reset", key=f"cmb_reset_{GEN}"):
        st.session_state.pop("cmb_state", None)
        st.rerun()

    if not groups:
        st.warning("Every response has been dropped, so there is nothing to build.")
        return

    kept = sum(int(df[source].isin(g["values"]).sum()) for g in groups)
    dropped = int(df[source].notna().sum()) - kept
    if dropped:
        st.info(f"{dropped:,} cases are not in any group and would be missing.")

    name = st.text_input("New variable name", value=f"{source}_grp",
                         key=f"cmb_name_{GEN}")
    if st.button("Create variable", key=f"cmb_make_{GEN}", type="primary"):
        labels = [g["label"] for g in groups]
        problem = name_problem(name)
        if len(set(labels)) != len(labels):
            problem = "Two groups share a name - rename one of them."
        if problem:
            st.error(problem)
        else:
            st.session_state.recipes.append({
                "kind": "combine", "name": name, "source": source,
                "groups": [[g["label"], list(g["values"])] for g in groups],
            })
            reset_creator()
            st.rerun()


def creator_interlock():
    """Cross two or more variables into one set of cells."""
    sources = st.multiselect(
        "Variables to cross", options=usable, key=f"ilk_src_{GEN}",
        help="Every combination becomes one category.",
    )
    if len(sources) < 2:
        st.caption("Pick at least two variables.")
        return

    complete = df[sources].notna().all(axis=1)
    cells = df.loc[complete, sources].astype(str).agg(SEP_X.join, axis=1)
    sizes = cells.value_counts()

    st.dataframe(
        pd.DataFrame({"Cell": sizes.index, "n": sizes.to_numpy()}),
        hide_index=True, width="stretch", height=240,
    )

    thin = int((sizes < BLOCK_NODE_N).sum())
    st.caption(
        f"{len(sizes)} cells with cases, smallest {int(sizes.min()):,}, "
        f"{int((~complete).sum()):,} cases missing on at least one variable."
    )
    if thin:
        st.warning(
            f"{thin} cell(s) under {BLOCK_NODE_N} cases. Interlocking spreads "
            "the sample thinly - consider grouping a source variable first, "
            "or weighting these variables separately as marginals."
        )

    name = st.text_input(
        "New variable name", value="_x_".join(sources)[:40], key=f"ilk_name_{GEN}"
    )
    if st.button("Create variable", key=f"ilk_make_{GEN}", type="primary"):
        problem = name_problem(name)
        if problem:
            st.error(problem)
        else:
            st.session_state.recipes.append({
                "kind": "interlock", "name": name, "sources": list(sources),
            })
            reset_creator()
            st.rerun()


def suggest_cuts(values: pd.Series, how: str, parts: int) -> list:
    """Cut points for equal width or equal count bands."""
    low, high = float(values.min()), float(values.max())
    if how == "Equal width":
        step = (high - low) / parts
        cuts = [round(low + step * i) for i in range(int(parts))]
    else:
        cuts = [round(values.quantile(i / parts)) for i in range(int(parts))]
    return sorted(dict.fromkeys(cuts))


def whole_numbers(values: pd.Series) -> bool:
    """True when a variable only holds whole numbers, so bounds can be
    inclusive without leaving anything between two bands."""
    present = values.dropna()
    return bool(len(present)) and bool((present % 1 == 0).all())


def creator_band():
    """Turn a numeric variable into a banded single-select."""
    options = numeric_vars(df)
    if not options:
        st.caption("No continuous numeric variables in this file.")
        return

    source = st.selectbox(
        "Numeric variable", options=options, index=None,
        placeholder="Choose a variable", key=f"bnd_src_{GEN}",
    )
    if not source:
        return

    values = pd.to_numeric(df[source], errors="coerce")
    low, high = float(values.min()), float(values.max())
    inclusive = whole_numbers(values)
    st.caption(
        f"Range {low:g} to {high:g}, {int(values.notna().sum()):,} cases "
        "with a value."
    )

    how = st.radio(
        "Bands",
        ["Custom", "Equal width", "Equal count"],
        horizontal=True,
        key=f"bnd_how_{GEN}",
    )
    parts = None
    if how != "Custom":
        parts = st.number_input("Number of bands", 2, 12, 4, key=f"bnd_parts_{GEN}")

    # Rebuild the table when the choice above it changes; leave the user's own
    # edits alone in between. Custom always starts empty.
    signature = (source, how, parts)
    state = st.session_state.get("bnd_state")
    if not state or state.get("signature") != signature:
        if how == "Custom":
            rows = [[None, None]]
        else:
            cuts = suggest_cuts(values, how, parts)
            rows = []
            for i, cut in enumerate(cuts):
                if i + 1 < len(cuts):
                    upper = cuts[i + 1] - 1 if inclusive else cuts[i + 1]
                    rows.append([cut, upper])
                else:
                    rows.append([cut, None])
        state = {
            "source": source, "signature": signature,
            "rows": [list(r) for r in rows],
            "version": (state or {}).get("version", 0) + 1,
        }
        st.session_state.bnd_state = state

    if inclusive:
        upper_col = "To"
        st.caption(
            "Each band includes both its From and its To. Leave the last To "
            "blank for an open ended band. Bands may have gaps - those cases "
            "become missing."
        )
    else:
        upper_col = "Up to (excl.)"
        st.caption(
            f"{source} has decimal values, so each band runs from its From up "
            "to, but not including, its upper bound. Leave the last one blank "
            "for an open ended band."
        )

    custom = how == "Custom"
    grid = {}
    if custom:
        grid[""] = [False] * len(state["rows"])
    grid["From"] = [r[0] for r in state["rows"]]
    grid[upper_col] = [r[1] for r in state["rows"]]

    edited = st.data_editor(
        pd.DataFrame(grid),
        key=f"bnd_grid_{GEN}_{state['version']}",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "": st.column_config.CheckboxColumn(width="small"),
            "From": st.column_config.NumberColumn(
                format="%g", help="Lowest value in this band"),
            upper_col: st.column_config.NumberColumn(
                format="%g",
                help=("Highest value in this band" if inclusive
                      else "First value of the next band"),
            ),
        },
    )
    state["rows"] = [[a, b] for a, b in zip(edited["From"], edited[upper_col])]

    if custom:
        picked = [i for i, tick in enumerate(edited[""]) if tick]
        a1, a2 = st.columns(2)
        if a1.button("Add band", key=f"bnd_add_{GEN}", icon=":material/add:"):
            state["rows"].append([None, None])
            state["version"] += 1
            st.rerun()
        if a2.button("Remove selected", key=f"bnd_del_{GEN}",
                     icon=":material/remove:",
                     disabled=not picked or len(picked) >= len(state["rows"]),
                     help="Cases in a removed range become missing"):
            state["rows"] = [
                r for i, r in enumerate(state["rows"]) if i not in picked
            ]
            state["version"] += 1
            st.rerun()

    # Each row becomes its own half-open interval internally, whatever the
    # bounds are called on screen. Bands need not touch; gaps are deliberate.
    filled = [
        (i, a, b) for i, (a, b) in enumerate(state["rows"])
        if not (
            (a is None or pd.isna(a)) and (b is None or pd.isna(b))
        )
    ]
    if not filled:
        st.info("Add a band to begin.")
        return

    last_filled = filled[-1][0]
    intervals, labels = [], []
    for index, lower, upper in filled:
        if lower is None or pd.isna(lower):
            st.error(f"Band {index + 1} needs a From value.")
            return
        lower = float(lower)
        if upper is None or pd.isna(upper):
            if index != last_filled:
                st.error(
                    f"Only the last band may be open ended - band "
                    f"{index + 1} needs an upper value."
                )
                return
            intervals.append([lower, None])
            labels.append(f"{lower:g}+")
        else:
            upper = float(upper)
            top = upper + 1 if inclusive else upper
            if top <= lower:
                st.error(
                    f"Band {index + 1}: the upper value must be "
                    + ("at least the From." if inclusive else "above the From.")
                )
                return
            intervals.append([lower, top])
            labels.append(f"{lower:g}-{upper:g}" if inclusive
                          else f"{lower:g} to <{upper:g}")

    starts = [i[0] for i in intervals]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        st.error("From values must increase down the table.")
        return

    overlaps = [
        i + 1 for i in range(len(intervals) - 1)
        if intervals[i][1] is not None
        and intervals[i][1] > intervals[i + 1][0]
    ]
    if overlaps:
        st.error(
            f"Band {overlaps[0]} overlaps the one below it. Bands may have "
            "gaps between them, but they cannot overlap."
        )
        return

    banded = pd.Series(None, index=df.index, dtype="object")
    for (lower, upper), label in zip(intervals, labels):
        top = float("inf") if upper is None else upper
        banded = banded.mask((values >= lower) & (values < top), label)
    sizes = banded.value_counts().reindex(labels).fillna(0).astype(int)

    st.dataframe(
        pd.DataFrame({"Band": labels, "n": sizes.to_numpy()}),
        hide_index=True, width="stretch",
    )

    gaps = []
    for i in range(len(intervals) - 1):
        top, next_start = intervals[i][1], intervals[i + 1][0]
        if top is not None and top < next_start:
            gaps.append(
                f"{top:g} to {next_start - 1:g}" if inclusive
                else f"{top:g} to {next_start:g}"
            )
    outside = int(values.notna().sum() - sizes.sum())
    if outside:
        detail = f"  Not covered: {', '.join(gaps)}." if gaps else ""
        st.warning(
            f"{outside:,} cases fall outside the bands and would be missing "
            f"on the new variable.{detail}"
        )

    name = st.text_input("New variable name", value=f"{source}_band",
                         key=f"bnd_name_{GEN}")
    if st.button("Create variable", key=f"bnd_make_{GEN}", type="primary"):
        problem = name_problem(name)
        if problem:
            st.error(problem)
        else:
            st.session_state.recipes.append({
                "kind": "band", "name": name, "source": source,
                "intervals": intervals, "labels": labels,
            })
            reset_creator()
            st.rerun()


def name_problem(name: str):
    name = (name or "").strip()
    if not name:
        return "Give the variable a name."
    if name in df.columns:
        return f"{name} already exists in this file."
    return None


# Loading belongs at the top: it is the first thing you do after uploading.
with st.expander("Load a project", expanded=False):
    st.caption(
        "Reuse a saved method - derived variable recipes, the tree, its "
        "targets and units. Every item is checked against this file and "
        "anything that no longer applies is reported rather than guessed at."
    )
    incoming = st.file_uploader(
        "Project file", type=["json"], key="project_upload",
        label_visibility="collapsed",
    )
    if incoming is not None:
        fingerprint = (incoming.name, incoming.size)
        if st.session_state.loaded_project != fingerprint:
            try:
                payload = json.loads(incoming.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that file: {exc}")
            else:
                st.session_state.project_notes = apply_project(payload)
                st.session_state.loaded_project = fingerprint
                st.rerun()

    report = st.session_state.project_notes
    if report:
        dropped = sum(1 for row in report if row["Status"] == "dropped")
        st.caption(
            f"{len(report)} item(s) checked against this file, "
            f"{len(report) - dropped} restored, {dropped} dropped."
        )
        st.dataframe(pd.DataFrame(report), hide_index=True, width="stretch")

GEN = st.session_state.creator_gen

with st.expander("Create a variable", expanded=False, key=f"creator_{GEN}"):
    for note in recipe_notes:
        st.warning(note)

    if st.session_state.recipes:
        st.caption("Derived variables")
        for index, recipe in enumerate(list(st.session_state.recipes)):
            row, action = st.columns([5, 1], vertical_alignment="center")
            row.markdown(
                f"**{recipe['name']}** - {var_labels.get(recipe['name'], '')}"
            )
            if action.button("Remove", key=f"rmrec_{index}",
                             icon=":material/delete:"):
                drop_variable_everywhere(recipe["name"])
                st.session_state.recipes.pop(index)
                st.rerun()
        st.divider()

    kind = st.radio(
        "What to build",
        ["Group categories", "Interlock", "Band a numeric"],
        horizontal=True,
        key=f"creator_kind_{GEN}",
    )
    if kind == "Group categories":
        creator_combine()
    elif kind == "Interlock":
        creator_interlock()
    else:
        creator_band()


# --------------------------------------------------------------- build tree

def build_nodes():
    """Depth-first walk of the whole tree.

    Every node is returned so state pruning sees the full picture; `visible`
    says whether a collapsed ancestor is hiding it from the tree.
    """
    out = []

    def walk(path, depth, visible):
        container = path_key(path)
        mask = scope_mask(df, path)
        scope = df[mask]

        for var in vars_at(container):
            if var not in df.columns:
                continue
            cats, counts = cats_and_counts(scope, var)
            base = int(scope[var].notna().sum())
            key = node_key(container, var)
            pct, _ = resolve(key, counts)
            expanded = key in st.session_state.expanded
            out.append({
                "kind": "variable", "id": key, "var": var, "path": path,
                "container": container, "depth": depth, "n": base,
                "cats": cats, "done": pct is not None, "visible": visible,
                "expanded": expanded, "parent": container,
            })
            for cat, n_cases in zip(cats, counts):
                if n_cases == 0:
                    continue   # nothing to weight inside an empty response
                child_path = path + [(var, cat)]
                child_key = path_key(child_path)
                n = int(scope_mask(df, child_path).sum())
                out.append({
                    "kind": "response", "id": child_key, "path": child_path,
                    "name": str(cat), "depth": depth + 1, "n": n,
                    "children": len(vars_at(child_key)),
                    "visible": visible and expanded, "parent": key,
                })
                if len(child_path) < MAX_DEPTH:
                    walk(child_path, depth + 2, visible and expanded)

    out.append({
        "kind": "root", "id": ROOT, "path": [], "depth": 0,
        "name": "All respondents", "n": len(df), "visible": True,
        "parent": None,
    })
    walk([], 1, True)
    return out


nodes = build_nodes()
by_id = {n["id"]: n for n in nodes}

if st.session_state.sel not in by_id:
    st.session_state.sel = ROOT

# Drop state belonging to branches that no longer exist.
live_containers = {ROOT} | {n["id"] for n in nodes if n["kind"] == "response"}
live_nodes = {n["id"] for n in nodes if n["kind"] == "variable"}
st.session_state.vars = {
    k: v for k, v in st.session_state.vars.items() if k in live_containers
}
for store in ("targets", "units", "versions"):
    st.session_state[store] = {
        k: v for k, v in st.session_state[store].items() if k in live_nodes
    }
st.session_state.expanded &= live_nodes


# ------------------------------------------------------------------ styling

INDENT = 20          # px per level of tree depth
RAIL = 9             # px from a level's left edge to its vertical rail

# The tree lines are drawn by us, so they follow the theme the user picked in
# Settings -> Appearance rather than assuming one.
try:
    DARK = st.context.theme.type == "dark"
except Exception:  # older Streamlit without st.context.theme
    DARK = True
GUIDE = "rgba(170, 180, 196, 0.80)" if DARK else "rgba(70, 82, 100, 0.55)"

visible_nodes = [n for n in nodes if n.get("visible")]

siblings = {}
for n in nodes:
    siblings.setdefault(n.get("parent"), []).append(n["id"])
last_child = {ids[-1] for ids in siblings.values()}


def ancestors_of(node):
    chain, parent = [], node.get("parent")
    while parent is not None:
        chain.append(by_id[parent])
        parent = by_id[parent].get("parent")
    return list(reversed(chain))


def row_style(index: int, node: dict) -> str:
    """Indent one row and paint its tree lines.

    Rails sit a fixed distance left of each level's content, and heights are
    percentages of the row, so they meet the rows above and below.
    """
    depth = node["depth"]
    rules = [f"padding-left: {depth * INDENT}px !important;"]
    if depth == 0:
        return f'div[class*="st-key-row_{index}"] {{ {" ".join(rules)} }}'

    images, positions, sizes = [], [], []

    def line(x_px, y, w, h):
        images.append(f"linear-gradient({GUIDE}, {GUIDE})")
        positions.append(f"{x_px}px {y}")
        sizes.append(f"{w} {h}")

    # carry on the rail of every ancestor that still has siblings below it;
    # that rail sits at the ancestor's own stem, one level left of it
    for ancestor in ancestors_of(node):
        a_depth = ancestor["depth"]
        if a_depth < 1 or a_depth >= depth:
            continue
        if ancestor["id"] in last_child:
            continue
        line((a_depth - 1) * INDENT + RAIL, "0", "1.5px", "100%")

    # this row's own stem, stopping at the elbow when it is the last child
    x = (depth - 1) * INDENT + RAIL
    line(x, "0", "1.5px", "50%" if node["id"] in last_child else "100%")
    line(x, "50%", f"{INDENT - RAIL - 1}px", "1.5px")   # elbow into the label

    rules += [
        f"background-image: {', '.join(images)} !important;",
        f"background-position: {', '.join(positions)} !important;",
        f"background-size: {', '.join(sizes)} !important;",
        "background-repeat: no-repeat !important;",
    ]
    return f'div[class*="st-key-row_{index}"] {{ {" ".join(rules)} }}'


# The gap rules have to hit the flex containers themselves, not only their
# descendants, or Streamlit's own row spacing breaks the lines up.
st.markdown(
    """
    <style>
    div[class*="st-key-treebox"],
    div[class*="st-key-treebox"] > div[data-testid="stVerticalBlock"],
    div[class*="st-key-row_"],
    div[class*="st-key-row_"] div[data-testid="stVerticalBlock"],
    div[class*="st-key-row_"] div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        row-gap: 0 !important;
    }
    div[class*="st-key-row_"] {
        min-height: 30px;
        margin: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[class*="st-key-row_"] div[data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[class*="st-key-row_"] button {
        padding: 0.18rem 0.5rem !important;
        min-height: 0 !important;
        line-height: 1.3 !important;
        border-radius: 7px !important;
    }
    div[class*="st-key-row_"] button p {
        font-size: 0.86rem !important;
        letter-spacing: 0.005em;
    }
    div[class*="st-key-row_"] button:hover {
        background: rgba(128, 128, 128, 0.12) !important;
    }
    div[class*="st-key-row_"] div[data-testid="stHorizontalBlock"]
        > div:has(div[class*="st-key-tg_"]) {
        flex: 0 0 24px !important;
        width: 24px !important;
        min-width: 24px !important;
    }
    div[class*="st-key-tg_"] button {
        padding: 0.18rem 0 !important;
        width: 24px !important;
        min-width: 24px !important;
        justify-content: center !important;
    }
    div[class*="st-key-tg_"] button p {
        font-size: 0.95rem !important;
        opacity: 0.75;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<style>"
    + "".join(row_style(i, n) for i, n in enumerate(visible_nodes))
    + "</style>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- tree + panel

left, right = st.columns([1, 2], gap="large")

with left:
    st.caption("Weighting tree")

    tree = st.container(key="treebox")
    with tree:
        for i, node in enumerate(visible_nodes):
            with st.container(key=f"row_{i}"):
                expandable = node["kind"] == "variable" and node["cats"]

                # only expandable rows get a toggle column, so response rows
                # sit directly against their elbow
                if expandable:
                    toggle_col, label_col, count_col = st.columns(
                        [0.3, 3.1, 1.1], vertical_alignment="center"
                    )
                    sign = "\u2212" if node["expanded"] else "+"
                    if toggle_col.button(sign, key=f"tg_{i}",
                                         type="tertiary"):
                        if node["expanded"]:
                            st.session_state.expanded.discard(node["id"])
                        else:
                            st.session_state.expanded.add(node["id"])
                        st.rerun()
                else:
                    label_col, count_col = st.columns(
                        [3.4, 1.1], vertical_alignment="center"
                    )

                if node["kind"] == "variable":
                    icon = (":material/check_circle:" if node["done"]
                            else ":material/radio_button_unchecked:")
                    label = node["var"]
                    count = "" if node["expanded"] else f"{len(node['cats'])} cats"
                elif node["kind"] == "response":
                    icon, label = None, node["name"]
                    count = f"{node['n']:,}"
                    if node["children"]:
                        word = "var" if node["children"] == 1 else "vars"
                        count += f"  ({node['children']} {word})"
                else:
                    icon, label, count = None, node["name"], f"{node['n']:,}"

                is_sel = node["id"] == st.session_state.sel
                if label_col.button(
                    label,
                    key=f"tn_{i}",
                    icon=icon,
                    type="secondary" if is_sel else "tertiary",
                ):
                    # selecting only - expanding is the toggle's job
                    st.session_state.sel = node["id"]
                    st.rerun()

                if count:
                    count_col.markdown(
                        "<div style='text-align:right; opacity:0.45; "
                        "font-size:0.76rem; padding-top:0.3rem; "
                        f"font-variant-numeric:tabular-nums'>{count}</div>",
                        unsafe_allow_html=True,
                    )

    if len(nodes) == 1:
        st.caption("Empty. Add a variable on the right to start.")

    constraints = sum(1 for n in nodes if n["kind"] == "variable")
    if constraints:
        st.caption(f"{constraints} target set(s) in the tree")

    if st.session_state.vars or st.session_state.recipes:
        st.download_button(
            "Save project",
            data=json.dumps(build_project(), indent=2, default=str),
            file_name="weighting-project.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
            help="The method only - reload it on another wave's file",
        )


def paste_panel(container_key: str, path: list):
    """Offer whatever is on the clipboard for pasting into this node."""
    clip = st.session_state.clipboard
    if not clip:
        return

    snap = clip["snapshot"]
    if snap["var"] in [v for v, _ in path]:
        return
    if snap["var"] in vars_at(container_key):
        return

    with st.container(border=True):
        has_children = bool(snap["children"])
        st.caption(
            f"Clipboard: {snap['var']}"
            + (f"  (from {clip['origin']})" if clip["origin"] else "")
        )
        c1, c2 = st.columns(2)
        with_targets = c1.checkbox(
            "Include targets", value=True, key=f"pt_{container_key}"
        )
        with_children = c2.checkbox(
            "Include sub-variables", value=has_children,
            disabled=not has_children, key=f"pc_{container_key}"
        )
        if st.button("Paste here", key=f"paste_{container_key}",
                     icon=":material/content_paste:"):
            report = []
            restore(container_key, path, snap, with_targets, with_children,
                    report)
            st.session_state.paste_report = report
            forget_pickers()
            for i in range(len(path)):
                st.session_state.expanded.add(
                    node_key(path_key(path[:i]), path[i][0])
                )
            st.rerun()

    for note in st.session_state.get("paste_report", []):
        st.warning(note)
    st.session_state.paste_report = []


def variable_picker(container_key: str, path: list, help_text: str):
    """Add or remove the weighting variables held at one container node."""
    used_on_path = [v for v, _ in path]
    current = vars_at(container_key)
    options = sorted(set(
        [c for c in usable if c not in used_on_path] + current
    ))
    chosen = st.multiselect(
        "Weighting variables",
        options=options,
        default=current,
        key=f"pick_{container_key}",
        help=help_text,
    )
    paste_panel(container_key, path)
    if chosen != current:
        st.session_state.vars[container_key] = chosen
        # opening a node deeper in the tree implies its ancestors are open
        for i in range(len(path)):
            ancestor = node_key(path_key(path[:i]), path[i][0])
            st.session_state.expanded.add(ancestor)
        st.rerun()


def target_grid(key: str, rows: list, observed: list):
    """Observed n, Observed %, Target - entered as percentages or counts.

    Categories with no cases are shown so the full response list is visible,
    marked and unable to take a target.
    """
    base = sum(observed)
    total_obs = base or 1
    empty = [r for r, n in zip(rows, observed) if n == 0]
    current_unit = st.session_state.units.get(key, "Percentages")

    unit = st.radio(
        "Targets as",
        ["Percentages", "Counts"],
        horizontal=True,
        index=0 if current_unit == "Percentages" else 1,
        key=f"unit_{key}",
    )
    if unit != current_unit:
        st.session_state.units[key] = unit
        set_targets(key, [None] * len(rows))
        st.rerun()
    st.session_state.units[key] = unit

    values = stored(key, len(rows))
    target_col = "Target n" if unit == "Counts" else "Target %"
    version = st.session_state.versions.get(key, 0)

    edited = st.data_editor(
        pd.DataFrame({
            "Category": [
                f"{r}  (no cases)" if n == 0 else str(r)
                for r, n in zip(rows, observed)
            ],
            "Observed n": observed,
            "Observed %": [round(o / total_obs * 100, 2) for o in observed],
            target_col: values,
        }),
        key=f"grid_{key}_{version}",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Category", "Observed n", "Observed %"],
        column_config={
            "Category": st.column_config.TextColumn(width="medium"),
            "Observed n": st.column_config.NumberColumn(
                format="%d", width="small"),
            "Observed %": st.column_config.NumberColumn(
                format="%.2f%%", width="small"),
            target_col: st.column_config.NumberColumn(
                min_value=0.0,
                format="%.2f" if unit == "Percentages" else "%.0f",
                width="small",
                help=("Must add to 100." if unit == "Percentages"
                      else f"Must add to the base here ({base:,})."),
            ),
        },
    )
    entered = list(edited[target_col])

    # anything typed against an empty category is dropped rather than obeyed
    stray = [
        str(r) for r, n, v in zip(rows, observed, entered)
        if n == 0 and v is not None and not pd.isna(v)
    ]
    if stray:
        entered = [
            None if n == 0 else v for n, v in zip(observed, entered)
        ]
    st.session_state.targets[key] = entered

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Use observed", key=f"copy_{key}",
                 help="Copy the achieved distribution into the targets"):
        set_targets(key, [
            None if n == 0
            else (float(n) if unit == "Counts" else round(n / total_obs * 100, 2))
            for n in observed
        ])
        st.rerun()
    if b2.button("Clear", key=f"clear_{key}"):
        set_targets(key, [None] * len(rows))
        st.rerun()

    if empty:
        st.info(
            "No cases for: " + ", ".join(str(e) for e in empty[:5])
            + ". Weighting cannot create respondents, so these take no target "
            "and the others carry the full 100%."
        )
    if stray:
        st.warning(
            "Ignored a target on " + ", ".join(stray[:3])
            + " - that category has no cases."
        )

    _, problem = resolve(key, observed)
    if problem:
        st.warning(problem)
    else:
        live = [
            float(v) for v, n in zip(st.session_state.targets[key], observed)
            if n > 0 and v is not None and not pd.isna(v)
        ]
        total = sum(live)
        st.caption(f"Total: {total:,.0f} of {base:,}" if unit == "Counts"
                   else f"Total: {total:,.2f}%")


with right:
    node = by_id[st.session_state.sel]

    if node["kind"] == "root":
        st.caption("All respondents")
        st.write(
            f"{len(df):,} cases. Variables added here are weighted across the "
            "whole sample."
        )
        variable_picker(ROOT, [], "Each becomes a node in the tree.")

    elif node["kind"] == "response":
        trail = " / ".join(f"{v}: {c}" for v, c in node["path"])
        st.caption(trail)
        st.write(
            f"{node['n']:,} cases. Variables added here are weighted only "
            "within this response."
        )
        if node["n"] < BLOCK_NODE_N:
            st.error(
                f"Only {node['n']} cases - too few to weight within. "
                "Combine categories first."
            )
        elif node["n"] < WARN_NODE_N:
            st.warning(
                f"Only {node['n']} cases - expect wide weights if you weight "
                "inside this response."
            )
        if len(node["path"]) < MAX_DEPTH:
            variable_picker(
                node["id"], node["path"],
                "Targets here are relative to this response, not the sample.",
            )
        else:
            st.info(f"Maximum depth of {MAX_DEPTH} variables reached.")

    else:  # variable node
        scope = df[scope_mask(df, node["path"])]
        cats, counts = cats_and_counts(scope, node["var"])
        trail = " / ".join(f"{v}: {c}" for v, c in node["path"])
        st.caption(f"{trail} / {node['var']}" if trail else node["var"])
        st.write(
            f"{node['n']:,} cases with a value"
            + (f", within {trail}." if trail else " across the sample.")
        )
        target_grid(node["id"], cats, counts)

        st.divider()
        copy_col, _ = st.columns([1, 2])
        if copy_col.button("Copy", key=f"cp_{node['id']}",
                           icon=":material/content_copy:",
                           help="Copy this variable, its targets and anything "
                                "beneath it"):
            trail = " / ".join(f"{v}: {c}" for v, c in node["path"])
            st.session_state.clipboard = {
                "snapshot": snapshot(node["container"], node["var"],
                                     node["path"]),
                "origin": trail or "all respondents",
            }
            st.rerun()

        below = sum(
            len(vars_at(path_key(node["path"] + [(node["var"], c)])))
            for c in node["cats"]
        )
        warn = (f" This also removes {below} variable(s) added under its "
                "responses." if below else "")
        st.caption(f"Remove {node['var']} from this node.{warn}")
        if st.button(f"Remove {node['var']}", key=f"rm_{node['id']}",
                     icon=":material/delete:"):
            remaining = [v for v in vars_at(node["container"])
                         if v != node["var"]]
            st.session_state.vars[node["container"]] = remaining
            st.session_state.expanded.discard(node["id"])
            forget_pickers()
            st.session_state.sel = node["container"]
            st.rerun()


# ------------------------------------------------------- flatten to margins

def collect(path: list, share: float, margins: dict, derived: dict,
            required: list, problems: list, order: list):
    """Walk the tree, turning each variable node into a raking margin.

    `share` is the node's share of the whole sample, so targets entered
    relative to a response become absolute before they reach the engine.
    """
    container = path_key(path)
    mask = scope_mask(df, path)
    scope = df[mask]

    for var in vars_at(container):
        cats, counts = cats_and_counts(scope, var)
        key = node_key(container, var)
        trail = " / ".join(f"{v}: {c}" for v, c in path)
        name = f"{trail} / {var}" if trail else var

        pct, problem = resolve(key, counts)
        if problem:
            problems.append(f"{name}: {problem}")
            continue

        required.append((mask, var))

        live = [(c, p) for c, p, n in zip(cats, pct, counts) if n > 0]
        order.append({
            "var": var, "path": list(path), "name": name,
            "targets": dict(live),
        })

        if not path:
            margins[var] = dict(live)
        else:
            col = f"_m{len(margins)}"
            derived[col] = (var, list(path), name)
            targets = {c: share * p / 100 for c, p in live}
            targets[OUTSIDE] = 100 - share
            margins[col] = targets

        for cat, p in live:
            child_path = path + [(var, cat)]
            if len(child_path) < MAX_DEPTH and vars_at(path_key(child_path)):
                n = int(scope_mask(df, child_path).sum())
                if n < BLOCK_NODE_N:
                    problems.append(
                        f"{name} / {cat}: only {n} cases, too few to weight "
                        "within"
                    )
                    continue
                collect(child_path, share * p / 100, margins, derived,
                        required, problems, order)


st.divider()

margins, derived, required, problems = {}, {}, [], []
order = []          # variable nodes in the order they appear in the tree
collect([], 100.0, margins, derived, required, problems, order)

if not margins and not problems:
    problems.append("No weighting variables yet.")

if problems:
    st.info("Not ready to run:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

if st.button("Run weighting", type="primary"):
    # Exclude cases missing on a variable that applies to them.
    keep = pd.Series(True, index=df.index)
    for mask, var in required:
        keep &= ~(mask & df[var].isna())

    # The engine only needs the columns the margins are built from, so the
    # working frame carries those rather than a copy of the whole dataset.
    needed = sorted(
        {v for v in margins if v in df.columns}
        | {var for var, _, _ in derived.values()}
        | {var for _, (var, path, _) in derived.items() for var, _ in path}
    )
    work = df.loc[keep, needed].copy()
    for col, (var, path, _) in derived.items():
        inside = scope_mask(work, path)
        work[col] = np.where(inside, work[var].astype("object"), OUTSIDE)

    try:
        scheme = wp.scheme_from_dict(margins)
        weighted = wp.weight_dataframe(work, scheme, weight_column="weights")
    except Exception as exc:  # noqa: BLE001 - surface engine errors to the user
        st.error(f"Weighting failed: {exc}")
        st.stop()

    # Only the weights are kept. Diagnostics and export read their columns
    # from the frame that is already in memory instead of a second copy.
    all_weights = pd.Series(np.nan, index=df.index, name="weights")
    all_weights.loc[keep] = weighted["weights"].to_numpy()

    st.session_state.result = {
        "all_weights": all_weights,
        "dropped": int((~keep).sum()),
        "weights": weighted["weights"],
        "checks": order,
        "scheme": {
            "margins": margins,
            "derived": {k: {"variable": v[0], "within": v[2]}
                        for k, v in derived.items()},
        },
    }


# ---------------------------------------------------------------- diagnostics

result = st.session_state.result
if result is None:
    st.stop()

weights = result["weights"]
eff = wp.weighting_efficiency(weights)
eff_base = weights.sum() ** 2 / (weights ** 2).sum()

if result["dropped"]:
    st.warning(
        f"{result['dropped']:,} cases were left unweighted - missing on a "
        "variable that applied to them."
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Efficiency", f"{eff:.1f}%")
m2.metric("Effective base", f"{eff_base:,.0f}")
m3.metric("Weight range", f"{weights.min():.2f} - {weights.max():.2f}")
m4.metric("Weighted cases", f"{len(weights):,}")

if eff < 70:
    st.warning(
        "Efficiency below 70% suggests the targets are a long way from the "
        "achieved sample, or that the tree is carrying more constraints than "
        "this sample can support."
    )

st.caption(
    "Each target set, in tree order. Weighted % should land on the target; "
    "where it doesn't, that constraint lost out to the others."
)

all_weights = result["all_weights"]
has_weight = all_weights.notna()

for entry in result["checks"]:
    var, path = entry["var"], entry["path"]
    # build a two column frame for this check rather than carrying a copy of
    # the whole dataset around
    columns = sorted({var} | {v for v, _ in path})
    scope = df.loc[has_weight, columns].assign(
        weights=all_weights[has_weight]
    )
    if path:
        scope = scope[scope_mask(scope, path)]
    cats, counts = cats_and_counts(scope, var)

    weight_sums = scope.groupby(var, observed=True)["weights"].sum()
    unw_base = sum(counts) or 1
    wtd_base = weight_sums.sum() or 1

    rows = []
    for cat, n in zip(cats, counts):
        w_n = float(weight_sums.get(cat, 0.0))
        unw_pct = n / unw_base * 100
        wtd_pct = w_n / wtd_base * 100
        target = entry["targets"].get(cat)
        rows.append({
            "Category": str(cat),
            "Unweighted %": round(unw_pct, 2),
            "Weighted %": round(wtd_pct, 2),
            "Target %": round(target, 2) if target is not None else None,
            "Diff pp": round(wtd_pct - unw_pct, 2),
        })

    st.markdown(f"**{entry['name']}**")
    st.caption(f"Base size: {sum(counts):,}")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Unweighted %": st.column_config.NumberColumn(format="%.2f%%"),
            "Weighted %": st.column_config.NumberColumn(format="%.2f%%"),
            "Target %": st.column_config.NumberColumn(format="%.2f%%"),
            "Diff pp": st.column_config.NumberColumn(
                format="%+.2f",
                help="Weighted minus unweighted - how hard the weights worked",
            ),
        },
    )

# Weights are exported against the respondent id only - a slim file to merge
# back onto the source data, rather than a second copy of it.
ids = id_candidates(df)
if st.session_state.id_var not in ids:
    st.session_state.id_var = None

st.markdown("**Export**")
if not ids:
    st.warning(
        "No variable in this file is both complete and unique, so weights "
        "cannot be exported against an identifier."
    )
id_var = st.selectbox(
    "Respondent identifier",
    options=ids,
    index=ids.index(st.session_state.id_var)
    if st.session_state.id_var in ids else None,
    placeholder="Choose the variable that identifies a respondent",
    key="pick_id",
    help="Only complete variables with one row per value are offered.",
)
st.session_state.id_var = id_var

if id_var:
    export = pd.DataFrame({
        id_var: as_id_series(df, id_var),
        "weight": all_weights.round(6),
    })
    blank = int(export["weight"].isna().sum())

    st.download_button(
        f"Download {id_var} + weight (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="weights.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption(
        f"{len(export):,} rows"
        + (f", {blank:,} with no weight (excluded from the run)."
           if blank else ".")
    )
