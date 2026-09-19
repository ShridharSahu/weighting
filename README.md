# Weighting

A web tool for rim weighting survey data. Upload an SPSS `.sav`, build a
weighting scheme, and export the weights.

## What it does

**Builds a weighting tree.** Add the variables you want to weight on and set
their targets. Under any response of a variable, you can add further
variables that apply only to the respondents who gave that answer — so age
can be weighted within the North without touching the rest of the sample.
A tree with no nesting is plain unsegmented weighting.

**Creates the variables you need.** Group a variable's responses into fewer
categories, cross two variables into one set of cells, or cut a numeric
variable into bands. These are stored as definitions rather than as columns,
so they rebuild from the source data every time.

**Takes targets as percentages or counts**, typed in or pasted from Excel.
Percentages must reach 100; counts must reach the base of the node they sit
in. Every category needs a target before a run will start.

**Reports what the weights did.** Efficiency, effective base, weight range,
and for each target set the unweighted and weighted distribution against the
target you asked for. Weighted percentages should land on the target; where
they don't, that constraint lost out to the others.

**Exports the respondent id and the weight**, ready to merge back onto the
source data.

**Saves the method to a project file.** Recipes, the tree and its targets,
with no respondent data. Load it against another wave and everything that
still applies is restored; anything that doesn't is reported with the
reason — a variable no longer in the file, a target set whose categories
have changed.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Things worth knowing

Weighting can only reweight the respondents you have. A category with no
cases takes no target — the grid shows it with `n = 0` and the remaining
categories carry the full 100%.

Nesting means conditional targets, not interlocking. Age weighted within the
North constrains only the North. For joint targets across every combination,
build an interlocked variable and weight on that instead.

Nodes under 30 cases can't be weighted within, and under 100 they warn.
Every conditional target is one more constraint competing in the same
raking, so the tree shows a running count of them.

Cases missing on a variable that applies to them are excluded from the run
and counted, never silently coerced.

Weights are not capped. An extreme weight shows up in the weight range in
the report, where it can be judged, rather than being trimmed behind the
scenes while the targets still appear to be met.

## Data handling

Uploaded files are held for the session only and are not written to
permanent storage. Each browser session is separate, so people using the
tool at the same time don't see each other's data. Parsed files sit in a
shared cache for at most two hours.

## Built with

[weightipy](https://github.com/Sky-UK/weightipy) for the rim algorithm,
[pyreadstat](https://github.com/Roche/pyreadstat) for reading SPSS files,
and [Streamlit](https://streamlit.io) for the interface.
