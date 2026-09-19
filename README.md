# RIM weighting demo

Upload an SPSS `.sav`, pick the variables to weight on, paste target
percentages from Excel, run rim weighting via `weightipy`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload `sample_data.sav` to try it.

## The interface

A tree on the left, and the selected node's panel on the right. There are
only two things you ever do:

- **Click a variable** to enter its targets, scoped to wherever it sits.
- **Click a response** to add further variables that apply only to the cases
  giving that response.

Variables added at the root are weighted across the whole sample. Variables
added under a response are weighted only within it. A tree of root-level
variables alone is plain unsegmented weighting, so there is no mode to
choose. You can add variables under some responses and not others, and
branches can be different depths, up to three variables deep.

## Targets

Each grid has three columns - Observed n, Observed % and Target - with a
toggle between percentages and counts. Paste a column from Excel and it
fills down the rows in order.

Targets are always relative to the node they sit in: age under North is
entered as percentages of North, and in counts mode must add to North's own
base, shown next to the running total.

Nesting here means *conditional targets*, not *interlocking*. An interlocked
age x gender target is a combined variable weighted as a single target,
which is a separate feature.

## How the tree reaches the engine

The tree is flattened into a single set of raking margins before it runs.
A root-level variable is a plain margin. A variable under a response becomes
a derived margin spanning the whole sample: it carries that variable's
categories inside the response and one lump category outside it, with
targets scaled by the response's own share of the sample.

This keeps conditional and unconditional targets in one rim run. A global
gender target can hold at 49/51 across the sample while age is separately
constrained within North - which a segmented scheme cannot express. Shares
multiply down the path, so a target three levels deep is scaled by the
product of its ancestors.

## Copying variables

Select a variable and press Copy. It takes the variable, its targets, and
any variables beneath its responses. Then select the node you want it in and
a clipboard panel appears, with switches for whether to bring the targets and
the sub-variables.

Paste refuses what it cannot honour rather than guessing, and says why:
targets are skipped when the categories differ in the new node, a variable
already used higher up the branch is rejected, and pasted counts are flagged
because the new node's base is different.

## Appearance

Light and dark both work - switch from the "..." menu at the top right,
under Settings, Appearance. `.streamlit/config.toml` sets the starting
theme. The tree's connector lines read the active theme so they stay legible
either way.

## Creating variables

The "Create a variable" panel above the tree builds three kinds of derived
variable, which then appear in the pickers like any other:

- **Group categories** - tick the responses that belong together and press
  Group selected. Rename a group by typing over its name; Ungroup puts it
  back; Drop excludes those cases from the new variable.
- **Interlock** - cross two or more variables into one set of cells, which is
  how you set joint targets (age x gender as cells) rather than the
  conditional targets the tree gives you.
- **Band a numeric** - Custom builds bands one at a time with your own From
  and To; Equal width and Equal count fill the table for you to edit.

  When the source holds only whole numbers, both bounds are inclusive, so
  18 to 24 means exactly that and bands read as they do in a published
  table. When the source has decimals, the upper bound is exclusive instead,
  since inclusive bounds would leave values between two bands unassigned.
  The column heading says which applies.

  Bands are independent intervals and need not touch. Remove one and those
  cases become missing rather than being folded into a neighbour. Gaps and
  cases falling outside are counted and named; overlaps are rejected. Leave
  the last upper value blank for an open ended band.

Definitions are stored as recipes rather than as columns, so they are rebuilt
from the source data on every run, can be replayed on the next wave's file,
and travel with the scheme. Recipes apply in order, so one can build on
another - band an age variable, then group the bands.

Removing a recipe also removes that variable from the tree, along with any
targets set on it.

## Categories shown

Grids list every category the file declares, in code order, not only the ones
with respondents. A declared response nobody picked appears marked
"(no cases)" with n = 0 and cannot take a target - raking multiplies the
cases that exist, so no factor turns zero respondents into a share. The
remaining categories therefore carry the full 100%, which the grid says out
loud rather than leaving you to infer.

Codes SPSS declares as user-missing are left out entirely. They arrive as
missing in the data and are not responses to weight towards.

## Exporting

Weights export as a two column file, the respondent id and the weight, ready
to merge back onto the source data. Pick the id under "Respondent id and
project file"; only variables that are complete and have one row per value
are offered. Whole-number ids are written without a decimal point so they
match on merge. Cases excluded from the run appear with a blank weight, and
the count is shown.

## Project files

A project file carries the method: derived variable recipes, the tree, its
targets and units. Nothing in it is tied to the file it was built on, and it
holds no respondent data, so the point of it is to reuse a scheme on the next
wave. The source name and save date are recorded as provenance only and are
never checked.

Weights are not stored. They would go stale against an edited tree with no
way to tell. Load the project, press Run, and the same numbers come back.

Load a project from the panel at the top of the page, which opens by itself
while the tree is still empty. Save one with the button under the tree, as
soon as there is anything worth keeping.

Loading checks every stored item against the file that is open and reports
each one as restored or dropped, with the reason: a recipe whose source
variable is gone, a tree variable not in this file, a target set whose
category count no longer matches. Everything that still applies is kept, so
a scheme survives a wave that adds or removes variables.

The respondent identifier is not part of the project file. It belongs to the
data rather than the method, so it is chosen at export.

## Guardrails

- Every category needs a target; percentages must reach 100 and counts must
  reach the node's base before the run unlocks.
- Responses under 30 cases can't have variables weighted inside them; under
  100 they warn. Thin nodes are the main way nesting produces unusable
  weights.
- The tree shows a running count of target sets, since every conditional
  target is one more constraint competing in the same raking.
- Each target set is checked back within its own node after the run, so you
  can see whether a conditional target actually landed.
- Cases missing on a variable that applies to them are excluded and counted,
  never silently coerced.
- The scheme downloads as JSON alongside the weighted data.

## Not in this version

Interlocked targets, category combining, weight capping, scheme upload, and
`.sav` export. All additive on top of this structure.

`sample_data.sav` has gender, age band, region, urbanicity and social grade
for building trees, plus three numeric variables for the banding tool: `age`
(whole numbers, so inclusive bounds), and `income` and `tv_hours` (decimals,
so exclusive upper bounds). Gender carries a user-missing code and an unused
label, and income has 48 refusals, so the missing-data handling is visible
too.

## Sessions and memory

Each browser session has its own state, so two people using the app at once
never see each other's file, tree or results.

Labelled columns are read as pandas categoricals rather than text, which is
the difference between a large tracker fitting in memory and not: a 20,000 x
200 labelled file takes about 4MB this way against 89MB as strings. The app
also avoids copying the dataset - weighting runs on just the columns the
margins need, and only the weights are kept afterwards.

Parsed files are held in a process-wide cache keyed on the file's own bytes.
Nobody can reach another person's entry, but the entries do share memory, so
the cache is capped at three files and two hours (CACHE_TTL and
CACHE_ENTRIES at the top of app.py). Change file clears the session's copy
immediately.

## Before putting real data near it

This holds uploaded data in the server process for the session. On shared or
free hosting that means respondent data sitting on someone else's
infrastructure — get IT and data protection sign-off first, and add auth and
an explicit deletion policy before it goes anywhere near live client work.
