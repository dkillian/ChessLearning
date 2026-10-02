# Activity Setup Guide

A reference for initializing and documenting a new activity within any workspace.

Last mod: May 14, 2026

---

## What Is an Activity?

An **activity** is a well-defined analytical or creative project living within a workspace. Each activity occupies its own folder and maintains its own documentation, session records, and code — separate from other activities.

---

## Step 0: Create the Workspace and AGENTS.md

When creating a new RStudio project, ask the assistant to create an `AGENTS.md` file at the project root. This file is not generated automatically by RStudio. It should include at minimum:

- A pointer to this Activity Setup Guide
- A project overview
- Coding style and conventions for the workspace

---

## Step 1: Create the Activity Folder

Create a named folder under the workspace root. Use a clear, descriptive name (e.g., `Presidential speeches`, `Conditional expectations as projections`).

Inside, establish the standard subfolder structure:

```
[Activity Name]/
├── scripts/    # All R (or other language) code files
├── data/       # Raw and processed data
├── viz/        # Visualizations
├── tables/     # Output tables
└── models/     # Saved models (if applicable)
```

Not all subfolders need to exist from the start — create them as needed.

---

## Step 2: Create the Three Documentation Files

Each activity maintains three files at the activity root (not in a subfolder):

### 1. `[ActivityName]Documentation.md`

The internal reference document. Written for the assistant and for your future self. Update it as the project evolves. Should include:

- **Overview** — What is this activity? What is the research or analytical question?
- **Methods** — What approach is being used?
- **Folder convention** — Pointer to the standard structure above
- **File locations** — A table of all key files: path, description, schema
- **Pipeline description** — How does the workflow run from raw data to output?
- **Custom schemes or codebooks** — Any classification systems, vocabularies, or codes developed during the work
- **Broader goals** — Long-term aims (e.g., generalizing to a full corpus)
- **Script naming convention** — See Step 3

This file may eventually double as a public-facing GitHub README if the project is shared — in that case, write a separate `README.md` with a different tone.

### 2. `[ActivityName]Log.md`

A structured, session-by-session record. Append a new entry at the end of each working session. Each entry should include:

```markdown
## Session N — [Date]

### What We Worked On
### What Was Decided
### What Was Created
### Risks & Uncertainties
### Steps for Next Session
```

The log is the primary tool for reconstructing context at the start of a new session.

### 3. `[ActivityName]Conversations.md`

A running transcript of assistant conversations. Captures the meander of the analytical process — decisions reconsidered, approaches tried and revised, questions asked. Useful for understanding *why* things are the way they are, not just *what* was done.

Format each session as:

```markdown
## Session N — [Date]

**User:** ...
**Assistant:** ...
```

---

## Step 3: Follow the Script Naming Convention

Scripts follow a numbered stage prefix:

| Prefix | Stage | Purpose |
|--------|-------|---------|
| `01_clean_*` | Cleaning | Parse, fix, segment, save clean objects to `data/` |
| `02_analyze_*` | Analysis | Load clean data, produce findings, save to `tables/` or `models/` |
| `03_report_*` | Reporting | Load findings, produce outputs in `viz/` or as Quarto documents |

All code produced in a session goes into a script unless explicitly noted otherwise. Scripts should be written for generalizability — using function arguments for file paths, not hard-coded values — so that workflows can scale from a test case to a full corpus or dataset.

---

## Step 4: Load Documentation at the Start of Each Session

At the start of a new session, ask the assistant to read both the Log and the Conversations file before proceeding. This reconstructs context efficiently:

> "Let's resume work on [activity name]."

The assistant will read the documentation files and summarize where things stand.

---

## Step 5: Close Each Session Properly

At the end of a session, ask the assistant to:

1. Append the session transcript to `[ActivityName]Conversations.md`
2. Append a new entry to `[ActivityName]Log.md`
3. Update `[ActivityName]Documentation.md` if new files, methods, or conventions were introduced

> "Let's close this session. Please save the transcript and update the log."

---

## Quick Start Prompts

Use these prompts in order when setting up a new activity. Copy and paste as needed.

**1. New workspace (if no AGENTS.md exists yet):**
> "I've created a new RStudio project called [name]. Please create an AGENTS.md file at the project root. The project is about [brief description]. Use the coding conventions from my other projects."

**2. New activity within an existing workspace:**
> "I'm starting a new activity called [name]. Please create the three documentation files — Documentation, Log, and Conversations — in the activity folder at [path]."

**3. Resume an existing activity:**
> "Let's resume work on [activity name]. Please read the Log and Conversations files and summarize where we left off."

**4. Close a session:**
> "Let's close this session. Please append the transcript to the Conversations file, add a new entry to the Log, and update the Documentation file if anything changed."

---

## Notes

- Keep one activity per folder. Do not mix unrelated work.
- Documentation files live at the activity root, not in `scripts/` or `data/`.
- The `viz/`, `tables/`, and `models/` subdirectories can be created on demand; they do not need to exist from day one.
- If an activity will be shared publicly on GitHub, write a separate `README.md` with a different audience in mind — the documentation file is for internal use.
