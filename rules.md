# DemandIQ --- Working Rules

## Purpose

These rules define how we build, review, and maintain DemandIQ. They
apply to every implementation task.

## Roles

### ChatGPT --- Technical Lead

-   Own the architecture, technical decisions, methodology, evaluation,
    and acceptance criteria.
-   Write the implementation task for Antigravity.
-   Evaluate Antigravity's work independently.
-   Reject weak, unnecessary, or unsupported implementation choices.
-   Do not change a frozen decision without documenting the reason and
    getting user approval.

### Antigravity --- Worker

-   Implement only the requested task.
-   Do not act as the technical decision maker.
-   Inspect the repository before changing anything.
-   State the files it will create or modify and why.
-   Implement the smallest clean solution that satisfies the task.
-   Run relevant tests and static checks when they are safe and
    reasonably fast.
-   Report exactly what changed and what passed or failed.
-   Never wait for a known long-running command.
-   If a long-running command is required, prepare the exact command,
    explain why it is expected to take time, and stop.

### User --- Reviewer / Approver

-   Review the proposed decisions and results.
-   Run long commands locally when requested.
-   Return the command output to ChatGPT.
-   Approve major decisions before the next work package begins.

## Working cycle

1.  ChatGPT defines one focused work package.
2.  ChatGPT writes the exact Antigravity prompt.
3.  Antigravity inspects and implements.
4.  Antigravity validates the implementation.
5.  Antigravity reports the result.
6.  ChatGPT evaluates the result.
7.  User reviews and approves.
8.  Only then do we start the next work package.

Do not combine unrelated work merely to reduce the number of prompts.

## Long-running commands

Predict expensive operations before Antigravity starts them.

Examples: - downloading large datasets, - database imports, - full-data
feature generation, - model training, - rolling validation, - bootstrap
or statistical evaluation, - full test suites that are expected to run
for a long time.

Protocol:

1.  Antigravity prepares the code and exact command.
2.  Antigravity does not execute the expensive command if it is expected
    to take significant time.
3.  Antigravity gives the user the exact command.
4.  Antigravity stops.
5.  The user runs the command and returns the output.
6.  ChatGPT evaluates the output before the next task.

Never spend quota waiting for a command that the user can run directly.

## Repository discipline

Keep the repository clean at all times.

Allowed project files should have a clear purpose. Do not create: -
scratch scripts, - duplicate implementations, - debug dumps, - temporary
CSVs, - copied datasets, - abandoned notebooks, - `final_v2` or similar
duplicate files, - generated caches, - local environment folders, -
unnecessary dependencies.

Use `.gitignore` for local artifacts.

Before finishing a work package, inspect the repository and remove
anything created by the task that is not required.

## Decision discipline

Do not select a model, feature, metric, or method because we expect it
to win.

Use evidence.

Every important change must answer: - What changed? - Why? - What
evidence supports it? - What alternative did we consider? - What effect
does it have on the project?

Frozen decisions cannot be changed casually.

## Claims and results

Never inflate findings.

Do not claim: - causality from observational promotion data, - revenue
improvement without revenue data, - inventory-cost reduction without
inventory/cost data, - production readiness when the required production
infrastructure does not exist, - statistical significance without an
appropriate statistical test, - generalization beyond what the dataset
supports.

Report the actual result, uncertainty, limitations, and assumptions.

## Checkpoint communication

Before each work package, ChatGPT will give the user: - the objective, -
the decision being implemented, - what will change, - what success looks
like.

After completion, ChatGPT will give: - the key result, - an honest
verdict, - any problems, - the next work package.

GitHub-facing project documents should use professional workstream names
rather than internal labels such as "Phase 1", "Step 1", or "Checkpoint
1".

## Approval gates

Major decisions require user approval before implementation continues.

Examples: - changing the project objective, - changing the target
definition, - changing the final test-period policy, - adding a new
model family, - adding a new dataset, - adding a major technology, -
changing the evaluation methodology, - adding a production/deployment
component.

## Scope

The project is a business Data Science forecasting project.

Do not add technologies just to make the stack look larger.

The project should remain centered on: - demand analytics, - time-series
forecasting, - supervised ML, - rigorous validation, - promotion
analysis, - hierarchical consistency, - sparse-demand analysis, -
forecast error analysis, - business decision support.

PostgreSQL is supporting infrastructure, not the identity of the
project. Use it where it provides real analytical or data-management
value.

A presentation/dashboard layer is optional and must earn its place
through usefulness.

## Quality bar

Treat the project as a resume portfolio project that must survive a
technical interview.

Prefer: - simple methods that are correct, - reproducible workflows, -
clear evidence, - small focused modules, - meaningful tests, - honest
limitations.

Avoid: - technology collection, - unnecessary abstraction, - huge
hyperparameter searches, - unvalidated claims, - copied research
implementations, - decorative SQL.
