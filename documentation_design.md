# DemandIQ --- Documentation Design Standard

## Goal

Make every project-facing document easy to understand for a new
engineer, recruiter, or interviewer.

Write in: - simple English, - active voice, - short sentences, -
professional daily-use language, - direct statements.

Avoid: - marketing language, - exaggerated claims, - academic filler, -
unnecessary jargon, - long introductory paragraphs.

## README

The README should answer these questions quickly:

1.  What problem does DemandIQ solve?
2.  What data does it use?
3.  What does the pipeline do?
4.  What models does it compare?
5.  How do we validate the models?
6.  What are the key results?
7.  What are the main limitations?
8.  How can another person run the project?

Recommended structure:

``` text
# DemandIQ

Short description

## Problem
## Approach
## Architecture
## Data
## Models
## Evaluation
## Results
## Business Findings
## Limitations
## Project Structure
## Setup
## Usage
```

Only include sections that contain useful information.

## Execution plan

The execution plan is the technical contract.

It should describe: - objective, - scope, - architecture, - data flow, -
methodology, - evaluation, - acceptance criteria, - final deliverables.

Do not fill it with implementation logs.

Do not use internal labels such as: - Phase 1, - Step 1, - Checkpoint 1.

Use professional workstream names.

## Reports

Reports must separate: - observed result, - interpretation, - business
implication, - limitation.

Do not write a conclusion before showing the evidence.

Every important number should identify: - metric, - dataset period, -
model, - comparison, - evaluation method.

## Result language

Prefer: \> XGBoost achieved lower WAPE than the seasonal baseline on the
validation period.

Avoid: \> XGBoost dramatically transformed forecasting performance.

Prefer: \> Promotion features improved validation accuracy under the
assumption that the planned promotion is known at forecast time.

Avoid: \> Promotions cause demand to increase.

## Figures

Every figure needs: - clear title, - readable axis labels, - units where
relevant, - useful legend, - short interpretation.

Do not create charts merely because a chart is possible.

## Tables

Keep tables focused.

Show only metrics or fields needed to support the decision.

## Reproducibility

Document: - setup, - dependencies, - data source, - commands, -
configuration, - expected outputs.

Never include credentials or secrets.

## Limitations

State important limitations clearly.

For this project, likely limitations include: - public historical
dataset, - limited product/category scope, - planned-promotion
availability assumption, - no direct inventory/lead-time/cost data, -
difficult intermittent-demand tail, - final conclusions limited to the
available data.

## Documentation quality gate

Before committing a document: - remove duplicate content, - remove
temporary notes, - check all commands, - check filenames, - check result
numbers, - avoid unsupported claims, - ensure a new reader can
understand the document without private chat context.
