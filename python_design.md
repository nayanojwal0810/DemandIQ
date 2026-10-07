# DemandIQ --- Python Design Standard

## Goal

Write Python that is modular, readable, testable, deterministic, and
easy for another engineer to maintain.

## File naming

Use `snake_case` for every Python file.

Examples: - `data_loader.py` - `data_validator.py` -
`feature_builder.py` - `train_xgboost.py` - `evaluate_forecasts.py`

Never use: - `DataLoader.py` - `trainModel.py` - `test2.py` -
`final_v2.py`

## Module design

Prefer small modules with one clear responsibility.

Good:

``` text
src/
├── ingestion/
│   └── data_loader.py
├── validation/
│   └── data_validator.py
├── features/
│   └── feature_builder.py
├── forecasting/
│   ├── xgboost_model.py
│   └── lightgbm_model.py
└── evaluation/
    └── forecast_metrics.py
```

Avoid one large script containing ingestion, feature engineering,
training, evaluation, and reporting.

## Functions

Keep functions small and focused.

A function should normally: - perform one meaningful operation, - have
explicit inputs, - return explicit outputs, - avoid hidden global
state, - be easy to test.

Do not create tiny functions that only wrap one obvious line unless the
name adds real meaning.

## Classes and OOP

Use classes only when they improve: - state management, - model
lifecycle, - configuration, - reusable interfaces, - or clear separation
of responsibilities.

Do not force OOP into simple transformations.

Functions are preferred for stateless operations such as: - metric
calculation, - feature transformation, - validation, - aggregation.

## Imports

Keep imports clean and deterministic.

Order them as: 1. standard library, 2. third-party packages, 3. project
modules.

Do not use wildcard imports.

## Configuration

Keep configuration deterministic.

Do not hard-code: - local absolute paths, - credentials, -
machine-specific directories, - random seeds in multiple places, -
hidden model parameters.

Use a single documented configuration approach.

Never commit secrets.

## Reproducibility

Set random seeds explicitly wherever randomness is used.

Record: - model parameters, - feature configuration, - training
period, - validation period, - test period, - random seed, - software
dependencies.

## Logging

Use Python's `logging` module.

Do not use casual progress prints such as:

``` python
print("done")
```

Use professional messages such as:

``` text
[INFO] Model training completed. WAPE: 12.18%
```

Log useful events: - data loaded, - validation completed, - feature
generation completed, - model training completed, - evaluation
completed, - errors and warnings.

Do not flood logs with row-level messages.

## Docstrings

Add concise docstrings to public functions, classes, and modules where
useful.

Document: - purpose, - important parameters, - return value, - important
assumptions.

Do not write essays inside docstrings.

## Comments

Write comments only when they explain: - a non-obvious decision, - a
leakage safeguard, - a business assumption, - or an implementation
constraint.

Do not comment obvious code.

## Error handling

Fail clearly when required inputs are invalid.

Do not silently: - drop unexpected data, - replace errors with zeros, -
ignore missing columns, - continue after schema failures.

Use explicit validation before expensive operations.

## Data handling

Do not mutate shared data unexpectedly.

Use explicit transformations.

Validate: - expected columns, - data types, - date ranges, - duplicate
keys, - missingness, - value constraints.

## Testing

Add tests for important logic, especially: - metric calculations, -
date/time feature logic, - lag/rolling feature generation, - leakage
safeguards, - hierarchy aggregation/reconciliation, - validation
utilities.

Tests should be deterministic.

## Notebooks

Do not create notebooks unless the execution plan explicitly requires
one.

The main project workflow should run through modular Python and SQL.

## Dependencies

Add a dependency only when the project actually needs it.

Do not add libraries for features that standard Python or existing
dependencies already provide.

Keep `requirements.txt` minimal and pinned where appropriate.

## Code completion standard

Before reporting a Python task as complete: - format/compile-check the
changed code, - run relevant tests, - verify imports, - inspect the
changed files, - remove temporary artifacts, - confirm deterministic
behavior where applicable.
