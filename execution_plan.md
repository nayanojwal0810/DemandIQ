# DemandIQ --- Execution Plan

## Project

**DemandIQ --- Promotion-Aware Hierarchical Demand Forecasting**

DemandIQ builds a practical retail demand forecasting workflow for 118
SKUs using the UCI Hierarchical Sales dataset. The project combines
demand analytics, time-series forecasting, XGBoost, LightGBM, promotion
analysis, hierarchical reconciliation, sparse-demand analysis, and
business decision support.

## Objective

Build and evaluate a leakage-safe next-day SKU demand forecasting
system.

The system must answer:

1.  How accurately can we forecast next-day demand for each SKU?
2.  Does known planned-promotion information improve forecast accuracy?
3.  How do global tree-based models compare with strong non-ML
    baselines?
4.  Which SKUs remain difficult to forecast?
5.  Can forecasts remain coherent across SKU, brand, and total levels?
6.  What practical forecasting policy should a retailer use?

Do not claim revenue or inventory-cost improvement because the dataset
does not contain the required financial and inventory fields.

## Dataset

Use the UCI Hierarchical Sales Data.

Known characteristics: - 118 daily SKU series. - 2014--2018 daily
observations. - Four pasta brands. - Promotion indicators. - Natural
three-level hierarchy. - No missing values in the published dataset. -
Small enough for normal CPU development.

The source and dataset schema must be verified during ingestion rather
than assumed from memory.

## Frozen evaluation boundary

Keep **2018 completely sealed** until the final blind evaluation.

Development and model decisions use the earlier period.

No: - tuning on 2018, - feature selection using 2018, - threshold
selection using 2018, - model selection using 2018, - repeated
inspection of 2018 followed by methodology changes.

The final test must be run once the methodology is frozen.

## Architecture

``` text
UCI Hierarchical Sales Data
            |
            v
      Raw Data Validation
            |
            v
     PostgreSQL / SQL Layer
     (only where useful)
            |
            v
     Canonical Sales Data
            |
            v
      Demand Analytics
            |
            v
   Leakage-Safe Feature Data
            |
       +----+----+
       |         |
       v         v
   Baselines   Tree Models
       |        XGBoost
       |        LightGBM
       +----+----+
            |
            v
      Time-Based Evaluation
            |
     +------+------+------+
     |             |      |
     v             v      v
Promotion      Hierarchy  Sparse
Analysis       Reconciliation Demand
     |             |      |
     +------+------+------+
            |
            v
       Error Analysis
            |
            v
     Business Recommendations
```

## Repository

Keep the repository minimal.

``` text
DemandIQ/
├── README.md
├── execution_plan.md
├── rules.md
├── python_design.md
├── documentation_design.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── sql/
│   ├── schema/
│   ├── staging/
│   ├── dimensions/
│   ├── facts/
│   └── analytics/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── features/
│   ├── forecasting/
│   ├── evaluation/
│   ├── hierarchy/
│   └── business/
├── tests/
└── reports/
    └── figures/
```

Do not create folders until they have a real purpose.

## Data layer

Start with: - source download, - source verification, - schema
inspection, - row/column checks, - date-range checks, - SKU and brand
checks, - promotion checks, - duplicate checks, - aggregation checks.

If PostgreSQL is used, use it for real data management and analytical
SQL. Do not add decorative database work.

The canonical analytical model may contain: - date dimension, -
SKU/brand dimension, - sales fact, - promotion information.

The exact schema must follow the verified source structure.

## Demand analytics

Analyze: - total demand over time, - SKU demand distribution, - brand
contribution, - zero-demand frequency, - volatility, - weekday
effects, - seasonal patterns, - promotion frequency, -
promotion/non-promotion demand patterns, - SKU-level heterogeneity.

Use these analyses to define the forecasting strategy.

Do not infer causal promotion effects from observational data.

## Forecast target

Primary target:

> Next-day SKU demand.

Construct the target only from information that would be available at
the forecast cutoff.

## Feature engineering

Candidate feature groups:

### Lag features

-   lag 1
-   lag 7
-   lag 14
-   lag 28

### Rolling features

-   rolling mean
-   rolling standard deviation
-   recent demand statistics

All rolling features must use past observations only.

### Calendar features

-   day of week
-   month
-   time position where justified

### Promotion features

Use target-date promotion only under the explicit operational assumption
that the planned promotion is known when the forecast is generated.

Also evaluate a historical-promotion-only scenario where appropriate.

Do not leak future sales through any feature.

## Baselines

At minimum compare: - simple historical baseline, - seasonal/day-of-week
baseline, - ETS or another strong classical benchmark.

The final baseline set should remain small and defensible.

The purpose is to prove whether ML adds value.

## ML models

Use exactly two primary tree-based ML models:

-   XGBoost
-   LightGBM

Do not add random forests, CatBoost, neural networks, or other model
families unless a later evidence-based decision explicitly approves
them.

Use global models across SKUs where the data supports that approach.

Model configuration must be reproducible.

Do not perform an unnecessarily large hyperparameter search.

## Validation

Use chronological time-based validation.

Development validation should use multiple expanding/rolling windows
where practical.

Never randomly split the time series.

Primary metrics: - WAPE - MAE

Use SKU-level metrics and portfolio-level metrics.

Report: - aggregate performance, - distribution across SKUs, -
difficult-SKU performance, - comparison with baselines.

Where statistical comparisons are used, select an appropriate paired
method and state its assumptions.

Do not call overlapping rolling windows "independent folds."

## Promotion analysis

Compare model performance with and without target-date promotion
information.

Report: - portfolio-level change, - SKU-level distribution, -
consistency across validation windows.

Phrase the result as predictive evidence under the planned-promotion
assumption.

Do not call it causal evidence.

## Hierarchical forecasting

The hierarchy is:

``` text
Total
├── Brand A
│   ├── SKU ...
│   └── SKU ...
├── Brand B
├── Brand C
└── Brand D
```

Do not assume hierarchy-derived ML features improve prediction.

Test hierarchical coherence separately from raw predictive feature
value.

Evaluate an appropriate reconciliation method, including MinT if
implementation and data support it.

Compare: - unreconciled forecasts, - coherent forecasts, - impact on SKU
and aggregate accuracy.

If MinT adds complexity without meaningful value, report that honestly.

## Sparse-demand analysis

Identify SKUs with high zero-demand frequency.

Evaluate whether the primary model performs differently for this group.

Use one focused intermittent-demand benchmark if it provides useful
evidence.

Do not create a large collection of intermittent-demand models.

The objective is to determine whether a separate policy is justified for
the difficult tail.

## Error forensics

Investigate: - largest absolute errors, - promotion-day errors, -
high-volatility SKUs, - intermittent SKUs, - systematic weekday
errors, - seasonal periods.

Separate: - model limitations, - data limitations, - operational
assumptions.

## Uncertainty

Where practical, estimate uncertainty around aggregate performance and
key comparisons.

Use bootstrap or another defensible method.

Do not make probabilistic claims that the method cannot support.

## Business decision layer

Translate results into practical recommendations such as: - default
forecasting approach, - promotion-aware forecasting policy, - treatment
of sparse SKUs, - hierarchy reporting policy, - monitoring
recommendations.

If a replenishment simulation is added, label it explicitly as a
scenario simulation based on stated assumptions. Do not present
simulated savings as observed business outcomes.

## Final blind evaluation

Before opening 2018: - freeze features, - freeze model configurations, -
freeze evaluation metrics, - freeze comparison rules, - freeze
reconciliation method, - freeze sparse-demand methodology.

Then evaluate on 2018.

Report: - baseline vs ML, - XGBoost vs LightGBM, - promotion impact, -
SKU-level distribution, - sparse-demand performance, - hierarchy
coherence, - major errors.

If the final result is weaker than validation, report it. Do not change
the methodology after seeing the test and call the changed result final.

## Final project output

The completed repository should provide:

1.  Reproducible source code.
2.  SQL/data layer where justified.
3.  Model training and evaluation workflow.
4.  Final forecast output.
5.  Evaluation results.
6.  Key figures.
7.  Business findings.
8.  Limitations.
9.  Clear README.
10. A concise final report suitable for technical review.

## Acceptance criteria

The project is complete only when: - the data pipeline is
reproducible, - important transformations are tested, - leakage controls
are verified, - baselines are established, - both XGBoost and LightGBM
are evaluated, - chronological validation is complete, - promotion
analysis is complete, - hierarchy treatment is evaluated, -
sparse-demand behavior is documented, - error analysis is complete, -
the 2018 blind test is performed after freeze, - final conclusions match
the evidence, - the repository contains no unnecessary files, - another
person can understand and run the project from the README.

## Scope rule

Do not add technology for appearance.

The project should demonstrate strong Data Science judgement: **business
problem → data → analytics → statistics → forecasting → ML → evaluation
→ business decision.**
