# DemandIQ

**Promotion-aware retail demand forecasting using time-series analytics, XGBoost, and LightGBM across 118 SKUs.**

## Project Description

DemandIQ is a reproducible, leakage-safe retail demand forecasting system designed for grocery replenishment across 118 SKU daily time series from the UCI Hierarchical Sales dataset. The system evaluates whether planned promotion schedules improve next-day demand forecasts, benchmarks global gradient-boosted decision trees (XGBoost and LightGBM) against classical baselines, reconciles forecasts across a three-level hierarchy (SKU, Brand, Total), and analyzes performance on intermittent-demand items.

## Current Project Status

The following components are currently implemented and verified:
- Source data organization (`data/raw/`)
- Canonical SKU-day data ingestion (`data/processed/sku_demand_daily.csv`, `data/processed/sku_dimension.csv`)
- Deterministic data-contract validation and source-to-canonical mathematical reconciliation
- Development-period demand analytics (`demand_profile.csv`, `brand_summary.csv`, `promotion_summary.csv`)
- Leakage-safe forecasting feature construction (`data/processed/forecasting_features_development.csv`)
- PostgreSQL dimensional modeling, staging COPY, and analytical SQL warehouse (`sql/`, `src/sql/`)
- Exploratory data visualization and demand storytelling figures (`src/visualization/`, `reports/figures/`)
- Classical forecasting baselines and chronological validation (`src/forecasting/`, `src/evaluation/`, `data/processed/baseline_metrics.csv`, `data/processed/baseline_winner_summary.csv`)
- XGBoost & LightGBM global forecasting models (promotion-aware and promotion-unaware variants with inner chronological early-stopping) (`src/models/`, `data/processed/ml_metrics.csv`, `data/processed/ml_winner_summary.csv`)
- Chronological robustness evaluation and expanding-window rolling historical validation across 8 quarterly folds (`src/evaluation/rolling_validation.py`, `src/evaluation/robustness_analysis.py`, `data/processed/rolling_validation_summary.csv`, `data/processed/model_robustness_summary.csv`)
- Model rank stability, brand/SKU robustness analysis, and 2017 within-year quarterly evaluation (`data/processed/ml_2017_quarterly_metrics.csv`, `reports/figures/`)
- Promotion-condition and training-only sparse-demand analysis (`src/business/promotion_sparse_analysis.py`, `data/processed/promotion_condition_metrics.csv`, `data/processed/promotion_model_comparison.csv`, `data/processed/sparse_demand_metrics.csv`, `data/processed/sparse_demand_summary.csv`, `data/processed/promotion_sparse_metrics.csv`)
- Automated unit, integration, leakage, relational, visualization, baseline, tree model, and robustness evaluation test suite

The following stages are planned and not yet completed:
- Hierarchical reconciliation across levels (SKU, Brand, Total)
- Final blind 2018 holdout evaluation
- Business decision recommendations
- Final project documentation

## Planned Architecture

```text
UCI Hierarchical Sales Data
        ↓
Data Ingestion & Integrity Checks
        ↓
Canonical SKU-Day Data
        ↓
Demand Analytics & Visualization
        ↓
Leakage-Safe Forecasting Features
        ↓
SQL Analytics / Data Modeling
        ↓
Forecasting Baselines
        ↓
XGBoost + LightGBM
        ↓
Time-Based Evaluation
        ↓
Promotion / Sparse Demand / Hierarchy Analysis
        ↓
Business Decision Support
```

## Repository Structure

```text
DemandIQ/
├── README.md
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
│   ├── sql/
│   ├── visualization/
│   ├── forecasting/
│   ├── models/
│   ├── evaluation/
│   ├── hierarchy/
│   └── business/
├── tests/
└── reports/
    └── figures/
```
