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
- Automated unit, integration, leakage, relational, and visualization test suite

The following stages are planned and not yet completed:
- Classical forecasting baselines
- XGBoost and LightGBM model training
- Time-based rolling validation and error forensics
- Promotion / sparse-demand / hierarchy analysis
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
│   ├── evaluation/
│   ├── hierarchy/
│   └── business/
├── tests/
└── reports/
    └── figures/
```
