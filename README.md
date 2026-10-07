# DemandIQ

**Promotion-aware retail demand forecasting using time-series analytics, XGBoost, and LightGBM across 118 SKUs.**

## Project Description

DemandIQ is a reproducible, leakage-safe retail demand forecasting system designed for grocery replenishment across 118 SKU daily time series from the UCI Hierarchical Sales dataset. The system evaluates whether planned promotion schedules improve next-day demand forecasts, benchmarks global gradient-boosted decision trees (XGBoost and LightGBM) against classical baselines, reconciles forecasts across a three-level hierarchy (SKU, Brand, Total), and analyzes performance on intermittent-demand items.

## Current Project Status

The project is currently in the **foundation stage**. Repository structure, project control documents, coding standards, and minimal environment dependencies have been established. Data ingestion, modeling pipelines, and empirical evaluations have not yet commenced. No models have been trained and no forecast results exist yet.

## Architecture Overview

```text
UCI Hierarchical Sales Data
            |
            v
      Raw Data Validation
            |
            v
     PostgreSQL / SQL Layer
     (where analytically useful)
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
       Error Forensics
            |
            v
     Business Recommendations
```

## Execution Plan & Governance

Technical decisions, frozen evaluation boundaries, and development standards are defined in the project control documents:
- [execution_plan.md](execution_plan.md) — Technical project execution plan and acceptance criteria
- [rules.md](rules.md) — Working protocol and review rules
- [python_design.md](python_design.md) — Python modularity and code standards
- [documentation_design.md](documentation_design.md) — Documentation standards

## Repository Structure

```text
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
