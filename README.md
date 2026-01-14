# PredictIQ  
_A Global, Intermittent-Demand Forecasting System with Probabilistic Outputs_

---

## Overview

PredictIQ is a production-grade demand forecasting system designed for **intermittent, sparse, and volatile retail demand**.  
It uses feature-engineered time-series data, gradient-boosted tree models, and a **two-stage forecasting architecture** to produce both **point forecasts** and **uncertainty intervals**.  
The project emphasizes **correctness, robustness, and interpretability** over leaderboard-only optimization, following practices similar to global ML approaches used in large-scale retail forecasting (e.g., M5).
---

## Problem Statement

Retail demand forecasting is challenging because:

- Many products exhibit **intermittent demand** (long stretches of zero sales).
- Demand patterns vary widely across products and categories. 
- Forecasting errors are concentrated around **rare but high-impact demand spikes**.   
- **Point forecasts alone** are insufficient for operational decisions such as safety stock and service levels. 
PredictIQ addresses these challenges by explicitly modeling **demand occurrence**, **demand magnitude**, and **uncertainty**.

---

## Core Contributions

### 1. Global Forecasting Architecture

- A **single global model** learns shared demand patterns across **thousands of products**, instead of training one model per item. 
- This scales better computationally and leverages cross-series information, reflecting how real retail forecasting systems are built for large hierarchies (e.g., M5’s 30k+ series). 
---

### 2. Leakage-Safe Time-Series Feature Engineering

The system reframes raw sales history into a **causal supervised learning problem** using only information available at prediction time:

- **Lag features**: 1, 7, 14, 28 days (recent history at multiple horizons).
- **Rolling statistics**: 7, 14, 28-day rolling means and other aggregates capturing local levels and volatility. 
- **Cyclical calendar features**: day-of-week encoded as sine/cosine to preserve periodic structure.   
- **Price and promotion context** where available (e.g., price levels, discount flags). 

All features are constructed using only **past information**, ensuring **no time leakage** from the future into the past, which is critical for valid backtesting.

---

### 3. Two-Stage Intermittent Demand Modeling (Key Innovation)

Instead of a single regression model, PredictIQ uses a **two-stage regime-separated architecture**, mirroring recent research on lumpy and intermittent demand.

**Stage 1 — Demand Occurrence (Classification)**  
A classifier estimates the probability of any demand:

P(demand>0)

**Stage 2 — Demand Magnitude (Regression)**  
A regressor predicts the expected demand conditional on it being non-zero:

E(demand | demand>0)

**Final Forecast**

y ​= P(demand>0) × E(demand | demand>0)

This decoupled formulation reduces error on **non-zero demand days**, which dominate business risk and inventory cost. 

---

### 4. Probabilistic Forecasting with Uncertainty Bands

PredictIQ produces **quantile forecasts** using gradient boosting with quantile loss (e.g., LightGBM quantile regression).

- **P10** — optimistic / low-demand scenario  
- **P50** — median / central forecast  
- **P90** — conservative / high-demand scenario  

These probabilistic forecasts enable:

- **Risk-aware inventory planning** (service levels and stock-outs).  
- **Safety stock estimation** from forecast distribution rather than a single point.
- **Decision-making under uncertainty**, aligned with modern probabilistic forecasting practice.
---

### 5. Honest & Diagnostic Evaluation

Models are evaluated using:

- **RMSE / MAE** – absolute error metrics at the series or aggregate level.  
- **SMAPE** – a scale-free error metric robust to different demand magnitudes.  
- **Non-zero-only metrics** – focusing on days with demand to isolate spike prediction quality, motivated by intermittent-demand literature.

Key empirical finding (from controlled experiments):

| Model                    | RMSE  | MAE  | Non-Zero MAE |
|--------------------------|------:|-----:|-------------:|
| Single-Stage (LightGBM) | ~11.9 | ~4.7 | ~10.9        |
| Two-Stage (PredictIQ)   | ~5.7  | ~1.9 | ~4.6         |

Roughly **>50% reduction in error on non-zero demand days**, validating the two-stage design for intermittent demand. 

---

## Interpretation of Uncertainty

Qualitative behavior of the probabilistic forecasts:

- For most products, **uncertainty bands are narrow**, reflecting high confidence that demand will remain at or near zero.
- A smaller subset of items exhibits **wide prediction intervals**, corresponding to volatile, high-risk products with more erratic demand. 
- At higher aggregation levels (e.g., category or total), **uncertainty naturally shrinks** due to diversification across items, matching standard portfolio and forecasting theory.

This behavior signals **well-calibrated probabilistic forecasts** that align with how real retail demand aggregates in practice.

---

## Dataset

**M5 Forecasting Dataset** (Walmart)

- Daily retail sales across **multiple categories and stores**, organized in a rich cross-sectional hierarchy.
- Known for **extreme sparsity and intermittency**, with many item-store series containing long zero stretches. 
- Widely used as a benchmark for global, ML-based forecasting methods.

Due to hardware constraints, experiments are conducted on a **deterministic, demand-diverse subset** of the full hierarchy, while the pipeline is architected to **scale horizontally** to all ~30k time series when resources allow.

---

## Design Decisions & Limitations

Several explicit design choices prioritize **correctness and causality**:

- **Long lag and rolling windows** enforce strict causality (only past information used) but **shorten per-series evaluation windows**, since early dates are consumed by feature construction.  
- As a result, evaluation and visualization focus on **aggregate and distributional behavior** (across items and horizons), rather than pretty per-series trajectories.  
- Classical **walk-forward cross-validation** becomes difficult under a heavily feature-lagged global model; instead, PredictIQ uses **hold-out horizon evaluation**, consistent with many M5-style global ML solutions.

These decisions favor **methodological honesty** over cosmetic plots, aiming to reflect real-world forecasting constraints.

---

## Tech Stack

- **Python** – core implementation language  
- **Pandas / NumPy** – data manipulation and feature engineering  
- **XGBoost / LightGBM** – gradient-boosted tree models (including quantile regression for probabilistic outputs). 
- **Scikit-learn** – metrics, model interfaces, and preprocessing utilities  
- **Matplotlib** – evaluation and diagnostics visualization  

Optional extensions may use experiment tracking and additional time-series utilities.

---

##  Future Work

Planned and potential extensions include:

- **Promotion & price-change features**: modeling exogenous drivers like discounts and marketing. 
- **Hierarchical reconciliation**: ensuring consistency between item-, category-, and total-level forecasts.
- **Probabilistic evaluation metrics** (e.g., CRPS) to quantify full-distribution forecast quality.   
- **Online / streaming updates** for near real-time demand adaptation in production environments. 

---

## Key Takeaway

PredictIQ shows that **explicitly handling intermittency**—through **regime separation** (occurrence vs. magnitude) and **uncertainty modeling**—has a larger impact on real-world forecasting quality than simply increasing model complexity.

The project emphasizes:

- **Engineering discipline** (leakage-safe pipelines, scalable global models)  
- **Diagnostic evaluation** (non-zero metrics, uncertainty behavior, aggregate patterns)  
- **Alignment with real-world constraints** in retail forecasting systems.

---

## Author Note

This repository is intended as a **research-grade applied ML system**, not a leaderboard-optimized M5 solution.  
Design decisions are motivated by **robustness, interpretability, and scalability**, mirroring practices used in production retail forecasting teams.
