# SIH26184 — Distance-Metric Grouping of Cybercrime Complaint Patterns for Predictive Cash-Withdrawal Location Forecasting

**Student:** Lavanya Gupta · **Mapping ID:** SIH26184  
A reproducible research prototype that groups synthetic cybercrime complaints and forecasts likely cash-out regions. It is not a production policing or suspect-identification system.

## Architecture

```text
Synthetic / CSV complaints + confirmed withdrawals
                  |
          Preprocess + SQLite
                  |
       Feature engineering groups
   geographic | temporal | MO | categorical
                  |
       Euclidean / Cosine similarity
   KMeans | Agglomerative | DBSCAN diagnostics
                  |
       Cluster-conditioned forecaster
                  |
       Time holdout validation + ablation
                  |
          Streamlit dashboard / CSV / PDF
```

## Run

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
pytest -q
streamlit run app.py
python report.py
```

The first CLI run creates 10,000 complaints and linked withdrawal events in `data/raw/`, cleaned files and evaluation tables in `data/processed/`, a SQLite database at `data/cybercrime.sqlite`, and a fitted model at `models/pipeline.joblib`.

## Data schema

`complaints.csv`: `complaint_id, complaint_datetime, victim_city, victim_state, victim_lat, victim_lon, fraud_type, mo_description, amount_lost, bank_name, transaction_channel, number_of_hops`.

`confirmed_withdrawals.csv`: `complaint_id, atm_id, withdrawal_lat, withdrawal_lon, withdrawal_city, withdrawal_datetime, amount`.

To use real data, put schema-compatible CSV files in `data/raw/` (same filenames) before running `python main.py`. Ensure one confirmed withdrawal row per complaint for the current evaluation workflow. Real ingestion should include a documented data-quality and access-control review.

## Implemented objectives

1. Geographic (3D unit-sphere coordinates, state/hotspot encoding and nearest-hotspot distance), temporal (cyclic hour/day, weekend, time bucket, and incident-to-report delay when `fraud_datetime` is available), and MO (TF-IDF, fraud type, channel, bank, amount and hops) features are created in separately weighted feature groups.
2. Euclidean and cosine distance are compared using KMeans, Agglomerative clustering with precomputed distances, and DBSCAN. Silhouette, Davies–Bouldin, Calinski–Harabasz, cluster sizes, and an ablation table are generated. The dashboard map shows victim complaint coordinates coloured by cluster.
3. Cluster-conditioned location distributions provide top-k city scores, centroid/spread summaries, and a 1–6 hour synthetic time window. A chronological 80/20 holdout compares both metrics using Top-1/3/5 hit rate, precision/recall@5, mean haversine error, and ARI/NMI diagnostics.

## Outputs

- `algorithm_comparison.csv`: metric/algorithm diagnostics.
- `validation_metrics.csv`: temporal holdout forecast metrics.
- `ablation.csv`: geographic-only, temporal-only, MO-only and combined features.
- `summary.json`, `conclusion.txt`, `SIH26184_report.pdf`.
- `pipeline.joblib`: fitted feature builder and Euclidean/Cosine forecasters.

The printed Euclidean-vs-Cosine conclusion is generated from the actual validation output on each run. Do not treat a small synthetic-data score difference as evidence of real-world superiority.

## Limitations

- Synthetic hotspots and fraud-type associations are intentionally embedded to test whether the pipeline can recover signal; results will be optimistic and do not establish real-world predictive validity.
- The current baseline predicts city-level cash-out distributions, not a verified individual ATM. ATM identifiers are synthetic.
- Clustering metrics are internal diagnostics, not proof that clusters represent criminal networks. The ARI/NMI fields are reference-label diagnostics and should be interpreted with caution.
- Agglomerative and DBSCAN comparisons are bounded to a reproducible sample for memory/runtime reasons. The deployed predictor uses a KNN-style distance vote over cluster-conditioned distributions.
- The current baseline uses complaint time as its temporal signal; real data should distinguish incident time, report time, and transaction time.

## Ethical, safety and privacy notes

Use only lawfully obtained, appropriately authorized data. Remove direct identifiers and minimize location precision before analysis. Encrypt data at rest and in transit, apply role-based access, audit every query, define retention periods, and assess disparate impact across regions. Outputs are probabilistic investigative leads for analyst review only. They must never independently trigger enforcement, deny services, or establish guilt. Validate on representative, independently confirmed real-world data before any operational consideration.
