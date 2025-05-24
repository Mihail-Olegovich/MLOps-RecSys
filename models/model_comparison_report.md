# Model Comparison Report

## Performance Metrics

| Model | Recall@40 |
|-------|-----------|
| LightFM without Features | 0.1180 |
| ALS with Features | 0.1430 |
| ALS without Features | 0.1410 |

## Analysis

The best performing model is **ALS with Features** with a Recall@40 of 0.1430.

Using item features with ALS improves Recall@40 by 0.0020 (1.4%).

ALS without features outperforms LightFM without features by 0.0229 (19.4%).
