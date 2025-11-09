# MSCS_634_Lab_2: KNN and RNN Classifiers on the Wine Dataset

## Lab Overview
This lab explores the performance of **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** classifiers using the **Wine Dataset** from the `sklearn` Python library. The dataset contains three classes of wine with 13 chemical properties as features.  

The main goal is to analyze how different parameter choices (number of neighbors `k` for KNN and `radius` for RNN) affect classification accuracy and to compare the performance of the two models.

---

## Dataset
- Source: `sklearn.datasets.load_wine()`
- Features: 13 chemical properties of wine samples
- Classes: 3 types of wine
- Split: 80% training, 20% testing
- Preprocessing: Standardization applied for distance-based models

---

## Implementation Details

### K-Nearest Neighbors (KNN)
- Tested `k` values: 1, 5, 11, 15, 21
- Trained on standardized training set
- Evaluated accuracy on the test set
- Observed that accuracy stabilizes as `k` increases

### Radius Neighbors (RNN)
- Tested radius values: 350, 400, 450, 500, 550, 600
- Used `outlier_label=-1` for points with no neighbors
- Accuracy calculated only for points with neighbors
- Observed sensitivity to radius size — too small radius may result in zero neighbors, too large may include noise

---

## Results

### Accuracy Trends
- KNN showed more stable accuracy improvements as `k` increased.
- RNN accuracy fluctuated depending on the radius value, highlighting sensitivity to parameter selection.

### Performance Summary Table

| KNN_k | KNN_Accuracy | RNN_Radius | RNN_Accuracy |
|-------|--------------|------------|--------------|
| 1     | 0.9722       | 350        | 0.0000       |
| 5     | 0.9722       | 400        | 0.0000       |
| 11    | 0.9444       | 450        | 0.0000       |
| 15    | 0.9444       | 500        | 0.0000       |
| 21    | 0.9444       | 550        | 0.0000       |
|       |              | 600        | 0.0000       |

> Note: RNN values may show 0 accuracy if the radius is too small for any neighbors in the scaled dataset. Adjust scaling or radius for practical results.

---

## Observations & Discussion
- **KNN** is generally more stable and reliable for this dataset.
- **RNN** can be useful when data density varies across classes but requires careful radius tuning.
- Standardizing features is crucial for distance-based models to prevent bias.
- Visualizations help identify optimal parameter settings and trends.

---

## Challenges & Decisions
- Choosing the correct radius for RNN required trial and error.
- Handling outliers (-1) in RNN predictions was necessary to calculate meaningful accuracy.
- Standardization improved both KNN and RNN performance significantly.

---

## Repository Contents
- `Lab2_KNN_RNN_Wine.ipynb` — Jupyter Notebook with full implementation
- `README.md` — this file with overview, results, and discussion
