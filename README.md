# 🌸 Iris Flower Classification
### CodeAlpha Data Science Internship — Task 1 | May Batch 2026

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project trains and evaluates **three machine learning classifiers** to identify Iris flower species based on sepal and petal measurements. It is a complete, production-ready ML pipeline that covers **data exploration → preprocessing → model training → evaluation → visualization**.

---

## 📂 Project Structure

```
CodeAlpha__Iris-flowers1/
│
├── Iris.csv                  # Dataset (150 rows × 6 columns)
├── iris_classification.py    # 🔥 Main ML script (self-contained)
│
├── eda_scatter.png           # Scatter plots for all feature pairs
├── feature_boxplot.png       # Boxplots per feature, grouped by species
├── correlation_heatmap.png   # Feature correlation heatmap
├── confusion_matrices.png    # Confusion matrix for each model
├── model_comparison.png      # Bar chart comparing all model accuracies
│
└── README.md                 # You are here!
```

---

## 📊 Dataset Information

| Property        | Detail                              |
|----------------|--------------------------------------|
| Source          | UCI Machine Learning Repository     |
| File            | `Iris.csv`                           |
| Rows            | 150                                  |
| Columns         | 6 (Id, 4 features, 1 target)        |
| Target Classes  | 3 (setosa, versicolor, virginica)    |
| Samples/Class   | 50 (perfectly balanced)              |
| Missing Values  | None                                 |

### Feature Columns

| Column          | Description                    | Unit |
|----------------|--------------------------------|------|
| SepalLengthCm  | Length of the sepal            | cm   |
| SepalWidthCm   | Width of the sepal             | cm   |
| PetalLengthCm  | Length of the petal            | cm   |
| PetalWidthCm   | Width of the petal             | cm   |
| **Species**    | **Target label**               | —    |

---

## 🤖 Models Trained

| Model                   | Configuration                       |
|------------------------|--------------------------------------|
| K-Nearest Neighbors     | `n_neighbors=5`                     |
| Support Vector Machine  | `kernel='rbf'`, `random_state=42`   |
| Decision Tree           | `max_depth=5`, `random_state=42`    |

---

## 🏆 Model Results

| Model                   | Accuracy  |
|------------------------|-----------|
| K-Nearest Neighbors     | 96.67%   |
| Support Vector Machine  | 96.67%   |
| Decision Tree           | 96.67%   |

> ⭐ **Best Performing Model:** All three models achieve near-perfect classification on this dataset, showcasing the separability of Iris species by petal measurements.

---

## 🎨 Visualisations

| File                       | Description                                        |
|---------------------------|----------------------------------------------------|
| `eda_scatter.png`          | 2×3 grid of scatter plots for all 6 feature pairs |
| `feature_boxplot.png`      | 1×4 boxplots — distribution of each feature       |
| `correlation_heatmap.png`  | Heatmap showing inter-feature correlations        |
| `confusion_matrices.png`   | 1×3 confusion matrices with per-model accuracy   |
| `model_comparison.png`     | Bar chart comparing all 3 model accuracies        |

All plots use a **dark theme** (`#0A0A0F` background) with species-specific colour coding:
- 🟣 **Iris-setosa** → `#FF6B9D`
- 🔵 **Iris-versicolor** → `#00D4FF`
- 🟠 **Iris-virginica** → `#FFB347`

---

## ⚙️ Tech Stack

| Tool           | Version  | Purpose                        |
|---------------|----------|-------------------------------|
| Python         | 3.x      | Core language                  |
| pandas         | 2.x      | Data loading & manipulation    |
| numpy          | 1.x      | Numerical operations           |
| matplotlib     | 3.x      | Plotting engine                |
| seaborn        | 0.x      | Statistical visualizations     |
| scikit-learn   | 1.x      | ML models & preprocessing      |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/VilashAIPro/CodeAlpha__Iris-flowers1.git
cd CodeAlpha__Iris-flowers1
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Script
```bash
python iris_classification.py
```

### Expected Output
- Console prints for each pipeline stage (EDA → Preprocessing → Training → Evaluation)
- 5 `.png` plot files saved in the current directory
- Final summary with model comparison table

---

## 💡 Key Insights

1. **Iris-setosa is linearly separable** from the other two species using petal dimensions alone — all models classify it with 100% accuracy.

2. **Petal features are highly correlated** (PetalLength ↔ PetalWidth ≈ 0.96), making them the strongest predictors for species classification.

3. **Iris-versicolor and Iris-virginica** have overlapping feature ranges, making them slightly harder to distinguish — this is where model choice matters.

4. **SVM with RBF kernel** is theoretically the most robust for this task due to its ability to model non-linear class boundaries.

5. **All three models achieve ≥96% accuracy**, confirming that the Iris dataset is a well-suited benchmark problem for supervised classification.

---

## 👤 Author

**Vilash Kumar**  
Data Science Intern — CodeAlpha (May Batch 2026)  
GitHub: [@VilashAIPro](https://github.com/VilashAIPro)

---

## 📜 License

This project is licensed under the MIT License.

---

*Built with ❤️ as part of the CodeAlpha Data Science Internship Program*
