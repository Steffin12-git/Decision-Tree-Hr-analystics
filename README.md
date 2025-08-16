# 🧭 Decision Tree — HR Attrition Analytics 

**Repository:** *Decision Tree HR Analytics*
**Notebook analysed:** `Decision tree hr analystics 2 remodeled.ipynb`

---

## 🎯 Project Objective

Develop a **Decision Tree pipeline** to predict **employee attrition (left vs. stayed)** using HR operational features, with a focus on:

* Robust preprocessing & hyperparameter tuning
* Balanced evaluation on imbalanced data
* Clear interpretability of results
* Business insights for HR retention strategies

---

## 📦 Dataset

* **Source file (in notebook):** `HR_comma_sep.csv`
* **Shape (notebook output):** `14,999 rows × 10 columns`
* **Features:**
  `satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, Department, salary`

**Attrition distribution (target = `left`):**

```
0    11428  (76.2%)
1     3571  (23.8%)
```

⚠️ Class imbalance (\~24% attrition) means metrics beyond accuracy are critical.

---

## 🔎 Exploratory Data Analysis (EDA)

### Distribution of numeric features

Key features such as `satisfaction_level`, `time_spend_company`, and `last_evaluation` show strong signals of separation between attrition and non-attrition groups.

![Numeric Histograms](images/histogram%20plot%20of%20numerical%20columns.png)

---

### Salary vs Attrition

Lower salary categories correlate with higher attrition risk.

![Salary vs Target](images/salary%20vs%20target.png)

---

### Correlation Heatmap

`satisfaction_level` and `time_spend_company` emerge as the strongest predictors of attrition.

![Correlation Heatmap](images/Correlation%20metrics.png)

---

## 🔬 Preprocessing & Pipeline

* **Encoding:**

  * Department → One-Hot Encoding
  * Salary → Ordinal (`low < medium < high`)
* **Scaling:** StandardScaler for numerical features
* **Model:** Decision Tree Classifier inside a `Pipeline`
* **Tuning:** GridSearchCV (StratifiedKFold, 5-fold) over depth, min\_samples, and pruning (`ccp_alpha`)

---

## 🔁 Hyperparameter Tuning

* **Grid search combinations tested:** 768
* **Best CV f1\_macro:** `0.9677`
* **Best parameters:**

  * `max_depth = 7`
  * `min_samples_split = 2`
  * `ccp_alpha ≈ 0.0039`

**Cross-validation results (mean ± std):**

```
Accuracy:    0.9777 ± 0.0028
F1-macro:    0.9694 ± 0.0037
ROC-AUC:     0.9694 ± 0.0024
```

---

## ✅ Final Test Results

**Metrics (test set):**

```
Accuracy:   97.47%
Precision:  92.9%
Recall:     96.8%
F1-score:   94.8%
```

📌 Interpretation: The model **catches nearly all employees likely to leave (high recall)**, while maintaining strong precision.

---

## 📊 Evaluation Visuals

### Confusion Matrix

![Confusion Matrix](images/confusion%20metrics.png)

### ROC Curve

![ROC Curve](images/ROC%20curve.png)

### Precision-Recall Curve

![Precision-Recall](images/Precision-recall%20curve.png)

### Calibration Plot

![Calibration Plot](images/calibration%20plot.png)

---

## 🔑 Feature Importance

Top drivers of attrition from the tuned Decision Tree:

| Feature                | Importance |
| ---------------------- | ---------: |
| `satisfaction_level`   |       0.44 |
| `time_spend_company`   |       0.31 |
| `last_evaluation`      |       0.12 |
| `average_montly_hours` |       0.08 |
| `number_project`       |       0.03 |
| `salary`               |       0.01 |

📌 **Business Insight:** Employee satisfaction and tenure dominate attrition risk — these should be the main levers for HR retention strategies.

![Feature Importances](images/top%20features%20that%20makes%20prediction.png)

---

## 📦 Pipeline Export

The entire pipeline (preprocessing + model) is saved as:

```
hr_attrition_decision_tree_pipeline.joblib
```

Usage:

```python
from joblib import load
pipeline = load("hr_attrition_decision_tree_pipeline.joblib")
preds = pipeline.predict(X_new)
probs = pipeline.predict_proba(X_new)[:, 1]
```

---

## ⚖️ Limitations

* **Imbalanced target:** handled with stratified CV, but requires ongoing monitoring in production.
* **Model choice:** Decision Trees are interpretable but can overfit. Ensembles (Random Forest, XGBoost) may provide further robustness.
* **Feature scope:** More signals (engagement scores, surveys, manager ratings) could improve accuracy and fairness.
* **Ethics & privacy:** HR data requires compliance with privacy regulations and bias audits before deployment.

---

## ▶️ Reproduce the Project

```bash
# Clone repo
git clone <your-repo-url>
cd Decision-Tree-Hr-analystics

# Install dependencies
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn seaborn matplotlib joblib

# Run the notebook
jupyter notebook "Decision tree hr analystics 2 remodeled.ipynb"
```

---

## 📂 Repository Structure

```
📦 Decision-Tree-Hr-analystics
 ┣ 📜 Decision tree hr analystics 2 remodeled.ipynb
 ┣ 📜 HR_comma_sep.csv
 ┣ 📜 hr_attrition_decision_tree_pipeline.joblib
 ┣ 📂 images
 ┃ ┣ calibration plot.png
 ┃ ┣ confusion metrics.png
 ┃ ┣ Correlation metrics.png
 ┃ ┣ histogram plot of numerical columns.png
 ┃ ┣ Precision-recall curve.png
 ┃ ┣ ROC curve.png
 ┃ ┣ salary vs target.png
 ┃ ┗ top features that makes prediction.png
 ┗ 📜 README.md
```

---

## ✨ Project Pitch

* Developed and tuned a **Decision Tree model** for HR attrition prediction (15k records).
* Achieved **97.5% test accuracy** with **F1 ≈ 0.95** for the attrition class.
* Delivered **interpretable insights** — satisfaction and tenure are the strongest attrition drivers.
* Exported a **ready-to-deploy pipeline** (`joblib`), with reproducible EDA, evaluation plots, and business interpretations.

📌 This project showcases end-to-end ML workflow mastery — **EDA, pipeline engineering, hyperparameter tuning, interpretability, visualization, and deployment readiness** — highly relevant for **Data Science & HR Analytics roles**.

---
