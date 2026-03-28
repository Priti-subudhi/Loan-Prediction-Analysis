# 🏦 Loan Prediction – Machine Learning Project

A binary classification machine learning project that predicts whether a loan application will be **approved or rejected**, based on applicant demographics, financial profile, and credit history. Built using **Logistic Regression** on a dataset of **614 historical loan records**, the model achieves **78.86% accuracy** with strong approval detection (98.75% recall). A **Power BI dashboard** (`Loan_prediction.pbix`) complements the model with interactive visual analytics for business stakeholders.

---

## 📁 Repository Structure

```
loan-prediction/
│
├── loan_prediction.ipynb       # Jupyter Notebook: EDA, preprocessing, ML model
├── Loan_prediction.pbix        # Power BI dashboard for visual analytics
├── loan_data_set.csv           # Raw dataset (614 records, 13 columns)
├── cleaned_loan_data.csv       # Preprocessed dataset (output of notebook)
└── README.md
```

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Total Records | 614 loan applications |
| Total Features | 13 columns (12 input + 1 target) |
| Target Variable | `Loan_Status` — Y = Approved (68.7%), N = Rejected (31.3%) |
| Class Distribution | ~422 Approved / ~192 Rejected |
| Missing Values | Present in 7 out of 13 columns |

### Feature Dictionary

| Feature | Type | Description | Missing |
|---|---|---|---|
| `Loan_ID` | ID | Unique application identifier (dropped) | 0 |
| `Gender` | Categorical | Male / Female | 13 (2.1%) |
| `Married` | Categorical | Yes / No | 3 (0.5%) |
| `Dependents` | Categorical | 0, 1, 2, 3+ | 15 (2.4%) |
| `Education` | Categorical | Graduate / Not Graduate | 0 |
| `Self_Employed` | Categorical | Yes / No | 32 (5.2%) |
| `ApplicantIncome` | Numerical | Primary applicant monthly income | 0 |
| `CoapplicantIncome` | Numerical | Co-applicant monthly income | 0 |
| `LoanAmount` | Numerical | Requested loan amount (in thousands) | 22 (3.6%) |
| `Loan_Amount_Term` | Numerical | Repayment term in months | 14 (2.3%) |
| `Credit_History` | Binary | Meets credit guidelines: 1 = Yes, 0 = No | 50 (8.1%) |
| `Property_Area` | Categorical | Urban / Semiurban / Rural | 0 |
| `Loan_Status` | Target | Approved (Y=1) / Rejected (N=0) | 0 |

### Numerical Feature Statistics

| Feature | Mean | Median | Min | Max | Std Dev |
|---|---|---|---|---|---|
| ApplicantIncome | 5,403 | 3,813 | 150 | 81,000 | 6,109 |
| CoapplicantIncome | 1,621 | 1,189 | 0 | 41,667 | 2,926 |
| LoanAmount (₹K) | 146 | 128 | 9 | 700 | 86 |
| Loan_Amount_Term (months) | 342 | 360 | 12 | 480 | 65 |
| Credit_History (mean) | 0.842 | — | 0 | 1 | — |

> **Note:** ApplicantIncome is heavily right-skewed — the median (₹3,813) is significantly lower than the mean (₹5,403), indicating most applicants are in the low-to-middle income bracket with a small number of high earners pulling the mean up.

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected dataset shape, dtypes, and summary statistics via `df.info()` and `df.describe()`
- Identified missing values across 7 features — **Credit_History** had the most (50 missing, 8.1%)
- Visualized distributions across Loan Status, Gender, Credit History, Property Area, and Income

### 2. Data Preprocessing

**Missing Value Treatment:**

| Column | Strategy | Missing Count | % of Total |
|---|---|---|---|
| Gender | Mode imputation | 13 | 2.1% |
| Married | Mode imputation | 3 | 0.5% |
| Dependents | Mode imputation | 15 | 2.4% |
| Self_Employed | Mode imputation | 32 | 5.2% |
| Credit_History | Mode imputation | 50 | 8.1% |
| LoanAmount | Mean imputation (₹146K) | 22 | 3.6% |
| Loan_Amount_Term | Mean imputation (342 months) | 14 | 2.3% |

**Label Encoding:**

| Feature | Encoding |
|---|---|
| Gender | Male = 1, Female = 0 |
| Married | Yes = 1, No = 0 |
| Education | Graduate = 1, Not Graduate = 0 |
| Self_Employed | Yes = 1, No = 0 |
| Property_Area | Urban = 2, Semiurban = 1, Rural = 0 |
| Loan_Status | Y (Approved) = 1, N (Rejected) = 0 |
| Dependents | `3+` standardized → integer `3` |

- `Loan_ID` dropped as a non-predictive identifier
- Final dataset after cleaning: **614 records × 12 features, 0 null values**

### 3. Model Training
- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Train / Test Split:** 80% / 20% → **491 training samples**, **123 test samples** (`random_state=42`)
- ⚠️ Convergence Warning: `lbfgs` solver failed to converge within 100 iterations — feature scaling and higher `max_iter` recommended

### 4. Model Evaluation

**Overall Accuracy: 78.86%** — 97 correct predictions out of 123 test samples

**Confusion Matrix:**

|  | Predicted: Rejected (0) | Predicted: Approved (1) |
|---|---|---|
| **Actual: Rejected (0)** | 18 ✅ True Negative | 25 ❌ False Positive |
| **Actual: Approved (1)** | 1 ❌ False Negative | 79 ✅ True Positive |

**Per-Class Performance Metrics:**

| Metric | Approved (Class 1) | Rejected (Class 0) |
|---|---|---|
| Precision | 75.96% | 94.74% |
| Recall (Sensitivity) | **98.75%** | 41.86% |
| F1-Score | 0.859 | 0.582 |

> **Key Finding:** The model detects approved applications with 98.75% recall (misses only 1 out of 80), but correctly identifies only 41.86% of actual rejections — 25 risky applicants were incorrectly approved (False Positives). Reducing this is the top priority.

### 5. Custom Prediction Example
```python
new_data = [[1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2]]
# Features: Male, Married, 0 Dependents, Graduate, Not Self-Employed,
#           Income ₹5000, Co-income ₹2000, Loan ₹150K, Term 360 months,
#           Good Credit History, Semiurban Area
prediction = model.predict(new_data)
# Output: [1] → APPROVED
```

---

## 📈 Key Insights from EDA

### 1. 🔑 Credit History — Strongest Predictor
- **84.2% of applicants** have a good credit history (Credit_History mean = 0.842 across 564 non-null records)
- Applicants with good credit are approved at dramatically higher rates
- Bad credit history is the clearest indicator of rejection

### 2. 🏘️ Property Area — Location Matters
- **Semi-Urban applicants have the highest loan approval rate** among the three area types
- Urban applicants submit the most applications but at a proportionally lower approval rate than Semi-Urban
- Rural applicants have the fewest applications and the most mixed outcomes

### 3. 💰 Income — Right-Skewed Distribution
- Applicant income spans ₹150 to ₹81,000/month (std dev = ₹6,109)
- **Median income (₹3,813) is ₹1,590 below the mean (₹5,403)** — a clear right skew
- The majority of applicants earn between ₹2,878 (25th percentile) and ₹5,795 (75th percentile)
- A small cluster of high earners (>₹20,000) creates a long right tail

### 4. ⚖️ Class Imbalance — 68.7% vs 31.3%
- Approximately **422 applications (68.7%) were approved** vs **192 (31.3%) rejected**
- This imbalance causes the model to be biased toward predicting approvals, leading to high false positive rates for rejections

### 5. 👤 Gender Distribution
- Male applicants dominate the dataset (~80% of records)
- Both approved and rejected counts are higher for males, largely reflecting sample composition rather than gender bias in decisions

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| Python 3.x | Core programming language |
| pandas | Data loading, cleaning, imputation, encoding |
| scikit-learn | Logistic Regression, train_test_split, accuracy_score, confusion_matrix |
| seaborn / matplotlib | Countplots, histogram with KDE curve |
| Power BI | Interactive business intelligence dashboard |
| Jupyter Notebook | End-to-end development environment |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn seaborn matplotlib jupyter
```

### Run the Notebook
```bash
git clone https://github.com/your-username/loan-prediction.git
cd loan-prediction
jupyter notebook loan_prediction.ipynb
```

> ⚠️ **Important:** Update the CSV file path in **Cell 1** to point to your local copy of `loan_data_set.csv` before running.

---

## 📌 Future Improvements

- [ ] Apply **StandardScaler / MinMaxScaler** to fix lbfgs convergence warning and likely boost accuracy
- [ ] Increase `max_iter` to ≥1000 in Logistic Regression
- [ ] Address **31.3% minority class imbalance** using SMOTE or `class_weight='balanced'`
- [ ] Improve **rejection recall from 41.86%** to reduce the 25 false positive approvals
- [ ] Evaluate ensemble models: **Random Forest, XGBoost, Gradient Boosting**
- [ ] Perform **k-fold cross-validation** for more reliable performance estimates
- [ ] Tune hyperparameters via **GridSearchCV** or **RandomizedSearchCV**
- [ ] Engineer new features: income-to-loan ratio, total household income (applicant + co-applicant)
- [ ] Deploy as a **REST API** (Flask / FastAPI) for real-time loan scoring

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Priti**
Feel free to raise issues or submit pull requests for improvements.

