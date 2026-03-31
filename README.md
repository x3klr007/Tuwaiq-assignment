# Medical Insurance Cost Analysis Pipeline

A production-grade data analysis and machine learning pipeline for predicting medical insurance costs. This project demonstrates industry-standard practices in data cleaning, feature engineering, statistical analysis, and model validation.

## Project Overview

This capstone project implements a complete ML pipeline across 5 phases:

| Phase | Description | Deliverable |
|-------|-------------|-------------|
| Phase 1 | Data Cleaning and Validation | `01_cleaning.ipynb` |
| Phase 2 | Feature Engineering and VIF Analysis | `02_features.ipynb` |
| Phase 3 | Statistical Analysis and EDA | `03_eda.ipynb` |
| Phase 4 | Modeling and Dashboard | `04_modeling.ipynb` |
| Phase 5 | Model Fairness Analysis | `05_fairness.ipynb` |

## Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Gradient Boosting |
| **R² Score** | 0.901 (90.1% variance explained) |
| **RMSE** | $4,134 |
| **Cross-Validation** | 0.853 ± 0.028 (5-fold) |
| **Fairness Test** | p = 0.451 (no bias detected) |

## Visualizations

The project includes 18 publication-quality visualizations:

### Distribution & Relationship Analysis
- **Charges by smoker status** - Shows dramatic cost difference ($34,684 vs $8,203)
- **Age vs charges scatter** - Linear relationship visualization
- **BMI by region** - Geographic health patterns
- **Charges heatmap** - Smoker/region interaction
- **Age distribution** - Demographics breakdown
- **KDE distributions** - Kernel density for age, BMI, charges

### Feature Engineering
- **Outlier comparison** - Before/after 99th percentile capping
- **Correlation matrix** - Feature relationships
- **Feature correlations** - Top predictors ranked
- **Interaction heatmaps** - Age×BMI, Smoker×Age, Region×Children effects

### Model Analysis
- **Model performance comparison** - Gradient Boosting vs Random Forest vs Linear
- **Feature importance** - What drives predictions
- **Residual diagnostics** - 6-panel analysis with Cook's distance
- **Bias analysis** - Fairness across demographic groups

### Advanced Visualizations
- **Surface plot** - 3D-style contours showing charges by age/BMI for smokers/non-smokers
- **Regional radar** - 5-metric comparison across 4 regions
- **Demographic flow** - Age, BMI, smoker status → charge categories
- **Executive dashboard** - 2x2 summary for stakeholders

## How to Run

### Step 1: Navigate to Project

```bash
cd submission_HAMAD_ALDHUBAYB
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- scipy==1.11.2
- statsmodels==0.14.0
- jupyter==1.0.0

### Step 4: Run Notebooks (In Order)

```bash
cd notebooks

# Phase 1: Data Cleaning
jupyter notebook 01_cleaning.ipynb

# Phase 2: Feature Engineering
jupyter notebook 02_features.ipynb

# Phase 3: EDA and Statistical Analysis
jupyter notebook 03_eda.ipynb

# Phase 4: Modeling and Dashboard
jupyter notebook 04_modeling.ipynb

# Phase 5: Model Fairness Analysis
jupyter notebook 05_fairness.ipynb
```

### Step 5: Verify Outputs

After running all notebooks, check the following outputs:

- `data/processed/` - Cleaned and engineered datasets
- `reports/` - Generated visualizations and analysis charts

## What We Did and Why

### Phase 1: Data Cleaning

**What we did:**
- Built a robust `clean_data()` function with median/mode imputation
- Capped outliers at the 99th percentile to preserve data while reducing skew
- Implemented 3 automated validation checks (schema, missing values, data integrity)
- Removed 1 duplicate row (1,338 → 1,337 records)

**Why:**
- Real-world data contains errors and extreme values that can distort model training
- Median imputation is robust against outliers compared to mean imputation
- Outlier capping at 99th percentile retains 99% of data while preventing extreme values from dominating
- Validation checks ensure data quality before downstream processing

### Phase 2: Feature Engineering

**What we did:**
- Applied One-Hot Encoding for categorical variables (sex, smoker, region)
- Created 8 domain-specific features:
  - `is_smoker` - Binary smoking indicator
  - `sex_male` - Binary gender indicator  
  - `bmi_category` - Ordinal BMI classification (underweight/normal/overweight/obese)
  - `age_bmi_risk` - Interaction: age × BMI (compound risk factor)
  - `bmi_smoker_interaction` - Interaction: BMI × smoking status
  - `family_size` - Children + 1 (household scale)
  - `age_squared` - Non-linear age effect
  - `bmi_squared` - Non-linear BMI effect
- Performed VIF (Variance Inflation Factor) analysis to detect multicollinearity
- Pruned features with VIF > 10 to reduce redundancy

**Why:**
- Machine learning models require numeric input; encoding converts categories to numbers
- Interaction features capture complex relationships (e.g., BMI impact differs by age)
- VIF analysis identifies correlated features that provide redundant information
- Removing high-VIF features improves model interpretability and stability

### Phase 3: EDA & Statistical Analysis

**What we did:**
- Implemented manual NumPy functions for mean, standard deviation, and z-score
- Verified manual implementations against NumPy/sklearn using `np.allclose()`
- Built manual cosine similarity with zero-vector guard
- Calculated conditional probabilities (e.g., P(charges > 15k | smoker))
- Created 6 professional visualizations

**Why:**
- Manual implementation demonstrates understanding of underlying mathematics
- Verification ensures correctness of custom implementations
- Zero-vector guard prevents division-by-zero errors in similarity calculations
- Conditional probabilities reveal actionable insights (smoking has massive cost impact)
- Visualizations communicate findings effectively to stakeholders

### Phase 4: Modeling

**What we did:**
- Trained and compared Gradient Boosting, Random Forest, and Linear Regression
- Created a 2x2 dashboard with residual analysis
- Performed 5-fold cross-validation for stability assessment
- Analyzed feature importance across models

**Why:**
- Different algorithms capture different patterns; comparison reveals best performer
- Gradient Boosting typically outperforms linear models on tabular data with non-linear relationships
- Cross-validation ensures model generalizes beyond training data
- Residual analysis validates model assumptions (homoscedasticity, normality)
- Feature importance explains what drives predictions

### Phase 5: Model Fairness

**What we did:**
- Conducted T-tests on model residuals across smoker/non-smoker groups
- Compared error distributions between demographic segments

**Why:**
- Models can inherit biases from training data
- Statistical testing provides evidence of fairness (or flags potential bias)
- Residual analysis reveals if the model systematically under/over-predicts for specific groups
- Ensures the model is ethically sound for deployment

## Key Findings

**Top Model:** Gradient Boosting achieved an R² of 0.901 and an RMSE of $4,134 on the test split.

**Dominant Predictor:** Smoking status accounts for the vast majority of cost variance (Smoker Avg: $34,684 vs Non-Smoker Avg: $8,203).

**Cross-Validation Stability:** 5-fold CV confirmed model stability with a mean Gradient Boosting R² of 0.853.

**Fairness Audit:** A statistical T-test on model residuals confirmed NO systematic bias between smoker and non-smoker demographics (p = 0.451).

**Data Leakage Resolution:** Removed the 'price_per_bmi' feature prior to final training to prevent target-leakage and training-serving skew.

## Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Gradient Boosting | $4,134 | $1,953 | 0.901 |
| Random Forest | $4,181 | $2,036 | 0.899 |
| Linear Regression | $8,418 | $4,245 | 0.589 |

**Best Model:** Gradient Boosting (90.1% variance explained)

**Linear vs Non-Linear:** 52.9% improvement over linear baseline

## Data Quality Summary

| Check | Result |
|-------|--------|
| Initial Records | 1,338 |
| After Cleaning | 1,337 (1 duplicate removed) |
| Missing Values | 0 |
| Outliers Capped | 13 records (0.97%) at 99th percentile |
| Features Engineered | 8 new features |
| Validation Checks | 3/3 passed |

## Project Structure

```
submission_HAMAD_ALDHUBAYB/
|-- data/
|   |-- raw/
|   |   |-- insurance.csv              # Original dataset (1,338 rows)
|   |-- processed/
|       |-- insurance_cleaned.csv      # Phase 1 output (1,337 rows)
|       |-- insurance_engineered.csv   # Phase 2 output (20 columns)
|       |-- insurance_numeric.csv      # Numeric features (13 columns)
|-- notebooks/
|   |-- 01_cleaning.ipynb              # Phase 1: Data cleaning
|   |-- 02_features.ipynb              # Phase 2: Feature engineering + VIF
|   |-- 03_eda.ipynb                   # Phase 3: EDA + manual numpy
|   |-- 04_modeling.ipynb              # Phase 4: Dashboard + ML models
|   |-- 05_fairness.ipynb              # Phase 5: Model fairness analysis
|-- src/
|   |-- utils.py                       # Production utilities (882 lines)
|   |-- advanced_viz.py                # Additional visualization functions
|-- reports/
|   |-- FINAL_REPORT.md                # Technical whitepaper
|   |-- *.png                          # 18 generated visualizations
|-- README.md                          # This file
|-- requirements.txt                   # Locked dependencies
```

## Code Quality Standards

- **Type Hints:** Full function signatures with List, Dict, Tuple, Optional, Union
- **Error Handling:** Comprehensive validation and guards (zero-vector, empty arrays)
- **Documentation:** NumPy-style docstrings with examples
- **PEP 8:** Strict style compliance
- **Bessel's Correction:** Sample standard deviation uses (n-1) denominator

## Dataset

**Medical Insurance Cost Dataset**

- 1,338 rows (1,337 after cleaning)
- 7 base columns (4 numerical, 3 categorical)
- 20 columns after feature engineering
- Target: medical charges ($1,122 - $63,770)
- Features: age, sex, bmi, children, smoker, region

## Dependencies

See `requirements.txt` for version-locked dependencies:

- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- scipy==1.11.2
- statsmodels==0.14.0
- jupyter==1.0.0
- ipykernel==6.25.0

## Citation

```
Medical Insurance Cost Analysis Pipeline
Author: Data Science Capstone
Version: 2.0
Date: March 2026
```

For detailed analysis, see `reports/FINAL_REPORT.md`
