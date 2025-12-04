# Violence Against Women: Risk Prediction & Structural Determinants Analysis

## Intro:
The UN describes violence against women and girls (VAWG) as: “One of the most widespread, persistent, and devastating human rights violations in our world today. It remains largely unreported due to the impunity, silence, stigma, and shame surrounding it.”

In general terms, it manifests itself in physical, sexual, and psychological forms, encompassing:
• intimate partner violence (battering, psychological abuse, marital rape, femicide)
• sexual violence and harassment (rape, forced sexual acts, unwanted sexual advances,
child sexual abuse, forced marriage, street harassment, stalking, cyber-harassment),
human trafficking (slavery, sexual exploitation)
• female genital mutilation
• child marriage

## Project Overview
This project investigates Violence Against Women & Girls (VAWG) across three analytical levels — individual, country, and global — to uncover patterns, risk factors, and actionable insights.

A final Streamlit dashboard integrates all levels and enables interactive exploration, including:
- Country comparisons
- Risk factor interpretation
- Model predictions
- Policy recommendations generated from context prompts

## Goal
- Build reliable, interpretable models to analyze VAWG determinants at multiple scales
- Integrate heterogeneous datasets (survey, demographic, governance, socioeconomic, gender inequality)
- Apply regularized regression and ensembles for stability with small/medium datasets
- Create a robust feature engineering pipeline for consistency across levels

## Structure
|– 01_data.ipynb
|– 02_eda.ipynb
|– 03_feature_engineering.ipynb
|– 01_modelling.ipynb
|– vawg_dashboard/
    |– app.py
    |– setup.py
    |– data/
        |– individual_df.csv
        |– violence_df.csv
        |– global_df.csv
        |– global_stats.csv
    |– models/
        |– individual_model.pkl
        |– individual_feature_names.pkl
        |– individual_selected_features.pkl
        |– violence_model.pkl
        |– global_model.pkl
    
## Raw Data: 

- Domestic Violence Against Women (Individual Level)
    Variables: age, education, employment, income, marital status (yes/no)
- Violence Against Women & Girls (70 Countries)
    Variables: demographic, residence type, reason for violence
- WHO Prevalence Data (Country-Level)
    Variables: intimate partner & non-partner secual violence
- Human Freedom Index
    Variables: personal freedom, human freedom, economic freedom 
- Most Dangerous Countries for Women Index  
    Variables: safety, homicide, gender gap, discrimination
- World GDP (WorldBank)
    Variables: GDP per capita, growth & PPP
- Gender Wage Gap (OECD)
    Variables: wage gap/country
- Global Unemployment Data
    Variables: country, sex, age, rate
- World Educational Data
    Variables: country, female/male, primary/secondary school
- Gender Inequality Index (UNDP)
    Variables: country, development, GII, %fem parliament, f/m secundary education, f/m labour force
- Legal Frameworks Gender Equality
    Variables: country, index
- International Regulations Gender-Based Violence 

## Datasets

1. Individual-Level Dataset
- n = 325 individuals
- survey-style features (demographics, education, autonomy indicators)
- Target: Binary classification (experienced violence: yes/no)

2. Violence (Country-Level) Dataset
- n = 50 countries
- VAWG prevalence statistics
- Gender norms, governance, socioeconomic indicators
- Target: Regression (violence prevalence %)

3. Global Structural Dataset
- n = 195 countries
- Gender inequality indices 
- Economic and governance datasets
- Development indicators
- Target: Regression (structural VAWG risk composite)

## Key Challenges
1. Heterogeneous Data Sources
- Different units, scales, and missingness patterns
    → Required robust cleaning, median imputation, scaling, and feature harmonization.

2. Uneven Sample Sizes
- Individual level: reasonably sized (n=325)
- Country level: very small (n=50) → risk of overfitting
- Global level: moderate (n=195)
- This made a single unified model impossible — each level has different variance structures and requires tailored algorithms.

3. Multi-Level Causal Complexity
- Individual factors ≠ country factors ≠ global structural drivers
- Mixing levels risks statistical leakage
- Therefore: three separate pipelines with aligned features but isolated modeling.

## Feature Engineering
- Median imputation
- Min–Max scaling
- Composite indices
    - Economic security
    - Education 
    - Governance
    - Gender inequality
- Exploratory PCA
    Used to validate structure and redundancy; final models prioritized interpretability, not PCA components.
-  Interaction features
    Only where theoretically justified (e.g., inequality × governance).

## Modeling

1. Individual Model
- Model: LogisticRegressionCV with ElasticNet penalty
- Performance:
    - F1 = 0.94
    - Accuracy = 0.94
- Notes:
    - Excellent generalization
    - Key predictors: education, autonomy, partner behaviors, socioeconomic constraints

2. Violence (Country) Model
- Task: Regress violence prevalence rates
- Model: ElasticNetCV
- Performance:
    - Train R² = 1.00
    - Test R² = 1.00 (sample too small → model memorizes)
- Notes:
    - Useful for exploratory insight, not causal inference
    - Small n = structural limitation, not a model issue

3. Global Model
- Task: Predict global VAWG structural risk
- Model: ElasticNetCV + XGBoost Ensemble
- Performance:
    - Train R² = 0.487
    - Test R² = 0.264
- Notes:
    - Structural global risk is inherently noisy
    - Ensemble balances interpretability + nonlinearity
    - Reveals consistent global drivers (inequality, governance, economic security)

## Dashboard (Streamlit)
- Level selection (Individual / Violence / Global)
- Feature explanations
- Model predictions
- Country comparisons
- Automatically generated policy recommendations
- Summary of datapoints, models, and assumptions
A GPT-based module provides context-aware insights:
    - Key differences
    - Strengths of each country
    - Lessons learned
    - Policy recommendations
