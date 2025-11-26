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

## Problem: 
Identify the strongest structural, social, and individual predictors of violence against women across countries and predict risk levels for different demographic groups. Combine demographic micro-data with country-level freedom indices to understand how personal conditions intersect with political structures.

## Data: 
Public datasets on domestic violence reports, crime statistics, social services data, survey data on intimate partner violence.

- Domestic Violence Against Women (Individual Level)
    Variables: age, education, employment, income, marital status (yes/no)
- Violence Against Women & Girls (70 Countries)
    Variables: demographic, residence type, reason for violence
- WHO Prevalence Data (Country-Level)
    Variables: intimate partner & non-partner secual violence
Human Freedom Index
Variables: personal freedom, human freedom, economic freedom 
Most Dangerous Countries for Women Index
Variables: safety, homicide, gender gap, discrimination
World GDP (WorldBank)
Variables: GDP per capita, growth & PPP
Gender Wage Gap (OECD)
Variables: wage gap/country
Global Unemployment Data
Variables: country, sex, age, rate
World Educational Data
Variables: country, female/male, primary/secondary school
Gender Inequality Index (UNDP)
Variables: country, development, GII, %fem parliament, f/m secundary education, f/m labour force
Legal Frameworks Gender Equality
Variables: country, index
International Regulations Gender-Based Violence 

## EDA

### Individual-Level EDA
Which groups have highest risk? (violence against women & girls)
age
education
employment status
income
marital status
rural/urban
→ “risk profiles” based on demographics


### Country-Level EDA
WHO + Freedom Index + Dangerous Country Index
correlation heatmaps 
scatterplots:
freedom index vs violence prevalence
gender gap vs violence
homicide rate vs non-partner violence
GDP vs violence
etc
geospatial mapping
clustering countries into risk groups (k-mean)


## ML 

[...]
