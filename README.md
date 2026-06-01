# Teen Phone Addiction Prediction (Machine Learning Project)

**Author:** Kevin Danh  
**Program:** Master of Data Science – Bellevue University  
**Course:** DSC 550: Data Mining  

## Project Summary

This project applies data mining and machine learning techniques to analyze teen smartphone usage patterns and predict addiction risk levels. As smartphone use among teens continues to rise, concerns have grown around excessive screen time, social media dependence, and the impact on sleep, mental health, and academic performance. This project explores those behavioral patterns and builds classification models to categorize teens into low, medium, and high addiction risk levels.

## Project Type
Data Mining | Machine Learning | Multiclass Classification

## Key Objectives

- Explore behavioral and demographic patterns associated with smartphone addiction
- Engineer a categorical addiction-risk target (low / medium / high) from a continuous score
- Build and tune classification models to predict addiction risk level
- Identify the behavioral indicators most associated with higher risk

## Dataset

- Source: Kaggle – Teen Smartphone Usage and Addiction Impact Dataset
- Behavioral and demographic variables on teen smartphone use

Example features include:

- Daily phone usage hours
- Social media and gaming time
- Sleep hours
- Academic performance indicators
- Demographic variables (age, gender)

Target variable:

**Addiction risk level** — engineered from a continuous addiction score into three categories: low, medium, and high.

## Methods

### Exploratory Data Analysis
- Distribution analysis of daily usage and addiction scores
- Comparison of addiction levels across gender
- Relationship between phone usage, sleep, and addiction risk
- Analysis of combined social media and gaming engagement

### Data Preparation
- Converted a continuous addiction score into a categorical target (low / medium / high) for more interpretable classification
- Encoded categorical variables with LabelEncoder
- Standardized continuous features with StandardScaler
- Used a stratified train/test split to preserve class distribution in the imbalanced target

### Modeling
- Logistic Regression as a baseline model with `class_weight='balanced'`
- Scikit-learn Pipelines and FeatureUnion to combine preprocessing steps
- Hyperparameter tuning with RandomizedSearchCV and GridSearchCV
- Additional models (e.g. Random Forest, K-Nearest Neighbors) compared

## Evaluation

Because the target classes are imbalanced, evaluation focused on the weighted F1-score rather than accuracy alone, supported by classification reports and confusion matrices.

- The baseline Logistic Regression model achieved high overall accuracy with a strong weighted F1-score
- Class-level metrics revealed weaker performance on the smallest minority class, reflecting the challenge of imbalanced data
- Tuning and model comparison were used to find the best fit for the dataset

## Key Findings

- Smartphone addiction risk is driven primarily by behavioral usage patterns rather than demographics alone
- High daily usage, reduced sleep, and simultaneous high engagement in social media and gaming are key indicators of risk
- These behavioral signals provide a foundation for feature engineering and for designing targeted interventions

## Skills Demonstrated

**Data Mining & Data Science**
- Exploratory Data Analysis (EDA)
- Feature engineering (continuous-to-categorical target creation)
- Handling imbalanced classes
- Model comparison and evaluation

**Machine Learning**
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Scikit-learn Pipelines and FeatureUnion
- Hyperparameter tuning (RandomizedSearchCV, GridSearchCV)
- Stratified sampling and classification metrics

**Tools & Technologies**
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- GitHub

## Project Structure

```text
teen-phone-addiction-prediction-ml/
├── teen_phone_addiction_prediction.ipynb   # Main analysis notebook
├── teen_phone_addiction_dataset.csv        # Dataset
└── README.md
```

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `teen_phone_addiction_prediction.ipynb` in Jupyter and run all cells

## Ethical Considerations

This project analyzes behavioral data related to minors and mental health. The dataset is anonymized and used for educational purposes only. Predictions about addiction risk are intended as exploratory insight, not clinical or diagnostic assessment, and should not be used to label or make decisions about individuals.

## References

Sumedh. (2024). *Teen Smartphone Usage and Addiction Impact Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction
