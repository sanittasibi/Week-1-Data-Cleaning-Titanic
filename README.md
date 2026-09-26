# Week 1 - Titanic Data Cleaning and Preprocessing

This repository contains the work completed for Week 1 of the YUVA Intern Virtual Data Science with Python Trainee program.

## Objective

Acquire a public dataset, explore its quality, handle missing values and potential outliers, investigate duplicate/inconsistent records, preprocess categorical variables, and save an analysis-ready dataset.

## Dataset

Titanic dataset distributed through the Seaborn example-data repository:
https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv

The notebook loads the dataset with:

```python
sns.load_dataset("titanic")
```

## Main findings

- Initial shape: 891 rows x 15 columns
- Missing age values: 177 (19.87%)
- Missing embarked values: 2 (0.22%)
- Missing deck values: 688 (77.22%)
- Missing embark_town values: 2 (0.22%)
- Initial exact duplicate rows: 107
- Missing values after cleaning: 0
- Cleaned shape: 891 rows x 14 columns
- One-hot encoded shape: 891 rows x 18 columns
- Modelling-ready shape: 891 rows x 13 columns

## Cleaning decisions

1. Missing `age` values were replaced with the median age.
2. Missing `embarked` and `embark_town` values were replaced with the mode.
3. `deck` was removed because 77.22% of its values were missing.
4. Exact duplicates were investigated but not automatically deleted because the Seaborn version used does not provide a unique passenger identifier.
5. IQR-based outliers were retained because being statistically unusual does not prove that a value is erroneous.
6. The redundant `alive` variable was removed from the modelling-ready version because it duplicates the information represented by `survived`.

## Files

- `week1_data_cleaning_titanic.py` - consolidated Python workflow
- `titanic_cleaned_processed.csv` - generated output from Google Colab
- `Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx` - internship report

## Environment

Google Colab / Python / Pandas / NumPy / Seaborn / Matplotlib
