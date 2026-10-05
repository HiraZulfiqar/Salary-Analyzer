# Salary Analyzer

A beginner-friendly Machine Learning project that analyzes employee data and predicts employee salaries based on features such as age, gender, department, and experience.

## What This Project Does

* Loads employee data using Pandas
* Performs basic data analysis
* Checks missing values and duplicate rows
* Performs Exploratory Data Analysis (EDA)
* Creates data visualizations
* Separates features and target
* Splits data into training and testing sets
* Encodes categorical features
* Builds a preprocessing pipeline
* Uses Linear Regression to predict salaries
* Compares actual and predicted salaries

## Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn

## Project Structure

```text
Salary-Analyzer/
│
├── datasets/
│   └── employees.csv
│
├── main.py
├── README.md
└── requirements.txt
```

## Machine Learning Workflow

```text
Employee Data
      ↓
Data Analysis
      ↓
EDA & Visualization
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Encoding
      ↓
Preprocessing
      ↓
Linear Regression
      ↓
Salary Prediction
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

## Example

The model uses employee information such as:

* Age
* Gender
* Department
* Experience

to predict:

**Salary**

## Learning Goal

This project was created to practice the complete beginner Machine Learning workflow, from data analysis and visualization to preprocessing, pipelines, and model prediction.
