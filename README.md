# Logistics Data Analysis: Planning and Data Preparation

This repository holds the code for my data science internship tasks on logistics data analysis.

- **Week 1:** strategic plan for a last-mile delivery optimisation project
- **Week 2:** data collection, cleaning and preprocessing
- **Week 3:** exploratory data analysis and visualization
- **Week 4:** predictive modeling and optimization

The written reports for each week were submitted separately as Word documents. This repository holds the code that goes with them.

## Week 1: Strategic planning

**Scenario.** A fictional regional courier company delivers parcels from one warehouse across a city. Routes are planned by hand, which leads to late deliveries, high fuel cost and uneven vehicle loads.

**KPIs:** on-time delivery rate, cost per delivery, average distance per delivery, vehicle capacity utilisation, demand forecast error (MAPE).

**Techniques:** regression for demand forecasting, k-means clustering for delivery zones, and vehicle routing optimisation with Google OR-Tools.

**Code:** `logistics_analysis.py` shows the planned steps. By default it runs on a small synthetic dataset, so its numbers are only examples and say nothing about real logistics. It is a planning-stage script, not a finished analysis.

## Week 2: Data collection, cleaning and preprocessing

**Dataset.** DataCo Smart Supply Chain for Big Data Analysis (Kaggle). 180,519 rows and 53 columns, with orders, shipping times, customers and products.

**What `data_cleaning.py` does:**

1. Loads the file (it needs latin-1 encoding, not UTF-8)
2. Audits quality: missing values, duplicates, text problems, impossible values
3. Drops columns that are more than 50% empty and personal columns not needed for analysis
4. Strips extra spaces, converts dates, fills the few remaining missing values with the median
5. Detects outliers with the IQR rule and caps them (rows are not deleted)
6. Adds min-max and z-score scaled columns
7. Saves a cleaned file and prints a before/after summary

**Results on the real dataset:**

| Measure | Before | After |
|---|---|---|
| Rows | 180,519 | 180,519 |
| Columns | 53 | 56 |
| Missing cells | 336,209 | 0 |
| Text cells with extra spaces | 24,251 | 0 |

Outliers were found in Sales (488), Order Item Total (1,943) and Benefit per order (18,942). No duplicates or impossible values were found.

## Week 3: Exploratory data analysis and visualization

`eda_visualization.py` reads the cleaned file from Week 2 (`cleaned_logistics_data.csv`) and:

1. Prints summary statistics (mean, median, spread, skewness)
2. Compares late delivery rates by shipping mode, market and region
3. Summarises sales by market and product category, and orders over time
4. Calculates correlations between the key variables
5. Saves 8 charts (histogram, bar charts, boxplot, line chart, heatmap)

**Main findings on the DataCo data:**

- 54.8% of order items were delivered late
- Shipping mode matters most: late rate is 95.3% for First Class, 76.6% for Second Class, 45.7% for Same Day and 38.1% for Standard Class
- Late rate barely changes between markets (54.4% to 55.2%)
- Sales are concentrated: Europe and LATAM make up 57.2% of sales, and five categories make up 61.4%
- Order volume drops sharply from about autumn 2017, which needs checking before any forecasting

The charts are in the `charts/` folder.

## Week 4: Predictive modeling and optimization

`predictive_modeling.py` reads the cleaned file from Week 2 and works on two problems:

**Part A: predicting actual shipping days (regression)**

- Compares a baseline, linear regression, decision tree, random forest and gradient boosting
- Uses 5-fold cross-validation and grid search for hyperparameters, and reports MAE, RMSE and R-squared
- Leaves out delay, late flag and delivery status as inputs, because they already contain the answer
- Tests different promised delivery times for each shipping mode

**Part B: forecasting daily order volume**

- Uses lagged values and calendar features, with a time-based train/test split and time-series cross-validation
- Compares the forecast with a "same day last week" benchmark
- Uses the forecast for a vehicle capacity planning example (the 40 items per vehicle figure is an assumption)

**Main findings:**

- All models beat the baseline (RMSE about 1.27 days against 1.62), but they perform almost the same, so a simple model is enough
- Only shipping mode and scheduled days matter for shipping time; the other inputs add almost nothing
- Realistic promised times cut the late rate from 57.3% to 15.9% (75th percentile policy)
- The daily order forecast captures the average level of demand (MAPE about 6.4%) but not day-to-day swings

The model charts are in the `charts/` folder.

## Files

- `logistics_analysis.py`: Week 1 planning code
- `data_cleaning.py`: Week 2 cleaning pipeline
- `eda_visualization.py`: Week 3 analysis and charts
- `predictive_modeling.py`: Week 4 models and optimization
- `charts/`: chart images from Weeks 3 and 4
- `README.md`: this file

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn ortools

# Week 1 (runs on built-in synthetic data if no file is given)
python logistics_analysis.py

# Week 2 (put the Kaggle CSV in the same folder first)
python data_cleaning.py

# Week 3 (needs cleaned_logistics_data.csv from Week 2)
pip install seaborn
python eda_visualization.py

# Week 4 (needs cleaned_logistics_data.csv from Week 2)
python predictive_modeling.py
```

## Data

The dataset is not included in this repository because of its size. Download it from Kaggle and check its licence before use:

- DataCo Smart Supply Chain for Big Data Analysis (Constante, Silva and Pereira, 2019)
