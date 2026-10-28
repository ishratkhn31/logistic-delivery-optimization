# Last-Mile Delivery Optimisation: Strategic Planning (Week 1)

This repository belongs to the Week 1 task of my data science internship. The task was to plan a logistics data analysis project: define a scenario, choose KPIs, and outline how Python and data science techniques could be used to solve the problem.

The full written plan is in the Word report I submitted. This repository holds the code that illustrates the approach.

## Scenario

A fictional regional courier company, QuickCart Logistics, delivers parcels from one warehouse to customers across a city. Routes are planned by hand, which leads to late deliveries, high fuel cost and uneven vehicle loads. The aim is to use data to plan better routes and to forecast demand in advance.

The company and its figures are assumed for illustration. A public dataset would be used as a stand-in for real company records.

## KPIs

- On-time delivery rate
- Cost per delivery
- Average distance per delivery
- Vehicle capacity utilisation
- Demand forecast error (MAPE)

## Approach

1. **Data collection:** public datasets (for example Olist Brazilian E-Commerce and DataCo Smart Supply Chain on Kaggle) and OpenStreetMap road data
2. **Data cleaning:** fix dates, remove duplicates and missing values, remove impossible values
3. **Exploratory analysis:** late deliveries by zone, weekday and hour; demand trends
4. **Predictive modelling:** regression models to forecast demand per zone
5. **Zoning:** k-means clustering of delivery addresses
6. **Route optimisation:** capacitated vehicle routing problem solved with Google OR-Tools
7. **Evaluation:** compare optimised routes with the manual baseline using the KPIs

## Techniques and tools

- Regression (linear regression, random forest) for demand forecasting
- Clustering (k-means) for delivery zones
- Optimisation (vehicle routing with capacity limits) for route planning
- Python, pandas, NumPy, matplotlib, scikit-learn, OR-Tools

## Files

- `logistics_analysis.py` / `logistics_analysis.ipynb`: code snippets for cleaning, exploration, forecasting, clustering and routing
- `README.md`: this file

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn ortools
python logistics_analysis.py
```

The snippets use placeholder file and column names (for example `deliveries.csv`, `order_date`, `lat`, `lon`). Change these to match the dataset you download.

## Data

The data is not included in this repository. Download it from the original source and check its licence before use:

- Olist Brazilian E-Commerce Public Dataset (Kaggle)
- DataCo Smart Supply Chain for Big Data Analysis (Kaggle)

## Status

This is the planning stage. The code is meant to show how each step would be done and is not a finished analysis. Data collection and preprocessing are next (Week 2).

## References

- Clarke and Wright (1964). Scheduling of vehicles from a central depot to a number of delivery points.
- Dantzig and Ramser (1959). The truck dispatching problem.
- Toth and Vigo (2014). Vehicle Routing: Problems, Methods, and Applications.
- Hyndman and Athanasopoulos (2021). Forecasting: Principles and Practice.
- Google OR-Tools documentation: Vehicle Routing
