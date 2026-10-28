"""
Last-Mile Delivery Optimisation - Week 1 planning code

This script walks through the main steps of the plan:
    1. load / clean data
    2. exploratory analysis (late deliveries)
    3. demand forecasting (regression)
    4. zoning (k-means clustering)
    5. route optimisation (OR-Tools vehicle routing)
    6. KPI comparison against a simple baseline

If "deliveries.csv" is not found, a small synthetic dataset is generated
so the script can still run from start to finish. Replace it with a real
public dataset (e.g. Olist or DataCo from Kaggle) and change the column
names in load_data() to match.

Install:  pip install pandas numpy matplotlib scikit-learn ortools
Run:      python logistics_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to files instead of opening windows
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, silhouette_score

DATA_FILE = "deliveries.csv"
WAREHOUSE = (19.10, 72.88)   # placeholder depot location (lat, lon)
SEED = 42


# ---------------------------------------------------------------- 1. data
def make_synthetic_data(days=90, per_day=80):
    """Small made-up dataset so the script runs without external files."""
    rng = np.random.default_rng(SEED)
    rows = []
    oid = 0
    start = pd.Timestamp("2025-01-01")
    for d in range(days):
        date = start + pd.Timedelta(days=d)
        n = int(per_day * (1.3 if date.dayofweek >= 5 else 1.0) + rng.integers(-10, 10))
        for _ in range(n):
            oid += 1
            lat = WAREHOUSE[0] + rng.normal(0, 0.05)
            lon = WAREHOUSE[1] + rng.normal(0, 0.05)
            promised = date + pd.Timedelta(days=int(rng.integers(2, 5)))
            delay = int(rng.choice([-1, 0, 0, 1, 2], p=[0.2, 0.35, 0.25, 0.15, 0.05]))
            rows.append((oid, date, promised, promised + pd.Timedelta(days=delay), lat, lon))
    cols = ["order_id", "order_date", "promised_date", "delivered_date", "lat", "lon"]
    return pd.DataFrame(rows, columns=cols)


def load_data(path=DATA_FILE):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("Loaded", path, df.shape)
    else:
        df = make_synthetic_data()
        print("deliveries.csv not found - using synthetic data", df.shape)
    return df


def clean_data(df):
    for c in ["order_date", "promised_date", "delivered_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset="order_id")
    df = df.dropna(subset=["order_date", "delivered_date", "lat", "lon"])
    df["delivery_days"] = (df["delivered_date"] - df["order_date"]).dt.days
    df = df[df["delivery_days"] >= 0].copy()
    df["is_late"] = (df["delivered_date"] > df["promised_date"]).astype(int)
    print(f"Cleaning: {before} -> {len(df)} rows")
    return df


# ---------------------------------------------------------------- 2. EDA
def explore(df):
    on_time = 100 * (1 - df["is_late"].mean())
    print(f"On-time delivery rate: {on_time:.1f}%")

    df["weekday"] = df["order_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    late_by_day = (df.groupby("weekday")["is_late"].mean() * 100).reindex(order)

    late_by_day.plot(kind="bar")
    plt.ylabel("Late deliveries (%)")
    plt.title("Late delivery rate by weekday")
    plt.tight_layout()
    plt.savefig("late_by_weekday.png")
    plt.close()
    print("Saved late_by_weekday.png")
    return on_time


# ---------------------------------------------------------------- 3. zoning
def make_zones(df, k_range=range(3, 8)):
    coords = df[["lat", "lon"]].values
    sample = coords[:2000]  # silhouette is slow on big data
    best_k, best_score = None, -1
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sample)
        score = silhouette_score(sample, km.labels_)
        print(f"  k={k}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(coords)
    df["zone"] = km.labels_
    print("Chosen number of zones:", best_k)
    return df


# ---------------------------------------------------------------- 4. forecast
def forecast_demand(df):
    daily = (df.groupby([df["order_date"].dt.normalize(), "zone"])
               .size().reset_index(name="parcels"))
    daily = daily.sort_values(["zone", "order_date"])
    daily["dow"] = daily["order_date"].dt.dayofweek
    daily["month"] = daily["order_date"].dt.month
    daily["lag_7"] = daily.groupby("zone")["parcels"].shift(7)
    daily = daily.dropna().sort_values("order_date")

    features = ["dow", "month", "lag_7", "zone"]
    cut = int(len(daily) * 0.8)            # time-based split
    train, test = daily.iloc[:cut], daily.iloc[cut:]

    results = {}
    for name, model in [("Linear regression", LinearRegression()),
                        ("Random forest", RandomForestRegressor(n_estimators=200, random_state=SEED))]:
        model.fit(train[features], train["parcels"])
        pred = model.predict(test[features])
        mae = mean_absolute_error(test["parcels"], pred)
        mape = np.mean(np.abs(test["parcels"] - pred) / test["parcels"]) * 100
        results[name] = (mae, mape)
        print(f"  {name}: MAE={mae:.2f}  MAPE={mape:.1f}%")
    return results


# ---------------------------------------------------------------- 5. routing
def haversine_matrix(points):
    """Distance matrix in metres (integers, as OR-Tools needs)."""
    pts = np.radians(np.array(points))
    lat, lon = pts[:, 0][:, None], pts[:, 1][:, None]
    dlat, dlon = lat - lat.T, lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return (6371000 * 2 * np.arcsin(np.sqrt(a))).astype(int)


def route_distance(route, dist):
    return sum(dist[a][b] for a, b in zip(route[:-1], route[1:]))


def baseline_routes(n_stops, n_vehicles, capacity):
    """Naive plan: stops handed to vehicles in the order they appear."""
    stops = list(range(1, n_stops + 1))
    routes = []
    for v in range(n_vehicles):
        chunk = stops[v * capacity:(v + 1) * capacity]
        routes.append([0] + chunk + [0])
    return routes


def optimise_routes(dist, n_vehicles, capacity):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(dist)
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    demands = [0] + [1] * (n - 1)          # one parcel per stop
    def demand_cb(i):
        return demands[manager.IndexToNode(i)]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [capacity] * n_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    routes = []
    for v in range(n_vehicles):
        idx, route = routing.Start(v), []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(0)
        routes.append(route)
    return routes


def compare_routing(df, n_stops=40, capacity=10):
    # take one day of deliveries as a test case
    one_day = df[df["order_date"] == df["order_date"].iloc[0]].head(n_stops)
    n_stops = len(one_day)
    points = [WAREHOUSE] + list(zip(one_day["lat"], one_day["lon"]))
    dist = haversine_matrix(points)
    n_vehicles = int(np.ceil(n_stops / capacity)) + 1

    base = baseline_routes(n_stops, n_vehicles, capacity)
    base_km = sum(route_distance(r, dist) for r in base if len(r) > 2) / 1000

    try:
        opt = optimise_routes(dist, n_vehicles, capacity)
    except ImportError:
        print("ortools is not installed - skipping optimisation (pip install ortools)")
        return
    if opt is None:
        print("No feasible solution found")
        return
    opt_km = sum(route_distance(r, dist) for r in opt if len(r) > 2) / 1000

    print(f"Stops: {n_stops}, vehicles available: {n_vehicles}, capacity: {capacity}")
    print(f"Baseline total distance : {base_km:.1f} km  ({base_km / n_stops:.2f} km per delivery)")
    print(f"Optimised total distance: {opt_km:.1f} km  ({opt_km / n_stops:.2f} km per delivery)")
    print(f"Distance saved          : {100 * (base_km - opt_km) / base_km:.1f}%")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    data = clean_data(load_data())

    print("\n--- Exploratory analysis ---")
    explore(data)

    print("\n--- Zoning (k-means) ---")
    data = make_zones(data)

    print("\n--- Demand forecasting ---")
    forecast_demand(data)

    print("\n--- Route optimisation (baseline vs OR-Tools) ---")
    compare_routing(data)
