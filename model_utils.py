"""
Training + prediction helpers for the HouseWise AI app.

Run `python model_utils.py` to retrain and overwrite house_model.pkl.

Model design
------------
* Target  : log(Price)  -> predictions are converted back with exp().
* Features: Log_Area, Bedrooms, Bathrooms, Parking, Floors,
            City_<city> one-hot columns,
            SubArea_<city>|<sub-area> one-hot columns.
  Price grows roughly in proportion to area, so a log-log model fits much
  better than a plain linear one once localities are added.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "house_price_dataset_with_floors.csv"
MODEL_PATH = BASE_DIR / "house_model.pkl"

NUMERIC = ["Bedrooms", "Bathrooms", "Parking", "Floors"]


def subarea_column(city: str, sub_area: str) -> str:
    return f"SubArea_{city}|{sub_area}"


def build_features(df: pd.DataFrame, cities=None, pairs=None) -> pd.DataFrame:
    """Turn raw rows (Area, Bedrooms, ..., City, SubArea) into model features."""
    cities = sorted(df["City"].unique()) if cities is None else cities
    if pairs is None:
        pairs = sorted(
            df[["City", "SubArea"]].drop_duplicates().itertuples(index=False, name=None)
        )
    data = {"Log_Area": np.log(df["Area"].astype(float))}
    for col in NUMERIC:
        data[col] = df[col].astype(float)
    for city in cities:
        data[f"City_{city}"] = (df["City"] == city).astype(int)
    for city, sub in pairs:
        data[subarea_column(city, sub)] = (
            (df["City"] == city) & (df["SubArea"] == sub)
        ).astype(int)
    return pd.DataFrame(data, index=df.index)


def train(csv_path=CSV_PATH, save_path=None):
    df = pd.read_csv(csv_path)
    X = build_features(df)
    y = np.log(df["Price"].astype(float))

    # Honest check on unseen rows first ...
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    check = LinearRegression().fit(X_tr, y_tr)
    pred = np.exp(check.predict(X_te))
    actual = np.exp(y_te)
    metrics = {
        "r2": r2_score(actual, pred),
        "mape": mean_absolute_percentage_error(actual, pred),
    }

    # ... then fit the final model on all rows.
    model = LinearRegression().fit(X, y)
    if save_path:
        joblib.dump(model, save_path)
    return model, metrics


# ---------------------------------------------------------------------------
# Prediction helpers (used by app.py)
# ---------------------------------------------------------------------------
def parse_locations(feature_names):
    """Return ({city: [sub-areas]}, [all cities]) from the model's feature names."""
    cities = sorted(c[len("City_"):] for c in feature_names if c.startswith("City_"))
    sub_map = {city: [] for city in cities}
    for col in feature_names:
        if col.startswith("SubArea_") and "|" in col:
            city, sub = col[len("SubArea_"):].split("|", 1)
            sub_map.setdefault(city, []).append(sub)
    for city in sub_map:
        sub_map[city].sort()
    return sub_map, cities


def _row(model_features, area, bedrooms, bathrooms, parking, floors, city, sub_area):
    row = dict.fromkeys(model_features, 0)
    row["Log_Area"] = float(np.log(area))
    row["Bedrooms"] = bedrooms
    row["Bathrooms"] = bathrooms
    row["Parking"] = parking
    row["Floors"] = floors
    row[f"City_{city}"] = 1
    if sub_area is not None:
        row[subarea_column(city, sub_area)] = 1
    return row


def predict_price(model, area, bedrooms, bathrooms, parking, floors,
                  city=None, sub_area=None):
    """
    * city + sub_area      -> exact locality prediction
    * city only            -> average over that city's sub-areas
    * city None (unknown)  -> average of every known city's average
    """
    features = list(model.feature_names_in_)
    sub_map, cities = parse_locations(features)

    def city_average(c):
        subs = sub_map.get(c) or [None]
        rows = [_row(features, area, bedrooms, bathrooms, parking, floors, c, s)
                for s in subs]
        return float(np.exp(model.predict(pd.DataFrame(rows)[features])).mean())

    if city is None:
        return float(np.mean([city_average(c) for c in cities]))
    if sub_area is None:
        return city_average(city)
    row = _row(features, area, bedrooms, bathrooms, parking, floors, city, sub_area)
    return float(np.exp(model.predict(pd.DataFrame([row])[features])[0]))


if __name__ == "__main__":
    model, m = train(save_path=MODEL_PATH)
    print(f"Saved {MODEL_PATH.name}")
    print(f"Hold-out R2: {m['r2']:.4f} | Hold-out MAPE: {m['mape']:.2%}")
