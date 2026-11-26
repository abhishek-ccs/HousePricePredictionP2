"""
One-time script: adds a SubArea column to the house price CSV.

How it works
------------
* Every famous city below has 8 well-known localities (sub-areas).
* Each row of the original CSV is assigned one sub-area of its own city
  (evenly spread, fixed random seed, so the result is reproducible).
* Each sub-area has a relative price tier (1.0 = city average). The row's
  Price is scaled by that tier so the model can learn that, for example,
  Vasant Vihar is costlier than Laxmi Nagar. Tiers are normalised per city,
  so the overall price level of each city stays the same as before.

IMPORTANT: the tiers are approximate, based on how public property portals
(e.g. Square Yards, NoBroker) rank localities, compressed to a gentler range.
They are NOT real transaction prices. For production-grade accuracy,
replace this CSV with real listings that already contain a locality column.

Usage:  python add_subareas.py [input.csv] [output.csv]
"""
import sys

import numpy as np
import pandas as pd

# city -> {sub-area: relative price tier}
SUB_AREAS = {
    "Delhi": {
        "Vasant Vihar": 1.70, "Greater Kailash": 1.55, "Hauz Khas": 1.40,
        "Saket": 1.25, "Janakpuri": 1.00, "Dwarka": 0.80,
        "Rohini": 0.75, "Laxmi Nagar": 0.70,
    },
    "Mumbai": {
        "Worli": 1.80, "Bandra West": 1.70, "Powai": 1.15,
        "Andheri West": 1.10, "Chembur": 1.00, "Borivali": 0.85,
        "Thane": 0.70, "Navi Mumbai": 0.65,
    },
    "Bangalore": {
        "Indiranagar": 1.40, "Koramangala": 1.40, "Jayanagar": 1.25,
        "HSR Layout": 1.10, "Whitefield": 1.00, "Sarjapur Road": 0.90,
        "Yelahanka": 0.80, "Electronic City": 0.75,
    },
    "Hyderabad": {
        "Banjara Hills": 1.70, "Jubilee Hills": 1.70, "HITEC City": 1.20,
        "Gachibowli": 1.15, "Kondapur": 1.05, "Kukatpally": 0.85,
        "Miyapur": 0.80, "Uppal": 0.70,
    },
    "Chennai": {
        "Adyar": 1.40, "T. Nagar": 1.35, "Anna Nagar": 1.30,
        "Velachery": 1.00, "OMR": 0.90, "Porur": 0.85,
        "Perambur": 0.75, "Tambaram": 0.70,
    },
    "Pune": {
        "Koregaon Park": 1.55, "Kalyani Nagar": 1.40, "Baner": 1.25,
        "Kothrud": 1.15, "Hinjewadi": 0.95, "Hadapsar": 0.90,
        "Pimpri-Chinchwad": 0.70, "Wagholi": 0.65,
    },
    "Kolkata": {
        "Alipore": 1.70, "Ballygunge": 1.50, "Salt Lake": 1.15,
        "New Town": 1.00, "Tollygunge": 0.95, "Garia": 0.80,
        "Behala": 0.75, "Howrah": 0.65,
    },
    "Lucknow": {
        "Hazratganj": 1.50, "Gomti Nagar": 1.40, "Mahanagar": 1.15,
        "Indira Nagar": 1.05, "Aliganj": 1.00, "Alambagh": 0.80,
        "Chinhat": 0.75, "Rajajipuram": 0.70,
    },
    "Jaipur": {
        "C-Scheme": 1.60, "Malviya Nagar": 1.15, "Sodala": 1.15,
        "Vaishali Nagar": 1.10, "Mansarovar": 0.95, "Tonk Road": 0.90,
        "Jagatpura": 0.85, "Ajmer Road": 0.70,
    },
    "Noida": {
        "Sector 150": 1.25, "Sector 128": 1.20, "Sector 18": 1.15,
        "Sector 50": 1.05, "Sector 62": 1.00, "Sector 137": 1.00,
        "Sector 76": 0.95, "Greater Noida West": 0.70,
    },
}


def add_subareas(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["SubArea"] = ""
    multiplier = pd.Series(1.0, index=out.index)

    for city, tiers in SUB_AREAS.items():
        names = list(tiers)
        mean_tier = np.mean(list(tiers.values()))  # keep city average unchanged
        idx = out.index[out["City"] == city].to_numpy().copy()
        rng.shuffle(idx)
        for i, row_index in enumerate(idx):
            name = names[i % len(names)]
            out.at[row_index, "SubArea"] = name
            multiplier.at[row_index] = tiers[name] / mean_tier

    out["Price"] = (out["Price"] * multiplier / 10_000).round().astype(int) * 10_000
    cols = ["Area", "Bedrooms", "Bathrooms", "Parking", "City", "SubArea", "Price", "Floors"]
    return out[cols]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "house_price_dataset_original.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "house_price_dataset_with_floors.csv"
    data = pd.read_csv(src)
    if "SubArea" in data.columns:
        sys.exit("This CSV already has a SubArea column - nothing to do.")
    add_subareas(data).to_csv(dst, index=False)
    print(f"Saved {dst}")
