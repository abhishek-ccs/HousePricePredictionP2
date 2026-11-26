"""
Builds house_price_dataset_with_floors.csv (and locality_rates.csv).

WHERE THE NUMBERS COME FROM
---------------------------
Each (city, sub-area) below has a 2026 average asking price per sq ft taken
from public property portals (HomeBazaar, Square Yards, NoBroker, Housing.com)
and market reports. Where portals gave a range, the midpoint is used.

From that real rate, individual houses are generated:
  * bedrooms -> realistic area range (1 BHK ~ 450-750 sq ft ... 5 BHK ~ 2600-4000)
  * bathrooms, parking and floors follow bedrooms
  * price = area x locality rate x random project-to-project variation (about +/-9%)

IMPORTANT: individual rows are simulated, NOT scraped listings. What is real is
the price level of every locality. If you later get a real listings dataset
(Kaggle, 99acres/MagicBricks export) with City + Locality columns, drop it in
with the same column names and run `python model_utils.py` to retrain.
"""
import numpy as np
import pandas as pd

# (city, sub-area, avg Rs per sq ft 2026, source note)
HB = "HomeBazaar 2026"
SY = "Square Yards 2026"
MID = "midpoint of portal ranges 2026"
RATES = [
    # ---- Delhi (HomeBazaar, Oct 2026) ----
    ("Delhi", "Jor Bagh", 56800, HB), ("Delhi", "Shanti Niketan", 45150, HB),
    ("Delhi", "Hauz Khas", 31250, HB), ("Delhi", "Greater Kailash II", 31000, HB),
    ("Delhi", "Vasant Kunj", 22150, HB), ("Delhi", "Rajouri Garden", 19150, HB),
    ("Delhi", "Pitampura", 18900, HB), ("Delhi", "Mayur Vihar Phase 1", 17200, HB),
    ("Delhi", "Dwarka (Sector 10)", 15900, HB), ("Delhi", "Rohini (Sector 24)", 11000, HB),
    ("Delhi", "Dilshad Garden", 8750, HB), ("Delhi", "Laxmi Nagar", 7300, HB),
    # ---- Mumbai (Square Yards / NoBroker / market reports) ----
    ("Mumbai", "Worli", 65000, SY), ("Mumbai", "Bandra West", 60000, MID),
    ("Mumbai", "Powai", 42000, SY), ("Mumbai", "Andheri West", 40000, MID),
    ("Mumbai", "Malad", 28000, MID), ("Mumbai", "Goregaon", 26000, MID),
    ("Mumbai", "Borivali", 23000, MID), ("Mumbai", "Thane", 19000, "NoBroker 2026"),
    ("Mumbai", "Navi Mumbai", 10500, MID),
    # ---- Bangalore (HomeBazaar, Oct 2026) ----
    ("Bangalore", "Indiranagar", 20200, HB), ("Bangalore", "Koramangala", 16650, HB),
    ("Bangalore", "Bellandur", 15400, HB), ("Bangalore", "Hebbal", 14450, HB),
    ("Bangalore", "Whitefield", 14050, HB), ("Bangalore", "Jayanagar", 15000, HB),
    ("Bangalore", "HSR Layout", 12750, HB), ("Bangalore", "Marathahalli", 12300, HB),
    ("Bangalore", "JP Nagar", 10550, HB), ("Bangalore", "Yelahanka", 10350, HB),
    ("Bangalore", "Electronic City", 7200, HB),
    # ---- Hyderabad (HomeBazaar, midpoints) ----
    ("Hyderabad", "Jubilee Hills", 12600, MID), ("Hyderabad", "Banjara Hills", 12500, MID),
    ("Hyderabad", "Hi-Tech City", 12000, HB), ("Hyderabad", "Gachibowli", 11000, HB),
    ("Hyderabad", "Madhapur", 10800, HB), ("Hyderabad", "Kondapur", 10150, HB),
    ("Hyderabad", "Kukatpally", 6700, MID), ("Hyderabad", "Uppal", 6050, HB),
    # ---- Chennai (HomeBazaar, May 2026) ----
    ("Chennai", "Besant Nagar", 20900, HB), ("Chennai", "T. Nagar", 19950, HB),
    ("Chennai", "Adyar", 17250, HB), ("Chennai", "Anna Nagar", 16150, HB),
    ("Chennai", "Thiruvanmiyur", 14500, HB), ("Chennai", "Vadapalani", 11600, HB),
    ("Chennai", "Velachery", 9400, HB), ("Chennai", "Porur", 8500, "NoBroker 2026"),
    ("Chennai", "OMR (Sholinganallur)", 7800, MID), ("Chennai", "Tambaram", 6900, HB),
    # ---- Pune (HomeBazaar, May 2026) ----
    ("Pune", "Koregaon Park", 18400, HB), ("Pune", "Kalyani Nagar", 16300, HB),
    ("Pune", "Kothrud", 14400, HB), ("Pune", "Aundh", 12500, HB),
    ("Pune", "Baner", 11700, HB), ("Pune", "Hadapsar", 11400, HB),
    ("Pune", "Kharadi", 11150, HB), ("Pune", "Hinjewadi", 8200, HB),
    ("Pune", "Wakad", 8200, HB), ("Pune", "Pimpri-Chinchwad", 7000, MID),
    ("Pune", "Wagholi", 6500, HB),
    # ---- Kolkata (Square Yards + HomeBazaar) ----
    ("Kolkata", "Park Street", 15000, MID), ("Kolkata", "Alipore", 13500, MID),
    ("Kolkata", "Ballygunge", 12300, MID), ("Kolkata", "Salt Lake", 11500, MID),
    ("Kolkata", "New Alipore", 11500, SY), ("Kolkata", "New Town", 8000, MID),
    ("Kolkata", "Dhakuria", 6750, HB), ("Kolkata", "Joka", 5650, HB),
    ("Kolkata", "Behala", 4600, MID), ("Kolkata", "Howrah", 3100, SY),
    # ---- Lucknow (Square Yards) ----
    ("Lucknow", "Mahanagar", 9850, SY), ("Lucknow", "Vibhuti Khand", 8000, SY),
    ("Lucknow", "Gomti Nagar", 7250, SY), ("Lucknow", "Sushant Golf City", 7000, SY),
    ("Lucknow", "Hazratganj", 6000, SY), ("Lucknow", "Sultanpur Road", 5500, SY),
    ("Lucknow", "Aliganj", 5300, SY), ("Lucknow", "Jankipuram", 5200, SY),
    ("Lucknow", "Alambagh", 4530, SY), ("Lucknow", "Indira Nagar", 4400, SY),
    # ---- Noida (Square Yards / market reports) ----
    ("Noida", "Sector 128", 17000, SY), ("Noida", "Sector 50", 14100, "RealtyHunting 2026"),
    ("Noida", "Sector 142", 13300, SY), ("Noida", "Sector 150", 12200, SY),
    ("Noida", "Sector 62", 10500, MID), ("Noida", "Sector 76", 10450, "Housing.com 2026"),
    ("Noida", "Sector 137", 10200, SY), ("Noida", "Greater Noida West", 8500, SY),
    # ---- Jaipur (Square Yards) ----
    ("Jaipur", "C-Scheme", 13300, SY), ("Jaipur", "Siddharth Nagar", 12150, SY),
    ("Jaipur", "Bani Park", 8240, SY), ("Jaipur", "Malviya Nagar", 7820, SY),
    ("Jaipur", "Vaishali Nagar", 5600, SY), ("Jaipur", "Tonk Road", 5350, SY),
    ("Jaipur", "Jagatpura", 5000, SY), ("Jaipur", "Mansarovar", 4380, SY),
    ("Jaipur", "Ajmer Road", 4250, SY),
    # ---- Gurgaon (Square Yards / Sobha / RealtyHunting) ----
    ("Gurgaon", "Golf Course Road", 27200, MID), ("Gurgaon", "DLF Phase 4", 21500, SY),
    ("Gurgaon", "Golf Course Extension Road", 19100, MID), ("Gurgaon", "Dwarka Expressway", 14000, MID),
    ("Gurgaon", "Sohna Road", 13500, MID), ("Gurgaon", "Udyog Vihar", 10100, MID),
    ("Gurgaon", "Sector 82", 9300, SY),
]

# typical flat sizes (sq ft) per bedroom count
AREA_RANGE = {1: (450, 750), 2: (750, 1200), 3: (1150, 1800),
              4: (1700, 2800), 5: (2600, 4000)}
# metro flats are smaller, tier-2 cities larger
SIZE_FACTOR = {"Mumbai": 0.85, "Lucknow": 1.15, "Jaipur": 1.15}
ROWS_PER_LOCALITY = 30


def generate(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for city, sub, rate, _ in RATES:
        for _ in range(ROWS_PER_LOCALITY):
            bed = rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.35, 0.33, 0.15, 0.05])
            lo, hi = AREA_RANGE[bed]
            area = int(rng.uniform(lo, hi) * SIZE_FACTOR.get(city, 1.0))
            bath = int(np.clip(bed + rng.choice([0, 1, -1], p=[0.55, 0.30, 0.15]), 1, 6))
            park = int(np.clip(round(rng.normal(0.4 + 0.5 * bed, 0.6)), 0, 4))
            floors = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.13, 0.07]))
            ppsf = (rate
                    * rng.lognormal(0, 0.09)           # project-to-project spread
                    * (area / 1200) ** 0.06            # bigger homes: slightly higher rate
                    * (1 + 0.012 * park)
                    * (1 + 0.005 * (floors - 1)))
            price = int(round(area * ppsf / 10_000)) * 10_000
            rows.append((area, bed, bath, park, city, sub, price, floors))
    cols = ["Area", "Bedrooms", "Bathrooms", "Parking", "City", "SubArea", "Price", "Floors"]
    return pd.DataFrame(rows, columns=cols).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    generate().to_csv("house_price_dataset_with_floors.csv", index=False)
    pd.DataFrame(RATES, columns=["City", "SubArea", "Rate_per_sqft_INR", "Source"]).to_csv(
        "locality_rates.csv", index=False)
    print("Saved house_price_dataset_with_floors.csv and locality_rates.csv")
