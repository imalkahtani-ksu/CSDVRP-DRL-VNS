"""
build_riyadh_case.py  —  CSDVRP-CS-Riyadh case study
Depot : NUPCO Central Warehouse, As Sahafah District, Riyadh
Nodes : 30 hospital nodes (Ministry of Health Saudi Arabia public data)
Fleet : 4 refrigerated vans, Q = 50 units each  (~75% utilisation)
Source: MOH interactive map; NUPCO logistics network (www.nupco.com)
"""
import os, sys, json, math
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEPOT_LAT = 24.7060
DEPOT_LON = 46.6720

# (hospital_name, lat, lon, beds_approx, tier)
HOSPITALS = [
    # Tier-1 — major tertiary / university hospitals (demand 7–18, U_max=3)
    ("King Abdulaziz Medical City (NGHA)",     24.6890, 46.6842, 1500, 1),
    ("King Faisal Specialist Hospital",         24.6886, 46.7069, 1200, 1),
    ("King Khalid University Hospital",         24.5906, 46.7117,  800, 1),
    ("Prince Sultan Military Medical City",     24.6580, 46.7380,  700, 1),
    ("King Fahad Medical City",                 24.5860, 46.7130, 1200, 1),
    ("King Salman Hospital",                    24.6650, 46.6920,  450, 1),
    ("King Abdulaziz Hospital MOH",             24.7110, 46.6350,  380, 1),
    # Tier-2 — specialized / secondary (demand 3–6, U_max=2)
    ("Security Forces Hospital",                24.6810, 46.7250,  350, 2),
    ("Saudi German Hospital Al Olaya",          24.7380, 46.6720,  300, 2),
    ("Dallah Hospital",                         24.6800, 46.6500,  240, 2),
    ("Dr. Sulaiman Al Habib Al Olaya",          24.6990, 46.6800,  280, 2),
    ("Al Mouwasat Hospital",                    24.6700, 46.6900,  200, 2),
    ("Al Hamad International Medical Center",   24.7050, 46.7180,  180, 2),
    ("Saudi German Hospital Al Nakheel",        24.7700, 46.6700,  200, 2),
    ("Specialized Medical Center Hospital",     24.6900, 46.6700,  180, 2),
    ("Al Zahrawi Hospital",                     24.7200, 46.7100,  160, 2),
    ("National Hospital Riyadh",                24.7000, 46.6950,  200, 2),
    # Tier-3 — primary care / day hospitals (demand 1.5–2.5, U_max=1)
    ("Al Yamamah Hospital",                     24.7450, 46.7350,  110, 3),
    ("Soliman Fakeeh Hospital",                 24.6950, 46.7300,   80, 3),
    ("Al Buhayra Health Centre",                24.6600, 46.7100,   65, 3),
    ("Al Rawdah Health Centre",                 24.7300, 46.6900,   70, 3),
    ("Al Nafisa Primary Hospital",              24.6400, 46.7200,   55, 3),
    ("Al Andalus Health Centre",                24.7550, 46.7050,   60, 3),
    ("Sulaimaniyah Primary",                    24.6850, 46.7150,   70, 3),
    ("Al Muraba Primary",                       24.6750, 46.7050,   55, 3),
    ("Al Shifa Primary",                        24.6950, 46.7450,   60, 3),
    ("Al Nuzha Primary",                        24.7600, 46.7450,   50, 3),
    ("Al Malaz Primary",                        24.6500, 46.7300,   55, 3),
    ("Al Hamra Primary",                        24.6350, 46.6800,   60, 3),
    ("Al Aziziyah Primary",                     24.6200, 46.7000,   65, 3),
]

KM_PER_LAT = 111.0
KM_PER_LON = 111.0 * math.cos(math.radians(DEPOT_LAT))

all_lats = [DEPOT_LAT] + [h[1] for h in HOSPITALS]
all_lons = [DEPOT_LON] + [h[2] for h in HOSPITALS]
xs_km = [(lo - DEPOT_LON) * KM_PER_LON for lo in all_lons]
ys_km = [(la - DEPOT_LAT) * KM_PER_LAT for la in all_lats]

x_min, x_max = min(xs_km), max(xs_km)
y_min, y_max = min(ys_km), max(ys_km)
span = max(x_max - x_min, y_max - y_min)

def norm(v, vmin, sp, lo=5.0, hi=95.0):
    return round(lo + (v - vmin) / sp * (hi - lo), 2)

xs_n = [norm(v, x_min, span) for v in xs_km]
ys_n = [norm(v, y_min, span) for v in ys_km]

def beds_to_demand(beds, tier):
    if tier == 1:
        return round(min(18.0, max(7.0, beds / 60.0)), 1)
    elif tier == 2:
        return round(min(6.0,  max(3.0, beds / 50.0)), 1)
    else:
        return round(min(2.5,  max(1.5, beds / 44.0)), 1)

def tier_params(tier):
    if tier == 1: return 3, 1, 1.15
    if tier == 2: return 2, 1, 0.95
    return 1, 0, 0.80

rows = [{
    "node_id": 0, "name": "NUPCO Central Warehouse",
    "x": xs_n[0], "y": ys_n[0],
    "demand": 0.0, "alpha": 0, "U_max": 0, "eta": 0.0,
    "node_type": "depot",
}]

for idx, (name, lat, lon, beds, tier) in enumerate(HOSPITALS, start=1):
    umax, alpha, eta = tier_params(tier)
    rows.append({
        "node_id": idx, "name": name,
        "x": xs_n[idx], "y": ys_n[idx],
        "demand": beds_to_demand(beds, tier),
        "alpha": alpha, "U_max": umax, "eta": eta,
        "node_type": "customer",
    })

df = pd.DataFrame(rows)
nodes_out = df[["node_id","x","y","demand","alpha","U_max","eta","node_type"]]
nodes_out.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)

K, Q = 4, 50
total_d = df[df["node_type"]=="customer"]["demand"].sum()
params = {
    "instance": "CSDVRP-CS-Riyadh",
    "n": 30, "K": K, "Q": Q, "delta": 15, "mu": 1000, "cr": 1.0,
    "size_class": "M",
    "description": (
        "NUPCO Central Warehouse to 30 hospitals in Riyadh, Saudi Arabia. "
        "7 Tier-1 tertiary (demand 7-18, U_max=3), "
        "10 Tier-2 specialized (demand 3-6, U_max=2), "
        "13 Tier-3 primary (demand 1.5-2.5, U_max=1). "
        f"Total demand {total_d:.1f} / fleet capacity {K*Q} = "
        f"{total_d/(K*Q)*100:.1f}% utilisation."
    ),
    "data_source": (
        "Hospital locations: Ministry of Health Saudi Arabia interactive map "
        "(https://www.moh.gov.sa/en/eServices/interactive-maps/Pages/default.aspx). "
        "Depot: NUPCO As Sahafah District, Riyadh (www.nupco.com). "
        "Bed counts: MOH Annual Statistical Book 2023 and public hospital profiles."
    ),
}
with open(os.path.join(OUT_DIR, "params.json"), "w") as f:
    json.dump(params, f, indent=2)

pd.DataFrame([{"vehicle_id": k, "capacity": Q, "cost_per_km": 1.0,
                "vtype": "refrigerated_van"} for k in range(K)]).to_csv(
    os.path.join(OUT_DIR, "vehicles.csv"), index=False)

meta = df.copy()
meta["lat"]  = [DEPOT_LAT] + [h[1] for h in HOSPITALS]
meta["lon"]  = [DEPOT_LON] + [h[2] for h in HOSPITALS]
meta["tier"] = [0] + [h[4] for h in HOSPITALS]
meta["beds"] = [0] + [h[3] for h in HOSPITALS]
meta.to_csv(os.path.join(OUT_DIR, "nodes_full.csv"), index=False)

print(f"CSDVRP-CS-Riyadh  n=30  K={K}  Q={Q}")
print(f"Depot NUPCO: ({xs_n[0]}, {ys_n[0]})")
print(f"Total demand: {total_d:.1f}  Fleet cap: {K*Q}  Util: {total_d/(K*Q)*100:.1f}%")
print(f"Tier breakdown: T1={sum(1 for h in HOSPITALS if h[4]==1)}  "
      f"T2={sum(1 for h in HOSPITALS if h[4]==2)}  "
      f"T3={sum(1 for h in HOSPITALS if h[4]==3)}")
print(f"x=[{df['x'].min()}, {df['x'].max()}]  y=[{df['y'].min()}, {df['y'].max()}]")
cust = df[df["node_type"]=="customer"]
print(f"Demand range: [{cust['demand'].min()}, {cust['demand'].max()}]  "
      f"mean {cust['demand'].mean():.2f}")
