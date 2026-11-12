"""External features joined to any set of (latitude, longitude) points. No Airbnb column is used except coordinates.
Sources: IDFM stations, OpenStreetMap POIs, DVF sales, IGN/INSEE IRIS + Filosofi 2021 income, SSMSI recorded crime."""
import json, gzip, zipfile, io, numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from common import station_features, EARTH

GROUPS = {  # OSM tag -> feature group
    "food": [("amenity", "restaurant"), ("amenity", "cafe"), ("amenity", "fast_food")],
    "nightlife": [("amenity", "bar"), ("amenity", "pub"), ("amenity", "nightclub")],
    "culture": [("amenity", "theatre"), ("amenity", "cinema"), ("tourism", "museum"), ("tourism", "gallery")],
    "attraction": [("tourism", "attraction"), ("tourism", "viewpoint"), ("tourism", "museum")],
    "hotel": [("tourism", "hotel")],
    "park": [("leisure", "park")],
    "daily_shop": [("shop", "supermarket"), ("shop", "bakery"), ("shop", "convenience")],
}
CRIME = {"Vols sans violence contre des personnes": "theft", "Vols violents sans arme": "violent_theft",
         "Violences physiques hors cadre familial": "assault", "Cambriolages de logement": "burglary",
         "Usage de stupéfiants": "drug_use", "Destructions et dégradations volontaires": "vandalism"}


def _q(lat, lon): return np.radians(np.c_[np.asarray(lat, float), np.asarray(lon, float)])


def load_osm(path):
    rows = []
    for e in json.load(open(path))["elements"]:
        lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else (e.get("center", {}).get("lat"), e.get("center", {}).get("lon"))
        if lat is None: continue
        t = e.get("tags", {})
        for g, pairs in GROUPS.items():
            if any(t.get(k) == v for k, v in pairs): rows.append((g, lat, lon))
    return pd.DataFrame(rows, columns=["group", "lat", "lon"]).drop_duplicates()


def osm_features(lat, lon, osm):
    q = _q(lat, lon); out = {}
    for g in GROUPS:
        p = osm[osm.group == g]; t = BallTree(_q(p.lat, p.lon), metric="haversine")
        for r in (250, 500, 1000):
            out[f"osm_{g}_n{r}"] = t.query_radius(q, r=r / 1000 / EARTH, count_only=True)
        out[f"osm_{g}_nearest_km"] = t.query(q, k=1)[0][:, 0] * EARTH
    a = osm[osm.group == "attraction"]; d, _ = BallTree(_q(a.lat, a.lon), metric="haversine").query(q, k=min(200, len(a)))
    d *= EARTH; out["osm_attraction_index"] = (np.where(d <= 3, 1 / np.maximum(d, 0.1), 0)).sum(1)
    return pd.DataFrame(out)


def load_dvf(paths):
    d = pd.concat([pd.read_csv(p, low_memory=False, usecols=["id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
                   "type_local", "surface_reelle_bati", "longitude", "latitude"]) for p in paths], ignore_index=True)
    d = d[d.nature_mutation == "Vente"]
    k = d.groupby("id_mutation").type_local.agg(lambda s: tuple(sorted(set(s.dropna()))))
    ok = k[k.isin([("Appartement",), ("Appartement", "Dépendance")])].index      # one sale = apartments (+ cellar/parking)
    a = d[d.id_mutation.isin(ok) & (d.type_local == "Appartement")]
    a = a.groupby("id_mutation").agg(price=("valeur_fonciere", "first"), surface=("surface_reelle_bati", "sum"), n_apt=("type_local", "size"),
                                     lat=("latitude", "first"), lon=("longitude", "first"), date=("date_mutation", "first"))
    a = a[(a.n_apt == 1) & (a.surface >= 9) & a.lat.notna()]
    a["eur_m2"] = a.price / a.surface
    return a[a.eur_m2.between(3000, 40000)].reset_index()


def dvf_features(lat, lon, dvf):
    q = _q(lat, lon); t = BallTree(_q(dvf.lat, dvf.lon), metric="haversine"); v = dvf.eur_m2.values; s = dvf.surface.values; out = {}
    for r in (250, 500):
        idx = t.query_radius(q, r=r / 1000 / EARTH)
        out[f"dvf_eur_m2_med_{r}m"] = np.array([np.median(v[i]) if len(i) >= 5 else np.nan for i in idx])
        out[f"dvf_sales_n_{r}m"] = np.array([len(i) for i in idx])
    _, idx = t.query(q, k=50)
    out["dvf_eur_m2_med_knn50"] = np.median(v[idx], 1)
    out["dvf_eur_m2_p90_knn50"] = np.percentile(v[idx], 90, axis=1)
    out["dvf_surface_med_knn50"] = np.median(s[idx], 1)
    return pd.DataFrame(out)


def load_iris(geojson, disp_zip, dec_zip):
    g = json.load(open(geojson)); polys, codes, arr, typ = [], [], [], []
    for f in g["features"]:
        p = f["properties"]; polys.append(shape(f["geometry"])); codes.append(p["iris_code"][0]); arr.append(p["com_arm_code"][0]); typ.append(p["iris_type"])
    def rd(z):
        zf = zipfile.ZipFile(z); n = [x for x in zf.namelist() if not x.startswith("meta")][0]
        return pd.read_csv(zf.open(n), sep=";", dtype={"IRIS": str}, decimal=",", na_values=["s", "so", "nd", "ns"])
    disp, dec = rd(disp_zip), rd(dec_zip)
    inc = disp[["IRIS", "DISP_MED21", "DISP_TP6021", "DISP_RD21", "DISP_GI21", "DISP_PPEN21", "DISP_PPAT21"]].merge(
        dec[["IRIS", "DEC_MED21", "DEC_D921", "DEC_PIMP21"]], on="IRIS", how="outer")
    meta = pd.DataFrame({"iris_code": codes, "arr_code": arr, "iris_type": typ})
    return polys, meta, inc


def iris_features(lat, lon, polys, meta, inc):
    tree = STRtree(polys); pts = [Point(x, y) for x, y in zip(lon, lat)]
    hit = np.full(len(pts), -1)
    pi, gi = tree.query(pts, predicate="within")       # point index, polygon index
    hit[pi] = gi
    miss = np.where(hit < 0)[0]                          # points just outside any polygon (coordinate jitter at the edge)
    for i in miss:
        hit[i] = int(np.atleast_1d(tree.query_nearest(pts[i]))[0])
    m = meta.iloc[hit].reset_index(drop=True)
    f = m.merge(inc, left_on="iris_code", right_on="IRIS", how="left").drop(columns=["IRIS"])
    f.columns = ["iris_code", "arr_code", "iris_type"] + [f"inc_{c.lower()}" for c in f.columns[3:]]
    return f, len(miss)


def load_crime(path, year="2024"):
    c = pd.read_csv(path, sep=";", dtype=str); c = c[(c.annee == year) & c.indicateur.isin(CRIME) & (c.CODGEO_2026 != "75056")]
    c["rate"] = pd.to_numeric(c.taux_pour_mille.str.replace(",", "."), errors="coerce")
    w = c.pivot_table(index="CODGEO_2026", columns="indicateur", values="rate").rename(columns=lambda k: f"crime_{CRIME[k]}_per1000")
    return w


def build(lat, lon, ext_dir, stations):
    osm = load_osm(f"{ext_dir}/osm_paris_poi.json")
    dvf = load_dvf([f"{ext_dir}/dvf_75_{y}.csv.gz" for y in (2022, 2023, 2024, 2025)])
    polys, meta, inc = load_iris(f"{ext_dir}/iris_paris.geojson", f"{ext_dir}/filosofi_iris_2021_disp.zip", f"{ext_dir}/filosofi_iris_2021_dec.zip")
    crime = load_crime(f"{ext_dir}/crime_paris.csv")
    st = station_features(lat, lon, stations); st.columns = ["ext_" + c for c in st.columns]
    ir, nmiss = iris_features(np.asarray(lat), np.asarray(lon), polys, meta, inc)
    cr = ir[["arr_code"]].merge(crime, left_on="arr_code", right_index=True, how="left").drop(columns="arr_code").reset_index(drop=True)
    info = {"osm_points": len(osm), "dvf_sales": len(dvf), "iris_polygons": len(polys), "points_outside_iris_snapped": int(nmiss),
            "iris_with_income": float(ir.inc_disp_med21.notna().mean()), "points_with_crime": float(cr.notna().all(1).mean())}
    o = osm_features(lat, lon, osm); o.columns = ["ext_" + c for c in o.columns]
    d = dvf_features(lat, lon, dvf); d.columns = ["ext_" + c for c in d.columns]
    i = ir.drop(columns=["iris_code", "arr_code"]); i.columns = ["ext_" + c for c in i.columns]
    c = cr.copy(); c.columns = ["ext_" + x for x in c.columns]
    X = pd.concat([st, o, d, i, c], axis=1); X["ext_iris_type"] = X["ext_iris_type"].astype("category")
    groups = {"stations": list(st.columns), "osm": list(o.columns), "dvf": list(d.columns), "iris_income": list(i.columns), "crime": list(c.columns)}
    return X, groups, info, dvf
