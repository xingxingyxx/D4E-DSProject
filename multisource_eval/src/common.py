"""Shared helpers: external features from coordinates, grouped CV, metrics."""
import numpy as np, pandas as pd, warnings
from sklearn.neighbors import BallTree
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, mean_absolute_error, d2_tweedie_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
EARTH = 6371.0088
SEED = 42
HP = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=20, min_samples_leaf=30, l2_regularization=4.0,
          categorical_features="from_dtype", random_state=SEED)

# Placeholder for OSM tourism=attraction: 12 hand-coded major landmarks (NOT a dataset; to be replaced by OSM).
LANDMARKS = {"Tour Eiffel": (48.8584, 2.2945), "Louvre": (48.8606, 2.3376), "Notre-Dame": (48.8530, 2.3499),
             "Sacre-Coeur": (48.8867, 2.3431), "Arc de Triomphe": (48.8738, 2.2950), "Centre Pompidou": (48.8606, 2.3522),
             "Musee d'Orsay": (48.8600, 2.3266), "Opera Garnier": (48.8720, 2.3316), "Pantheon": (48.8462, 2.3464),
             "Bastille": (48.8532, 2.3692), "Champs-Elysees": (48.8698, 2.3078), "Jardin du Luxembourg": (48.8462, 2.3372)}


def _tree(lat, lon):
    return BallTree(np.radians(np.c_[lat, lon]), metric="haversine")


def station_features(lat, lon, stations):
    """Official IDFM stations -> access features. No raw coordinates are returned."""
    q = np.radians(np.c_[lat, lon]); out = {}
    met, rer = stations[stations.metro == 1], stations[stations.rer == 1]
    tm, tr, ta = _tree(met.latitude, met.longitude), _tree(rer.latitude, rer.longitude), _tree(stations.latitude, stations.longitude)
    d, _ = tm.query(q, k=3); d *= EARTH
    out["st_metro_nearest_km"] = d[:, 0]; out["st_metro_mean3_km"] = d.mean(1)
    out["st_rer_nearest_km"] = tr.query(q, k=1)[0][:, 0] * EARTH
    for r in (300, 500, 1000):
        out[f"st_metro_n_{r}m"] = tm.query_radius(q, r=r / 1000 / EARTH, count_only=True)
    out["st_rer_n_1000m"] = tr.query_radius(q, r=1.0 / EARTH, count_only=True)
    out["st_any_n_1000m"] = ta.query_radius(q, r=1.0 / EARTH, count_only=True)
    lines = stations.res_com.astype(str).values
    for r in (500, 1000):
        idx = ta.query_radius(q, r=r / 1000 / EARTH)
        out[f"st_lines_{r}m"] = np.array([len(set(lines[i])) for i in idx])
    return pd.DataFrame(out)


def landmark_features(lat, lon):
    q = np.radians(np.c_[lat, lon]); P = np.array(list(LANDMARKS.values())); t = _tree(P[:, 0], P[:, 1])
    d, _ = t.query(q, k=len(P)); d *= EARTH
    return pd.DataFrame({"lm_nearest_km": d[:, 0], "lm_n_1km": (d <= 1).sum(1), "lm_n_2km": (d <= 2).sum(1),
                         "lm_attr_index": (1 / np.maximum(d, 0.1)).sum(1)})


def cell_1km(lat, lon):
    y = np.floor((np.asarray(lat) - 48.80) * 111.2).astype(int)
    x = np.floor((np.asarray(lon) - 2.22) * 111.2 * np.cos(np.radians(48.86))).astype(int)
    return y * 1000 + x


def block_3km(lat, lon):
    y = np.floor((np.asarray(lat) - 48.80) * 111.2 / 3).astype(int)
    x = np.floor((np.asarray(lon) - 2.22) * 111.2 * np.cos(np.radians(48.86)) / 3).astype(int)
    return y * 100 + x


def folds(df, protocol, n=5):
    """host: no host in both train and test. spatial: 1 km cells held out; block: 3 km blocks held out.
    For spatial and block, hosts that appear in the test fold are also removed from training."""
    g = {"host": df["host_id"].values, "spatial": df["cell"].values, "block": df["block"].values}[protocol]
    for tr, te in GroupKFold(n_splits=n).split(df, groups=g):
        if protocol in ("spatial", "block"):
            tr = tr[~np.isin(df["host_id"].values[tr], df["host_id"].values[te])]
        yield tr, te


def oof(df, cols, y, kind, protocol):
    """Out-of-fold predictions. kind: reg | poisson | clf"""
    X = df[cols]; p = np.full(len(df), np.nan); fid = np.zeros(len(df), int)
    for k, (tr, te) in enumerate(folds(df, protocol)):
        if kind == "clf":
            m = HGC(**HP).fit(X.iloc[tr], y[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
        else:
            m = HGR(loss="poisson" if kind == "poisson" else "squared_error", **HP).fit(X.iloc[tr], y[tr]); p[te] = m.predict(X.iloc[te])
        fid[te] = k
    return p, fid


def score(y, p, kind):
    if kind == "reg":
        return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p)}
    if kind == "poisson":
        return {"D2_poisson": d2_tweedie_score(y, np.maximum(p, 1e-6), power=1), "MAE": mean_absolute_error(y, p), "Spearman": spearmanr(y, p)[0]}
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}


def run_task(name, df, y, kind, sets, out_rows, protocols=("host", "spatial"), only=None):
    """sets: dict label -> column list. Records pooled OOF metrics and per-fold primary metric.
    only: optional dict protocol -> list of set labels to run for that protocol."""
    prim = {"reg": "R2", "poisson": "D2_poisson", "clf": "AUC"}[kind]; preds = {}
    for protocol in protocols:
        for label, cols in sets.items():
            if only and protocol in only and label not in only[protocol]: continue
            p, fid = oof(df, cols, y, kind, protocol); preds[(protocol, label)] = p
            s = score(y, p, kind)
            fs = [score(y[fid == k], p[fid == k], kind)[prim] for k in range(5)]
            out_rows.append({"task": name, "protocol": protocol, "features": label, "n": len(df), "n_features": len(cols),
                             "primary": prim, **s, "fold_min": min(fs), "fold_max": max(fs), "folds": ";".join(f"{v:.4f}" for v in fs)})
            print(f"{name:28s} {protocol:8s} {label:34s} " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)
    return preds


def cat(df, cols):
    for c in cols:
        df[c] = df[c].astype("category")
    return df
