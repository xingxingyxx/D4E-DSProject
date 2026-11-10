"""
Paris Airbnb host revenue forecast - preliminary analysis.

Inputs (Inside Airbnb, Paris, detailed files from one snapshot):
    listings.csv.gz   detailed listings
    reviews.csv.gz    detailed reviews (listing_id, date, ...)

Usage:
    python prelim_analysis.py --listings listings.csv.gz --reviews reviews.csv.gz --out results/

Design (see project plan):
  - cutoff = snapshot date - 12 months
  - features: listing attributes + review history BEFORE cutoff
  - target:   estimated revenue in the 12 months AFTER cutoff
              nights = min(reviews / r * max(L, min_nights), 0.70 * 365)  [Inside Airbnb "San Francisco model"]
              revenue = nights * listed nightly price
  - scenario A caps nights at 90 (Paris primary-residence limit since 2025); scenario B no cap
  - models: naive persistence baseline vs gradient boosting (quantile P10/P50/P90)
Only pandas / numpy / scikit-learn / matplotlib are needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

REVIEW_RATE = 0.50          # share of stays that leave a review
STAY_NIGHTS = 5.2           # Paris value implied by Inside Airbnb (est. occupancy / reviews = 10.4 nights per review at r = 50%); their generic default is 3
OCC_CAP = 0.70              # occupancy cap
LEGAL_CAP = 90              # Paris primary residence, nights / year (from 2025)
CENTER = (48.8530, 2.3499)  # Notre-Dame, used as city-centre reference
SEED = 42


def parse_price(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def nights_from_reviews(n_reviews, min_nights, review_rate=REVIEW_RATE, stay=None):
    stay = STAY_NIGHTS if stay is None else stay
    stay_len = np.maximum(stay, min_nights.fillna(stay).clip(upper=30))
    return np.minimum(n_reviews / review_rate * stay_len, OCC_CAP * 365)


def load(listings_path, reviews_path):
    L = pd.read_csv(listings_path, low_memory=False)
    R = pd.read_csv(reviews_path, usecols=["listing_id", "date"], parse_dates=["date"])
    snapshot = pd.to_datetime(L["last_scraped"]).max()
    return L, R, snapshot


def build_frame(L, R, snapshot):
    cutoff = snapshot - pd.DateOffset(months=12)
    df = L.copy()
    df["price_eur"] = parse_price(df["price"])
    funnel = {"all_listings": int(len(df))}
    df = df[df["room_type"] == "Entire home/apt"]; funnel["entire_home"] = int(len(df))
    df = df[df["price_eur"].notna()]; funnel["with_price"] = int(len(df))
    df = df[df["price_eur"].between(20, 1000)]; funnel["price_20_1000"] = int(len(df))
    # minimum stay >= 30 nights = mid-term / mobility lease, not tourist rental
    df = df[df["minimum_nights"].fillna(1) < 30]; funnel["min_nights_lt_30"] = int(len(df))

    pre = R[R["date"] < cutoff]
    post = R[(R["date"] >= cutoff) & (R["date"] < snapshot)]
    last12_pre = pre[pre["date"] >= cutoff - pd.DateOffset(months=12)]

    agg = pd.DataFrame({
        "rev_total_pre": pre.groupby("listing_id").size(),
        "rev_12m_pre": last12_pre.groupby("listing_id").size(),
        "first_review_pre": pre.groupby("listing_id")["date"].min(),
        "rev_12m_post": post.groupby("listing_id").size(),
    })
    df = df.merge(agg, left_on="id", right_index=True, how="left")
    # keep listings with history before the cutoff (cold-start handled separately)
    df = df[df["rev_total_pre"].fillna(0) > 0].copy(); funnel["reviewed_before_cutoff"] = int(len(df))
    for c in ["rev_12m_pre", "rev_12m_post"]:
        df[c] = df[c].fillna(0)
    df["months_active_pre"] = (cutoff - df["first_review_pre"]).dt.days / 30.44

    df["nights_pre"] = nights_from_reviews(df["rev_12m_pre"], df["minimum_nights"])
    df["nights_post"] = nights_from_reviews(df["rev_12m_post"], df["minimum_nights"])
    df["rev_eur_post_B"] = df["nights_post"] * df["price_eur"]
    df["rev_eur_post_A"] = np.minimum(df["nights_post"], LEGAL_CAP) * df["price_eur"]
    df["rev_eur_pre_B"] = df["nights_pre"] * df["price_eur"]
    df["rev_eur_pre_A"] = np.minimum(df["nights_pre"], LEGAL_CAP) * df["price_eur"]

    df["dist_center_km"] = haversine_km(df["latitude"], df["longitude"], *CENTER)
    df["bathrooms_n"] = pd.to_numeric(df.get("bathrooms_text", pd.Series(index=df.index, dtype=str))
                                      .astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    df["superhost"] = (df.get("host_is_superhost") == "t").astype(int)
    df["instant"] = (df.get("instant_bookable") == "t").astype(int)
    return df, cutoff, funnel


NUM = ["accommodates", "bedrooms", "beds", "bathrooms_n", "dist_center_km", "minimum_nights",
       "review_scores_rating", "review_scores_location", "number_of_reviews", "superhost", "instant",
       "calculated_host_listings_count", "price_eur", "rev_total_pre", "rev_12m_pre", "months_active_pre"]
CAT = ["neighbourhood_cleansed"]


def make_X(df):
    cols = [c for c in NUM if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    if "number_of_reviews" in X:
        # number_of_reviews is as-of snapshot -> leaks post-cutoff reviews; drop it
        X = X.drop(columns="number_of_reviews")
    for c in CAT:
        X[c] = df[c].astype("category")
    return X


def fit_quantiles(Xtr, ytr, Xte):
    preds = {}
    models = {}
    for q in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400,
                                          learning_rate=0.05, categorical_features="from_dtype",
                                          random_state=SEED)
        m.fit(Xtr, ytr)
        preds[q] = m.predict(Xte)
        models[q] = m
    return models, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--stay", type=float, default=STAY_NIGHTS, help="average nights per stay")
    a = ap.parse_args()
    globals()["STAY_NIGHTS"] = a.stay
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    L, R, snapshot = load(a.listings, a.reviews)
    df, cutoff, funnel = build_frame(L, R, snapshot)
    X = make_X(df)
    summary = {
        "snapshot": str(snapshot.date()), "cutoff": str(cutoff.date()),
        "funnel": funnel,
    }
    if "estimated_occupancy_l365d" in df:
        summary["check_vs_insideairbnb_occupancy"] = {
            "spearman_corr": float(df["nights_post"].corr(df["estimated_occupancy_l365d"], method="spearman")),
            "median_ours": float(df["nights_post"].median()),
            "median_insideairbnb": float(df["estimated_occupancy_l365d"].median())}
    summary["overall"] = {
        "median_price_eur": float(df["price_eur"].median()),
        "median_nights": float(df["nights_post"].median()),
        "share_zero_reviews_post": float((df["rev_12m_post"] == 0).mean()),
        "median_revenue_B_eur": float(df["rev_eur_post_B"].median()),
        "median_revenue_A_eur": float(df["rev_eur_post_A"].median()),
        "p25_revenue_B_eur": float(df["rev_eur_post_B"].quantile(.25)),
        "p75_revenue_B_eur": float(df["rev_eur_post_B"].quantile(.75)),
        "share_over_90_nights": float((df["nights_post"] > LEGAL_CAP).mean())}

    # ---------- revenue model, scenario B (uncapped) and A (90-night cap) ----------
    results = {}
    for scen in ("B", "A"):
        y = np.log1p(df[f"rev_eur_post_{scen}"])
        idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED)
        models, p = fit_quantiles(X.loc[idx_tr], y.loc[idx_tr], X.loc[idx_te])
        y_true = df.loc[idx_te, f"rev_eur_post_{scen}"].values
        p50 = np.expm1(p[0.5])
        naive = df.loc[idx_te, f"rev_eur_pre_{scen}"].values
        p10, p90 = np.expm1(p[0.1]), np.expm1(p[0.9])
        results[scen] = {
            "MAE_model_eur": float(np.mean(np.abs(p50 - y_true))),
            "MAE_naive_last_year_eur": float(np.mean(np.abs(naive - y_true))),
            "median_actual_est_revenue_eur": float(np.median(y_true)),
            "MedAE_model_eur": float(np.median(np.abs(p50 - y_true))),
            "MedAE_naive_eur": float(np.median(np.abs(naive - y_true))),
            "spearman_model": float(pd.Series(p50).corr(pd.Series(y_true), method="spearman")),
            "spearman_naive": float(pd.Series(naive).corr(pd.Series(y_true), method="spearman")),
            "P10_P90_coverage": float(np.mean((y_true >= p10) & (y_true <= p90))),
        }
        if scen == "B":
            pi = permutation_importance(models[0.5], X.loc[idx_te], y.loc[idx_te], n_repeats=5,
                                        random_state=SEED, scoring="neg_mean_absolute_error")
            imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
            ax = imp.tail(12).plot.barh(figsize=(7, 5), color="#4C6EF5")
            ax.set_title("Permutation importance - revenue model (P50, scenario B)")
            ax.set_xlabel("MAE increase on log revenue when feature is shuffled")
            plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=150); plt.close()
            imp.sort_values(ascending=False).to_csv(out / "feature_importance.csv")
    summary["revenue_models"] = results

    # ---------- price model (no review history) ----------
    Xp = X.drop(columns=[c for c in ["price_eur", "rev_total_pre", "rev_12m_pre", "months_active_pre"] if c in X])
    yp = np.log(df["price_eur"])
    itr, ite = train_test_split(df.index, test_size=0.2, random_state=SEED)
    mp = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                       categorical_features="from_dtype", random_state=SEED)
    mp.fit(Xp.loc[itr], yp.loc[itr])
    pp = np.exp(mp.predict(Xp.loc[ite]))
    ytrue = df.loc[ite, "price_eur"].values
    ss_res = np.sum((np.log(ytrue) - np.log(pp)) ** 2)
    ss_tot = np.sum((np.log(ytrue) - np.log(ytrue).mean()) ** 2)
    summary["price_model"] = {"MAE_eur": float(np.mean(np.abs(pp - ytrue))),
                              "R2_log_price": float(1 - ss_res / ss_tot),
                              "median_price_eur": float(np.median(ytrue))}

    # ---------- by arrondissement ----------
    by = df.groupby("neighbourhood_cleansed").agg(
        listings=("id", "size"),
        median_price_eur=("price_eur", "median"),
        median_nights_est=("nights_post", "median"),
        median_revenue_B_eur=("rev_eur_post_B", "median"),
        median_revenue_A_eur=("rev_eur_post_A", "median"),
        share_over_90_nights=("nights_post", lambda s: float((s > LEGAL_CAP).mean())),
    ).sort_values("median_revenue_B_eur", ascending=False).round(1)
    by.to_csv(out / "by_arrondissement.csv")

    # ---------- sensitivity to review rate ----------
    sens = []
    for r in (0.30, 0.50, 0.72):
        n = nights_from_reviews(df["rev_12m_post"], df["minimum_nights"], review_rate=r)
        sens.append({"review_rate": r, "median_nights": float(np.median(n)),
                     "median_revenue_B_eur": float(np.median(n * df["price_eur"])),
                     "median_revenue_A_eur": float(np.median(np.minimum(n, LEGAL_CAP) * df["price_eur"])),
                     "share_over_90": float((n > LEGAL_CAP).mean())})
    summary["sensitivity_review_rate"] = sens

    # ---------- monthly seasonality (all reviews, last 5 years) ----------
    recent = R[R["date"] >= snapshot - pd.DateOffset(years=5)]
    season = recent.groupby(recent["date"].dt.month).size()
    season = (season / season.mean()).round(3)
    season.to_csv(out / "seasonality_index.csv", header=["index"])
    monthly = R.set_index("date").resample("MS").size()
    monthly = monthly[monthly.index >= snapshot - pd.DateOffset(years=6)]
    ax = monthly.plot(figsize=(8, 3.5), color="#4C6EF5")
    ax.set_title("Paris Airbnb reviews per month (proxy for stays)"); ax.set_ylabel("reviews")
    plt.tight_layout(); plt.savefig(out / "monthly_reviews.png", dpi=150); plt.close()

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(by.head(20).to_string())


if __name__ == "__main__":
    main()
