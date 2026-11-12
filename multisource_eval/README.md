# Paris Airbnb: does external open data improve price prediction?

Task design, data choices, checks, results and the work plan are in **[PROJECT_DESIGN.md](PROJECT_DESIGN.md)**.

## Headline

Predicting the nightly price of 31,044 Paris entire homes (Inside Airbnb, September 2026):

| Features | R², unseen hosts | R², unseen 3 km blocks |
|---|---|---|
| A: listing attributes only | 0.543 | 0.522 |
| B: A + five external sources, no coordinates | **0.655** | **0.639** |
| C: A + latitude, longitude, arrondissement | 0.654 | 0.610 |

Merging external data cuts the error by 11.4 EUR per night (95% CI 10.6–12.1) and generalises better than coordinates to unseen areas.

## Run it

```bash
pip install -r requirements.txt
```

1. **Inside Airbnb data.**
   - Put `listings_slim.csv.gz` and `reviews_slim.csv.gz` in `data/insideairbnb_slim/`.
   - Clone the classmate's repository `2026_DS_Airbnb` next to this one. It provides the cleaned price sample, the June panels and the IDFM stations.
2. **External data.** Run `notebooks/00_fetch_external_data.ipynb`. Downloaded files are checked against `data/external/manifest.json`.
3. **Analysis.** Run notebooks `01` → `06` in order. Results go to `results/`.
   - Runtime: 02 takes about 5 minutes; the whole sequence takes about 15 minutes on a laptop.

Paths can be changed with the environment variables `DSP_CLASSMATE_REPO`, `DSP_SLIM`, `DSP_EXT` and `DSP_OUT` (see `src/config.py`).

Raw data is not committed. Inside Airbnb data is not redistributed here; the external files are re-downloaded from their publishers.

## Sources

- Inside Airbnb (listings, reviews)
- Île-de-France Mobilités (stations)
- OpenStreetMap contributors (ODbL)
- DGFiP DVF géolocalisées (Etalab)
- IGN/INSEE IRIS and INSEE Filosofi 2021
- SSMSI recorded crime
