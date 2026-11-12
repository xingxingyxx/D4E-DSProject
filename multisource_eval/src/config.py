"""Paths used by every notebook. Override any of them with environment variables."""
import os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]                      # multisource_eval/
CLASSMATE_REPO = pathlib.Path(os.environ.get("DSP_CLASSMATE_REPO", ROOT.parent.parent / "2026_DS_Airbnb"))  # processed panels + stations
SLIM = pathlib.Path(os.environ.get("DSP_SLIM", ROOT / "data" / "insideairbnb_slim"))  # listings_slim.csv.gz, reviews_slim.csv.gz
EXT = pathlib.Path(os.environ.get("DSP_EXT", ROOT / "data" / "external"))             # output of 00_fetch_external_data
OUT = pathlib.Path(os.environ.get("DSP_OUT", ROOT / "results")); (OUT / "figures").mkdir(parents=True, exist_ok=True)
