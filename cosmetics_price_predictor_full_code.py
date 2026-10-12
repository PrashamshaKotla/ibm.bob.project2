"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          COSMETICS PRICE PREDICTOR — COMPLETE SOURCE CODE                  ║
║          Python · scikit-learn · FastAPI · Streamlit · Plotly              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This single file contains all project code organised into sections:       ║
║                                                                              ║
║   SECTION 1 — DATA PREPROCESSING & FEATURE ENGINEERING  (preprocess.py)   ║
║   SECTION 2 — MODEL TRAINING                             (models/train.py) ║
║   SECTION 3 — FASTAPI BACKEND                            (backend/app.py)  ║
║   SECTION 4 — STREAMLIT FRONTEND                   (frontend/streamlit_app)║
║                                                                              ║
║  ── HOW TO RUN ──────────────────────────────────────────────────────────  ║
║  1. pip install -r requirements.txt                                         ║
║  2. python models/train.py          (train & save model)                   ║
║  3. uvicorn backend.app:app --reload --port 8000  (start API)              ║
║  4. streamlit run frontend/streamlit_app.py       (start frontend)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
#  SECTION 1 — DATA PREPROCESSING & FEATURE ENGINEERING
#  File: cosmetics_price_predictor/data/preprocess.py
# ============================================================================

"""
preprocess.py
Feature engineering for the Cosmetics Price Predictor.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------------
# Key ingredient groups that influence premium pricing
# ------------------------------------------------------------------
PREMIUM_INGREDIENTS = [
    "gold", "caviar", "retinol", "retinyl", "vitamin c", "ascorbic acid",
    "niacinamide", "hyaluronic acid", "sodium hyaluronate", "peptide",
    "collagen", "ceramide", "squalane", "argan", "marula", "rosehip",
    "bakuchiol", "adenosine", "coenzyme", "resveratrol", "ferulic",
]

COMMON_FILLER_INGREDIENTS = [
    "water", "glycerin", "dimethicone", "phenoxyethanol",
    "carbomer", "xanthan gum", "sodium benzoate",
]


def count_ingredients(text: str) -> int:
    """Count the number of ingredients in a comma-separated list."""
    if not isinstance(text, str):
        return 0
    return len([i.strip() for i in text.split(",") if i.strip()])


def count_premium(text: str) -> int:
    """Count how many premium ingredient keywords appear."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for kw in PREMIUM_INGREDIENTS if kw in text_lower)


def count_fillers(text: str) -> int:
    """Count how many commodity/filler ingredient keywords appear."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for kw in COMMON_FILLER_INGREDIENTS if kw in text_lower)


def premium_ratio(text: str) -> float:
    """Ratio of premium ingredients to total ingredients."""
    total = count_ingredients(text)
    if total == 0:
        return 0.0
    return count_premium(text) / total


def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """
    Load the cosmetics CSV, clean it, and engineer features.

    Returns a DataFrame ready for model training / inference.
    Added columns:
      - ingredient_count      : total number of ingredients
      - premium_count         : premium ingredient keywords found
      - filler_count          : filler ingredient keywords found
      - premium_ratio         : premium / total
      - skin_type_sum         : sum of Combination+Dry+Normal+Oily+Sensitive
      - label_encoded         : integer-encoded product category
      - brand_encoded         : integer-encoded brand
      - brand_mean_price      : brand-level mean price (target-encoded)
      - log_price             : log1p(Price) — used as model target
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # --- Clean strings ---
    df["Label"] = df["Label"].str.strip()
    df["Brand"] = df["Brand"].str.strip().str.upper()
    df["Name"] = df["Name"].str.strip()
    df["Ingredients"] = df["Ingredients"].fillna("")

    # --- Ingredient-based features ---
    df["ingredient_count"] = df["Ingredients"].apply(count_ingredients)
    df["premium_count"]    = df["Ingredients"].apply(count_premium)
    df["filler_count"]     = df["Ingredients"].apply(count_fillers)
    df["premium_ratio"]    = df["Ingredients"].apply(premium_ratio)

    # --- Skin type aggregate ---
    skin_cols = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
    df["skin_type_sum"] = df[skin_cols].sum(axis=1)

    # --- Category & brand encoding ---
    le_label = LabelEncoder()
    df["label_encoded"] = le_label.fit_transform(df["Label"])

    le_brand = LabelEncoder()
    df["brand_encoded"] = le_brand.fit_transform(df["Brand"])

    # Brand-level mean price (target encoding)
    brand_mean = df.groupby("Brand")["Price"].mean().rename("brand_mean_price")
    df = df.merge(brand_mean, on="Brand", how="left")

    # --- Log target ---
    df["log_price"] = np.log1p(df["Price"])

    # Clip rank to valid range
    df["Rank"] = df["Rank"].clip(0, 5)

    return df


# Feature columns used by the model
FEATURE_COLS = [
    "label_encoded",
    "brand_encoded",
    "brand_mean_price",
    "Rank",
    "ingredient_count",
    "premium_count",
    "filler_count",
    "premium_ratio",
    "skin_type_sum",
    "Combination",
    "Dry",
    "Normal",
    "Oily",
    "Sensitive",
]

TARGET_COL = "log_price"


# ============================================================================
#  SECTION 2 — MODEL TRAINING
#  File: cosmetics_price_predictor/models/train.py
#  Run: python models/train.py
# ============================================================================

"""
train.py
Train Random Forest, GradientBoosting, and Ridge regressors on the Cosmetics
dataset. Saves the best model + encoders as pickle artefacts.
"""

# --- To run as a script, uncomment and execute this block independently ---
#
# import os
# import sys
# import pickle
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.preprocessing import LabelEncoder
#
# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))
# DATA_PATH  = ROOT / "data" / "cosmetics.csv"
# MODELS_DIR = ROOT / "models"
# MODELS_DIR.mkdir(exist_ok=True)
#
# def train():
#     print(">> Loading & engineering features ...")
#     df = load_and_engineer(str(DATA_PATH))
#
#     X = df[FEATURE_COLS].copy()
#     y = df[TARGET_COL].copy()
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.20, random_state=42
#     )
#
#     candidates = {
#         "RandomForest": RandomForestRegressor(
#             n_estimators=300, max_depth=12,
#             min_samples_leaf=3, random_state=42, n_jobs=-1,
#         ),
#         "GradientBoosting": GradientBoostingRegressor(
#             n_estimators=300, learning_rate=0.05,
#             max_depth=5, subsample=0.8, random_state=42,
#         ),
#         "Ridge": Ridge(alpha=1.0),
#     }
#
#     print("\n  Cross-validated R2 (5-fold) on training set:")
#     best_name, best_model, best_cv = None, None, -np.inf
#     cv_results = {}
#     for name, model in candidates.items():
#         cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
#         mean_cv = cv_scores.mean()
#         cv_results[name] = round(float(mean_cv), 4)
#         print(f"    {name:25s}  R2 = {mean_cv:.4f}  +/- {cv_scores.std():.4f}")
#         if mean_cv > best_cv:
#             best_cv = mean_cv
#             best_name = name
#             best_model = model
#
#     print(f"\n  Best model: {best_name}")
#     best_model.fit(X_train, y_train)
#
#     y_pred_log = best_model.predict(X_test)
#     y_pred     = np.expm1(y_pred_log)
#     y_true     = np.expm1(y_test)
#
#     mae = mean_absolute_error(y_true, y_pred)
#     r2  = r2_score(y_true, y_pred)
#     print(f"\n  Test-set MAE : ${mae:.2f}")
#     print(f"  Test-set R2  : {r2:.4f}")
#
#     brand_mean  = df.groupby("Brand")["Price"].mean().to_dict()
#     label_enc   = LabelEncoder().fit(df["Label"])
#     brand_enc   = LabelEncoder().fit(df["Brand"])
#
#     feat_importance = {}
#     if hasattr(best_model, "feature_importances_"):
#         feat_importance = dict(zip(FEATURE_COLS, best_model.feature_importances_.tolist()))
#
#     artefacts = {
#         "model":            best_model,
#         "label_encoder":    label_enc,
#         "brand_encoder":    brand_enc,
#         "brand_mean_price": brand_mean,
#         "feature_cols":     FEATURE_COLS,
#         "model_name":       best_name,
#         "metrics": {
#             "test_mae":   round(float(mae), 2),
#             "test_r2":    round(float(r2), 4),
#             "cv_results": cv_results,
#         },
#         "feature_importance": feat_importance,
#         "all_labels":  sorted(df["Label"].unique().tolist()),
#         "all_brands":  sorted(df["Brand"].unique().tolist()),
#         "price_stats": {
#             "min":    float(df["Price"].min()),
#             "max":    float(df["Price"].max()),
#             "mean":   float(df["Price"].mean()),
#             "median": float(df["Price"].median()),
#         },
#         "label_price_stats": df.groupby("Label")["Price"]
#                                .agg(["mean","min","max"])
#                                .rename(columns={"mean":"avg","min":"low","max":"high"})
#                                .to_dict(orient="index"),
#     }
#
#     save_path = MODELS_DIR / "artefacts.pkl"
#     with open(save_path, "wb") as f:
#         pickle.dump(artefacts, f)
#
#     with open(MODELS_DIR / "metrics.json", "w") as f:
#         json.dump({
#             "model_name":         best_name,
#             "test_mae":           round(float(mae), 2),
#             "test_r2":            round(float(r2), 4),
#             "cv_results":         cv_results,
#             "feature_importance": {k: round(v, 6) for k, v in feat_importance.items()},
#         }, f, indent=2)
#
#     print(f"\nArtefacts saved to {save_path}")
#     return artefacts
#
# if __name__ == "__main__":
#     train()


# ============================================================================
#  SECTION 3 — FASTAPI BACKEND
#  File: cosmetics_price_predictor/backend/app.py
#  Run: uvicorn backend.app:app --reload --port 8000
# ============================================================================

"""
FastAPI backend for the Cosmetics Price Predictor.
Exposes three endpoints:
  GET  /          — health check
  GET  /metadata  — available labels, brands, price stats
  POST /predict   — price prediction
"""

import sys
import pickle
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup (run from cosmetics_price_predictor/ root) ──
_BACKEND_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
_PROJECT_ROOT = _BACKEND_ROOT.parent if _BACKEND_ROOT.name == "backend" else _BACKEND_ROOT
_ARTEFACT_PATH = _PROJECT_ROOT / "models" / "artefacts.pkl"


def _load_artefacts():
    if not _ARTEFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artefacts not found at {_ARTEFACT_PATH}. "
            "Run `python models/train.py` first."
        )
    with open(_ARTEFACT_PATH, "rb") as f:
        return pickle.load(f)


_artefacts      = _load_artefacts()
_model          = _artefacts["model"]
_label_enc      = _artefacts["label_encoder"]
_brand_enc      = _artefacts["brand_encoder"]
_brand_mean_d   = _artefacts["brand_mean_price"]


# ── Pydantic schemas ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    label:       str   = Field(...,  example="Moisturizer")
    brand:       str   = Field(...,  example="LA MER")
    rank:        float = Field(4.0,  ge=0.0, le=5.0)
    ingredients: str   = Field("",  description="Comma-separated INCI list")
    combination: int   = Field(0,    ge=0, le=1)
    dry:         int   = Field(0,    ge=0, le=1)
    normal:      int   = Field(0,    ge=0, le=1)
    oily:        int   = Field(0,    ge=0, le=1)
    sensitive:   int   = Field(0,    ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_price:       float
    predicted_price_range: dict
    model_name:            str
    confidence_note:       str


# ── FastAPI application ─────────────────────────────────────────────────────

app = FastAPI(
    title="Cosmetics Price Predictor API",
    version="1.0.0",
    description="Predict cosmetic product retail prices from product features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper encoders ─────────────────────────────────────────────────────────

def _enc_label(label: str) -> int:
    return int(_label_enc.transform([label])[0]) if label in _label_enc.classes_ else 0


def _enc_brand(brand: str) -> int:
    b = brand.strip().upper()
    return int(_brand_enc.transform([b])[0]) if b in _brand_enc.classes_ else 0


def _get_brand_mean(brand: str) -> float:
    b = brand.strip().upper()
    return float(_brand_mean_d.get(b, np.mean(list(_brand_mean_d.values()))))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — returns model name and performance metrics."""
    return {
        "status":   "ok",
        "model":    _artefacts["model_name"],
        "test_r2":  _artefacts["metrics"]["test_r2"],
        "test_mae": _artefacts["metrics"]["test_mae"],
    }


@app.get("/metadata", tags=["Metadata"])
def metadata():
    """Return available labels, brands, price statistics, feature importances."""
    return {
        "labels":             _artefacts["all_labels"],
        "brands":             _artefacts["all_brands"],
        "price_stats":        _artefacts["price_stats"],
        "label_price_stats":  _artefacts["label_price_stats"],
        "feature_importance": _artefacts.get("feature_importance", {}),
        "metrics":            _artefacts["metrics"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict the retail price of a cosmetic product.

    Request body example:
    {
      "label": "Moisturizer",
      "brand": "LA MER",
      "rank": 4.1,
      "ingredients": "Water, Glycerin, Niacinamide, Hyaluronic Acid",
      "combination": 1, "dry": 1, "normal": 1, "oily": 0, "sensitive": 1
    }
    """
    ing_count  = count_ingredients(req.ingredients)
    prem_count = count_premium(req.ingredients)
    fill_count = count_fillers(req.ingredients)
    prem_ratio = premium_ratio(req.ingredients)
    skin_sum   = req.combination + req.dry + req.normal + req.oily + req.sensitive

    features = np.array([[
        _enc_label(req.label),
        _enc_brand(req.brand),
        _get_brand_mean(req.brand),
        req.rank,
        ing_count,
        prem_count,
        fill_count,
        prem_ratio,
        skin_sum,
        req.combination,
        req.dry,
        req.normal,
        req.oily,
        req.sensitive,
    ]])

    log_pred = _model.predict(features)[0]
    price    = float(round(max(1.0, np.expm1(log_pred)), 2))

    return PredictResponse(
        predicted_price=price,
        predicted_price_range={
            "low":  round(price * 0.80, 2),
            "high": round(price * 1.20, 2),
        },
        model_name=_artefacts["model_name"],
        confidence_note=(
            f"Model R2={_artefacts['metrics']['test_r2']}, "
            f"MAE=+/-${_artefacts['metrics']['test_mae']}"
        ),
    )


# ── Entry point (run directly for dev server) ────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# ============================================================================
#  SECTION 4 — STREAMLIT FRONTEND
#  File: cosmetics_price_predictor/frontend/streamlit_app.py
#  Run: streamlit run frontend/streamlit_app.py
# ============================================================================

"""
Streamlit multi-page frontend for the Cosmetics Price Predictor.

Pages:
  Home            — KPIs, category chart, price histogram, top brands
  EDA             — Box plots, scatter plots, skin type analysis, heatmap
  Predict         — Interactive prediction form with result visualisation
  Model Insights  — CV comparison, feature importance, actual vs predicted
"""

import sys
import pickle
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split

# ── Path setup ───────────────────────────────────────────────────────────────
_FRONTEND_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
_APP_ROOT      = _FRONTEND_ROOT.parent if _FRONTEND_ROOT.name == "frontend" else _FRONTEND_ROOT
_PKL_PATH      = _APP_ROOT / "models" / "artefacts.pkl"
_CSV_PATH      = _APP_ROOT / "data"   / "cosmetics.csv"
_API_URL       = "http://localhost:8000"


# ── Page config & custom CSS ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Cosmetics Price Predictor",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  :root {--accent:#c0596e; --soft:#fdf6f7; --card:#fff0f3;}
  .block-container {padding-top:1.5rem;}
  h1,h2,h3 {color: #c0596e !important;}
  .stButton>button {
      background:#c0596e !important; color:white !important;
      border-radius:8px; border:none;
  }
  .stButton>button:hover {background:#a0445a !important;}
  div[data-testid="stMetricValue"] {color:#c0596e !important; font-size:2rem !important;}
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource
def _load_art():
    with open(_PKL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def _load_df():
    return load_and_engineer(str(_CSV_PATH))


# ── Prediction helpers ───────────────────────────────────────────────────────

def _predict_api(payload: dict):
    """Try the FastAPI backend; return (result_dict, None) or (None, error_str)."""
    try:
        r = requests.post(f"{_API_URL}/predict", json=payload, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _predict_local(art, payload: dict) -> dict:
    """Fallback: run inference directly from the loaded artefacts (no server needed)."""
    model    = art["model"]
    lenc     = art["label_encoder"]
    benc     = art["brand_encoder"]
    bm       = art["brand_mean_price"]

    def el(v):  return int(lenc.transform([v])[0]) if v in lenc.classes_ else 0
    def eb(v):  u = v.strip().upper(); return int(benc.transform([u])[0]) if u in benc.classes_ else 0
    def bmu(v): u = v.strip().upper(); return float(bm.get(u, np.mean(list(bm.values()))))

    ing = payload.get("ingredients", "")
    sk  = sum([payload["combination"], payload["dry"],
               payload["normal"],      payload["oily"],
               payload["sensitive"]])

    feats = np.array([[
        el(payload["label"]),
        eb(payload["brand"]),
        bmu(payload["brand"]),
        payload["rank"],
        count_ingredients(ing),
        count_premium(ing),
        count_fillers(ing),
        premium_ratio(ing),
        sk,
        payload["combination"],
        payload["dry"],
        payload["normal"],
        payload["oily"],
        payload["sensitive"],
    ]])

    price = float(round(max(1.0, np.expm1(model.predict(feats)[0])), 2))
    return {
        "predicted_price":       price,
        "predicted_price_range": {"low": round(price*0.8,2), "high": round(price*1.2,2)},
        "model_name":            art["model_name"],
        "confidence_note":       (
            f"Model R2={art['metrics']['test_r2']}, "
            f"MAE=+/-${art['metrics']['test_mae']}"
        ),
    }


# ── Sidebar navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/lipstick.png", width=80)
    st.title("Cosmetics\nPrice Predictor")
    st.caption("ML-powered price estimation")
    st.divider()
    page = option_menu(
        menu_title=None,
        options=["Home", "EDA", "Predict", "Model Insights"],
        icons=["house", "bar-chart", "magic", "graph-up"],
        default_index=0,
        styles={
            "container":         {"padding": "0"},
            "nav-link":          {"font-size": "14px"},
            "nav-link-selected": {"background-color": "#c0596e"},
        },
    )


# ── Load shared data ─────────────────────────────────────────────────────────

_art = _load_art()
_df  = _load_df()


# ============================================================
# PAGE: HOME
# ============================================================

if page == "Home":
    st.title("Cosmetics Price Predictor")
    st.markdown(
        "An end-to-end **Machine Learning** application that predicts "
        "cosmetic product prices based on brand, category, rating, "
        "and ingredient composition."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products",    f"{len(_df):,}")
    col2.metric("Categories",  _df["Label"].nunique())
    col3.metric("Brands",      _df["Brand"].nunique())
    col4.metric("Avg Price",   f"${_df['Price'].mean():.0f}")

    st.divider()

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Products by Category")
        counts = _df["Label"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.bar(counts, x="Category", y="Count",
                     color="Count", color_continuous_scale=["#f7cdd5","#c0596e"])
        fig.update_layout(showlegend=False, margin=dict(t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.subheader("Price Distribution")
        fig2 = px.histogram(_df, x="Price", nbins=50,
                            color_discrete_sequence=["#c0596e"])
        fig2.update_layout(margin=dict(t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Priciest Brands (avg, min 3 products)")
    top_b = (
        _df.groupby("Brand")["Price"]
           .agg(["mean","count"])
           .query("count >= 3")
           .sort_values("mean", ascending=False)
           .head(10)
           .reset_index()
    )
    top_b.columns = ["Brand","Avg Price","# Products"]
    fig3 = px.bar(top_b, x="Brand", y="Avg Price", color="Avg Price",
                  color_continuous_scale=["#f7cdd5","#c0596e"], text="Avg Price")
    fig3.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig3.update_layout(margin=dict(t=20,b=0), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Raw Dataset Preview (first 50 rows)"):
        st.dataframe(
            _df[["Label","Brand","Name","Price","Rank",
                 "ingredient_count","premium_count","skin_type_sum"]].head(50),
            use_container_width=True,
        )


# ============================================================
# PAGE: EDA
# ============================================================

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price by Category", "Brand Analysis", "Ingredients", "Skin Types"
    ])

    # ── Tab 1: Price by Category ─────────────────────────────
    with tab1:
        st.subheader("Price Distribution by Category")
        fig = px.box(_df, x="Label", y="Price", color="Label",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     points="outliers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        avg = _df.groupby("Label")["Price"].mean().sort_values(ascending=False).reset_index()
        avg.columns = ["Category","Avg Price ($)"]
        st.dataframe(avg, use_container_width=True)

    # ── Tab 2: Brand Analysis ────────────────────────────────
    with tab2:
        n = st.slider("Top N brands", 5, 30, 15)
        bs = (
            _df.groupby("Brand")["Price"]
               .agg(["mean","median","min","max","count"])
               .sort_values("mean", ascending=False)
               .head(n)
               .reset_index()
        )
        bs.columns = ["Brand","Mean","Median","Min","Max","Count"]
        fig = px.bar(bs, x="Brand", y="Mean",
                     error_y=bs["Max"] - bs["Mean"],
                     color="Mean", color_continuous_scale=["#f7cdd5","#c0596e"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bs, use_container_width=True)

    # ── Tab 3: Ingredients ───────────────────────────────────
    with tab3:
        st.subheader("Ingredient Count vs Price")
        fig = px.scatter(_df, x="ingredient_count", y="Price", color="Label",
                         hover_data=["Brand","Name"], opacity=0.6, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Premium Ingredient Count vs Price")
        fig2 = px.scatter(_df, x="premium_count", y="Price", color="Label",
                          hover_data=["Brand","Name"], opacity=0.6, trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Avg Ingredient Count",    f"{_df['ingredient_count'].mean():.1f}")
        c2.metric("Avg Premium Ingredients", f"{_df['premium_count'].mean():.1f}")

    # ── Tab 4: Skin Types ────────────────────────────────────
    with tab4:
        st.subheader("Price by Skin Type Compatibility")
        skin_types = ["Combination","Dry","Normal","Oily","Sensitive"]
        rows_st = []
        for sc in skin_types:
            rows_st.append({
                "Skin Type":    sc,
                "Suitable":     _df[_df[sc]==1]["Price"].mean(),
                "Not Suitable": _df[_df[sc]==0]["Price"].mean(),
            })
        skin_df = pd.DataFrame(rows_st)
        fig = px.bar(
            skin_df.melt(id_vars="Skin Type", var_name="Suitable For", value_name="Avg Price"),
            x="Skin Type", y="Avg Price", color="Suitable For", barmode="group",
            color_discrete_sequence=["#c0596e","#f7cdd5"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Skin Type Coverage")
        ss = _df[skin_types].sum().reset_index()
        ss.columns = ["Skin Type","Count"]
        fig2 = px.pie(ss, names="Skin Type", values="Count",
                      color_discrete_sequence=px.colors.sequential.RdPu, hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price vs Rating")
    fig_r = px.scatter(_df, x="Rank", y="Price", color="Label",
                       size="ingredient_count", hover_data=["Brand","Name"],
                       opacity=0.65, trendline="ols")
    st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = ["Price","Rank","ingredient_count","premium_count",
                 "filler_count","premium_ratio","skin_type_sum"]
    corr = _df[corr_cols].corr()
    fig_h = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values,2), texttemplate="%{text}",
    ))
    fig_h.update_layout(height=400)
    st.plotly_chart(fig_h, use_container_width=True)


# ============================================================
# PAGE: PREDICT
# ============================================================

elif page == "Predict":
    st.title("Predict Product Price")
    st.markdown("Fill in the product details below to get an estimated market price.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            label = st.selectbox("Product Category", sorted(_df["Label"].unique()))
            brand = st.selectbox("Brand", ["(New / Unknown)"] + sorted(_df["Brand"].unique()))
            rank  = st.slider("Product Rating", 0.0, 5.0, 4.0, 0.1)
        with c2:
            st.markdown("**Skin Type Suitability**")
            combination = st.checkbox("Combination")
            dry         = st.checkbox("Dry")
            normal      = st.checkbox("Normal")
            oily        = st.checkbox("Oily")
            sensitive   = st.checkbox("Sensitive")

        ingredients = st.text_area(
            "Ingredients (comma-separated — leave blank to use defaults)",
            height=120,
            placeholder="Water, Glycerin, Niacinamide, Hyaluronic Acid, Retinol ...",
        )
        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        bv = brand if brand != "(New / Unknown)" else "UNKNOWN"
        payload = {
            "label":       label,
            "brand":       bv,
            "rank":        rank,
            "ingredients": ingredients,
            "combination": int(combination),
            "dry":         int(dry),
            "normal":      int(normal),
            "oily":        int(oily),
            "sensitive":   int(sensitive),
        }

        result, err = _predict_api(payload)
        if result is None:
            result = _predict_local(_art, payload)
            st.warning("Backend API offline — using local model.")
        else:
            st.success("Predicted via FastAPI backend.")

        st.divider()
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Predicted Price",  f"${result['predicted_price']:.2f}")
        pc2.metric("Low Estimate",     f"${result['predicted_price_range']['low']:.2f}")
        pc3.metric("High Estimate",    f"${result['predicted_price_range']['high']:.2f}")
        st.caption(f"{result['model_name']} | {result['confidence_note']}")

        # Price distribution with prediction marker
        cat_prices = _df[_df["Label"] == label]["Price"]
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=cat_prices, name="Category prices",
                                     marker_color="#f7cdd5", opacity=0.8))
        fig_d.add_vline(x=result["predicted_price"], line_dash="dash",
                        line_color="#c0596e", line_width=2,
                        annotation_text=f"  Predicted ${result['predicted_price']:.0f}",
                        annotation_font_color="#c0596e")
        fig_d.update_layout(title=f"Price distribution: {label}",
                            xaxis_title="Price ($)", yaxis_title="Count",
                            margin=dict(t=40,b=20))
        st.plotly_chart(fig_d, use_container_width=True)

        st.subheader("Similar Products")
        similar = _df[_df["Label"] == label].copy()
        similar["diff"] = abs(similar["Price"] - result["predicted_price"])
        similar = similar.sort_values("diff").head(5)
        st.dataframe(
            similar[["Brand","Name","Price","Rank","ingredient_count"]].reset_index(drop=True),
            use_container_width=True,
        )


# ============================================================
# PAGE: MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":
    st.title("Model Insights")

    m = _art["metrics"]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Best Model", _art["model_name"])
    mc2.metric("Test R2",    f"{m['test_r2']:.4f}")
    mc3.metric("Test MAE",   f"${m['test_mae']:.2f}")

    st.divider()

    # CV model comparison
    st.subheader("Model Comparison — 5-Fold CV R2")
    cv = m["cv_results"]
    cv_df = pd.DataFrame({"Model": list(cv.keys()), "CV R2": list(cv.values())}) \
              .sort_values("CV R2", ascending=False)
    fig_cv = px.bar(cv_df, x="Model", y="CV R2", color="CV R2",
                    color_continuous_scale=["#f7cdd5","#c0596e"], text="CV R2")
    fig_cv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_cv.update_layout(yaxis_range=[0,1], margin=dict(t=20,b=0), showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)

    # Feature importance
    fi = _art.get("feature_importance", {})
    if fi:
        st.subheader("Feature Importance")
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}) \
                  .sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#f7cdd5","#c0596e"])
        fig_fi.update_layout(margin=dict(t=20,b=0), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices — Test Set")
    X  = _df[FEATURE_COLS]
    y  = _df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_pred_log = _art["model"].predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    sc_df = pd.DataFrame({"Actual": y_true.values, "Predicted": y_pred})
    fig_sc = px.scatter(sc_df, x="Actual", y="Predicted", opacity=0.5,
                        color_discrete_sequence=["#c0596e"], trendline="ols")
    fig_sc.add_shape(type="line", x0=0, y0=0,
                     x1=sc_df["Actual"].max(), y1=sc_df["Actual"].max(),
                     line=dict(color="gray", dash="dash"))
    fig_sc.update_layout(xaxis_title="Actual Price ($)",
                         yaxis_title="Predicted Price ($)",
                         margin=dict(t=20,b=20))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Residual distribution
    st.subheader("Residual Distribution")
    residuals = y_true.values - y_pred
    fig_res = px.histogram(x=residuals, nbins=50,
                           color_discrete_sequence=["#c0596e"],
                           labels={"x": "Residual (Actual - Predicted)"})
    fig_res.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_res, use_container_width=True)

    # Label price stats
    st.subheader("Label Price Statistics")
    lps = _art.get("label_price_stats", {})
    if lps:
        lps_df = pd.DataFrame(lps).T.reset_index()
        lps_df.columns = ["Category","Avg Price","Min Price","Max Price"]
        st.dataframe(lps_df.sort_values("Avg Price", ascending=False),
                     use_container_width=True)
