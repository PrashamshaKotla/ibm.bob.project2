# 💄 Cosmetics Price Predictor

A complete end-to-end **Machine Learning** web application that predicts cosmetic product prices using brand, product category, customer rating, and ingredient composition as features.

---

## 📁 Project Structure

```
cosmetics_price_predictor/
├── data/
│   ├── cosmetics.csv          # Raw dataset (1,472 products)
│   └── preprocess.py          # Feature engineering pipeline
├── models/
│   ├── train.py               # Model training script
│   ├── artefacts.pkl          # Saved model + encoders (generated)
│   └── metrics.json           # Model metrics (generated)
├── backend/
│   └── app.py                 # FastAPI REST API
├── frontend/
│   └── streamlit_app.py       # Streamlit multi-page frontend
└── requirements.txt
```

---

## 🗃️ Dataset

**Source:** Sephora cosmetics dataset  
**Size:** 1,472 products × 11 columns

| Column | Description |
|---|---|
| Label | Product category (Moisturizer, Cleanser, Face Mask, …) |
| Brand | Brand name |
| Name | Product name |
| Price | Target variable — price in USD |
| Rank | Customer rating (0–5) |
| Ingredients | Full ingredient list |
| Combination/Dry/Normal/Oily/Sensitive | Binary skin-type flags |

**Price range:** $3 – $370 · **Mean:** $55.58 · **Median:** $42.50

---

## 🤖 ML Model

### Feature Engineering
| Feature | Description |
|---|---|
| `label_encoded` | Integer-encoded product category |
| `brand_encoded` | Integer-encoded brand |
| `brand_mean_price` | Brand-level average price (target encoding) |
| `Rank` | Product rating |
| `ingredient_count` | Total number of ingredients |
| `premium_count` | Count of luxury ingredient keywords |
| `filler_count` | Count of commodity ingredient keywords |
| `premium_ratio` | premium_count / ingredient_count |
| `skin_type_sum` | Sum of all five skin type flags |
| `Combination/Dry/Normal/Oily/Sensitive` | Individual skin type flags |

### Model Comparison (5-fold CV R²)
| Model | CV R² |
|---|---|
| **Random Forest** ✅ | **0.6823** |
| Gradient Boosting | 0.6643 |
| Ridge Regression | 0.5533 |

### Test-Set Performance
- **R²**: 0.5918
- **MAE**: $15.44

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd cosmetics_price_predictor
pip install -r requirements.txt
```

### 2. Train the model
```bash
python models/train.py
```

### 3. Start the FastAPI backend
```bash
uvicorn backend.app:app --reload --port 8000
```
API docs available at **http://localhost:8000/docs**

### 4. Launch the Streamlit frontend
```bash
streamlit run frontend/streamlit_app.py
```
Opens at **http://localhost:8501**

> **Note:** The Streamlit app works in standalone mode (without the backend) — it falls back to loading the model directly.

---

## 🌐 API Reference

### `GET /`
Health check — returns model name and metrics.

### `GET /metadata`
Returns all available labels, brands, price statistics, feature importances.

### `POST /predict`
Predict price for a product.

**Request body:**
```json
{
  "label":       "Moisturizer",
  "brand":       "LA MER",
  "rank":        4.1,
  "ingredients": "Water, Glycerin, Niacinamide, Hyaluronic Acid",
  "combination": 1,
  "dry":         1,
  "normal":      1,
  "oily":        0,
  "sensitive":   1
}
```

**Response:**
```json
{
  "predicted_price":       87.50,
  "predicted_price_range": {"low": 70.00, "high": 105.00},
  "model_name":            "RandomForest",
  "confidence_note":       "Model R²=0.5918, MAE=±$15.44"
}
```

---

## 📊 Frontend Pages

| Page | Description |
|---|---|
| 🏠 Home | Overview KPIs, category distribution, price histogram, top brands |
| 📊 EDA | Price by category, brand analysis, ingredient scatter plots, skin type charts, correlation heatmap |
| 🔮 Predict | Interactive prediction form — see predicted price, price range, similar products |
| 🧠 Model Insights | CV comparison, feature importance, actual vs predicted scatter, residual distribution |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | pandas, numpy, scikit-learn |
| ML Models | RandomForestRegressor, GradientBoostingRegressor, Ridge |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + streamlit-option-menu |
| Visualisation | Plotly |
