# 🌿 AgriMind AI – Smart Agriculture System

A full-stack AI-powered agriculture platform with crop prediction, price forecasting, disease detection, profit optimization, and a bilingual voice assistant (English + Telugu).

---

## 📁 Folder Structure

```
AgriAI/
├── index.html              ← Complete React frontend (single file, open in browser)
│
├── backend/                ← Node.js Express API Gateway
│   ├── server.js
│   └── package.json
│
├── ml_api/                 ← Python Flask ML Server
│   ├── app.py              ← Flask API routes
│   ├── train_models.py     ← Model training script
│   ├── requirements.txt
│   └── models/             ← (auto-created) Saved .pkl and .h5 files
│
└── README.md
```

---

## 🚀 Quick Start (Frontend Only)

The `index.html` file works **standalone** with built-in mock AI — no backend needed to explore the UI:

1. Open `index.html` in any modern browser
2. Sign up / Log in (any credentials work)
3. Explore all 6 features with simulated AI responses

---

## 🔧 Full Stack Setup

### Step 1 – Python Flask ML API

```bash
cd ml_api

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Train models
# First download datasets:
# - Crop Recommendation: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
# - PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease
# Place CSVs in ml_api/data/

python train_models.py

# Start Flask server
python app.py
# → Running at http://localhost:5001
```

### Step 2 – Node.js Express Backend

```bash
cd backend
npm install
npm run dev
# → Running at http://localhost:5000
```

### Step 3 – Frontend (React)

**Option A – Open directly:**
Just open `index.html` in your browser.

**Option B – Point to real backend:**
In `index.html`, find `mockPredict` and replace with Axios calls:
```js
// Replace mockPredict("crop", form) with:
const response = await axios.post("http://localhost:5000/predict-crop", {
  N: form.N, P: form.P, K: form.K,
  temperature: form.temp, humidity: form.humidity,
  ph: form.ph, rainfall: form.rainfall
});
const r = response.data;
```

---

## 🌐 API Endpoints

### Node.js (port 5000)

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/predict-crop` | `{N, P, K, temperature, humidity, ph, rainfall}` | `{crop, confidence, alternatives}` |
| POST | `/predict-price` | `{crop, state?, year?}` | `{current_price, trend, forecast}` |
| POST | `/predict-disease` | form-data: `image` | `{plant, disease, solution, confidence}` |
| POST | `/calculate-profit` | `{cost, land_area, selling_price_per_kg, crop?}` | `{profit, revenue, margin}` |
| POST | `/voice-query` | `{query, language}` | `{response, confidence}` |

### Flask (port 5001) — same endpoints, direct ML

---

## 🤖 ML Models Used

| Feature | Algorithm | Dataset |
|---------|-----------|---------|
| Crop Prediction | Random Forest Classifier | Crop Recommendation (Kaggle) |
| Price Forecast | Gradient Boosting Regressor | Synthetic + APMC market data |
| Disease Detection | MobileNetV2 (Transfer Learning) | PlantVillage (54,000+ images) |
| Profit Calculator | Rule-based + ML | Custom formulas |
| Voice Assistant | Web Speech API + NLP | Pattern matching / LLM |

---

## 🎨 UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Login / Sign Up | `/` | Auth page with Google OAuth option |
| Dashboard | `#dashboard` | Stats, quick actions, soil monitor |
| Crop Prediction | `#crop` | NPK + climate → crop recommendation |
| Price Forecast | `#price` | Crop dropdown → price chart |
| Disease Detection | `#disease` | Upload leaf image → diagnosis |
| Profit Calculator | `#profit` | Cost/land/price → profit & ROI |
| Voice Assistant | `#voice` | Bilingual AI voice interface |

---

## 📦 Dependencies

**Frontend:** React 18, Chart.js, Google Fonts (Plus Jakarta Sans, Syne)  
**Backend:** Express, Axios, Multer, CORS  
**ML API:** Flask, scikit-learn, TensorFlow 2.15, Pillow, NumPy, Pandas  

---

## 🌍 Voice Assistant Languages

- **English** – Web Speech API with `en-IN` locale
- **Telugu** – Web Speech API with `te-IN` locale (Chrome/Edge recommended)

---

## 📞 Support

Built with ❤️ for Indian farmers.  
For production deployment: use Nginx + Gunicorn (Flask) + PM2 (Node).

```bash
# Production Flask
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# Production Node
npm install -g pm2
pm2 start server.js --name agrimind-api
```
