// AgriMind AI - Node.js Express Backend
// Run: npm install && node server.js

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 5000;
const FLASK_URL = "http://localhost:5001"; // Python Flask ML API

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Multer config for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = "./uploads";
    if (!fs.existsSync(dir)) fs.mkdirSync(dir);
    cb(null, dir);
  },
  filename: (req, file, cb) =>
    cb(null, Date.now() + path.extname(file.originalname)),
});
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png|webp/;
    cb(null, allowed.test(file.mimetype));
  },
});

// ─── HEALTH CHECK ───────────────────────────────────────────────
app.get("/", (req, res) =>
  res.json({ status: "AgriMind API running", version: "1.0.0" })
);

// ─── CROP PREDICTION ────────────────────────────────────────────
// POST /predict-crop
// Body: { N, P, K, temperature, humidity, ph, rainfall }
app.post("/predict-crop", async (req, res) => {
  try {
    const { N, P, K, temperature, humidity, ph, rainfall } = req.body;

    // Validate
    const fields = { N, P, K, temperature, humidity, ph, rainfall };
    for (const [key, val] of Object.entries(fields)) {
      if (val === undefined || val === "") {
        return res.status(400).json({ error: `Missing field: ${key}` });
      }
    }

    // Forward to Flask ML API
    const response = await axios.post(`${FLASK_URL}/predict-crop`, {
      N: parseFloat(N),
      P: parseFloat(P),
      K: parseFloat(K),
      temperature: parseFloat(temperature),
      humidity: parseFloat(humidity),
      ph: parseFloat(ph),
      rainfall: parseFloat(rainfall),
    });

    res.json(response.data);
  } catch (err) {
    console.error("Crop prediction error:", err.message);
    res.status(500).json({ error: "Prediction failed. Check ML server." });
  }
});

// ─── PRICE PREDICTION ───────────────────────────────────────────
// POST /predict-price
// Body: { crop, state, year }
app.post("/predict-price", async (req, res) => {
  try {
    const { crop, state = "Telangana", year = 2024 } = req.body;
    if (!crop) return res.status(400).json({ error: "Crop name required" });

    const response = await axios.post(`${FLASK_URL}/predict-price`, {
      crop,
      state,
      year: parseInt(year),
    });

    res.json(response.data);
  } catch (err) {
    console.error("Price prediction error:", err.message);
    res.status(500).json({ error: "Price prediction failed." });
  }
});

// ─── DISEASE DETECTION ──────────────────────────────────────────
// POST /predict-disease
// Form-data: image file
app.post("/predict-disease", upload.single("image"), async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "No image uploaded" });

    const FormData = require("form-data");
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path), {
      filename: req.file.filename,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${FLASK_URL}/predict-disease`, form, {
      headers: form.getHeaders(),
    });

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json(response.data);
  } catch (err) {
    console.error("Disease detection error:", err.message);
    if (req.file?.path) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: "Disease detection failed." });
  }
});

// ─── PROFIT CALCULATOR ──────────────────────────────────────────
// POST /calculate-profit
// Body: { cost, land_area, selling_price_per_kg, crop }
app.post("/calculate-profit", async (req, res) => {
  try {
    const { cost, land_area, selling_price_per_kg, crop } = req.body;

    if (!cost || !land_area || !selling_price_per_kg) {
      return res.status(400).json({ error: "cost, land_area, selling_price_per_kg required" });
    }

    const response = await axios.post(`${FLASK_URL}/calculate-profit`, {
      cost: parseFloat(cost),
      land_area: parseFloat(land_area),
      selling_price_per_kg: parseFloat(selling_price_per_kg),
      crop: crop || "Rice",
    });

    res.json(response.data);
  } catch (err) {
    console.error("Profit calc error:", err.message);
    res.status(500).json({ error: "Profit calculation failed." });
  }
});

// ─── VOICE QUERY ────────────────────────────────────────────────
// POST /voice-query
// Body: { query, language }  language: "en" | "te"
app.post("/voice-query", async (req, res) => {
  try {
    const { query, language = "en" } = req.body;
    if (!query) return res.status(400).json({ error: "Query text required" });

    const response = await axios.post(`${FLASK_URL}/voice-query`, {
      query,
      language,
    });

    res.json(response.data);
  } catch (err) {
    console.error("Voice query error:", err.message);
    res.status(500).json({ error: "Voice query failed." });
  }
});

// ─── GLOBAL ERROR HANDLER ───────────────────────────────────────
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`\n🌿 AgriMind API Server running at http://localhost:${PORT}`);
  console.log(`📡 Forwarding AI requests to Flask at ${FLASK_URL}\n`);
});
