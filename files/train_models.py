# AgriMind AI - ML Model Training Script
# Run: python train_models.py
# This trains and saves all ML models used by the Flask API

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 1. CROP RECOMMENDATION MODEL (Random Forest Classifier)
# Dataset: Crop Recommendation Dataset (Kaggle)
# Features: N, P, K, temperature, humidity, ph, rainfall
# ─────────────────────────────────────────────────────────────────
def train_crop_model():
    print("🌱 Training Crop Recommendation Model...")

    # Load dataset (download from Kaggle: crop-recommendation-dataset)
    # kaggle datasets download -d atharvaingle/crop-recommendation-dataset
    DATA_PATH = "data/Crop_recommendation.csv"

    if not os.path.exists(DATA_PATH):
        print("  ⚠ Dataset not found. Generating synthetic data...")
        # Generate synthetic training data
        np.random.seed(42)
        crops = ["Rice", "Wheat", "Maize", "Chickpea", "Lentil", "Cotton",
                 "Mango", "Banana", "Grapes", "Soybean", "Coconut", "Papaya"]
        n = 1000

        # Feature ranges per crop
        params = {
            "Rice":     (80, 50, 40, 25, 85, 6.5, 200),
            "Wheat":    (60, 40, 40, 20, 65, 7.0, 70),
            "Maize":    (90, 50, 45, 22, 70, 6.2, 85),
            "Chickpea": (40, 70, 80, 22, 18, 7.0, 70),
            "Lentil":   (18, 40, 40, 24, 64, 6.5, 45),
            "Cotton":   (120, 40, 20, 28, 80, 7.0, 80),
            "Mango":    (15, 10, 30, 31, 50, 6.0, 100),
            "Banana":   (100, 75, 50, 27, 80, 6.0, 105),
            "Grapes":   (20, 125, 200, 24, 80, 6.5, 70),
            "Soybean":  (20, 70, 20, 30, 65, 6.8, 65),
            "Coconut":  (22, 16, 30, 27, 94, 5.5, 150),
            "Papaya":   (50, 50, 50, 33, 92, 6.5, 145),
        }

        rows = []
        for crop, (N, P, K, temp, hum, ph, rain) in params.items():
            for _ in range(n // len(crops)):
                rows.append({
                    "N": N + np.random.normal(0, 5),
                    "P": P + np.random.normal(0, 5),
                    "K": K + np.random.normal(0, 5),
                    "temperature": temp + np.random.normal(0, 2),
                    "humidity": hum + np.random.normal(0, 5),
                    "ph": ph + np.random.normal(0, 0.3),
                    "rainfall": rain + np.random.normal(0, 15),
                    "label": crop
                })
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(DATA_PATH)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ✅ Accuracy: {acc*100:.2f}%")

    # Save models
    for obj, name in [(model, "crop_model.pkl"), (scaler, "crop_scaler.pkl"), (le, "crop_encoder.pkl")]:
        with open(os.path.join(MODELS_DIR, name), "wb") as f:
            pickle.dump(obj, f)

    print("  💾 Saved: crop_model.pkl, crop_scaler.pkl, crop_encoder.pkl")


# ─────────────────────────────────────────────────────────────────
# 2. PRICE PREDICTION MODEL (Gradient Boosting Regressor)
# Features: crop_encoded, month, year
# ─────────────────────────────────────────────────────────────────
def train_price_model():
    print("\n💰 Training Price Prediction Model...")

    BASE_PRICES = {
        0: 2200, 1: 2150, 2: 1800, 3: 5000, 4: 4200,
        5: 6500, 6: 350, 7: 1200, 8: 800
    }

    rows = []
    np.random.seed(42)
    for crop_id, base in BASE_PRICES.items():
        for year in range(2018, 2025):
            for month in range(1, 13):
                seasonal = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                yearly = 1 + 0.04 * (year - 2018)
                noise = np.random.normal(0, base * 0.03)
                price = base * seasonal * yearly + noise
                rows.append({"crop_id": crop_id, "year": year, "month": month, "price": price})

    df = pd.DataFrame(rows)
    X = df[["crop_id", "year", "month"]].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  ✅ MAE: ₹{mae:.2f}/quintal")

    with open(os.path.join(MODELS_DIR, "price_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("  💾 Saved: price_model.pkl")


# ─────────────────────────────────────────────────────────────────
# 3. DISEASE DETECTION (CNN using TensorFlow/Keras)
# Based on PlantVillage dataset
# ─────────────────────────────────────────────────────────────────
def train_disease_model():
    print("\n🌿 Training Disease Detection CNN...")

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        # Check if dataset exists
        DATA_DIR = "data/PlantVillage"
        if not os.path.exists(DATA_DIR):
            print("  ⚠ PlantVillage dataset not found.")
            print("  Download: https://www.kaggle.com/datasets/emmarex/plantdisease")
            print("  Place in: ml_api/data/PlantVillage/")
            print("  Skipping CNN training. Using mock predictions.")
            return

        # Data pipeline
        IMG_SIZE = 224
        BATCH_SIZE = 32

        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE
        )

        num_classes = len(train_ds.class_names)
        print(f"  📁 Classes found: {num_classes}")

        # Use MobileNetV2 transfer learning
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)

        # Normalize
        normalization_layer = layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        model.fit(train_ds, validation_data=val_ds, epochs=10)

        save_path = os.path.join(MODELS_DIR, "disease_model.h5")
        model.save(save_path)
        print(f"  💾 Saved: disease_model.h5")

    except ImportError:
        print("  ⚠ TensorFlow not installed. Run: pip install tensorflow")


if __name__ == "__main__":
    print("🌿 AgriMind AI - Model Training Pipeline")
    print("=" * 50)
    train_crop_model()
    train_price_model()
    train_disease_model()
    print("\n✅ All models trained and saved to ./models/")
    print("🚀 Start the API: python app.py")
