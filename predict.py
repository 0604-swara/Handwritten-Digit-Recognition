
import sys
import json
import joblib
import numpy as np
from pathlib import Path

# Get the directory where predict.py is located
BASE_DIR = Path(__file__).resolve().parent

# Correct model location
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

def predict_digit(pixel_matrix):

    image = np.asarray(pixel_matrix, dtype=float)

    # Validate dimensions
    if image.shape != (8, 8):
        raise ValueError("Input must be an 8x8 matrix.")

    # Validate pixel values
    if not np.isfinite(image).all():
        raise ValueError("Input contains invalid values.")

    if np.any(image < 0) or np.any(image > 16):
        raise ValueError("Pixel values must be between 0 and 16.")

    # Convert 8x8 image into 64 features
    features = image.reshape(1, 64)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict digit
    prediction = model.predict(features)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(features)[0]

    return {
        "predicted_digit": int(prediction),
        "probabilities": {
            str(i): round(float(probabilities[i]), 4)
            for i in range(10)
        }
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py example_input.json")
        sys.exit(1)

    try:
        with open(sys.argv[1], "r") as f:
            pixel_matrix = json.load(f)

        result = predict_digit(pixel_matrix)
        print(json.dumps(result, indent=4))

    except (ValueError, TypeError) as e:
        print("Invalid Input:", e)

    except FileNotFoundError as e:
        print("File not found:", e)
