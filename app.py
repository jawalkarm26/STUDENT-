from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Model is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Example:
        # {
        #   "features": [5.1, 3.5, 1.4, 0.2]
        # }

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
