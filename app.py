from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {prediction[0]}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
