from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("src/sentiment_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment Analysis API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expecting JSON input
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = model.predict([text])[0]
    return jsonify({"sentiment": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
