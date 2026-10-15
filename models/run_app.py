import argparse
import os

from flask import Flask, jsonify, render_template, request, send_from_directory
import joblib

from main import BASE_DIR, predict_emotion, train_model


MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.joblib")


def load_or_train_model(model_path=MODEL_PATH, retrain=False):
    if os.path.isfile(model_path) and not retrain:
        print(f"Loading saved model from {model_path}")
        artifact = joblib.load(model_path)
        return artifact["model"], artifact["vectorizer"], artifact["accuracy"]

    print("Training model from the dataset...")
    model, vectorizer, accuracy = train_model()
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "accuracy": accuracy},
        model_path,
    )
    print(f"Saved trained model to {model_path}")
    return model, vectorizer, accuracy


def create_app(predictor, accuracy):
    web_app = Flask(__name__, template_folder=BASE_DIR, static_folder=None)

    @web_app.get("/")
    def home():
        return render_template("index2.html", accuracy=f"{accuracy * 100:.2f}")

    @web_app.get("/app.js")
    def javascript():
        return send_from_directory(BASE_DIR, "app.js", mimetype="text/javascript")

    @web_app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return jsonify(error="Send a JSON object with a text string."), 400

        text = payload["text"].strip()
        if not text:
            return jsonify(error="Enter a sentence first."), 400
        if len(text) > 1000:
            return jsonify(error="Text must be 1000 characters or fewer."), 400

        return jsonify(predictor(text))

    return web_app


def main():
    parser = argparse.ArgumentParser(description="Run the emotion detector web app.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train again from the dataset and replace the saved model.",
    )
    args = parser.parse_args()

    try:
        model, vectorizer, accuracy = load_or_train_model(retrain=args.retrain)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    web_app = create_app(
        lambda text: predict_emotion(text, model, vectorizer),
        accuracy,
    )
    print("Emotion detector running at http://127.0.0.1:5000")
    web_app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()