import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# Load Models & Vectorizers at startup
try:
    rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_rating_model.pkl"))
    tfidf_rating = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    tag_model = joblib.load(os.path.join(BASE_DIR, "tag_prediction_model.pkl"))
    tag_tfidf = joblib.load(os.path.join(BASE_DIR, "tag_tfidf_vectorizer.pkl"))
    tag_mlb = joblib.load(os.path.join(BASE_DIR, "tag_mlb.pkl"))
    print("✅ All ML models and vectorizers loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    rf_model, tfidf_rating, tag_model, tag_tfidf, tag_mlb = None, None, None, None, None


def get_rank_info(rating: int):
    """Map Codeforces rating to tier title and brand color."""
    if rating < 1200:
        return "Newbie", "#808080"
    elif rating < 1400:
        return "Pupil", "#008000"
    elif rating < 1600:
        return "Specialist", "#03A89E"
    elif rating < 1900:
        return "Expert", "#0000FF"
    elif rating < 2100:
        return "Candidate Master", "#AA00AA"
    elif rating < 2300:
        return "Master", "#FF8C00"
    elif rating < 2400:
        return "International Master", "#FF8C00"
    elif rating < 2600:
        return "Grandmaster", "#FF0000"
    elif rating < 3000:
        return "Int. Grandmaster", "#FF0000"
    else:
        return "Legendary Grandmaster", "#FF0000"


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": rf_model is not None and tag_model is not None
    })


@app.route("/predict-tags", methods=["POST"])
def predict_tags_endpoint():
    if not tag_model or not tag_tfidf or not tag_mlb:
        return jsonify({"error": "Tag models are not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    statement = data.get("statement", "").strip()
    title = data.get("title", "").strip()
    prob_input = data.get("input", "").strip()
    prob_output = data.get("output", "").strip()

    if not statement:
        return jsonify({"error": "Problem statement is required."}), 400

    tag_text = f"Title: {title} Statement: {statement} Input: {prob_input} Output: {prob_output}"
    X_tag = tag_tfidf.transform([tag_text])
    y_tag_pred = tag_model.predict(X_tag)
    predicted_tags = list(tag_mlb.inverse_transform(y_tag_pred)[0])

    # Calculate probabilities if available
    probabilities = {}
    try:
        y_probs = tag_model.predict_proba(X_tag)[0]
        for tag, prob in zip(tag_mlb.classes_, y_probs):
            if prob >= 0.35 or tag in predicted_tags:
                probabilities[tag] = round(float(prob), 4)
    except Exception:
        pass

    return jsonify({
        "tags": predicted_tags,
        "probabilities": probabilities
    })


@app.route("/predict-rating", methods=["POST"])
def predict_rating_endpoint():
    if not rf_model or not tfidf_rating:
        return jsonify({"error": "Rating models are not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    statement = data.get("statement", "").strip()
    title = data.get("title", "").strip()
    prob_input = data.get("input", "").strip()
    prob_output = data.get("output", "").strip()
    tags = data.get("tags", [])

    if not statement:
        return jsonify({"error": "Problem statement is required."}), 400

    tags_str = str(tags) if isinstance(tags, list) else str(tags)
    rating_text = f"Title: {title} Statement: {statement} Input: {prob_input} Output: {prob_output} Tags: {tags_str}"
    
    X_rating = tfidf_rating.transform([rating_text])
    pred = rf_model.predict(X_rating)[0]
    rating_int = max(800, min(3500, int(round(pred))))
    rank_title, color = get_rank_info(rating_int)

    return jsonify({
        "rating": rating_int,
        "title": rank_title,
        "color": color
    })


@app.route("/predict-all", methods=["POST"])
def predict_all():
    if not rf_model or not tfidf_rating or not tag_model or not tag_tfidf or not tag_mlb:
        return jsonify({"error": "Prediction models are not loaded properly."}), 500

    data = request.get_json(force=True, silent=True) or {}
    statement = data.get("statement", "").strip()
    title = data.get("title", "").strip()
    prob_input = data.get("input", "").strip()
    prob_output = data.get("output", "").strip()

    if not statement:
        return jsonify({"error": "Problem statement is required."}), 400

    try:
        # 1. Tag Prediction
        tag_text = f"Title: {title} Statement: {statement} Input: {prob_input} Output: {prob_output}"
        X_tag = tag_tfidf.transform([tag_text])
        y_tag_pred = tag_model.predict(X_tag)
        predicted_tags = list(tag_mlb.inverse_transform(y_tag_pred)[0])

        # 2. Rating Prediction
        tags_str = str(predicted_tags)
        rating_text = f"Title: {title} Statement: {statement} Input: {prob_input} Output: {prob_output} Tags: {tags_str}"
        X_rating = tfidf_rating.transform([rating_text])
        pred_rating = rf_model.predict(X_rating)[0]
        rating_int = max(800, min(3500, int(round(pred_rating))))
        rank_title, color = get_rank_info(rating_int)

        # Probabilities
        probabilities = {}
        try:
            y_probs = tag_model.predict_proba(X_tag)[0]
            for tag, prob in zip(tag_mlb.classes_, y_probs):
                if prob >= 0.35 or tag in predicted_tags:
                    probabilities[tag] = round(float(prob), 4)
        except Exception:
            pass

        return jsonify({
            "rating": rating_int,
            "title": rank_title,
            "color": color,
            "tags": predicted_tags,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 CPVision AI starting on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
