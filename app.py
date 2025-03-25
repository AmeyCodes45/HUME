from flask import Flask, request, jsonify
import requests
import json
import time
import os
import logging

# 📚 Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# ✅ Replace with your Hume AI API key (Ensure API Key is set correctly)
HUME_API_KEY = os.environ.get("HUME_API_KEY")

if not HUME_API_KEY:
    logging.error("❗️ HUME_API_KEY is not set. Please configure it in environment variables.")
    exit(1)

# 🎯 Define emotion level thresholds
def get_level(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"


@app.route("/process_hume", methods=["GET"])
def process_hume():
    # ✅ Get job_id from request query parameters (GET method fix)
    job_id = request.args.get("job_id")

    if not job_id:
        logging.error("❗️ Job ID is required but not provided.")
        return jsonify({"error": "Job ID is required"}), 400

    logging.info(f"✅ Received job_id: {job_id}")

    # 🕰️ Poll Hume API for results with retries
    url = f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions"
    headers = {"X-Hume-Api-Key": HUME_API_KEY}

    max_retries = 12  # Try for 120 seconds max (12 x 10 sec)
    response = None
    results = []

    for i in range(max_retries):
        logging.info(f"⏳ Attempt {i + 1}/{max_retries} - Fetching results from Hume API...")
        try:
            response = requests.get(url, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ API request failed: {str(e)}")
            return jsonify({"error": "Failed to connect to Hume API"}), 500

        # ✅ Log raw response for troubleshooting
        logging.debug(f"API Response: {response.status_code}, {response.text[:500]}")

        if response.status_code == 200:
            try:
                results = response.json()
                if isinstance(results, list):
                    state = results[0].get("state", "unknown")
                else:
                    logging.error("⚠️ Unexpected response format: results should be a list.")
                    return jsonify({"error": "Unexpected API response format"}), 500

                if state == "done":
                    logging.info("🎉 Hume API results ready!")
                    break
                else:
                    logging.info(f"⏳ Current state: {state}. Retrying...")
            except json.JSONDecodeError:
                logging.error("⚠️ Error parsing JSON response from Hume API")
                return jsonify({"error": "Failed to parse Hume API response"}), 500
        elif response.status_code == 404:
            logging.error("⚠️ Invalid job_id or results not found.")
            return jsonify({"error": "Invalid job_id or results not ready"}), 404

        time.sleep(10)  # Wait before retrying

    if not response or response.status_code != 200 or state != "done":
        logging.error("❌ Failed to retrieve data from Hume API after retries.")
        return jsonify({"error": "Failed to retrieve data from Hume AI"}), 500

    # 🧠 Process API response
    try:
        predictions = (
            results[0]["results"]["predictions"][0]["models"]["face"]["grouped_predictions"][0]["predictions"]
        )
        if not predictions:
            logging.error("⚠️ No predictions found in the API response.")
            return jsonify({"error": "No predictions found in Hume API results"}), 500
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"⚠️ Invalid data format or missing predictions: {str(e)}")
        return jsonify({"error": "Invalid data format from Hume API"}), 500

    # 🔍 Initialize variables for analysis
    emotion_scores = {}
    engagement_sum, nervousness_sum, confidence_sum = 0, 0, 0
    frame_count = len(predictions)

    # 📊 Process each frame to calculate scores
    for frame in predictions:
        emotions = frame["emotions"]
        for emotion in emotions:
            name = emotion["name"]
            score = emotion["score"]
            if name not in emotion_scores or score > emotion_scores[name]:
                emotion_scores[name] = score

        # Engagement Score → Concentration, Interest, Excitement
        engagement_sum += sum(
            [emotion["score"] for emotion in emotions if emotion["name"] in ["Concentration", "Interest", "Excitement"]]
        ) / 3

        # Nervousness Score → Anxiety, Distress, Doubt
        nervousness_sum += sum(
            [emotion["score"] for emotion in emotions if emotion["name"] in ["Anxiety", "Distress", "Doubt"]]
        ) / 3

        # Confidence Score → Determination, Pride, Satisfaction
        confidence_sum += sum(
            [emotion["score"] for emotion in emotions if emotion["name"] in ["Determination", "Pride", "Satisfaction"]]
        ) / 3

    # 🥇 Get top emotion
    top_emotion = max(emotion_scores, key=emotion_scores.get, default="Neutral")
    top_emotion_score = emotion_scores.get(top_emotion, 0)

    # 📈 Calculate average scores (avoid division by zero)
    if frame_count > 0:
        engagement_score = round(engagement_sum / frame_count, 2)
        nervousness_score = round(nervousness_sum / frame_count, 2)
        confidence_score = round(confidence_sum / frame_count, 2)
    else:
        engagement_score, nervousness_score, confidence_score = 0, 0, 0

    # 🎁 Generate response
    result = {
        "top_emotion": f"{top_emotion} ({'Excellent' if top_emotion_score > 0.7 else 'Good' if top_emotion_score > 0.4 else 'Average'})",
        "engagement_score": engagement_score,
        "engagement_level": get_level(engagement_score),
        "nervousness_score": nervousness_score,
        "nervousness_level": get_level(nervousness_score),
        "confidence_score": confidence_score,
        "confidence_level": get_level(confidence_score),
    }

    logging.info(f"✅ Generated result: {result}")
    return jsonify(result), 200


if __name__ == "__main__":
    # ✅ Fetch port dynamically for deployment
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
