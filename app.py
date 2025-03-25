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

    max_retries = 18  # Increased to 3 minutes (18 x 10 sec)
    results = None

    for i in range(max_retries):
        logging.info(f"⏳ Attempt {i + 1}/{max_retries} - Fetching results from Hume API...")
        try:
            response = requests.get(url, headers=headers, timeout=30)  # Increased timeout to 30 sec
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ API request failed: {str(e)}")
            return jsonify({"error": "Failed to connect to Hume API"}), 500

        # ✅ Log raw response for troubleshooting
        logging.debug(f"API Response: {response.status_code}, {response.text[:500]}")

        if response.status_code == 200:
            try:
                results = response.json()

                # ✅ Check if results are in the expected format
                if isinstance(results, list) and len(results) > 0 and "results" in results[0]:
                    logging.info("🎉 Hume API results ready! Proceeding to processing...")
                    break
                else:
                    logging.error("⚠️ Unexpected response format or no results found.")
                    return jsonify({"error": "Unexpected API response format or no results found"}), 500
            except json.JSONDecodeError:
                logging.error("⚠️ Error parsing JSON response from Hume API")
                return jsonify({"error": "Failed to parse Hume API response"}), 500
        elif response.status_code == 404:
            logging.error("⚠️ Invalid job_id or results not found.")
            return jsonify({"error": "Invalid job_id or results not ready"}), 404
        elif response.status_code == 401:
            logging.error("❗️ Unauthorized - Invalid Hume API Key.")
            return jsonify({"error": "Unauthorized - Check your API Key"}), 401
        elif response.status_code == 500:
            logging.error("⚠️ Hume API internal error.")
            return jsonify({"error": "Hume API internal error"}), 500

        time.sleep(10)  # Wait before retrying

    if not results:
        logging.error("❌ Failed to retrieve data from Hume API after retries.")
        return jsonify({"error": "Failed to retrieve data from Hume AI"}), 500

    # 🧠 Process API response
    try:
        predictions = (
            results[0]["results"]["predictions"][0]["models"]["face"]["grouped_predictions"][0]["predictions"]
        )
        if not predictions or not isinstance(predictions, list):
            logging.error("⚠️ No valid predictions in the API response.")
            return jsonify({"error": "No predictions found in Hume API results"}), 500
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"⚠️ Invalid data format or missing predictions: {str(e)}")
        return jsonify({"error": "Invalid data format from Hume API"}), 500

    # ✅ Define correct emotion categories
    confidence_emotions = ["calm", "focused", "content"]
    nervousness_emotions = ["nervous", "worried", "tense"]

    # 🔍 Initialize variables for analysis
    emotion_scores = {}
    top_emotions = []
    confidence_sum, nervousness_sum = 0, 0
    frame_count = len(predictions)

    # 📊 Process each frame to calculate scores efficiently
    for frame in predictions:
        emotions = frame.get("emotions", [])
        if not emotions:
            continue

        # ✅ Get top emotion per frame
        top_emotion_frame = max(emotions, key=lambda x: x["score"])
        top_emotions.append(top_emotion_frame["name"])
        emotion_scores[top_emotion_frame["name"]] = (
            emotion_scores.get(top_emotion_frame["name"], 0) + 1
        )

        # ✅ Confidence Score
        confidence_scores = [e["score"] for e in emotions if e["name"].lower() in confidence_emotions]
        confidence_sum += sum(confidence_scores) / len(confidence_emotions) if confidence_scores else 0

        # ✅ Nervousness Score
        nervousness_scores = [e["score"] for e in emotions if e["name"].lower() in nervousness_emotions]
        nervousness_sum += sum(nervousness_scores) / len(nervousness_emotions) if nervousness_scores else 0

    # ✅ Top Emotion - Most Frequent
    final_top_emotion = max(emotion_scores, key=emotion_scores.get, default="Neutral")

    # ✅ Engagement Score - Emotion Switches Count
    switches = sum(1 for i in range(1, len(top_emotions)) if top_emotions[i] != top_emotions[i - 1])
    engagement_score = switches / (len(top_emotions) - 1) if len(top_emotions) > 1 else 0

    # 📈 Calculate average scores (avoid division by zero)
    confidence_score = round(confidence_sum / frame_count, 2) if frame_count > 0 else 0
    nervousness_score = round(nervousness_sum / frame_count, 2) if frame_count > 0 else 0
    engagement_score = round(engagement_score, 2)

    # 🎁 Generate response
    result = {
        "top_emotion": f"{final_top_emotion} ({'Excellent' if emotion_scores[final_top_emotion] > 0.7 else 'Good' if emotion_scores[final_top_emotion] > 0.4 else 'Average'})",
        "engagement_score": engagement_score,
        "engagement_level": get_level(engagement_score),
        "nervousness_score": nervousness_score,
        "nervousness_level": get_level(nervousness_score),
        "confidence_score": confidence_score,
        "confidence_level": get_level(confidence_score),
    }

    logging.info(f"✅ Corrected result: {result}")
    return jsonify(result), 200


if __name__ == "__main__":
    # ✅ Fetch port dynamically for deployment
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
