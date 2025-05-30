from flask import Flask, request, jsonify, send_file
import requests
import json
import time
import os
import logging
import ijson

# 📚 Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# ✅ Replace with your Hume AI API key (Ensure API Key is set correctly)
HUME_API_KEY = os.environ.get("HUME_API_KEY")

if not HUME_API_KEY:
    logging.error("❗️ HUME_API_KEY is not set. Please configure it in environment variables.")
    exit(1)

# 🎯 Emotion thresholds
def get_level(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"

confidence_emotions = {"calm", "focused", "content"}
nervousness_emotions = {"nervous", "worried", "tense"}

@app.route("/process_hume", methods=["GET"])
def process_hume():
    job_id = request.args.get("job_id")

    if not job_id:
        logging.error("❗️ Job ID is required but not provided.")
        return jsonify({"error": "Job ID is required"}), 400

    logging.info(f"✅ Received job_id: {job_id}")
    url = f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions"
    headers = {"X-Hume-Api-Key": HUME_API_KEY}

    max_retries = 18
    results = None

    for i in range(max_retries):
        logging.info(f"⏳ Attempt {i + 1}/{max_retries} - Fetching results from Hume API...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ API request failed: {str(e)}")
            return jsonify({"error": "Failed to connect to Hume API"}), 500

        logging.debug(f"API Response: {response.status_code}, {response.text[:500]}")

        if response.status_code == 200:
            try:
                results = response.json()
                if isinstance(results, list) and len(results) > 0 and "results" in results[0]:
                    with open("hume_api_response.json", "w") as f:
                        json.dump(results, f, indent=4)
                    logging.info("✅ Full Hume API JSON saved.")
                    break
                else:
                    logging.error("⚠️ Unexpected response format or no results.")
                    return jsonify({"error": "Unexpected API response format or no results found"}), 500
            except json.JSONDecodeError:
                logging.error("⚠️ Error parsing JSON response from Hume API")
                return jsonify({"error": "Failed to parse Hume API response"}), 500
        elif response.status_code == 404:
            return jsonify({"error": "Invalid job_id or results not ready"}), 404
        elif response.status_code == 401:
            return jsonify({"error": "Unauthorized - Check your API Key"}), 401
        elif response.status_code == 500:
            return jsonify({"error": "Hume API internal error"}), 500

        time.sleep(10)

    if not results:
        return jsonify({"error": "Failed to retrieve data from Hume AI after retries"}), 500

    # ✅ Stream JSON file instead of loading full into memory
    file_path = "hume_api_response.json"
    if not os.path.exists(file_path):
        return jsonify({"error": "hume_api_response.json not found"}), 404

    confidence_sum = 0
    nervousness_sum = 0
    frame_count = 0
    top_emotions = []
    emotion_counts = {}

    try:
        with open(file_path, "r") as f:
            frames = ijson.items(f, "item.results.predictions.item.models.face.grouped_predictions.item.predictions.item")

            for frame in frames:
                emotions = frame.get("emotions", [])
                if not emotions:
                    continue

                top_emotion = max(emotions, key=lambda e: e["score"])
                top_name = top_emotion["name"]
                top_emotions.append(top_name)
                emotion_counts[top_name] = emotion_counts.get(top_name, 0) + 1

                confidence_scores = [e["score"] for e in emotions if e["name"].lower() in confidence_emotions]
                if confidence_scores:
                    confidence_sum += sum(confidence_scores) / len(confidence_scores)

                nervousness_scores = [e["score"] for e in emotions if e["name"].lower() in nervousness_emotions]
                if nervousness_scores:
                    nervousness_sum += sum(nervousness_scores) / len(nervousness_scores)

                frame_count += 1

        if frame_count == 0:
            return jsonify({"error": "No valid frames found"}), 400

        switches = sum(1 for i in range(1, len(top_emotions)) if top_emotions[i] != top_emotions[i - 1])
        engagement_score = switches / (len(top_emotions) - 1) if len(top_emotions) > 1 else 0

        top_emotion = max(emotion_counts, key=emotion_counts.get, default="Neutral")

        result = {
            "confidence_score": round(confidence_sum / frame_count, 2),
            "confidence_level": get_level(confidence_sum / frame_count),
            "nervousness_score": round(nervousness_sum / frame_count, 2),
            "nervousness_level": get_level(nervousness_sum / frame_count),
            "engagement_score": round(engagement_score, 2),
            "engagement_level": get_level(engagement_score),
            "top_emotion": f"{top_emotion} (Average)"
        }

        logging.info(f"✅ Final Result: {result}")
        return jsonify(result), 200

    except Exception as e:
        logging.error(f"❌ Error during processing: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/get_hume_json", methods=["GET"])
def get_hume_json():
    file_path = "hume_api_response.json"
    if not os.path.exists(file_path):
        return jsonify({"error": "hume_api_response.json not found"}), 404
    try:
        return send_file(file_path, as_attachment=True, download_name="hume_api_response.json")
    except Exception as e:
        return jsonify({"error": "Error while sending the file"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
