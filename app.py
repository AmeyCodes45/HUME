from flask import Flask, request, jsonify, send_file
import requests
import json
import time
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

HUME_API_KEY = os.environ.get("HUME_API_KEY")
if not HUME_API_KEY:
    logging.error("HUME_API_KEY not set.")
    exit(1)

confidence_emotions = {"calm", "focused", "content"}
nervousness_emotions = {"nervous", "worried", "tense"}

def get_level(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"

@app.route("/process_hume", methods=["GET"])
def process_hume():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400

    logging.info(f"✅ Received job_id: {job_id}")
    url = f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions"
    headers = {"X-Hume-Api-Key": HUME_API_KEY}

    results = None
    for attempt in range(18):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            logging.debug(f"API Response: {response.status_code}")
            if response.status_code == 200:
                results = response.json()
                with open("hume_api_response.json", "w") as f:
                    json.dump(results, f)
                break
            elif response.status_code == 404:
                return jsonify({"error": "Job not found"}), 404
            elif response.status_code == 401:
                return jsonify({"error": "Unauthorized"}), 401
        except Exception as e:
            logging.error(f"Request failed: {e}")
        time.sleep(10)

    if not results:
        return jsonify({"error": "Timeout or API error"}), 500

    # Process the saved file
    try:
        with open("hume_api_response.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read JSON: {e}"}), 500

    # Safe navigation
    try:
        predictions_raw = data[0]["results"]["predictions"]
        predictions = []
        for pred in predictions_raw:
            face_preds = pred["models"]["face"]["grouped_predictions"]
            for group in face_preds:
                predictions.extend(group["predictions"])
    except Exception as e:
        return jsonify({"error": f"Malformed prediction structure: {e}"}), 500

    frame_count = 0
    confidence_sum = 0
    nervousness_sum = 0
    top_emotions = []
    emotion_counts = {}

    for frame in predictions:
        emotions = frame.get("emotions", [])
        if not emotions:
            continue

        top = max(emotions, key=lambda e: e["score"])
        top_name = top["name"]
        top_emotions.append(top_name)
        emotion_counts[top_name] = emotion_counts.get(top_name, 0) + 1

        conf = [e["score"] for e in emotions if e["name"].lower() in confidence_emotions]
        nerv = [e["score"] for e in emotions if e["name"].lower() in nervousness_emotions]
        if conf:
            confidence_sum += sum(conf) / len(conf)
        if nerv:
            nervousness_sum += sum(nerv) / len(nerv)

        frame_count += 1

    if frame_count == 0:
        return jsonify({"error": "No valid emotion data found"}), 400

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

@app.route("/get_hume_json", methods=["GET"])
def get_hume_json():
    try:
        return send_file("hume_api_response.json", as_attachment=True, download_name="hume_api_response.json")
    except Exception as e:
        return jsonify({"error": f"Could not send file: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
