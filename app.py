from flask import Flask, request, jsonify
import requests
import json
import time
import os

app = Flask(__name__)

# Replace with your Hume AI API key
HUME_API_KEY = os.environ.get("HUME_API_KEY", "YOUR_HUME_API_KEY")

# Define levels based on thresholds
def get_level(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"


@app.route("/process_hume", methods=["GET"])
def process_hume():
    # Get job_id from the request
    data = request.json
    job_id = data.get("job_id")

    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400

    print(f"Received job_id: {job_id}")  # ✅ Debug Log

    # Delay to allow Hume API to process results
    time.sleep(15)  # ✅ Wait for 15 seconds

    # Fetch results from Hume API
    url = f"https://api.hume.ai/v1/jobs/{job_id}/predictions"
    headers = {"X-Hume-Api-Key": " {HUME_API_KEY}"}

    print("Fetching results from Hume API...")  # ✅ Debug Log
    response = requests.get(url, headers=headers)
    print(f"API Response: {response.status_code}, {response.text}")  # ✅ Debug Log

    if response.status_code != 200:
        return jsonify({"error": "Failed to retrieve data from Hume AI"}), 500

    # Parse the JSON result
    results = response.json()

    # Handle invalid format
    try:
        predictions = results["results"]["predictions"][0]["models"]["face"]["grouped_predictions"][0]["predictions"]
    except KeyError:
        return jsonify({"error": "Invalid data format"}), 500

    # Initialize variables
    emotion_scores = {}
    engagement_sum, nervousness_sum, confidence_sum = 0, 0, 0
    frame_count = len(predictions)

    # Process each frame to calculate the scores
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

    # Get top emotion
    top_emotion = max(emotion_scores, key=emotion_scores.get)
    top_emotion_score = emotion_scores[top_emotion]

    # Calculate average scores
    engagement_score = round(engagement_sum / frame_count, 2)
    nervousness_score = round(nervousness_sum / frame_count, 2)
    confidence_score = round(confidence_sum / frame_count, 2)

    # Generate response
    result = {
        "top_emotion": f"{top_emotion} ({'Excellent' if top_emotion_score > 0.7 else 'Good' if top_emotion_score > 0.4 else 'Average'})",
        "engagement_score": engagement_score,
        "engagement_level": get_level(engagement_score),
        "nervousness_score": nervousness_score,
        "nervousness_level": get_level(nervousness_score),
        "confidence_score": confidence_score,
        "confidence_level": get_level(confidence_score),
    }

    print(f"Generated result: {result}")  # ✅ Debug Log
    return jsonify(result), 200


if __name__ == "__main__":
    # ✅ Dynamically fetch port to work with Render
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
