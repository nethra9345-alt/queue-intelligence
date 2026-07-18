from flask import Flask, render_template, jsonify, Response
import json
from detector import generate_frames

app = Flask(__name__)

@app.route("/")
def home():

    try:
        with open("output/queue_data.json", "r") as f:
            data = json.load(f)
    except:
        data = {
            "queue_count": 0,
            "people_served": 0,
            "average_wait": 0,
            "longest_wait": 0,
            "status": "Low",
            "recommendation": "Queue Normal",
            "fps": 0,
            "confidence": 0,
            "throughput": 0
        }

    return render_template(
        "dashboard.html",
        queue_count=data.get("queue_count",0),
        wait_time=data.get("average_wait",0),
        longest_wait=data.get("longest_wait",0),
        risk=data.get("status","Low"),
        people_served=data.get("people_served",0),
        recommendation=data.get("recommendation","Queue Normal"),
        fps=data.get("fps",0),
        confidence=data.get("confidence",0),
        throughput=data.get("throughput",0)
    )

@app.route("/data")
def data():

    with open("output/queue_data.json","r") as f:
        return jsonify(json.load(f))

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__=="__main__":
    app.run(debug=True)