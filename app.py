from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Temporary data (later we'll replace this with live values)
queue_data = {
    "queue_count": 0,
    "people_served": 0,
    "avg_wait": 0,
    "status": "Normal"
}

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/data")
def data():
    return jsonify(queue_data)

if __name__ == "__main__":
    app.run(debug=True)