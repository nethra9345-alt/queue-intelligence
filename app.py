from flask import Flask, jsonify, render_template
import json

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/data")
def data():
    with open("output/queue_data.json","r") as f:
        return jsonify(json.load(f))

if __name__=="__main__":
    app.run(debug=True)