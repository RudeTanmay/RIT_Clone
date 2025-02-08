from flask import Flask, render_template, request, jsonify
from chat import get_response
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data["message"]
    response = get_response(message)
    return jsonify({"answer": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
