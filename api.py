from flask import Flask, request
from transformers import pipeline
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "current_model"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} does not exist")

nlp = pipeline(
    "text-classification",
    model=str(MODEL_PATH)
)


@app.route("/intent_recognition", methods=["POST"])
def intent_recognition():

    data = request.get_json()

    query = data.get("query", "")

    if len(query) <= 5:
        return {
            "error": "Query must be at least 6 characters."
        }, 400

    result = nlp(query)[0]

    return {
        "query": query,
        "predicted_class": result["label"],
        "confidence": result["score"],
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )