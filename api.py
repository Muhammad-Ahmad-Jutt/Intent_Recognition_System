from flask import Flask, request
from transformers import pipeline
from pathlib import Path
from dotenv import load_dotenv


app = Flask(__name__)

current = Path(__file__).resolve().parent

model_path = current / "current_model"

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )


nlp = pipeline(
    "text-classification",
    model=str(model_path)
)


@app.route('/intent_recognition', methods=['POST'])
def intent_recognition():

    data = request.get_json()

    query = data.get("query", "")

    if len(query) <= 5:
        return {
            "error": "Query must be at least 6 characters required."
        }, 400

    return predict_intent(query)


def predict_intent(query):

    result = nlp(query)[0]
    return {
        "query": query,
        "predicted_class": result['label'],
        "confidence": result["score"]
    }



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)