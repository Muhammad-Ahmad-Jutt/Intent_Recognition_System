import json
import os

import torch
from flask import Flask, request, Response
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_DIR = os.environ.get(
    "SM_MODEL_DIR",
    "/opt/ml/model"
)


app = Flask(__name__)


tokenizer = None
model = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global tokenizer
    global model

    print(
        f"Loading model from: {MODEL_DIR}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    print(
        "Model loaded successfully."
    )


# ============================================================
# PING
# ============================================================

@app.route(
    "/ping",
    methods=["GET"]
)
def ping():

    if model is None:

        return Response(
            response="Model not loaded",
            status=500
        )

    return Response(
        response="OK",
        status=200
    )


# ============================================================
# INVOCATIONS
# ============================================================

@app.route(
    "/invocations",
    methods=["POST"]
)
def invocations():

    if model is None:

        return Response(
            response="Model not loaded",
            status=500
        )

    request_data = request.get_json()

    if request_data is None:

        return Response(
            response=json.dumps({
                "error": "Invalid JSON request"
            }),
            status=400,
            mimetype="application/json"
        )

    if "text" not in request_data:

        return Response(
            response=json.dumps({
                "error": "Request must contain 'text'"
            }),
            status=400,
            mimetype="application/json"
        )

    text = request_data["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    label = model.config.id2label.get(
        predicted_id,
        str(predicted_id)
    )

    result = {
        "intent": label,
        "confidence": confidence,
        "label_id": predicted_id
    }

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    load_model()

    app.run(
        host="0.0.0.0",
        port=8080
    )