import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as T
from models.cnn import IntelCNN_PyTorch

# Config
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 150
PYTORCH_WEIGHTS = os.path.join(BASE_DIR, "geraud_model.pth")
KERAS_WEIGHTS   = os.path.join(BASE_DIR, "geraud_model.keras")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

_pytorch_model = None
_keras_model   = None

def get_pytorch_model():
    global _pytorch_model
    if _pytorch_model is None:
        if not os.path.exists(PYTORCH_WEIGHTS):
            raise FileNotFoundError(f"PyTorch weights not found: {PYTORCH_WEIGHTS}")
        _pytorch_model = IntelCNN_PyTorch(num_classes=6).to(DEVICE)
        _pytorch_model.load_state_dict(
            torch.load(PYTORCH_WEIGHTS, map_location=DEVICE, weights_only=True)  
        )
        _pytorch_model.eval()
        print(f" PyTorch model loaded ({DEVICE})")
    return _pytorch_model

def get_keras_model():
    global _keras_model
    if _keras_model is None:
        if not os.path.exists(KERAS_WEIGHTS):
            raise FileNotFoundError(f"Keras weights not found: {KERAS_WEIGHTS}")
        import tensorflow as tf
        _keras_model = tf.keras.models.load_model(KERAS_WEIGHTS)
        print(" Keras model loaded")
    return _keras_model

# Preprocessing
_torch_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_torch(pil_img):
    return _torch_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

def preprocess_keras(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    backend = request.form.get("backend", "pytorch").lower()
    file    = request.files["image"]

    # Verify of the the file is image
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        if backend == "keras":
            model  = get_keras_model()
            tensor = preprocess_keras(img)
            probs  = model.predict(tensor, verbose=0)[0]
            used   = "Keras"
        else:
            model  = get_pytorch_model()
            with torch.no_grad():
                logits = model(preprocess_torch(img))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            used = "PyTorch"

        results = [
            {"class": cls, "confidence": round(float(p) * 100, 2)}
            for cls, p in zip(CLASSES, probs)
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all":        results,
            "backend":    used,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def available_models():
    return jsonify({
        "PyTorch": os.path.exists(PYTORCH_WEIGHTS),
        "Keras":   os.path.exists(KERAS_WEIGHTS),
    })

#Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
