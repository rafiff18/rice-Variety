import io
import os
import time
import base64
import pathlib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

app = FastAPI(title="RiceVarietyVision", version="1.0.0")

# Setup paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATASET_DIR = ROOT_DIR / "rice-dataset"
TFLITE_PATH = ROOT_DIR / "tflite" / "model.tflite"
LABEL_PATH = ROOT_DIR / "tflite" / "label.txt"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load TFLite Model
print(f"Loading TFLite model from {TFLITE_PATH}...")
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Labels
with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]

print(f"Model loaded successfully with classes: {CLASS_NAMES}")

# Rich Rice Varieties Knowledge Base
VARIETY_METADATA = {
    "Arborio": {
        "tagline": "Italian Short/Medium Grain Classic",
        "origin": "Po Valley, Northern Italy",
        "shape": "Short, oval, plump with distinct pearly center",
        "texture": "Creamy, rich starch exterior with firm al dente core",
        "best_uses": "Classic Risotto, Arancini, Rice Puddings, Soups",
        "cooking_ratio": "1 : 3 (Liquid absorption is very high)",
        "glycemic_index": "Medium-High (~70)",
        "accent_color": "#38bdf8"
    },
    "Basmati": {
        "tagline": "Crown Jewel of Fragrant Long Grains",
        "origin": "Himalayan Foothills (India & Pakistan)",
        "shape": "Extra-long, needle-slender, tapers at grain ends",
        "texture": "Dry, light, exceptionally fluffy, grains double in length without sticking",
        "best_uses": "Royal Biryani, Dum Pulao, Persian Rice, Indian Curries",
        "cooking_ratio": "1 : 1.75 to 2.0",
        "glycemic_index": "Low-Medium (50–55, diabetic friendly)",
        "accent_color": "#10b981"
    },
    "Ipsala": {
        "tagline": "Thracian Premium Baldo-Grade Grain",
        "origin": "Meriç Plain, Ipsala, Edirne (Turkey)",
        "shape": "Large, elongated, crystalline with high translucency",
        "texture": "Firm, resilient bite that absorbs heavy broths without becoming mushy",
        "best_uses": "Turkish Wedding Pilaf, Stuffed Vegetables (Dolma), Mediterranean Casseroles",
        "cooking_ratio": "1 : 2.0 to 2.25",
        "glycemic_index": "Medium (~62)",
        "accent_color": "#8b5cf6"
    },
    "Jasmine": {
        "tagline": "Aromatic Southeast Asian Staple",
        "origin": "Northeastern Thailand (Thai Hom Mali)",
        "shape": "Long, translucent, slightly rounded shoulders",
        "texture": "Silky, tender, naturally sweet floral pandan aroma with gentle clinginess",
        "best_uses": "Pad Krapow, Thai Green Curry, Nasi Uduk, Garlic Fried Rice",
        "cooking_ratio": "1 : 1.25 to 1.5",
        "glycemic_index": "High (~68–78)",
        "accent_color": "#f59e0b"
    },
    "Karacadag": {
        "tagline": "Volcanic Soil Heritage Grain",
        "origin": "Mount Karacadağ, Diyarbakir (Anatolia)",
        "shape": "Short, robust, rounded pebble-like profile with chalky nucleus",
        "texture": "Deep earthy aroma, exceptionally high oil and mineral content, chewy resilience",
        "best_uses": "Traditional Diyarbakir Pilaf, Slow-cooked Lamb Stews, Soups",
        "cooking_ratio": "1 : 2.0 to 2.5",
        "glycemic_index": "Low-Medium (~54)",
        "accent_color": "#ec4899"
    }
}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image matching training pipeline (150x150, RGB, Rescaling [0, 1])."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Aspect-ratio preserving crop & resize (similar to crop_to_aspect_ratio=True)
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((150, 150), Image.BILINEAR)
    
    img_array = np.array(img_resized, dtype=np.float32)
    # Model has Rescaling(1./255) inside or accepts [0, 255] float32
    # In our trained inference model: model.layers[0] is Rescaling(1./255)
    # So input to interpreter is raw RGB pixel values [0, 255] float32
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))

@app.get("/api/samples")
async def get_sample_images():
    """Returns 1 sample image per class encoded as base64 for instant 1-click testing."""
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / cls
        if cls_dir.exists():
            image_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            if image_files:
                sample_file = image_files[0]
                with open(sample_file, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                samples.append({
                    "class_name": cls,
                    "filename": sample_file.name,
                    "image_b64": f"data:image/jpeg;base64,{b64_data}"
                })
    return JSONResponse(content={"samples": samples})

@app.post("/api/predict")
async def predict_rice_variety(file: UploadFile = File(...)):
    """Inference endpoint using the optimized TFLite model."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
    
    contents = await file.read()
    start_time = time.time()
    
    try:
        input_data = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    
    latency_ms = (time.time() - start_time) * 1000
    
    pred_idx = int(np.argmax(output_data))
    top_class = CLASS_NAMES[pred_idx]
    confidence_pct = float(output_data[pred_idx] * 100)
    
    # Build complete breakdown list sorted by probability descending
    probabilities = []
    for idx, cls in enumerate(CLASS_NAMES):
        prob = float(output_data[idx] * 100)
        probabilities.append({
            "class_name": cls,
            "probability": round(prob, 2),
            "percentage_str": f"{prob:.2f}%",
            "accent_color": VARIETY_METADATA.get(cls, {}).get("accent_color", "#38bdf8")
        })
    probabilities.sort(key=lambda x: x["probability"], reverse=True)
    
    metadata = VARIETY_METADATA.get(top_class, {})
    
    return JSONResponse({
        "success": True,
        "prediction": {
            "class_name": top_class,
            "confidence": round(confidence_pct, 2),
            "confidence_str": f"{confidence_pct:.2f}%",
            "is_high_confidence": confidence_pct >= 85.0
        },
        "probabilities": probabilities,
        "metadata": metadata,
        "latency_ms": round(latency_ms, 2)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
