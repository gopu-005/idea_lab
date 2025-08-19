import os
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from pymongo import MongoClient
import gridfs
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv
import re

load_dotenv()

# CONFIG
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "face_welcome_db")
SAVE_FOLDER = os.path.join("static", "faces")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Mongo + GridFS
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)
meta_collection = db["faces_meta"]

# Haar cascade path
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def sanitize_filename(name):
    sanitized = re.sub(r'[^\w\s-]', '', name)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.strip('_').lower()

def decode_base64_image(data_url):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    payload = request.get_json(force=True)
    data_url = payload.get("image")
    name = payload.get("name", "").strip()

    if not data_url:
        return jsonify({"success": False, "error": "No image received"}), 400
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    try:
        frame = decode_base64_image(data_url)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not decode image: {e}"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create circular mask matching overlay size (200px diameter)
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    radius = 100  # matches front-end overlay
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    faces = face_cascade.detectMultiScale(
        masked_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return jsonify({
            "success": False,
            "error": "No face detected inside the circle. Please align your face properly."
        }), 200

    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = largest

    margin = int(0.3 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    face_img = frame[y1:y2, x1:x2]

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).convert("RGB")

    safe_name = sanitize_filename(name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.jpg"
    save_path = os.path.join(SAVE_FOLDER, filename)
    pil_img.save(save_path, format="JPEG", quality=95)

    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        grid_id = fs.put(
            buf.read(),
            filename=filename,
            contentType="image/jpeg",
            uploaded_at=datetime.utcnow()
        )

    doc = {
        "filename": filename,
        "gridfs_id": grid_id,
        "name": name,
        "safe_name": safe_name,
        "timestamp": datetime.utcnow(),
        "face_detected": True,
        "face_count": len(faces)
    }
    meta_collection.insert_one(doc)

    import random
    welcome_messages = [
        f"Welcome to our institution, {name}! 🎉",
        f"Hello {name}! Great to have you here! ✨",
        f"Welcome {name}! Hope you have a wonderful time! 🌟",
        f"Greetings {name}! Welcome aboard! 🚀"
    ]
    welcome_message = random.choice(welcome_messages)

    face_url = url_for("static", filename=f"faces/{filename}")

    return jsonify({
        "success": True,
        "message": welcome_message,
        "face_url": face_url,
        "name": name,
        "filename": filename,
        "meta_id": str(doc.get("_id"))
    }), 200

@app.route("/api/visitors", methods=["GET"])
def get_visitors():
    try:
        recent_visitors = list(
            meta_collection.find()
            .sort("timestamp", -1)
            .limit(10)
        )
        for visitor in recent_visitors:
            visitor["_id"] = str(visitor["_id"])
            visitor["gridfs_id"] = str(visitor["gridfs_id"])
            visitor["timestamp"] = visitor["timestamp"].isoformat()
        return jsonify({"success": True, "visitors": recent_visitors})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
