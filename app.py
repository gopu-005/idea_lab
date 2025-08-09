import os
import io
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from pymongo import MongoClient
import gridfs
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()  # if you use a .env file for MONGO_URI

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

# Haar cascade path (comes with opencv)
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def decode_base64_image(data_url):
    # expected "data:image/png;base64,AAAA..."
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Expect JSON:
    {
      "image": "data:image/png;base64,...",
      "name": "Optional Name"
    }
    """
    payload = request.get_json(force=True)
    data_url = payload.get("image")
    name = payload.get("name", "").strip()

    if not data_url:
        return jsonify({"success": False, "error": "No image sent"}), 400

    # decode to cv2 image
    try:
        frame = decode_base64_image(data_url)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not decode image: {e}"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces (tweak scaleFactor/minSize as you like)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    if len(faces) == 0:
        return jsonify({"success": False, "error": "No face detected. Ask the person to face the camera."}), 200

    # pick the largest face (likely the person)
    largest = max(faces, key=lambda r: r[2]*r[3])
    x, y, w, h = largest
    margin = int(0.25 * max(w, h))  # add some margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    face_img = frame[y1:y2, x1:x2]

    # convert BGR -> RGB and save via PIL
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).convert("RGB")

    # filename
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}.jpg"
    save_path = os.path.join(SAVE_FOLDER, filename)
    pil_img.save(save_path, format="JPEG", quality=90)

    # Save a copy to GridFS
    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        grid_id = fs.put(buf.read(), filename=filename, contentType="image/jpeg", uploaded_at=datetime.utcnow())

    # Save metadata
    doc = {
        "filename": filename,
        "gridfs_id": grid_id,
        "name": name if name else None,
        "timestamp": datetime.utcnow(),
    }
    meta_collection.insert_one(doc)

    # build welcome message
    display_name = name if name else "Welcome Guest"
    # construct accessible URL for the saved face file
    face_url = url_for("static", filename=f"faces/{filename}")

    return jsonify({
        "success": True,
        "message": f"Welcome, {display_name}!",
        "face_url": face_url,
        "meta_id": str(doc.get("_id"))
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
