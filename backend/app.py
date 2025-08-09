# backend/app.py
import os
import io
import json
import uuid
import pickle
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)



DATA_DIR = "faces"
DB_FILE = os.path.join(DATA_DIR, "faces_db.pkl")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# DB structure: list of dicts: {id, name, embedding (np.array), image_path, created_at}
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        faces_db = pickle.load(f)
else:
    faces_db = []

def save_db():SAVE_DIR = "captured_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    name = data.get("name")
    img_data = data.get("image")

    if not name or not img_data:
        return jsonify({"status": "error", "message": "Name or image missing"}), 400

    # Decode Base64 image
    img_bytes = base64.b64decode(img_data.split(",")[1])
    img_array = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
    )

    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, img_array)

    return jsonify({"status": "ok", "message": f"Saved {filename}"})

if __name__ == "__main__":
    app.run(debug=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump(faces_db, f)

def find_match(embedding, threshold=0.55):
    if len(faces_db) == 0:
        return None, None
    db_embeddings = np.array([entry['embedding'] for entry in faces_db])
    # compute distances (euclidean)
    dists = np.linalg.norm(db_embeddings - embedding, axis=1)
    idx = np.argmin(dists)
    if dists[idx] <= threshold:
        return faces_db[idx], float(dists[idx])
    return None, float(dists[idx])

@app.route("/save_face", methods=["POST"])
def save_face():
    """
    Accepts multipart form:
      - file: image (face crop or full frame)
      - name: optional (string) -- if provided, store that name; if not and matched, return matched name
      - register: optional "1" or "0" - if 1, force registration (create new record)
    Returns JSON:
      { "status": "ok", "message": "...", "name": "...", "matched": True/False }
    """
    if 'file' not in request.files:
        return jsonify({"status":"error", "message":"no file provided"}), 400
    file = request.files['file']
    name = request.form.get("name", "").strip()
    register_flag = request.form.get("register", "0") == "1"

    image_bytes = file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"status":"error", "message":"cannot open image"}), 400

    np_img = np.array(pil_image)
    # attempt to compute face embedding(s)
    face_locations = face_recognition.face_locations(np_img, model="hog")
    if len(face_locations) == 0:
        # no face found; try encoding whole image (may fail)
        return jsonify({"status":"error", "message":"no face detected in image"}), 400

    # if multiple faces, use the first detected
    encodings = face_recognition.face_encodings(np_img, face_locations)
    if len(encodings) == 0:
        return jsonify({"status":"error", "message":"could not compute face encoding"}), 400

    embedding = encodings[0]

    # Try to match with existing DB unless forced registration
    matched = False
    matched_name = None
    matched_dist = None
    if not register_flag and len(faces_db) > 0:
        match_entry, dist = find_match(embedding)
        if match_entry:
            matched = True
            matched_name = match_entry['name']
            matched_dist = dist

    # If a name is provided or register_flag True and not matched, create new record
    if name or (register_flag and not matched):
        person_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()
        img_filename = f"{person_id}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        pil_image.save(img_path, format="JPEG")

        entry = {
            "id": person_id,
            "name": name if name else f"Guest-{person_id}",
            "embedding": embedding,
            "image_path": img_path,
            "created_at": timestamp
        }
        faces_db.append(entry)
        save_db()
        return jsonify({
            "status":"ok",
            "message": f"Registered {entry['name']}",
            "name": entry['name'],
            "matched": False
        }), 200

    if matched:
        # Return welcome message using matched name
        return jsonify({
            "status":"ok",
            "message": f"Welcome back, {matched_name}!",
            "name": matched_name,
            "matched": True,
            "distance": matched_dist
        }), 200

    # No name and not matched => register as anonymous guest (auto-create)
    person_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()
    img_filename = f"{person_id}.jpg"
    img_path = os.path.join(IMAGES_DIR, img_filename)
    pil_image.save(img_path, format="JPEG")

    entry = {
        "id": person_id,
        "name": f"Guest-{person_id}",
        "embedding": embedding,
        "image_path": img_path,
        "created_at": timestamp
    }
    faces_db.append(entry)
    save_db()
    return jsonify({
        "status":"ok",
        "message": f"Welcome, {entry['name']}! You've been registered.",
        "name": entry['name'],
        "matched": False
    }), 200

@app.route("/list_faces", methods=["GET"])
def list_faces():
    # Return metadata (not embeddings)
    out = [{"id":e["id"], "name":e["name"], "image_path": e["image_path"], "created_at":e["created_at"]} for e in faces_db]
    return jsonify(out)

if __name__ == "__main__":
    # Run in debug for development; use a proper server in production
    app.run(host="0.0.0.0", port=5000, debug=True)
