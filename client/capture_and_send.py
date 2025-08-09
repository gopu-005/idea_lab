# client/capture_and_send.py
import cv2
import requests
import time
import os

# Backend endpoint (change to your server IP if needed)
BACKEND_URL = "http://127.0.0.1:5000/save_face"

# You can optionally provide a name to register known people
AUTO_REGISTER = False  # If True, unknown faces become new records automatically
NAME_TO_REGISTER = ""  # if non-empty, will register with this name when requested

# Haar cascade for face detection (fast). Adjust path if needed.
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)  # default webcam
if not cap.isOpened():
    print("Could not open webcam")
    exit(1)

print("Press 'r' to register current face with a name, 'q' to quit.")

last_sent = 0
SEND_INTERVAL = 2.0  # seconds between sends to avoid spamming backend

def send_face_image(cropped_bgr, register=False, name=""):
    # Encode as JPEG
    retval, buf = cv2.imencode('.jpg', cropped_bgr)
    if not retval:
        return None
    files = {'file': ('face.jpg', buf.tobytes(), 'image/jpeg')}
    data = {}
    if register:
        data['register'] = "1"
        if name:
            data['name'] = name
    else:
        if name:
            data['name'] = name
    try:
        resp = requests.post(BACKEND_URL, files=files, data=data, timeout=5)
        return resp.json()
    except Exception as e:
        return {"status":"error", "message": str(e)}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    # Draw boxes
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Entrance Camera - press 'r' to register, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    if key == ord('q'):
        break
    if key == ord('r') and len(faces) > 0:
        # Register first detected face with a name prompt
        (x,y,w,h) = faces[0]
        crop = frame[y:y+h, x:x+w].copy()
        name = input("Enter name to register for this face: ").strip()
        print("Registering...")
        res = send_face_image(crop, register=True, name=name)
        print("Server response:", res)
    # Auto-send first face every SEND_INTERVAL seconds to check for welcome
    if len(faces) > 0 and (current_time - last_sent) > SEND_INTERVAL:
        (x,y,w,h) = faces[0]
        crop = frame[y:y+h, x:x+w].copy()
        # optionally AUTO_REGISTER with empty name will create Guest if unknown
        res = send_face_image(crop, register=(AUTO_REGISTER), name=NAME_TO_REGISTER)
        if res:
            if res.get("status") == "ok":
                message = res.get("message", "Welcome!")
            else:
                message = res.get("message", "Error")
            # draw message on frame
            cv2.putText(frame, message, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Entrance Camera - press 'r' to register, 'q' to quit", frame)
            print("Server:", message)
        last_sent = current_time

cap.release()
cv2.destroyAllWindows()
