from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
import base64  # Import base64 to handle image data

app = Flask(__name__)

# Directory to store known faces
known_faces_dir = 'known_faces'
known_faces = {}
attendance = []

# Load known faces
def load_known_faces():
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces[filename.split('.')[0]] = encoding  # Use the filename (without extension) as the name

load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    image_data = data['image']

    # Decode the image data
    image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

    # Recognize faces
    face_encodings = face_recognition.face_encodings(image)
    print(f"Found {len(face_encodings)} face(s) in the image.")  # Debug statement

    if face_encodings:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]
                if name not in attendance:
                    attendance.append(name)  # Mark attendance

    print(f"Current attendance: {attendance}")  # Debug statement
    return jsonify({'attendance': attendance})

if __name__ == '__main__':
    app.run(debug=True)
