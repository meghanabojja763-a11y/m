import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import glob
from pathlib import Path

# Paths for storing data
USERS_DIR = "registered_users"
DOCS_DIR = "teacher_documents"

# Utility to save image from OpenCV frame
def save_user_photo(user_id, frame):
    user_folder = os.path.join(USERS_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)
    photo_path = os.path.join(user_folder, "photo.jpg")
    cv2.imwrite(photo_path, frame)
    return photo_path

def load_known_faces():
    known_encodings = []
    known_ids = []
    if not os.path.exists(USERS_DIR):
        return known_encodings, known_ids
    for user_folder in os.listdir(USERS_DIR):
        photo_path = os.path.join(USERS_DIR, user_folder, "photo.jpg")
        if os.path.exists(photo_path):
            image = face_recognition.load_image_file(photo_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_ids.append(user_folder)
    return known_encodings, known_ids

def display_documents(user_id):
    user_doc_folder = os.path.join(DOCS_DIR, user_id)
    if not os.path.exists(user_doc_folder):
        st.warning("No documents found for this user")
        return
    ppts = glob.glob(f"{user_doc_folder}/*.ppt*")
    pdfs = glob.glob(f"{user_doc_folder}/*.pdf")
    st.write(f"### Documents for {user_id}:")
    for ppt in ppts:
        if st.button(f"Open {os.path.basename(ppt)}"):
            os.system(f'start "" "{ppt}"')  # Windows command to open file
    for pdf in pdfs:
        if st.button(f"Open {os.path.basename(pdf)}"):
            os.system(f'start "" "{pdf}"')

def register_user():
    st.subheader("User Registration")
    college_name = st.text_input("Enter your College/Office name")
    user_id = st.text_input("Create a unique User ID (no spaces)")
    
    start_camera = st.button("Start Camera for Photo Capture")
    FRAME_WINDOW = st.image([])
    
    if start_camera and college_name and user_id:
        cap = cv2.VideoCapture(0)
        st.info("Press 'Capture Photo' when ready")
        
        captured = False
        while not captured:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if st.button("Capture Photo"):
                save_path = save_user_photo(user_id, frame)
                st.success(f"Photo saved to {save_path}")
                captured = True
                cap.release()
        if captured:
            # Create user folder for documents as well
            os.makedirs(os.path.join(DOCS_DIR, user_id), exist_ok=True)
            st.success(f"Registration complete for user '{user_id}' from '{college_name}'")
            st.stop()
    else:
        if start_camera:
            st.warning("Please enter College/Office name and User ID before starting camera")

def recognize_user():
    known_encodings, known_ids = load_known_faces()
    if not known_encodings:
        st.warning("No registered users found. Please register first.")
        return None

    st.subheader("User Login - Face Recognition")
    run = st.checkbox('Start camera to recognize your face')
    FRAME_WINDOW = st.image([])
    recognized_user = None

    if run:
        cap = cv2.VideoCapture(0)
        process_frame = True
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break
            
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_frame = small_frame[:, :, ::-1]

            if process_frame:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_ids[best_match_index]
                    face_names.append(name)

                if face_names and face_names[0] != "Unknown":
                    recognized_user = face_names[0]
                    st.success(f"User recognized: {recognized_user}")
                    cap.release()
                    break

            process_frame = not process_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,0), 1)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    return recognized_user

def main():
    st.title("Teacher Face Recognition App with Registration & Document Access")

    menu = ["Register", "Login & Open Docs"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        register_user()
    elif choice == "Login & Open Docs":
        user = recognize_user()
        if user:
            display_documents(user)

if __name__ == "__main__":
    main()
