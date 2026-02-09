import streamlit as st
import pandas as pd
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# --- CONFIGURATION ---
ATTENDANCE_FILE = 'attendance.csv'
KNOWN_FACES_DIR = 'known_faces'

# Ensure directories exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

if not os.path.exists(ATTENDANCE_FILE):
    # Create the CSV with headers if it doesn't exist
    df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
    df.to_csv(ATTENDANCE_FILE, index=False)

# --- FUNCTIONS ---

@st.cache_data
def load_known_faces():
    """
    Loads all images from the 'known_faces' folder and encodes them.
    Cached so the app remains fast.
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        return [], []

    files = os.listdir(KNOWN_FACES_DIR)
    
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, file)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    # Use filename (without extension) as the name
                    name = os.path.splitext(file)[0]
                    known_names.append(name)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
                
    return known_encodings, known_names

def mark_attendance(name):
    """
    Logs the attendance into the CSV file.
    """
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Check if this person is already in the log for TODAY
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_entry = pd.DataFrame({'Name': [name], 'Time': [time_str], 'Date': [date_str]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"✅ Attendance marked for: {name}")
    else:
        st.info(f"ℹ️ {name} is already marked present for today.")

# --- UI LAYOUT ---

st.set_page_config(page_title="Smart Attendance", layout="wide")
st.title("📸 Smart Attendance System")

# Sidebar for Registration
with st.sidebar:
    st.header("Register New Person")
    reg_name = st.text_input("Enter Name")
    reg_photo = st.file_uploader("Upload Face Image", type=['jpg', 'png', 'jpeg'])
    
    if st.button("Register"):
        if reg_name and reg_photo:
            save_path = os.path.join(KNOWN_FACES_DIR, f"{reg_name}.jpg")
            with open(save_path, "wb") as f:
                f.write(reg_photo.getbuffer())
            
            st.success(f"Registered {reg_name} successfully!")
            # Clear cache so the new face is loaded next time
            load_known_faces.clear()
        else:
            st.error("Please provide both a name and a photo.")

# Main Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mark Attendance")
    # Native Streamlit Camera Input (Works on Cloud)
    camera_image = st.camera_input("Take a photo to mark attendance")

    if camera_image is not None:
        # Load known faces (uses cache for speed)
        known_encodings, known_names = load_known_faces()
        
        if not known_encodings:
            st.warning("No registered faces found! Please register someone in the sidebar first.")
        else:
            # Convert the file to an opencv image
            bytes_data = camera_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the current frame
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

            if not face_encodings:
                st.error("No face detected in the webcam image.")
            else:
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    name = "Unknown"

                    # Calculate distance to find best match
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        mark_attendance(name)
                    else:
                        st.error("Face not recognized. Please register first.")

with col2:
    st.subheader("Attendance Log")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.write("No attendance data yet.")
