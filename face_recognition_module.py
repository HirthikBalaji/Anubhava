# face_recognition_module.py
"""
Face Recognition Module (PyQt6 Compatible)
Handles camera capture, face detection, and user identification.
"""

import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
from PyQt6.QtCore import QMutex


class FaceRecognitionManager:
    def __init__(self, database_path="user_database.pkl"):
        self.database_path = database_path
        self.known_encodings = []
        self.known_names = []
        self.mutex = QMutex()  # Thread safety
        self.load_database()

        # Face detection parameters
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.recognition_threshold = 0.6  # Lower is more strict

    def load_database(self):
        """Load known faces from database"""
        self.mutex.lock()
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                print(f"Loaded {len(self.known_names)} known faces from database")
            else:
                print("No existing database found. Starting fresh.")
                self.known_encodings = []
                self.known_names = []
        except Exception as e:
            print(f"Error loading database: {e}")
            self.known_encodings = []
            self.known_names = []
        finally:
            self.mutex.unlock()

    def save_database(self):
        """Save known faces to database"""
        self.mutex.lock()
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'timestamp': datetime.now().isoformat()
            }

            # Create backup
            if os.path.exists(self.database_path):
                backup_path = f"{self.database_path}.backup"
                os.rename(self.database_path, backup_path)

            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
            print("Database saved successfully")

        except Exception as e:
            print(f"Error saving database: {e}")
        finally:
            self.mutex.unlock()

    def identify_face(self, frame):
        """Identify face in the given frame with improved accuracy"""
        if frame is None or frame.size == 0:
            return None

        # Resize frame for faster processing while maintaining quality
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces in current frame
        self.face_locations = face_recognition.face_locations(
            rgb_small_frame, model="hog"  # Use HOG model for better performance
        )

        if not self.face_locations:
            return None

        self.face_encodings = face_recognition.face_encodings(
            rgb_small_frame, self.face_locations
        )

        self.face_names = []

        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=self.recognition_threshold
            )
            name = None

            if True in matches and len(self.known_encodings) > 0:
                # Find the best match using face distance
                face_distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index] and face_distances[best_match_index] < self.recognition_threshold:
                    name = self.known_names[best_match_index]

            self.face_names.append(name)

        # Draw rectangles and labels on the original frame
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations (we used 0.5 scale)
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Choose color based on recognition
            if name:
                color = (0, 255, 0)  # Green for recognized
                display_name = name
            else:
                color = (0, 0, 255)  # Red for unknown
                display_name = "Unknown"

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, display_name, (left + 6, bottom - 6),
                        font, 0.6, (255, 255, 255), 1)

        # Return the first recognized name or None
        if self.face_names and self.face_names[0]:
            return self.face_names[0]
        return None

    def register_new_user(self, name, max_attempts=10):
        """Register a new user with improved face capture"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        best_encoding = None
        best_confidence = float('inf')

        try:
            # Capture multiple frames to get the best encoding
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if len(face_locations) == 1:  # Ensure exactly one face
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        encoding = face_encodings[0]

                        # Calculate confidence (distance to existing faces)
                        if len(self.known_encodings) > 0:
                            distances = face_recognition.face_distance(
                                self.known_encodings, encoding
                            )
                            min_distance = np.min(distances)
                        else:
                            min_distance = 0

                        # Keep the encoding with best quality
                        if min_distance < best_confidence:
                            best_confidence = min_distance
                            best_encoding = encoding

            if best_encoding is not None:
                # Add to database
                self.known_encodings.append(best_encoding)
                self.known_names.append(name)
                self.save_database()
                return True

        except Exception as e:
            print(f"Error during registration: {e}")
        finally:
            cap.release()

        return False

    def remove_user(self, name):
        """Remove a user from the database"""
        self.mutex.lock()
        try:
            if name in self.known_names:
                index = self.known_names.index(name)
                self.known_names.pop(index)
                self.known_encodings.pop(index)
                self.save_database()
                return True
        except Exception as e:
            print(f"Error removing user: {e}")
        finally:
            self.mutex.unlock()
        return False

    def get_user_count(self):
        """Get number of registered users"""
        return len(self.known_names)

    def get_all_users(self):
        """Get list of all registered users"""
        return self.known_names.copy()