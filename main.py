# main.py
"""
Face Chat Application - Main Entry Point (PyQt6 Version)
Combines facial recognition with conversational AI for personalized interactions.
"""

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QLineEdit, QSplitter, QFrame, QMessageBox,
                             QInputDialog, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from face_recognition_module import FaceRecognitionManager
from chatbot import ChatbotManager


class VideoThread(QThread):
    """Thread for handling video capture and face recognition"""
    frame_ready = pyqtSignal(np.ndarray)
    user_detected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.face_manager = FaceRecognitionManager()
        self.running = False
        self.cap = None

    def run(self):
        """Main video capture loop"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for face recognition
                user_name = self.face_manager.identify_face(frame)

                # Emit signals
                self.frame_ready.emit(frame)
                if user_name:
                    self.user_detected.emit(user_name)
                else:
                    self.user_detected.emit("")

            self.msleep(50)  # 20 FPS

        if self.cap:
            self.cap.release()

    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()

    def register_user(self, name):
        """Register a new user"""
        return self.face_manager.register_new_user(name)


class ChatWidget(QFrame):
    """Custom chat widget with modern styling"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_styling()

    def setup_ui(self):
        """Setup the chat interface"""
        layout = QVBoxLayout()

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.chat_display)

        # Input section
        input_layout = QHBoxLayout()

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your message here...")
        self.input_line.setFont(QFont("Segoe UI", 10))

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)
        self.setLayout(layout)

    def setup_styling(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Segoe UI';
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Segoe UI';
            }
            QLineEdit:focus {
                border: 2px solid #0d6efd;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
        """)

    def add_message(self, sender, message, is_user=False):
        """Add a message to the chat display"""
        if is_user:
            html = f"""
            <div style="margin: 10px 0; text-align: right;">
                <span style="background-color: #0d6efd; color: white; padding: 8px 12px; 
                           border-radius: 18px; display: inline-block; max-width: 70%;">
                    <strong>{sender}:</strong> {message}
                </span>
            </div>
            """
        else:
            html = f"""
            <div style="margin: 10px 0;">
                <span style="background-color: #e9ecef; color: #212529; padding: 8px 12px; 
                           border-radius: 18px; display: inline-block; max-width: 70%;">
                    <strong>{sender}:</strong> {message}
                </span>
            </div>
            """

        self.chat_display.append(html)
        self.chat_display.ensureCursorVisible()

    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.clear()


class VideoWidget(QLabel):
    """Custom video display widget"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 300)
        self.setStyleSheet("""
            QLabel {
                background-color: #212529;
                border: 2px solid #495057;
                border-radius: 8px;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Initializing camera...")
        self.setFont(QFont("Segoe UI", 12))

    def update_frame(self, cv_img):
        """Update the video frame"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale image to fit widget
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class FaceChatMainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_user = None
        self.chatbot = ChatbotManager()
        self.user_greeted = set()

        self.setup_ui()
        self.setup_video_thread()
        self.apply_dark_theme()

    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Face Chat Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Video and controls
        left_panel = QFrame()
        left_panel.setFixedWidth(450)
        left_layout = QVBoxLayout()

        # Video section
        video_label = QLabel("Camera Feed")
        video_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(video_label)

        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget)

        # User info
        self.user_info_label = QLabel("No user detected")
        self.user_info_label.setFont(QFont("Segoe UI", 12))
        self.user_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_info_label.setStyleSheet("""
            QLabel {
                background-color: #495057;
                color: white;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
            }
        """)
        left_layout.addWidget(self.user_info_label)

        # Control buttons
        self.register_button = QPushButton("Register New User")
        self.register_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.register_button.clicked.connect(self.register_new_user)

        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.clear_chat_button.clicked.connect(self.clear_chat)

        self.refresh_button = QPushButton("Refresh Camera")
        self.refresh_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.refresh_button.clicked.connect(self.refresh_camera)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.register_button)
        buttons_layout.addWidget(self.clear_chat_button)
        buttons_layout.addWidget(self.refresh_button)

        left_layout.addLayout(buttons_layout)
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right panel - Chat interface
        right_panel = QFrame()
        right_layout = QVBoxLayout()

        chat_label = QLabel("Chat Interface")
        chat_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        chat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(chat_label)

        self.chat_widget = ChatWidget()
        right_layout.addWidget(self.chat_widget)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        # Connect chat input
        self.chat_widget.input_line.returnPressed.connect(self.send_message)
        self.chat_widget.send_button.clicked.connect(self.send_message)

        # Style panels
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #343a40;
                border-radius: 8px;
                margin: 5px;
            }
        """)

        right_panel.setStyleSheet("""
            QFrame {
                background-color: #343a40;
                border-radius: 8px;
                margin: 5px;
            }
        """)

        # Initial chat message
        self.chat_widget.add_message("System",
                                     "Face Chat App initialized. Please look at the camera for identification.")

    def setup_video_thread(self):
        """Initialize and start video thread"""
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.user_detected.connect(self.update_current_user)
        self.video_thread.start()

    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #212529;
                color: #f8f9fa;
            }
            QLabel {
                color: #f8f9fa;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
        """)

    def update_video_display(self, frame):
        """Update video display with new frame"""
        self.video_widget.update_frame(frame)

    def update_current_user(self, user_name):
        """Update current user information"""
        if user_name != self.current_user:
            self.current_user = user_name if user_name else None

            if self.current_user:
                self.user_info_label.setText(f"Hello, {self.current_user}!")
                self.user_info_label.setStyleSheet("""
                    QLabel {
                        background-color: #198754;
                        color: white;
                        padding: 10px;
                        border-radius: 6px;
                        margin: 10px 0;
                    }
                """)

                # Greet user if not already greeted
                if self.current_user not in self.user_greeted:
                    self.chat_widget.add_message("System",
                                                 f"Welcome back, {self.current_user}! How can I help you today?")
                    self.user_greeted.add(self.current_user)
            else:
                self.user_info_label.setText("No user detected")
                self.user_info_label.setStyleSheet("""
                    QLabel {
                        background-color: #495057;
                        color: white;
                        padding: 10px;
                        border-radius: 6px;
                        margin: 10px 0;
                    }
                """)

    def register_new_user(self):
        """Register a new user"""
        name, ok = QInputDialog.getText(self, 'Register User', 'Enter your name:')

        if ok and name.strip():
            success = self.video_thread.register_user(name.strip())
            if success:
                self.chat_widget.add_message("System", f"Successfully registered {name}!")
                QMessageBox.information(self, "Success",
                                        f"User {name} registered successfully!")
            else:
                QMessageBox.warning(self, "Error",
                                    "Failed to register user. Make sure your face is visible in the camera.")

    def send_message(self):
        """Send message to chatbot"""
        message = self.chat_widget.input_line.text().strip()
        if not message:
            return

        self.chat_widget.input_line.clear()

        # Add user message to chat
        user_name = self.current_user or "User"
        self.chat_widget.add_message(user_name, message, is_user=True)

        # Get chatbot response
        try:
            response = self.chatbot.get_response(message, self.current_user)
            self.chat_widget.add_message("Assistant", response)
        except Exception as e:
            self.chat_widget.add_message("System", f"Error: {str(e)}")

    def clear_chat(self):
        """Clear chat history"""
        self.chat_widget.clear_chat()
        self.chat_widget.add_message("System", "Chat cleared.")
        self.user_greeted.clear()

    def refresh_camera(self):
        """Refresh camera connection"""
        self.video_thread.stop()
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.user_detected.connect(self.update_current_user)
        self.video_thread.start()
        self.chat_widget.add_message("System", "Camera refreshed.")

    def closeEvent(self, event):
        """Handle application closing"""
        self.video_thread.stop()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Face Chat Application")
    app.setApplicationVersion("2.0")

    # Set application icon and style
    app.setStyle('Fusion')  # Modern look

    window = FaceChatMainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
