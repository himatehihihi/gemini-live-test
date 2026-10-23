# --- Core Imports ---
import asyncio
import base64
import io
import os
import sys
import traceback
import json
import websockets
import argparse
import threading
from html import escape
import subprocess
import webbrowser
import math
import urllib.parse
import urllib.request
import re
import time # Added for audio stream management
import speech_recognition as sr

# --- PySide6 GUI Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLabel,
                               QVBoxLayout, QWidget, QLineEdit, QHBoxLayout,
                               QSizePolicy, QPushButton, QScrollArea)
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer, QUrl, QDateTime
from PySide6.QtGui import (QImage, QPixmap, QFont, QFontDatabase, QTextCursor, 
                           QPainter, QPen, QVector3D, QMatrix4x4, QColor, QBrush)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEnginePage
from tool_bridge import (
    GEMINI_TOOL_DECLARATIONS,
    dispatch_tool_call,
    KNOWN_TOOL_NAMES,
)

# --- Media and AI Imports ---
import cv2
import pyaudio
import PIL.Image
from google import genai
from dotenv import load_dotenv
from PIL import ImageGrab
import numpy as np
import numpy as np
import openwakeword
from openwakeword.model import Model
from kokoro import KPipeline # Added Kokoro TTS


# --- Load Environment Variables ---
load_dotenv()
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") # Removed
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create kokoro cache directory
os.makedirs('./kokoro_cache', exist_ok=True)
os.environ['KOKORO_MODEL_CACHE'] = './kokoro_cache'

if not GEMINI_API_KEY:
    sys.exit("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
# if not ELEVENLABS_API_KEY:
#     sys.exit("Error: ELEVENLABS_API_KEY not found. Please check your .env file.")

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "gemini-live-2.5-flash-preview"
# VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku' # Removed
SELECTED_VOICE = 'bm_daniel'  # Using Kokoro's bm_daniel voice
DEFAULT_MODE = "none"  # Options: "camera", "screen", "none"
MAX_OUTPUT_TOKENS = 100
DEBUG_AUDIO = True

# --- Initialize Clients ---
pya = pyaudio.PyAudio()

# ==============================================================================
# AI Animation Widget
# ==============================================================================
# ==============================================================================
# AI Animation Widget (Web-Based Avatar)
# ==============================================================================
class AvatarWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Transparent background
        self.setStyleSheet("background: transparent;")
        self.page().setBackgroundColor(Qt.transparent)
        
        # Load the HTML file
        html_path = os.path.abspath("avatar_renderer.html")
        self.load(QUrl.fromLocalFile(html_path))
        
        # Settings
        self.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        self.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        
    def start_speaking_animation(self):
        self.page().runJavaScript("setTalking(true);")
        
    def stop_speaking_animation(self):
        self.page().runJavaScript("setTalking(false);")

# ==============================================================================
# AI BACKEND LOGIC
# ==============================================================================
class AI_Core(QObject):
    """
    Handles all backend operations. Inherits from QObject to emit signals
    for thread-safe communication with the GUI.
    """
    text_received = Signal(str)
    end_of_turn = Signal()
    frame_received = Signal(QImage)
    search_results_received = Signal(list)
    code_being_executed = Signal(str, str)
    file_list_received = Signal(str, list)
    video_mode_changed = Signal(str)
    speaking_started = Signal()
    speaking_stopped = Signal()
    audio_generation_started = Signal()
    audio_generation_finished = Signal()
    audio_playback_started = Signal()
    audio_playback_finished = Signal(str)
    connection_status_changed = Signal(str)
    video_requested = Signal(str)
    media_control_requested = Signal(str)
    wake_word_activated = Signal()
    wake_word_deactivated = Signal()
    image_generated = Signal(str)  # Emits path to generated image

    def __init__(self, video_mode=DEFAULT_MODE):
        super().__init__()
        self.video_mode = video_mode
        self.is_running = True
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        create_folder = {
            "name": "create_folder",
            "description": "Creates a new folder at the specified path relative to the script's root directory.",
            "parameters": {
                "type": "OBJECT",
                "properties": { "folder_path": { "type": "STRING", "description": "The path for the new folder (e.g., 'new_project/assets')."}},
                "required": ["folder_path"]
            }
        }

        create_file = {
            "name": "create_file",
            "description": "Creates a new file with specified content at a given path.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": { "type": "STRING", "description": "The path for the new file (e.g., 'new_project/notes.txt')."},
                    "content": { "type": "STRING", "description": "The content to write into the new file."}
                },
                "required": ["file_path", "content"]
            }
        }

        edit_file = {
            "name": "edit_file",
            "description": "Appends content to an existing file at a specified path.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": { "type": "STRING", "description": "The path of the file to edit (e.g., 'project/notes.txt')."},
                    "content": { "type": "STRING", "description": "The content to append to the file."}
                },
                "required": ["file_path", "content"]
            }
        }

        list_files = {
            "name": "list_files",
            "description": "Lists all files and directories within a specified folder. Defaults to the current directory if no path is provided.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "directory_path": { "type": "STRING", "description": "The path of the directory to inspect. Defaults to '.' (current directory) if omitted."}
                }
            }
        }

        read_file = {
            "name": "read_file",
            "description": "Reads the entire content of a specified file.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": { "type": "STRING", "description": "The path of the file to read (e.g., 'project/notes.txt')."}
                },
                "required": ["file_path"]
            }
        }

        open_application = {
            "name": "open_application",
            "description": "Opens or launches a desktop application on the user's computer.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "application_name": { "type": "STRING", "description": "The name of the application to open (e.g., 'Notepad', 'Calculator', 'Chrome')."}
                },
                "required": ["application_name"]
            }
        }

        open_website = {
            "name": "open_website",
            "description": "Opens a given URL. Can open in the internal app window or the external default browser.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "url": { "type": "STRING", "description": "The full URL of the website to open (e.g., 'https://www.google.com')."},
                    "target": { "type": "STRING", "description": "Where to open the URL. Use 'internal' for YouTube, login, or keeping it inside the app. Use 'external' for general browsing or if the user asks to open in their browser.", "enum": ["internal", "external"]}
                },
                "required": ["url"]
            }
        }

        play_youtube = {
            "name": "play_youtube",
            "description": "Plays a song or video on YouTube using the internal player.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": { "type": "STRING", "description": "The song or video title to search for."}
                },
                "required": ["query"]
            }
        }

        control_media = {
            "name": "control_media",
            "description": "Controls the media playback (YouTube) in the internal player.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": { "type": "STRING", "description": "The action to perform.", "enum": ["play", "pause", "stop", "mute", "unmute", "next", "previous"]}
                },
                "required": ["action"]
            }
        }
        
        generate_image = {
            "name": "generate_image",
            "description": "Generates an image based on a text prompt using AI. The generated image will be displayed to the user.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "prompt": { "type": "STRING", "description": "A detailed description of the image to generate (e.g., 'a futuristic city at sunset with flying cars')."}
                },
                "required": ["prompt"]
            }
        }

        print(f">>> [DEBUG] Checking open_website: {type(open_website)}")

        try:
            tools = [
                {'google_search': {}},
                {'code_execution': {}},
                {
                    "function_declarations": [
                        create_folder,
                        create_file,
                        edit_file,
                        list_files,
                        read_file,
                        control_media,
                        generate_image,
                        *GEMINI_TOOL_DECLARATIONS,
                    ]
                }
            ]

            print(">>> [DEBUG] Tools list created successfully.")

        except NameError as e:
            print(f">>> [ERROR] NameError in tools definition: {e}")
            raise

        except Exception as e:
            print(f">>> [ERROR] Error in tools definition: {e}")
            raise
        
        self.config = {
            "response_modalities": ["TEXT"],
            "system_instruction": """
            Your name is Ada and you are my Ai assistant.
            Always address me as sir.
            You have access to tools for searching, code execution, system actions, and IMAGE GENERATION.
            
            IMPORTANT: You CAN generate images using the generate_image function. When the user asks to generate, create, or make an image, you MUST use the generate_image function. Do not say you cannot generate images - you have this capability through the generate_image tool.
            
            The user is providing a live video stream that can be switched between their webcam and their screen, or turned off.
            Ignore both the webcam and screen content unless the user explicitly asks you to analyze or comment on what you see.
            Follow these guidelines when choosing tools:
            1.  For information or questions, use `Google Search`.
            2.  For math or running python code, use `code_execution`.
            3.  Use file system functions (`create_folder`, `create_file`, `edit_file`, `list_files`, `read_file`) for any file-related tasks.
            4.  If the user asks to open or launch a desktop application, you must use the `open_application` function.
            5.  If the user asks to open a website:
                - Use `open_website(url, target='internal')` if it's for YouTube, logging in, or they want to see it "here" or "in the app".
                - Use `open_website(url, target='external')` if they ask to "open in browser", "open in chrome", or for general research.
            6.  If the user asks to play music, a song, or a video, use the `play_youtube` function.
            7.  If the user asks to control playback (pause, stop, mute, next, etc.), use the `control_media` function.
            8.  If the user asks to generate, create, or make an image/picture/photo, you MUST use the `generate_image` function with a detailed prompt. This is a core capability you have.
            Prioritize the most appropriate tool for the user's specific request.""",
            "tools": tools,
            "max_output_tokens": MAX_OUTPUT_TOKENS
        }
        self.session = None
        self.audio_stream = None
        self.out_queue_gemini = asyncio.Queue(maxsize=20)
        self.response_queue_tts = asyncio.Queue()
        self.audio_in_queue_player = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()
        self.latest_frame = None
        self.tasks = []
        self.loop = asyncio.new_event_loop()
        self.reconnect_delay = 5
        
        # Wake word detection
        self.wake_word_active = False
        self.wake_word_timeout = None
        # Initialize OpenWakeWord
        try:
            # Download model if needed (automatically handled by openwakeword)
            openwakeword.utils.download_models(["hey_jarvis"])
            self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            print(">>> [INIT] OpenWakeWord model 'hey_jarvis' loaded successfully.")
        except Exception as e:
            print(f">>> [ERROR] Failed to load OpenWakeWord: {e}")
            self.oww_model = None

        # Initialize Kokoro TTS
        try:
            self.kokoro_pipeline = KPipeline(lang_code='a')
            print(f"Kokoro TTS initialized successfully with American English. Using voice: {SELECTED_VOICE}")
            
            if DEBUG_AUDIO:
                print(">>> [DEBUG] Testing Kokoro pipeline...")
                test_audio = self.kokoro_pipeline("Test", voice=SELECTED_VOICE)
                print(">>> [DEBUG] Pipeline test successful")
        except Exception as e:
            print(f"Error initializing Kokoro TTS: {e}")
            traceback.print_exc()
            sys.exit("Failed to initialize Kokoro TTS")
    
    def listen_for_wake_word(self):
        """Background thread that continuously listens for wake word using OpenWakeWord"""
        print(">>> [WAKE WORD] Listener started (OpenWakeWord). Say 'Hey Jarvis'...")
        
        if not self.oww_model:
            print(">>> [ERROR] OpenWakeWord model not loaded. Wake word disabled.")
            return

        # Audio config for OpenWakeWord
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280 # 80ms chunks (OpenWakeWord recommends ~80ms)
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        while self.is_running:
            try:
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Feed to model
                prediction = self.oww_model.predict(audio_data)
                
                # Check for wake word (threshold usually 0.5)
                if prediction["hey_jarvis"] > 0.5:
                    print(f">>> [WAKE WORD] Detected! Score: {prediction['hey_jarvis']:.2f}")
                    
                    if not self.wake_word_active:
                        self.wake_word_active = True
                        self.wake_word_activated.emit()
                        
                        # Set timeout
                        if self.wake_word_timeout:
                            self.wake_word_timeout.cancel()
                        self.wake_word_timeout = threading.Timer(15.0, self.deactivate_wake_word)
                        self.wake_word_timeout.start()
                        
                        # Pause listener while active (optional, but good to avoid double triggers)
                        while self.wake_word_active and self.is_running:
                            time.sleep(0.1)
                        
                        print(">>> [WAKE WORD] Resuming listener...")
                        # Clear buffer to avoid processing old audio
                        if stream.is_active():
                            stream.stop_stream()
                            stream.start_stream()

            except Exception as e:
                print(f">>> [WAKE WORD] Error: {e}")
                time.sleep(1)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def deactivate_wake_word(self):
        """Deactivate wake word after timeout"""
        print(">>> [WAKE WORD] Timeout - deactivating...")
        self.wake_word_active = False
        self.wake_word_deactivated.emit()

    def _create_folder(self, folder_path):
        try:
            if not folder_path or not isinstance(folder_path, str): return {"status": "error", "message": "Invalid folder path provided."}
            if os.path.exists(folder_path): return {"status": "skipped", "message": f"The folder '{folder_path}' already exists."}
            os.makedirs(folder_path)
            return {"status": "success", "message": f"Successfully created the folder at '{folder_path}'."}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _create_file(self, file_path, content):
        try:
            if not file_path or not isinstance(file_path, str): return {"status": "error", "message": "Invalid file path provided."}
            if os.path.exists(file_path): return {"status": "skipped", "message": f"The file '{file_path}' already exists."}
            with open(file_path, 'w') as f: f.write(content)
            return {"status": "success", "message": f"Successfully created the file at '{file_path}'."}
        except Exception as e: return {"status": "error", "message": f"An error occurred while creating the file: {str(e)}"}

    def _edit_file(self, file_path, content):
        try:
            if not file_path or not isinstance(file_path, str): return {"status": "error", "message": "Invalid file path provided."}
            if not os.path.exists(file_path): return {"status": "error", "message": f"The file '{file_path}' does not exist. Please create it first."}
            with open(file_path, 'a') as f: f.write(f"\n{content}")
            return {"status": "success", "message": f"Successfully appended content to the file at '{file_path}'."}
        except Exception as e: return {"status": "error", "message": f"An error occurred while editing the file: {str(e)}"}

    def _list_files(self, directory_path):
        try:
            path_to_list = directory_path if directory_path else '.'
            if not isinstance(path_to_list, str): return {"status": "error", "message": "Invalid directory path provided."}
            if not os.path.isdir(path_to_list): return {"status": "error", "message": f"The path '{path_to_list}' is not a valid directory."}
            files = os.listdir(path_to_list)
            return {"status": "success", "message": f"Found {len(files)} items in '{path_to_list}'.", "files": files, "directory_path": path_to_list}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _read_file(self, file_path):
        try:
            if not file_path or not isinstance(file_path, str): return {"status": "error", "message": "Invalid file path provided."}
            if not os.path.exists(file_path): return {"status": "error", "message": f"The file '{file_path}' does not exist."}
            if not os.path.isfile(file_path): return {"status": "error", "message": f"The path '{file_path}' is not a file."}
            with open(file_path, 'r') as f: content = f.read()
            return {"status": "success", "message": f"Successfully read the file '{file_path}'.", "content": content}
        except Exception as e: return {"status": "error", "message": f"An error occurred while reading the file: {str(e)}"}

    def _open_application(self, application_name):
        print(f">>> [DEBUG] Attempting to open application: '{application_name}'")
        try:
            if not application_name or not isinstance(application_name, str):
                return {"status": "error", "message": "Invalid application name provided."}
            command, shell_mode = [], False
            if sys.platform == "win32":
                app_map = {"calculator": "calc:", "notepad": "notepad", "chrome": "chrome", "google chrome": "chrome", "firefox": "firefox", "explorer": "explorer", "file explorer": "explorer"}
                app_command = app_map.get(application_name.lower(), application_name)
                command, shell_mode = f"start {app_command}", True
            elif sys.platform == "darwin":
                app_map = {"calculator": "Calculator", "chrome": "Google Chrome", "firefox": "Firefox", "finder": "Finder", "textedit": "TextEdit"}
                app_name = app_map.get(application_name.lower(), application_name)
                command = ["open", "-a", app_name]
            else:
                command = [application_name.lower()]
            subprocess.Popen(command, shell=shell_mode)
            return {"status": "success", "message": f"Successfully launched '{application_name}'."}
        except FileNotFoundError: return {"status": "error", "message": f"Application '{application_name}' not found."}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _open_website(self, url, target="external"):
        print(f">>> [DEBUG] Attempting to open URL: '{url}' (Target: {target})")
        try:
            if not url or not isinstance(url, str): return {"status": "error", "message": "Invalid URL provided."}
            if not url.startswith(('http://', 'https://')): url = 'https://' + url
            
            if target == "internal":
                # Use internal browser
                self.video_requested.emit(url)
                return {"status": "success", "message": f"Opened '{url}' in internal window."}
            else:
                # Use external default browser
                webbrowser.open(url)
                return {"status": "success", "message": f"Opened '{url}' in default browser."}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _control_media(self, action):
        print(f">>> [DEBUG] Media control requested: {action}")
        try:
            self.media_control_requested.emit(action)
            return {"status": "success", "message": f"Executed media action: {action}"}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _generate_image(self, prompt):
        """Generate an image using Hugging Face Inference API (free tier)"""
        print(f">>> [IMAGE GEN] Generating image for prompt: '{prompt}'")
        try:
            import requests
            from datetime import datetime
            
            # Hugging Face API endpoint - using Stable Diffusion XL (currently available)
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            
            # Check if HF API key is set (optional for free tier, but recommended)
            HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
            headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
            
            # Make API request
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
            
            if response.status_code == 200:
                # Save image
                os.makedirs("generated_images", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.abspath(f"generated_images/image_{timestamp}.png")
                
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                print(f">>> [IMAGE GEN] Image saved to: {image_path}")
                self.image_generated.emit(image_path)
                return {"status": "success", "message": f"Image generated successfully: {image_path}"}
            else:
                try:
                    error_data = response.json()
                    error_msg = f"API Error ({response.status_code}): {error_data.get('error', 'Unknown error')}"
                except:
                    error_msg = f"API returned status {response.status_code}"
                
                if response.status_code == 503:
                    error_msg += " (Model is loading, please try again in 20-30 seconds)"
                elif response.status_code == 410:
                    error_msg += " (Model endpoint deprecated)"
                    
                print(f">>> [IMAGE GEN] Error Details: {response.text}")
                return {"status": "error", "message": error_msg}
                
        except Exception as e:
            print(f">>> [IMAGE GEN] Exception: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"Failed to generate image: {str(e)}"}

    def _play_youtube(self, query):
        print(f">>> [DEBUG] Attempting to play on YouTube: '{query}'")
        try:
            if not query or not isinstance(query, str): return {"status": "error", "message": "Invalid query provided."}
            
            # If it's already a URL, just open it
            if "youtube.com" in query or "youtu.be" in query:
                self.video_requested.emit(query)
                return {"status": "success", "message": f"Playing YouTube URL: '{query}'."}

            # Search for the video
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            
            try:
                # Attempt to scrape the first video ID
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                req = urllib.request.Request(search_url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    html = response.read().decode()
                    # Regex to find video IDs
                    video_ids = re.findall(r"watch\?v=(\S{11})", html)
                    if video_ids:
                        first_video_id = video_ids[0]
                        # Use Watch URL for better compatibility and account support
                        video_url = f"https://www.youtube.com/watch?v={first_video_id}&autoplay=1"
                        self.video_requested.emit(video_url)
                        return {"status": "success", "message": f"Playing first result for '{query}'."}
            except Exception as scrape_error:
                print(f">>> [WARNING] Failed to scrape YouTube video ID: {scrape_error}")
                # Fallback to search page
            
            self.video_requested.emit(search_url)
            return {"status": "success", "message": f"Showing YouTube search results for '{query}'."}
        except Exception as e: return {"status": "error", "message": f"An error occurred: {str(e)}"}

    @Slot(str)
    def set_video_mode(self, mode):
        """Sets the video source and notifies the GUI."""
        if mode in ["camera", "screen", "none"]:
            self.video_mode = mode
            print(f">>> [INFO] Switched video mode to: {self.video_mode}")
            if mode == "none":
                self.latest_frame = None
            self.video_mode_changed.emit(mode)

    async def stream_video_to_gui(self):
        video_capture = None
        while self.is_running:
            frame = None
            try:
                if self.video_mode == "camera":
                    if video_capture is None: video_capture = await asyncio.to_thread(cv2.VideoCapture, 0)
                    if video_capture.isOpened():
                        ret, frame = await asyncio.to_thread(video_capture.read)
                        if not ret:
                            await asyncio.sleep(0.01)
                            continue
                elif self.video_mode == "screen":
                    if video_capture is not None:
                        await asyncio.to_thread(video_capture.release)
                        video_capture = None
                    screenshot = await asyncio.to_thread(ImageGrab.grab)
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                else:
                    if video_capture is not None:
                        await asyncio.to_thread(video_capture.release)
                        video_capture = None
                    await asyncio.sleep(0.1)
                    continue
                if frame is not None:
                    self.latest_frame = frame
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    self.frame_received.emit(qt_image.copy())
                else: self.frame_received.emit(QImage())
                await asyncio.sleep(0.033)
            except Exception as e:
                print(f">>> [ERROR] Video streaming error: {e}")
                if video_capture is not None:
                    await asyncio.to_thread(video_capture.release)
                    video_capture = None
                await asyncio.sleep(1)
        if video_capture is not None: await asyncio.to_thread(video_capture.release)

    async def send_frames_to_gemini(self):
        while self.is_running:
            await asyncio.sleep(1.0)
            if self.video_mode != "none" and self.latest_frame is not None:
                frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(frame_rgb)
                pil_img.thumbnail([1024, 1024])
                image_io = io.BytesIO()
                pil_img.save(image_io, format="jpeg")
                gemini_data = {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.getvalue()).decode()}
                await self.out_queue_gemini.put(gemini_data)

    async def receive_text(self):
        while self.is_running:
            try:
                turn_urls, turn_code_content, turn_code_result, file_list_data = set(), "", "", None
                turn = self.session.receive()
                async for chunk in turn:
                    if chunk.tool_call and chunk.tool_call.function_calls:
                        function_responses = []
                        for fc in chunk.tool_call.function_calls:
                            args, result = fc.args, {}
                            if fc.name == "create_folder": result = self._create_folder(folder_path=args.get("folder_path"))
                            elif fc.name == "create_file": result = self._create_file(file_path=args.get("file_path"), content=args.get("content"))
                            elif fc.name == "edit_file": result = self._edit_file(file_path=args.get("file_path"), content=args.get("content"))
                            elif fc.name == "list_files":
                                result = self._list_files(directory_path=args.get("directory_path"))
                                if result.get("status") == "success": file_list_data = (result.get("directory_path"), result.get("files"))
                            elif fc.name == "read_file": result = self._read_file(file_path=args.get("file_path"))
                            elif fc.name == "open_application": result = self._open_application(application_name=args.get("application_name"))
                            elif fc.name == "open_website": result = self._open_website(url=args.get("url"), target=args.get("target", "external"))
                            elif fc.name == "play_youtube": result = self._play_youtube(query=args.get("query"))
                            elif fc.name == "control_media": result = self._control_media(action=args.get("action"))
                            elif fc.name == "generate_image": result = self._generate_image(prompt=args.get("prompt"))
                            function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                        await self.session.send_tool_response(function_responses=function_responses)
                        continue
                    if chunk.server_content:
                        if hasattr(chunk.server_content, 'grounding_metadata') and chunk.server_content.grounding_metadata:
                            for g_chunk in chunk.server_content.grounding_metadata.grounding_chunks:
                                if g_chunk.web and g_chunk.web.uri: turn_urls.add(g_chunk.web.uri)
                        if chunk.server_content.model_turn:
                            for part in chunk.server_content.model_turn.parts:
                                if part.executable_code: turn_code_content = part.executable_code.code
                                if part.code_execution_result: turn_code_result = part.code_execution_result.output
                    if chunk.text:
                        self.text_received.emit(chunk.text)
                        await self.response_queue_tts.put(chunk.text)
                if file_list_data: self.file_list_received.emit(file_list_data[0], file_list_data[1])
                elif turn_code_content: self.code_being_executed.emit(turn_code_content, turn_code_result)
                elif turn_urls: self.search_results_received.emit(list(turn_urls))
                else:
                    self.code_being_executed.emit("",""); self.search_results_received.emit([]); self.file_list_received.emit("",[])
                self.end_of_turn.emit()
                await self.response_queue_tts.put(None)
            except websockets.exceptions.ConnectionClosedError:
                print(">>> [WARNING] WebSocket connection closed in receive_text.")
                break # Exit loop to trigger reconnection in run()
            except Exception:
                if not self.is_running: break
                traceback.print_exc()
                await asyncio.sleep(1)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = pya.open(format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE, input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE)
        while self.is_running:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            if not self.is_running: break
            
            # Only send audio to Gemini when wake word is active
            if self.wake_word_active:
                await self.out_queue_gemini.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while self.is_running:
            msg = await self.out_queue_gemini.get()
            if not self.is_running: break
            await self.session.send(input=msg)
            self.out_queue_gemini.task_done()

    async def process_text_input_queue(self):
        while self.is_running:
            text = await self.text_input_queue.get()
            if text is None:
                self.text_input_queue.task_done(); break
            if self.session:
                for q in [self.response_queue_tts, self.audio_in_queue_player]:
                    while not q.empty(): q.get_nowait()
                await self.session.send_client_content(turns=[{"role": "user", "parts": [{"text": text or "."}]}])
            self.text_input_queue.task_done()

    async def tts(self):
        accumulated_text = ""
        accumulated_audio = []
        
        while self.is_running:
            text_chunk = await self.response_queue_tts.get()
            if text_chunk is None or not self.is_running:
                # End of response, process accumulated content
                if accumulated_text and accumulated_audio:
                    self.audio_generation_finished.emit()
                    # self.audio_playback_started.emit() # Handled in play_audio
                    # self.text_received.emit(accumulated_text) # Removed to prevent double printing
                    
                    audio = np.concatenate(accumulated_audio)
                    scaled_audio = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
                    audio_bytes = scaled_audio.tobytes()
                    
                    # Send audio chunks to player
                    chunk_size = CHUNK_SIZE * 2
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i:i+chunk_size]
                        if len(chunk) > 0:
                            await self.audio_in_queue_player.put(chunk)
                    
                    # Send end marker
                    await self.audio_in_queue_player.put(b'END_OF_AUDIO')
                
                # Reset accumulators
                accumulated_text = ""
                accumulated_audio = []
                self.response_queue_tts.task_done()
                continue
            
            if not text_chunk.strip():
                self.response_queue_tts.task_done()
                continue
                
            try:
                if DEBUG_AUDIO:
                    print(f">>> [DEBUG] TTS processing text: {text_chunk[:50]}...")
                
                # Signal start of audio generation
                if not accumulated_text:
                    self.audio_generation_started.emit()
                
                try:
                    if DEBUG_AUDIO:
                        print(">>> [DEBUG] Generating audio for text chunk...")
                        
                    # OPTIMIZATION 2: Increased speed from 1.3 to 1.4 for snappier response
                    for result in self.kokoro_pipeline(text_chunk, voice=SELECTED_VOICE, speed=1.4):
                        if hasattr(result, 'output') and hasattr(result.output, 'audio'):
                            chunk_audio = result.output.audio
                            if hasattr(chunk_audio, 'numpy'):
                                chunk_audio = chunk_audio.numpy()
                            
                            if DEBUG_AUDIO:
                                print(f">>> [DEBUG] Accumulated chunk shape: {chunk_audio.shape}")
                            
                            accumulated_audio.append(chunk_audio)
                    
                    # Accumulate text
                    accumulated_text += text_chunk
                    
                    if DEBUG_AUDIO:
                        print(">>> [DEBUG] Audio processing complete")
                
                except Exception as e:
                    print(f">>> [ERROR] TTS Error: {str(e)}")
                    traceback.print_exc()
                finally:
                    self.response_queue_tts.task_done()
                
            except Exception as e:
                print(f">>> [ERROR] TTS Error: {e}")
            finally:
                pass

    async def play_audio(self):
        if DEBUG_AUDIO:
            print(">>> [DEBUG] Audio player task started")
            
        stream = None
        last_activity = time.time()
        is_playing = False
        
        try:
            while self.is_running:
                current_time = time.time()
                if stream is None or current_time - last_activity > 60:
                    if stream:
                        await asyncio.to_thread(stream.stop_stream)
                        await asyncio.to_thread(stream.close)
                    
                    output_info = pya.get_default_output_device_info()
                    stream = await asyncio.to_thread(
                        pya.open,
                        format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RECEIVE_SAMPLE_RATE,
                        output=True,
                        output_device_index=output_info["index"],
                        frames_per_buffer=CHUNK_SIZE,
                    )
                    last_activity = current_time
                    print(">>> [INFO] Audio output stream opened")
                
                bytestream = await self.audio_in_queue_player.get()
                
                if bytestream == b'END_OF_AUDIO':
                    if is_playing:
                        self.audio_playback_finished.emit("Audio playback completed")
                        self.speaking_stopped.emit() # Update UI
                        is_playing = False
                    self.audio_in_queue_player.task_done()
                    continue
                
                if bytestream and self.is_running:
                    if not is_playing:
                        is_playing = True
                        self.audio_playback_started.emit()
                        self.speaking_started.emit() # Update UI
                    
                    await asyncio.to_thread(stream.write, bytestream)
                    last_activity = current_time
                
                self.audio_in_queue_player.task_done()
                
        except Exception as e:
            print(f">>> [ERROR] Audio playback error: {e}")
            traceback.print_exc()
        finally:
            if stream:
                await asyncio.to_thread(stream.stop_stream)
                await asyncio.to_thread(stream.close)
                print(">>> [INFO] Audio output stream closed")

    async def main_task_runner(self, session):
        self.session = session
        self.tasks = [
            asyncio.create_task(self.stream_video_to_gui()),
            asyncio.create_task(self.send_frames_to_gemini()),
            asyncio.create_task(self.listen_audio()),
            asyncio.create_task(self.send_realtime()),
            asyncio.create_task(self.receive_text()),
            asyncio.create_task(self.tts()),
            asyncio.create_task(self.play_audio()),
            asyncio.create_task(self.process_text_input_queue())
        ]
        
        # Wait for any task to complete (or fail)
        # If receive_text fails (connection lost), we want to exit and reconnect
        done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel all other tasks to clean up before reconnection
        for task in pending:
            task.cancel()
            
        # Wait for cancellations to finish
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def run(self):
        while self.is_running:
            try:
                print(">>> [INFO] Connecting to Gemini Live API...")
                async with self.client.aio.live.connect(model=MODEL, config=self.config) as session:
                    print(">>> [INFO] Connected to Gemini Live API.")
                    self.connection_status_changed.emit("CONNECTED")
                    await self.main_task_runner(session)
            except asyncio.CancelledError:
                print(f"\n>>> [INFO] AI Core run loop gracefully cancelled.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"\n>>> [WARNING] Connection closed (1011). Reconnecting in {self.reconnect_delay}s...")
                self.connection_status_changed.emit("RECONNECTING")
                await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                print(f"\n>>> [ERROR] AI Core run loop encountered an error: {type(e).__name__}: {e}")
                print(f">>> [INFO] Reconnecting in {self.reconnect_delay}s...")
                self.connection_status_changed.emit("ERROR")
                await asyncio.sleep(self.reconnect_delay)
            finally:
                pass # Loop handles restart unless cancelled
        
        if not self.is_running:
            self.stop()

    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run())

    @Slot(str)
    def handle_user_text(self, text):
        if self.is_running and self.loop.is_running(): asyncio.run_coroutine_threadsafe(self.text_input_queue.put(text), self.loop)

    async def shutdown_async_tasks(self):
        if self.text_input_queue: await self.text_input_queue.put(None)
        for task in self.tasks: task.cancel()
        await asyncio.sleep(0.1)

    def stop(self):
        if self.is_running and self.loop.is_running():
            self.is_running = False
            future = asyncio.run_coroutine_threadsafe(self.shutdown_async_tasks(), self.loop)
            try: future.result(timeout=5)
            except Exception as e: print(f">>> [ERROR] Timeout or error during async shutdown: {e}")
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream(); self.audio_stream.close()

# ==============================================================================
# NEW UI WIDGETS
# ==============================================================================
class TimeDateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(5)
        
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #AFFFDA; font-family: 'Segoe UI';")
        
        self.date_label = QLabel()
        self.date_label.setAlignment(Qt.AlignCenter)
        self.date_label.setStyleSheet("font-size: 14pt; color: #50C878; font-family: 'Segoe UI'; letter-spacing: 2px;")
        
        layout.addWidget(self.time_label)
        layout.addWidget(self.date_label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()

    def update_time(self):
        now = QDateTime.currentDateTime()
        self.time_label.setText(now.toString("hh:mm AP"))
        self.date_label.setText(now.toString("dddd, MMMM d, yyyy").upper())

class LocationWeatherTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 8, 15, 8)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Style like a tab/pill
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(13, 31, 23, 0.8);
                border: 1px solid #50C878;
                border-radius: 20px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        
        self.icon_label = QLabel("📍") # Simple icon
        self.icon_label.setStyleSheet("font-size: 14pt;")
        
        self.info_label = QLabel("Shah Alam • --°C")
        self.info_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #AFFFDA; margin-left: 5px;")
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.info_label)
        
        self.manager = QNetworkAccessManager(self)
        self.manager.finished.connect(self.on_weather_received)
        
        # Update weather every 15 minutes
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_weather)
        self.timer.start(15 * 60 * 1000)
        self.fetch_weather()

    def fetch_weather(self):
        # Shah Alam coordinates: 3.0738, 101.5183
        url = QUrl("https://api.open-meteo.com/v1/forecast?latitude=3.0738&longitude=101.5183&current=temperature_2m&timezone=Asia%2FSingapore")
        request = QNetworkRequest(url)
        self.manager.get(request)

    def on_weather_received(self, reply):
        if reply.error() == QNetworkReply.NoError:
            try:
                data = json.loads(str(reply.readAll(), 'utf-8'))
                temp = data.get("current", {}).get("temperature_2m", "--")
                self.info_label.setText(f"Shah Alam • {temp}°C")
            except Exception as e:
                print(f">>> [WEATHER] Error parsing data: {e}")
        else:
            print(f">>> [WEATHER] Error fetching data: {reply.errorString()}")
        reply.deleteLater()

class WeatherCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(13, 31, 23, 0.8);
                border: 1px solid rgba(80, 200, 120, 0.4);
                border-radius: 12px;
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)
        
        # Header with icon
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        weather_icon = QLabel("🌤")
        weather_icon.setStyleSheet("font-size: 16pt;")
        
        title = QLabel("Weather")
        title.setStyleSheet("color: #50C878; font-size: 12pt; font-weight: bold;")
        
        header_layout.addWidget(weather_icon)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Main temperature display
        self.temp_label = QLabel("25.2°C")
        self.temp_label.setStyleSheet("color: #AFFFDA; font-size: 36pt; font-weight: bold; margin: 10px 0;")
        layout.addWidget(self.temp_label)
        
        # Location and condition
        self.location_label = QLabel("Shah Alam")
        self.location_label.setStyleSheet("color: #8FE8B8; font-size: 11pt; font-weight: bold;")
        layout.addWidget(self.location_label)
        
        self.condition_label = QLabel("Overcast Clouds")
        self.condition_label.setStyleSheet("color: #6FD89C; font-size: 10pt;")
        layout.addWidget(self.condition_label)
        
        # Divider
        divider = QLabel()
        divider.setStyleSheet("background-color: rgba(80, 200, 120, 0.2); min-height: 1px; max-height: 1px; margin: 10px 0;")
        layout.addWidget(divider)
        
        # Details grid
        details_layout = QHBoxLayout()
        details_layout.setSpacing(15)
        
        # Humidity
        humidity_layout = QVBoxLayout()
        humidity_layout.setSpacing(3)
        humidity_title = QLabel("Humidity")
        humidity_title.setStyleSheet("color: #6FD89C; font-size: 9pt;")
        self.humidity_value = QLabel("90%")
        self.humidity_value.setStyleSheet("color: #AFFFDA; font-size: 11pt; font-weight: bold;")
        humidity_layout.addWidget(humidity_title)
        humidity_layout.addWidget(self.humidity_value)
        
        # Wind
        wind_layout = QVBoxLayout()
        wind_layout.setSpacing(3)
        wind_title = QLabel("Wind")
        wind_title.setStyleSheet("color: #6FD89C; font-size: 9pt;")
        self.wind_value = QLabel("3.8 m/s")
        self.wind_value.setStyleSheet("color: #AFFFDA; font-size: 11pt; font-weight: bold;")
        wind_layout.addWidget(wind_title)
        wind_layout.addWidget(self.wind_value)
        
        # Feels Like
        feels_layout = QVBoxLayout()
        feels_layout.setSpacing(3)
        feels_title = QLabel("Feels Like")
        feels_title.setStyleSheet("color: #6FD89C; font-size: 9pt;")
        self.feels_value = QLabel("26.3°C")
        self.feels_value.setStyleSheet("color: #AFFFDA; font-size: 11pt; font-weight: bold;")
        feels_layout.addWidget(feels_title)
        feels_layout.addWidget(self.feels_value)
        
        details_layout.addLayout(humidity_layout)
        details_layout.addLayout(wind_layout)
        details_layout.addLayout(feels_layout)
        
        layout.addLayout(details_layout)
        layout.addStretch()
        
        self.manager = QNetworkAccessManager(self)
        self.manager.finished.connect(self.on_weather_received)
        
        # Update every 15 minutes
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch_weather)
        self.timer.start(15 * 60 * 1000)
        self.fetch_weather()

    def fetch_weather(self):
        # Fetch comprehensive weather data including humidity, wind, and apparent temperature
        url = QUrl("https://api.open-meteo.com/v1/forecast?latitude=3.0738&longitude=101.5183&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=Asia%2FSingapore")
        request = QNetworkRequest(url)
        self.manager.get(request)

    def get_weather_description(self, code):
        """Convert WMO weather code to description"""
        weather_codes = {
            0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Foggy", 51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
            61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 71: "Light Snow", 73: "Snow",
            75: "Heavy Snow", 77: "Snow Grains", 80: "Light Showers", 81: "Showers",
            82: "Heavy Showers", 85: "Light Snow Showers", 86: "Snow Showers",
            95: "Thunderstorm", 96: "Thunderstorm with Hail", 99: "Heavy Thunderstorm"
        }
        return weather_codes.get(code, "Unknown")

    def on_weather_received(self, reply):
        if reply.error() == QNetworkReply.NoError:
            try:
                data = json.loads(str(reply.readAll(), 'utf-8'))
                current = data.get("current", {})
                
                temp = current.get("temperature_2m", "--")
                humidity = current.get("relative_humidity_2m", "--")
                wind = current.get("wind_speed_10m", "--")
                feels_like = current.get("apparent_temperature", "--")
                weather_code = current.get("weather_code", 0)
                condition = self.get_weather_description(weather_code)
                
                self.temp_label.setText(f"{temp}°C")
                self.humidity_value.setText(f"{humidity}%")
                self.wind_value.setText(f"{wind} m/s")
                self.feels_value.setText(f"{feels_like}°C")
                self.condition_label.setText(condition)
            except Exception as e:
                print(f">>> [WEATHER] Error parsing data: {e}")
                self.condition_label.setText("Weather data unavailable")
        else:
            print(f">>> [WEATHER] Error fetching data: {reply.errorString()}")
        reply.deleteLater()

# ==============================================================================
# STYLED GUI APPLICATION
# ==============================================================================
class MainWindow(QMainWindow):
    user_text_submitted = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("R.O.N.I. - Just Intelligently Being Operational")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1280, 720)

        # --- Load Custom Font ---
        font_path = os.path.abspath("fonts/nyxerin.otf")
        print(f">>> [DEBUG] Attempting to load font from: {font_path}")
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            print(f">>> [DEBUG] Font ID: {font_id}")
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                print(f">>> [DEBUG] Loaded font families: {font_families}")
                if font_families:
                    self.font_family = font_families[0]
                    print(f">>> [INFO] Using custom font: {self.font_family}")
                else:
                    print(">>> [WARNING] No font families found for loaded font.")
                    self.font_family = "'Segoe UI', 'Helvetica Neue', sans-serif"
            else:
                print(">>> [WARNING] Failed to load custom font (ID is -1).")
                self.font_family = "'Segoe UI', 'Helvetica Neue', sans-serif"
        else:
            print(f">>> [WARNING] Font file not found at: {font_path}")
            self.font_family = "'Segoe UI', 'Helvetica Neue', sans-serif"

        #        # Apply Stylesheet with Cohesive Emerald Green Theme
        self.setStyleSheet(f"""
            /* --- MAIN WINDOW --- */
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #0d1f17, stop:1 #1a332a);
                font-family: '{self.font_family}', 'Segoe UI', Arial, sans-serif;
            }}
            
            /* --- PANELS --- */
            QWidget#left_panel, QWidget#right_panel {{
                background-color: rgba(13, 31, 23, 0.85);
                border: 1px solid rgba(80, 200, 120, 0.4);
                border-radius: 12px;
            }}
            QWidget#center_panel {{
                background-color: transparent;
            }}
            
            /* --- HEADERS --- */
            QLabel#panel_header {{
                color: #50C878;
                font-family: '{self.font_family}';
                font-weight: bold;
                font-size: 14pt;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(80, 200, 120, 0.2), 
                                          stop:1 rgba(80, 200, 120, 0.05));
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom: 2px solid rgba(80, 200, 120, 0.5);
                text-transform: uppercase;
                letter-spacing: 3px;
                alignment: center;
            }}
            
            /* --- TEXT DISPLAY --- */
            QTextEdit#text_display {{
                background-color: rgba(0, 0, 0, 0.4);
                color: #6FD89C;
                font-family: '{self.font_family}';
                font-size: 11pt;
                border: none;
                padding: 15px;
                selection-background-color: #2d6e4a;
            }}
            QLabel#tool_activity_display {{
                background-color: transparent;
                color: #8FE8B8;
                font-family: '{self.font_family}';
                font-size: 10pt;
                padding: 15px;
            }}
            
            /* --- INPUT --- */
            QLineEdit#input_box {{
                background-color: rgba(13, 31, 23, 0.6);
                color: #6FD89C;
                font-family: '{self.font_family}';
                font-size: 12pt;
                border: 2px solid #50C878;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }}
            QLineEdit#input_box:focus {{
                border: 2px solid #8FE8B8;
                background-color: rgba(13, 31, 23, 0.9);
                color: #AFFFDA;
            }}
            
            /* --- VIDEO --- */
            QLabel#video_label {{
                background-color: #000000;
                border: 2px solid #50C878;
                border-radius: 15px;
            }}
            
            /* --- BUTTONS --- */
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(80, 200, 120, 0.15), 
                                          stop:1 rgba(80, 200, 120, 0.05));
                color: #6FD89C;
                font-family: '{self.font_family}';
                border: 1px solid #50C878;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 10pt;
                font-weight: bold;
                letter-spacing: 1px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(80, 200, 120, 0.4), 
                                          stop:1 rgba(80, 200, 120, 0.2));
                color: #AFFFDA;
                border: 1px solid #8FE8B8;
            }}
            QPushButton:pressed {{
                background-color: #50C878;
                color: #0d1f17;
            }}
            QPushButton#video_button_active {{
                background-color: #50C878;
                color: #0d1f17;
                border: 1px solid #6FD89C;
            }}
            
            /* --- SCROLLBAR --- */
            QScrollBar:vertical {{
                border: none;
                background: rgba(13, 31, 23, 0.5);
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #50C878, stop:1 #6FD89C);
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #6FD89C, stop:1 #8FE8B8);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        """)

        self.central_widget = QWidget(); self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # ==============================================================================
        # LEFT COLUMN: WEATHER + CAMERA
        # ==============================================================================
        self.left_panel = QWidget(); self.left_panel.setObjectName("left_panel")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10); self.left_layout.setSpacing(10)
        
        # Weather Card (Top of Left Panel)
        self.weather_card = WeatherCard()
        self.left_layout.addWidget(self.weather_card)
        
        # Camera Feed (Below Weather)
        self.video_container = QWidget()
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0,0,0,0)
        
        self.video_label = QLabel(); self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        video_layout.addWidget(self.video_label)
        self.left_layout.addWidget(self.video_container, 2) # Stretch factor for camera

        # ==============================================================================
        # CENTER COLUMN: VISUAL CORE (Time + Sphere)
        # ==============================================================================
        self.center_panel = QWidget(); self.center_panel.setObjectName("center_panel")
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0); self.center_layout.setSpacing(20)
        
        # Time and Date (Top Middle)
        self.time_date_widget = TimeDateWidget()
        self.center_layout.addWidget(self.time_date_widget)
        
        # AI Sphere Animation
        self.animation_widget = AvatarWidget()
        self.animation_widget.setMinimumHeight(300)
        self.animation_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.center_layout.addWidget(self.animation_widget)
        
        # Embedded Browser (Hidden by default)
        self.browser = QWebEngineView()
        self.browser.setVisible(False)
        self.browser.setMinimumHeight(200)
        
        # Enable Persistent Storage (Cookies/Login)
        profile = QWebEngineProfile("ada_profile", self.browser)
        profile.setPersistentStoragePath(os.path.abspath("web_profile"))
        profile.setCachePath(os.path.abspath("web_cache"))
        profile.setHttpUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        page = QWebEnginePage(profile, self.browser)
        self.browser.setPage(page)
        
        self.browser.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
        self.browser.settings().setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        self.browser.setStyleSheet("background: transparent;")
        self.browser.loadFinished.connect(self.on_browser_load_finished)
        self.center_layout.addWidget(self.browser)
        
        # Controls
        self.button_container = QHBoxLayout(); self.button_container.setSpacing(10)
        self.button_container.setAlignment(Qt.AlignCenter)
        self.webcam_button = QPushButton("CAM LINK"); self.webcam_button.setCursor(Qt.PointingHandCursor)
        self.screenshare_button = QPushButton("SCR LINK"); self.screenshare_button.setCursor(Qt.PointingHandCursor)
        self.off_button = QPushButton("OFFLINE"); self.off_button.setCursor(Qt.PointingHandCursor)
        self.close_browser_button = QPushButton("CLOSE VIDEO"); self.close_browser_button.setCursor(Qt.PointingHandCursor)
        self.close_browser_button.setVisible(False)
        self.close_browser_button.clicked.connect(self.stop_video)
        
        self.button_container.addWidget(self.webcam_button)
        self.button_container.addWidget(self.screenshare_button)
        self.button_container.addWidget(self.off_button)
        self.button_container.addWidget(self.close_browser_button)
        self.center_layout.addLayout(self.button_container)

        # ==============================================================================
        # RIGHT COLUMN: SYSTEM DATA + CHAT
        # ==============================================================================
        self.right_panel = QWidget(); self.right_panel.setObjectName("right_panel")
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(10, 10, 10, 10); self.right_layout.setSpacing(10)
        
        # System Data
        self.sys_header = QLabel("SYSTEM DATA"); self.sys_header.setObjectName("panel_header")
        self.sys_header.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.sys_header)
        
        self.tool_activity_title = QLabel("Initializing..."); self.tool_activity_title.setVisible(False) # Hidden, used for logic
        
        self.tool_activity_display = QLabel(); self.tool_activity_display.setObjectName("tool_activity_display")
        self.tool_activity_display.setWordWrap(True); self.tool_activity_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.tool_activity_display.setOpenExternalLinks(True); self.tool_activity_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tool_activity_display)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        self.right_layout.addWidget(scroll_area, 1)
        
        # Chat Interface
        self.comm_header = QLabel("COMM LINK"); self.comm_header.setObjectName("panel_header")
        self.comm_header.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.comm_header)
        
        self.text_display = QTextEdit(); self.text_display.setObjectName("text_display"); self.text_display.setReadOnly(True)
        self.right_layout.addWidget(self.text_display, 3) # Stretch factor 3
        
        self.input_box = QLineEdit(); self.input_box.setObjectName("input_box")
        self.input_box.setPlaceholderText("Execute Command...")
        self.input_box.returnPressed.connect(self.send_user_text)
        self.right_layout.addWidget(self.input_box)

        # --- Assemble Main Layout ---
        self.main_layout.addWidget(self.left_panel, 25)   # 25% width
        self.main_layout.addWidget(self.center_panel, 40) # 40% width
        self.main_layout.addWidget(self.right_panel, 35)  # 35% width
        self.is_first_ada_chunk = True
        self.current_video_mode = DEFAULT_MODE
        self.setup_backend_thread()
        
        # Log loaded font to UI
        self.tool_activity_display.setText(f"SYSTEM INITIALIZED\n> Loaded Font: {self.font_family}")

    def setup_backend_thread(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="pixels to stream from", choices=["camera", "screen", "none"])
        args, unknown = parser.parse_known_args()
        
        self.ai_core = AI_Core(video_mode=args.mode)
        self.ai_thread = threading.Thread(target=self.ai_core.start_event_loop, daemon=True)
        self.ai_thread.start()
        
        # Start wake word listener thread
        self.wake_word_thread = threading.Thread(target=self.ai_core.listen_for_wake_word, daemon=True)
        self.wake_word_thread.start()
        
        # Connect Signals
        self.user_text_submitted.connect(self.ai_core.handle_user_text)
        self.ai_core.text_received.connect(self.update_text)
        self.ai_core.search_results_received.connect(self.update_search_results)
        self.ai_core.code_being_executed.connect(self.display_executed_code)
        self.ai_core.file_list_received.connect(self.update_file_list)
        self.ai_core.end_of_turn.connect(self.add_newline)
        self.ai_core.frame_received.connect(self.update_frame)
        self.ai_core.video_mode_changed.connect(self.update_video_mode_ui)
        self.ai_core.speaking_started.connect(self.animation_widget.start_speaking_animation)
        self.ai_core.speaking_stopped.connect(self.animation_widget.stop_speaking_animation)
        self.ai_core.connection_status_changed.connect(self.update_connection_status)
        self.ai_core.video_requested.connect(self.play_video)
        self.ai_core.media_control_requested.connect(self.handle_media_control)
        self.ai_core.wake_word_activated.connect(self.on_wake_word_activated)
        self.ai_core.wake_word_deactivated.connect(self.on_wake_word_deactivated)
        self.ai_core.image_generated.connect(self.display_generated_image)
        
        # Button Connections
        self.webcam_button.clicked.connect(lambda: self.ai_core.set_video_mode("camera"))
        self.screenshare_button.clicked.connect(lambda: self.ai_core.set_video_mode("screen"))
        self.off_button.clicked.connect(lambda: self.ai_core.set_video_mode("none"))
        
        self.update_video_mode_ui(self.ai_core.video_mode)

    @Slot(str)
    def play_video(self, url):
        """Switches to browser view and plays the URL."""
        print(f">>> [GUI] Playing video: {url}")
        self.animation_widget.setVisible(False)
        self.browser.setVisible(True)
        self.close_browser_button.setVisible(True)
        self.browser.setUrl(QUrl(url))

    @Slot()
    def stop_video(self):
        """Stops video and returns to animation view."""
        print(">>> [GUI] Stopping video...")
        self.browser.setUrl(QUrl("about:blank"))
        self.browser.setVisible(False)
        self.close_browser_button.setVisible(False)
        self.animation_widget.setVisible(True)

    @Slot(str)
    def handle_media_control(self, action):
        print(f">>> [GUI] Handling media control: {action}")
        js_code = ""
        if action == "play" or action == "pause":
            js_code = "document.querySelector('.ytp-play-button').click();"
        elif action == "mute" or action == "unmute":
            js_code = "document.querySelector('.ytp-mute-button').click();"
        elif action == "next":
            js_code = "document.querySelector('.ytp-next-button').click();"
        elif action == "previous":
            js_code = "document.querySelector('.ytp-prev-button').click();"
        elif action == "stop":
            self.stop_video()
            return

        if js_code:
            self.browser.page().runJavaScript(js_code)

    @Slot(str)
    def update_connection_status(self, status):
        if status == "CONNECTED":
            self.sys_header.setText("SYSTEM ONLINE")
            self.sys_header.setStyleSheet("color: #50C878; background-color: rgba(80, 200, 120, 0.1);")
        elif status == "RECONNECTING":
            self.sys_header.setText("RECONNECTING...")
            self.sys_header.setStyleSheet("color: #ffaa00; background-color: rgba(255, 170, 0, 0.1);")
        elif status == "ERROR":
            self.sys_header.setText("SYSTEM ERROR")
            self.sys_header.setStyleSheet("color: #ff0000; background-color: rgba(255, 0, 0, 0.1);")
    
    @Slot()
    def on_wake_word_activated(self):
        """Update UI when wake word is detected"""
        print(">>> [GUI] Wake word activated!")
        self.sys_header.setText("LISTENING - ACTIVE")
        self.sys_header.setStyleSheet("color: #00ff00; background-color: rgba(0, 255, 0, 0.2);")
    
    @Slot()
    def on_wake_word_deactivated(self):
        """Update UI when wake word times out"""
        print(">>> [GUI] Wake word deactivated!")
        self.sys_header.setText("SYSTEM ONLINE")
        self.sys_header.setStyleSheet("color: #50C878; background-color: rgba(80, 200, 120, 0.1);")

    def on_browser_load_finished(self, success):
        if not success: return
        url = self.browser.url().toString()
        if "youtube.com/watch" in url:
            print(">>> [GUI] Injecting Theater Mode Script...")
            js_code = """
            (function() {
                function toggleTheater() {
                    var app = document.querySelector("ytd-watch-flexy");
                    var btn = document.querySelector(".ytp-size-button");
                    if (app && btn && !app.hasAttribute("theater")) {
                        btn.click();
                        console.log("Entered Theater Mode");
                    }
                }
                setTimeout(toggleTheater, 1000);
                setTimeout(toggleTheater, 3000);
            })();
            """
            self.browser.page().runJavaScript(js_code)

    def send_user_text(self):
        text = self.input_box.text().strip()
        if text:
            self.text_display.append(f"<p style='color:#00ffff; font-weight:bold;'>&gt; USER:</p><p style='color:#e0e0ff; padding-left: 10px;'>{escape(text)}</p>")
            self.user_text_submitted.emit(text)
            self.input_box.clear()

    @Slot(str)
    def update_video_mode_ui(self, mode):
        self.current_video_mode = mode
        self.webcam_button.setObjectName("")
        self.screenshare_button.setObjectName("")
        self.off_button.setObjectName("")

        if mode == "camera":
            self.webcam_button.setObjectName("video_button_active")
        elif mode == "screen":
            self.screenshare_button.setObjectName("video_button_active")
        elif mode == "none":
            self.off_button.setObjectName("video_button_active")
            self.video_label.clear()

        for button in [self.webcam_button, self.screenshare_button, self.off_button]:
            button.style().unpolish(button)
            button.style().polish(button)

    @Slot(str)
    def update_text(self, text):
        if self.is_first_ada_chunk:
            self.is_first_ada_chunk = False
            self.text_display.append(f"<p style='color:#00d1ff; font-weight:bold;'>&gt; R.O.N.I.:</p>")
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.text_display.verticalScrollBar().setValue(self.text_display.verticalScrollBar().maximum())

    @Slot(list)
    def update_search_results(self, urls):
        base_title = "SYSTEM ACTIVITY"
        if not urls:
            if "SEARCH" in self.tool_activity_title.text():
                self.tool_activity_display.clear(); self.tool_activity_title.setText(base_title)
            return
        self.tool_activity_display.clear()
        self.tool_activity_title.setText(f"{base_title} // SEARCH")
        html_content = ""
        for i, url in enumerate(urls):
            display_text = url.split('//')[1].split('/')[0] if '//' in url else url
            html_content += f'<p style="margin:0; padding: 4px;">{i+1}: <a href="{url}" style="color: #00ffff; text-decoration: none;">{display_text}</a></p>'
        self.tool_activity_display.setText(html_content)

    @Slot(str, str)
    def display_executed_code(self, code, result):
        base_title = "SYSTEM ACTIVITY"
        if not code:
            if "CODE EXEC" in self.tool_activity_title.text():
                 self.tool_activity_display.clear(); self.tool_activity_title.setText(base_title)
            return
        self.tool_activity_display.clear()
        self.tool_activity_title.setText(f"{base_title} // CODE EXEC")
        html = f'<pre style="white-space: pre-wrap; word-wrap: break-word; color: #e0e0ff; font-size: 9pt; line-height: 1.4;">{escape(code)}</pre>'
        if result:
            html += f'<p style="color:#00d1ff; font-weight:bold; margin-top:10px; margin-bottom: 5px;">&gt; OUTPUT:</p><pre style="white-space: pre-wrap; word-wrap: break-word; color: #90EE90; font-size: 9pt;">{escape(result.strip())}</pre>'
        self.tool_activity_display.setText(html)

    @Slot(str, list)
    def update_file_list(self, directory_path, files):
        base_title = "SYSTEM ACTIVITY"
        if not directory_path:
            if "FILESYS" in self.tool_activity_title.text():
                self.tool_activity_display.clear(); self.tool_activity_title.setText(base_title)
            return
        self.tool_activity_display.clear()
        self.tool_activity_title.setText(f"{base_title} // FILESYS")
        html = f'<p style="color:#00d1ff; margin-bottom: 5px;">DIR &gt; <strong>{escape(directory_path)}</strong></p>'
        if not files:
            html += '<p style="margin-top:5px; color:#a0a0ff;"><em>(Directory is empty)</em></p>'
        else:
            folders = sorted([i for i in files if os.path.isdir(os.path.join(directory_path, i))])
            file_items = sorted([i for i in files if not os.path.isdir(os.path.join(directory_path, i))])
            html += '<ul style="list-style-type:none; padding-left: 5px; margin-top: 5px;">'
            for folder in folders: html += f'<li style="margin: 2px 0; color: #87CEEB;">[+] {escape(folder)}</li>'
            for file_item in file_items: html += f'<li style="margin: 2px 0; color: #e0e0ff;">&#9679; {escape(file_item)}</li>'
            html += '</ul>'
        self.tool_activity_display.setText(html)

    @Slot()
    def add_newline(self):
        if not self.is_first_ada_chunk: self.text_display.append("")
        self.is_first_ada_chunk = True

    @Slot(QImage)
    def update_frame(self, image):
        if self.current_video_mode == "none":
            if self.video_label.pixmap():
                self.video_label.clear()
            return

        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.video_container.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        else:
            self.video_label.clear()
    
    @Slot(str)
    def display_generated_image(self, image_path):
        """Display generated image in the chat"""
        print(f">>> [GUI] Displaying generated image: {image_path}")
        if os.path.exists(image_path):
            # Fix path for HTML (replace backslashes with forward slashes)
            clean_path = image_path.replace("\\", "/")
            
            # Display image in chat with clickable link
            filename = os.path.basename(image_path)
            html = f'''
            <div style="margin: 10px 0; padding: 10px; background-color: rgba(80, 200, 120, 0.1); border-radius: 8px;">
                <p style="color: #00d1ff; font-weight: bold; margin-bottom: 5px;">🖼️ Generated Image:</p>
                <img src="file:///{clean_path}" style="max-width: 100%; border-radius: 8px; border: 1px solid #50C878;">
                <p style="color: #8FE8B8; font-size: 9pt; margin-top: 5px;">Saved to: {filename}</p>
            </div>
            '''
            self.text_display.append(html)
            
            # Auto-scroll to bottom
            self.text_display.verticalScrollBar().setValue(self.text_display.verticalScrollBar().maximum())
        else:
            print(f">>> [GUI] Image file not found: {image_path}")
            
    def closeEvent(self, event):
        print(">>> [GUI] Closing application...")
        # Cleanup WebEngine to avoid profile release warning
        if hasattr(self, 'browser'):
            self.browser.setPage(None)
        
        self.ai_core.stop()
        event.accept()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print(">>> [INFO] Application interrupted by user.")
    finally:
        pya.terminate()
        print(">>> [INFO] Application terminated.")


