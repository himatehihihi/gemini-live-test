# JARVIS - AI Assistant

A sophisticated, multi-modal AI assistant powered by Google's **Gemini 2.0 Flash Live** and features a dynamic **3D Animated Avatar**.

![JARVIS Preview](https://via.placeholder.com/800x450.png?text=JARVIS+AI+Assistant+Preview)

## 🚀 Features

- **Live Multimodal Interaction**: Real-time voice and vision interaction using Gemini Live.
- **Dynamic 3D Avatar**: A Three.js-powered 3D avatar that pulses and reacts when the AI is speaking.
- **Smart Tool Integration**:
  - **Vision-to-Action**: Analyze your screen or webcam to perform tasks.
  - **Web Search**: Real-time information retrieval via Google Search.
  - **Media Control**: YouTube playback and system media controls.
  - **Image Generation**: Built-in image generation capabilities.
  - **System Automation**: Open applications, manage files, and execute Python code.
- **Voice Capabilities**: High-quality TTS using Kokoro and wake word detection ("Hey Jarvis").

## 📋 Prerequisites

- Python 3.10+
- PySide6 (for the GUI)
- Google GenAI API Key
- Hugging Face API Key (for image generation)

## 🛠️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AimanRosman/Jarvis-AI-Assistant.git
   cd Jarvis-AI-Assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY = "YOUR_API_KEY"
   HUGGINGFACE_API_KEY = "YOUR_API_KEY"
   ```

4. **Run the application**:
   ```bash
   python jarvis.py
   ```

## 🖥️ Usage

- **Voice Commands**: Say "Hey Jarvis" to activate the assistant.
- **Visual Input**: Switch between Camera, Screen, or None in the UI to share visual context with the AI.
- **Tools**: JARVIS will automatically use its tools (Search, Code Execution, etc.) based on your requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ by Aiman Rosman*
