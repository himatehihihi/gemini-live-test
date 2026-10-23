# Gemini Live Test Assistant

This project is an experimental Windows desktop assistant built around Google's Gemini Live API. It combines a PySide6 interface, a web-based animated avatar, streaming microphone audio, optional camera or screen frames, wake-word handling, and a bridge to the local Jarvis tool system.

## What it does

- Opens a PySide6 desktop GUI with an embedded animated avatar.
- Maintains a bidirectional Gemini Live session for realtime voice interaction.
- Streams microphone audio to Gemini and plays returned audio through Kokoro TTS/audio playback paths.
- Supports `none`, `camera`, and `screen` visual-input modes.
- Uses a wake word and exposes connection, speaking, media, file, and image-generation events to the GUI.
- Includes Gemini function declarations for safe tools from the sibling `tools/` registry.
- Provides desktop actions such as opening applications and websites, media control, file operations, Google search, and selected Windows/WhatsApp automation.

## Repository layout

```text
google_jarvis/
  jarvis.py              Main PySide6 application and Gemini Live client
  tool_bridge.py         Gemini function-calling adapter
  avatar_renderer.html  Embedded avatar renderer
  fonts/                 Avatar font asset
  requirements.txt       Original pinned dependency snapshot
hands/                   Windows/browser/media/communication actions
tools/                   Shared registry, validation, routing, and executors
```

`google_jarvis/` contains the GUI/Live experiment. The top-level `hands/` and `tools/` packages are imported by `tool_bridge.py` so Gemini tool calls can reuse the existing validation and execution architecture.

## Prerequisites

- Windows 10/11
- Python 3.10+
- A working microphone and audio output device
- A Google Gemini API key with Gemini Live access
- The GUI, audio, computer-vision, wake-word, and automation dependencies used by `google_jarvis/jarvis.py`

The checked-in `google_jarvis/requirements.txt` is an older dependency snapshot and does not list every import currently used by `jarvis.py` (for example, the Google GenAI SDK, PySide6/WebEngine, PyAudio, python-dotenv, and openwakeword). Treat it as a starting point and resolve missing imports in your environment.

## Configuration

Create a local `.env` file in `google_jarvis/` or set the variable in PowerShell:

```env
GEMINI_API_KEY=your-key
```

Never commit `.env` or real credentials. The application exits at startup when `GEMINI_API_KEY` is missing.

## Run

From the repository root:

```powershell
cd google_jarvis
python jarvis.py
```

The application starts the desktop interface and backend event loop. Use the GUI to choose camera/screen/none visual mode and interact by voice or text.

## Tool bridge

`tool_bridge.py` converts safe entries from `tools.tool_registry` into Gemini function declarations. Incoming calls are validated and routed through `tools.tool_validator` and `tools.executor_router`.

For the initial Live demo, only tools marked `SAFE` are exposed. Confirmation-required, blocked, install, test, shutdown, delete, and Git operations are intentionally excluded because this experimental path does not yet provide an interactive confirmation flow.

Run the bridge sanity check from `google_jarvis/`:

```powershell
python tool_bridge.py
```

## Current limitations

- This is an experimental integration rather than a packaged application.
- The dependency snapshot needs updating for a clean reproducible install.
- Hardware/audio behavior is Windows- and device-dependent.
- Gemini Live, screen capture, image generation, and external automation require their respective runtime permissions and services.
- Several tool modules contain placeholders or environment-specific implementations.
- The avatar is a lightweight embedded renderer, not a full 3D asset pipeline.

## Relationship to `jarvis`

This repository is a focused Gemini Live desktop experiment. The separate [`jarvis`](https://github.com/himatehihihi/jarvis) repository is the more integrated assistant architecture with explicit Gemini-primary/local-fallback voice control, optional planner health, broader diagnostics, and a more formal tool-validation boundary.

## License

See [`google_jarvis/LICENSE`](google_jarvis/LICENSE) for the license included with the original GUI/Live experiment.
