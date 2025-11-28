# AI Manim Video Generator

An AI-powered educational video generation system that automatically creates animated explanations of algorithms and data structures using Manim (Mathematical Animation Engine). The system features a modern web interface built with NiceGUI and leverages large language models to plan, generate, and render educational content.

## 🎯 Features

- **Automated Video Generation**: Generate complete educational videos from simple topic descriptions
- **AI-Powered Planning**: Uses LLMs (Gemini, Claude, OpenAI) to create scene outlines and implementation plans
- **Manim Integration**: Automatically generates and renders Manim animation code
- **Error Recovery**: Built-in error detection and automatic code fixing with visual validation
- **Interactive Web UI**: Modern, Streamlit-inspired interface with dark mode support
- **Temporal Context Mapping (TCM)**: Fine-grained subtitle generation with concept tracking
- **Concurrent Processing**: Parallel scene generation and rendering for faster video production
- **AI Tutor Chat**: Interactive Q&A system that understands video context and timing
- **Text-to-Speech**: Integrated voice narration using Kokoro TTS engine
- **Project Viewer**: Detailed view of scene plans, code, and rendering status

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg
- Cairo and Pango libraries
- PortAudio (for audio processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LDSPrgrm/AlgoVision
cd AlgoVision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```env
GEMINI_API_KEY="your-gemini-api-key"
CLAUDE_API_KEY="your-claude-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### Running the Application

**IMPORTANT**: Before generating videos, set the Python path to avoid errors:

```bash
# Windows (CMD)
set PYTHONPATH=%cd%;%PYTHONPATH%

# Windows (PowerShell)
$env:PYTHONPATH="$pwd;$env:PYTHONPATH"

# Linux/Mac
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Then start the application:

```bash
python main.py
```

The web interface will be available at `http://localhost:8080`

## 📖 Usage

### Creating a New Video

1. **Open the sidebar** and navigate to "Create New Video"
2. **Enter topic details**:
   - Topic name (e.g., "Bubble Sort")
   - Detailed description of what to explain
3. **Configure settings**:
   - Select AI model (Gemini, Claude, or OpenAI)
   - Adjust scene concurrency for parallel processing
4. **Generate**:
   - Click "Generate Video" to start the pipeline
   - Monitor progress in real-time
   - View logs and status updates

### Video Generation Pipeline

The system follows a multi-stage pipeline:

1. **Scene Planning**: AI generates a structured outline breaking down the topic into scenes
2. **Implementation Design**: Each scene gets detailed technical and visual plans
3. **Code Generation**: Manim Python code is automatically generated for each scene
4. **Rendering**: Videos are rendered with automatic error recovery
5. **Combination**: All scenes are merged with synchronized subtitles and TCM

### Viewing Generated Videos

1. Navigate to "Project View"
2. Select a topic to view:
   - Combined final video with subtitles
   - Individual scene breakdowns
   - Implementation plans and code
   - Rendering logs and status
3. Use the AI Tutor to ask questions about specific moments in the video

### AI Tutor Chat

The AI Tutor provides context-aware assistance:

- Pause the video at any point
- Ask questions about the current concept
- Get explanations referencing exact visual elements
- Understand what's happening at specific timestamps

## 🛠️ Configuration

### Model Selection

Supported models (configured in `src/utils/models.json`):
- Google Gemini
- Anthropic Claude
- OpenAI

## 🎨 Features in Detail

### Temporal Context Map (TCM)

TCM provides fine-grained tracking of educational content:
- Maps narration to specific timestamps
- Tracks visual concepts and their durations
- Enables context-aware AI tutoring
- Generates accurate subtitles (SRT/VTT)

### Error Recovery System

Automatic error handling includes:
- Syntax error detection and fixing
- Runtime error recovery
- Visual validation of rendered frames
- Iterative code improvement with LLM feedback

## 🔧 Troubleshooting

### Common Issues

**Import Errors**:
- Ensure `PYTHONPATH` is set correctly before running
- Use the command provided in the Quick Start section

**Rendering Failures**:
- Check FFmpeg installation: `ffmpeg -version`
- Verify Cairo/Pango libraries are installed
- Review error logs in `output/<topic>/scene<N>/code/`

**API Rate Limits**:
- The system automatically detects and handles rate limits
- Consider using multiple API keys or different models
- Adjust scene concurrency to reduce parallel requests

## 📝 Output Structure

Generated content is organized as:

```
output/
└── <Topic Name>/
    └── <topic_slug>/
        ├── <topic_slug>_scene_outline.txt
        ├── <topic_slug>_combined.mp4
        ├── <topic_slug>_combined_tcm.json
        ├── <topic_slug>_combined.srt
        ├── <topic_slug>_combined.vtt
        ├── session_id.txt
        └── scene<N>/
            ├── <topic_slug>_scene<N>_implementation_plan.txt
            ├── proto_tcm.json
            ├── succ_rendered.txt
            ├── code/
            │   ├── <topic_slug>_scene<N>_v0.py
            │   └── <topic_slug>_scene<N>_v0_init_log.txt
            └── subplans/
                └── scene_trace_id.txt
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM provider support
- Enhanced error recovery strategies
- More Manim animation templates
- UI/UX improvements
- Performance optimizations

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) - Mathematical Animation Engine
- [NiceGUI](https://nicegui.io/) - Modern Python UI framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API interface
- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) - Text-to-speech engine

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Note**: This system requires API access to LLM providers (Gemini, Claude, or OpenAI). Costs may vary based on usage and selected models.
