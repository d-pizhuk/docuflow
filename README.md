# DocuFlow

DocuFlow is a Windows desktop application that turns a narrated computer task
into step-by-step documentation with screenshots.

The application records microphone audio, captures selected screen regions,
transcribes the narration locally with Whisper, and generates portable Markdown
and HTML documentation.

## Current workflow

1. Select a microphone and documentation language.
2. Start a recording session.
3. Explain the task while demonstrating it.
4. Hold `Ctrl` and drag with the mouse to capture screenshots.
5. Stop the recording.
6. Choose **Process Recording** or **Later**.
7. Follow the six-stage processing progress.
8. Open the generated HTML, Markdown, or session folder.

The processing pipeline performs:

```text
Audio recording
    → Whisper transcription
    → Screenshot timestamp alignment
    → Structured process generation
    → Screenshot descriptions
    → Markdown
    → Styled HTML
```

## Requirements

- Windows 10 or Windows 11
- Python 3.12
- A microphone
- FFmpeg available on `PATH`
- An OpenAI API key
- Internet access during the OpenAI processing stages
- Optional: an NVIDIA GPU with CUDA support for faster Whisper transcription

Whisper can run on the CPU, but the default `large-v3` model will be
significantly slower.

## Installation with Anaconda

Open Anaconda PowerShell Prompt:

```powershell
cd D:\repos\docuflow
conda create -n docuflow python=3.12 -y
conda activate docuflow
```

Install the desktop recording dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the AI and document-processing dependencies:

```powershell
python -m pip install openai-whisper openai python-dotenv pydantic markdown torch
```

Install FFmpeg in the active Conda environment:

```powershell
conda install -c conda-forge ffmpeg -y
```

Verify the installation:

```powershell
python -c "import PySide6, whisper, torch, openai, markdown; print('Dependencies OK')"
ffmpeg -version
```

For NVIDIA GPU acceleration, install a CUDA-compatible PyTorch build suitable
for your hardware instead of relying on the default `torch` installation.

## OpenAI API key

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by Git. Never commit a real API key.

## Running DocuFlow

Activate the environment and start the application from the repository root:

```powershell
conda activate docuflow
cd D:\repos\docuflow
python main.py
```

The repository root matters because recordings are stored using the relative
`recordings/` path.

### Recording controls

- **Test Mic** checks the selected microphone input.
- **Start Session** begins recording.
- Hold **Ctrl** and drag to capture a selected screen region.
- **Stop Session** finishes and saves the recording.
- `Ctrl+Alt+Q` is an emergency exit and may lose unsaved recording data.

DocuFlow records the selected microphone input. It does not intentionally
capture system audio. Audio playing through speakers can still be picked up by
the microphone.

## Documentation languages

German is selected by default. The setup dialog currently offers:

- German
- English
- French
- Spanish
- Italian
- Portuguese
- Dutch
- Polish
- Czech
- Japanese
- Korean
- Chinese (Simplified)

Whisper detects the spoken language automatically. The selected documentation
language controls the generated process steps, screenshot descriptions, and
final documents.

## Session files

Each session is stored under:

```text
recordings/
└── recording_session_YYYY-MM-DD_HH-MM-SS_microseconds/
    ├── session.json
    ├── recording.wav
    ├── screenshots/
    │   ├── screenshot_001.png
    │   └── screenshot_002.png
    └── output/
        ├── transcript.json
        ├── transcript_with_screenshots.txt
        ├── screenshot_assignments.json
        ├── process_documentation.json
        ├── screenshot_descriptions.json
        ├── descriptions/
        │   ├── screenshot_001.json
        │   └── screenshot_002.json
        └── final/
            ├── final_documentation.md
            ├── process_documentation.html
            └── screenshots/
```

`session.json` records the session settings, screenshot timestamps, artifact
paths, stage status, errors, and processing metadata.

The `recordings/` directory is ignored by Git.

## Processing an existing session

List available sessions:

```powershell
Get-ChildItem recordings -Directory
```

Run or resume the complete pipeline:

```powershell
python -m processing.pipeline "recordings\recording_session_2026-06-19_18-39-57_388518"
```

The saved documentation language is used automatically.

Override the output language:

```powershell
python -m processing.pipeline "recordings\recording_session_..." --output-language English
```

Restart one stage and every downstream stage:

```powershell
python -m processing.pipeline "recordings\recording_session_..." --restart-from process_documentation
```

Rerun every stage:

```powershell
python -m processing.pipeline "recordings\recording_session_..." --force
```

Use an actual session folder name in place of
`recording_session_...`. Do not enter `<session>` literally.

### Pipeline stages

Valid values for `--restart-from` are:

```text
transcription
screenshot_markers
process_documentation
screenshot_descriptions
markdown
html
```

Completed compatible stages are reused automatically. If processing fails,
running the pipeline again resumes from the first incomplete stage.

## Privacy and data processing

- Microphone recording and screenshot capture happen locally.
- Whisper transcription runs locally.
- When **Process Recording** is selected, the enriched transcript is sent to
  OpenAI to generate structured process steps.
- Referenced screenshots and nearby transcript context are sent to OpenAI to
  generate screenshot descriptions.
- Markdown and HTML generation happen locally.

Review screenshots before processing if they may contain passwords, personal
messages, API keys, customer data, or other sensitive information.

## Command-line stages

Each stage can also be run separately:

```powershell
python -m processing.transcribe "recordings\recording_session_..."
python -m processing.insert_screenshot_markers "recordings\recording_session_..."
python -m processing.generate_process_documentation "recordings\recording_session_..."
python -m processing.describe_screenshots "recordings\recording_session_..."
python -m processing.build_markdown "recordings\recording_session_..."
python -m processing.render_html "recordings\recording_session_..."
```

Use `python -m <module> --help` for all available options.

## Tests

Run the test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The suite covers session metadata, transcription behavior, screenshot
alignment, structured-output validation, screenshot context, Markdown, HTML,
pipeline resume/retry logic, and the Qt processing workflow.

## Troubleshooting

### `ModuleNotFoundError: No module named 'PySide6'`

Activate the correct environment and install the recording dependencies:

```powershell
conda activate docuflow
python -m pip install -r requirements.txt
```

### Whisper cannot find FFmpeg

Confirm:

```powershell
ffmpeg -version
```

If the command fails:

```powershell
conda install -c conda-forge ffmpeg -y
```

Then restart the terminal.

### CUDA runs out of memory

Use a smaller Whisper model from the command line:

```powershell
python -m processing.pipeline "recordings\recording_session_..." --whisper-model medium --restart-from transcription
```

Or force CPU processing:

```powershell
python -m processing.pipeline "recordings\recording_session_..." --device cpu --restart-from transcription
```

### `OPENAI_API_KEY is not set`

Confirm that `.env` exists in the repository root:

```powershell
Test-Path .env
```

The command should print `True`.

### The application remains open after processing

Wait for the active stage to finish, then close the processing dialog. The
dialog intentionally prevents closing while its background worker is running.

## Current limitations

- Windows-only desktop workflow
- No previous-session browser in the UI yet
- Processing cannot currently be cancelled safely
- OpenAI is required for process generation and screenshot descriptions
- Mixed-DPI and complex multi-monitor layouts need broader testing
- No packaged executable or installer yet
- Dependency setup is not yet distributed as one consolidated environment file

## License

This project is licensed under the terms in [LICENSE](LICENSE).
