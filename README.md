# DocuFlow

**AI-Powered Automatic Step-by-Step Documentation Generator**
*Just talk. We'll write the docs.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2B-lightgrey.svg)](https://github.com/)

DocuFlow is a desktop application that automatically generates clean, structured step-by-step documentation (text + screenshots) from a user verbally explaining and demonstrating a computer task. It records your microphone and screen while you talk through a workflow, then uses a hybrid local + cloud AI pipeline to transcribe, segment, structure, illustrate, and export the final guide.

> **Domain:** Productivity Tools and Software Engineering
> 
> **Authors:** David Radonic (k11906471) · Davyd Pizhuk (k12148477)
> 
> **License:** MIT

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How It Works — The Pipeline](#how-it-works--the-pipeline)
4. [System Architecture](#system-architecture)
5. [AI Strategy](#ai-strategy)
6. [Privacy & Consent](#privacy--consent)
7. [Project Structure](#project-structure)
8. [Requirements](#requirements)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Configuration](#configuration)
12. [Keyboard Shortcuts](#keyboard-shortcuts)
13. [Export Formats](#export-formats)
14. [Benchmark & Evaluation](#benchmark--evaluation)
15. [Testing](#testing)
16. [Scripts](#scripts)
17. [Troubleshooting](#troubleshooting)

---

## Overview

Writing documentation is slow, tedious, and frequently skipped. Engineers finish a task — and the docs never get written. DocuFlow closes that gap:

- **Just talk.** Speak your workflow aloud while demonstrating.
- **Capture the right moments.** Press a hotkey to grab annotated screenshots at the exact moments that matter.
- **Let AI structure it.** A local Whisper model transcribes everything privately on your machine. A cloud LLM converts the transcript into clean step-by-step instructions, and a Vision-Language Model (VLM) writes captions for each screenshot.
- **Review and export.** Edit inline with a live preview, then export to **HTML**, **PDF**, or **Markdown**.

**Target users:** Software engineers, IT support staff, technical writers, and students.

---

## Key Features

- 🎙️ **Local speech-to-text** with `faster-whisper` (Whisper large-v3-turbo, INT8 quantized, CPU). Audio never leaves your machine.
- 🖼️ **Manual screenshot capture** with a global Ctrl+drag overlay, timestamped and aligned to the transcript.
- 🧠 **Cloud LLM step structuring** — converts a messy spoken transcript into clean JSON steps (zero-shot, with strict extractive rules preserving speaker vocabulary).
- 👁️ **Cloud VLM screenshot captioning** — generates a title + 1–3 sentence description per screenshot, grounded in the transcript context.
- ✏️ **Side-by-side Review & Edit UI** — every step is labeled `AI` until the user edits it, with the original transcript and screenshot visible alongside.
- 📄 **Export to HTML, PDF, and Markdown** with embedded (base64) images for portable, self-contained files.
- 🌐 **12 supported documentation languages** (English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech, Japanese, Korean, Chinese).
- 🛡️ **Resilient API Gateway** with TLS 1.3, exponential-backoff retries, and a checkpoint-and-fallback safety net that always preserves your raw transcript + screenshots if the cloud goes down.
- 📊 **Built-in benchmark suite** — 10 reference workflows with WER and ROUGE-L evaluation, plus timing tests for the non-functional performance requirements.

---

## How It Works — The Pipeline

DocuFlow's documentation generation is a 9-stage pipeline:

```
┌─────────────────────┐
│ 1. Audio Capture     │  AudioRecorderThread records 16kHz mono audio
│                     │  from the selected microphone, slicing it into
│                     │  ~15s chunks on silence/max-size boundaries.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Whisper STT       │  Each chunk is queued to a local faster-whisper
│                     │  worker thread that transcribes it on CPU
│                     │  (INT8). Word-level timestamps are optional.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Screenshot Capture│  GlobalOverlay activates when the user holds
│                     │  Ctrl; a drag-select rectangle captures a
│                     │  region with `mss` and stamps it with the
│                     │  session-relative timestamp.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Transcript &     │  TranscriptAssembler builds a word-level timeline
│    Context Assembler │  from the Whisper segments and inserts
│                     │  [SCREENSHOT: filename.png] markers at the
│                     │  correct position based on timestamps.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. User Approval     │  The annotated transcript is shown to the user
│                     │  in a review dialog so they can fix any
│                     │  transcription errors before generation.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. LLM Step          │  StepStructurer sends the annotated transcript
│    Structurer        │  to a cloud LLM via tool-calling
│                     │  (emit_documentation), producing structured
│                     │  JSON steps with screenshot placeholders.
│                     │  Strict "extract, don't paraphrase" rules.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 7. VLM Screenshot    │  ScreenshotDescriber runs the VLM in parallel
│    Describer         │  (up to 3 concurrent workers) to write a title
│                     │  and 1–3 sentence description for each
│                     │  screenshot, grounded in the step context.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 8. JSON Doc Merger   │  DocMerger deterministically fuses the
│                     │  StructuredDoc + ScreenshotDescriptions
│                     │  into a single MergedDoc.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 9. Review, Edit &    │  ReviewEditWindow renders a side-by-side
│    Export             │  live preview, lets the user edit any field,
│                     │  and exports to HTML / PDF / Markdown.
└─────────────────────┘
```

---

## System Architecture

DocuFlow follows a strict separation between **Non-AI components** (local, deterministic) and **AI/ML components** (statistical, model-driven), with the System Boundary clearly enforced by the API Gateway.

### Non-AI Components

| Component | Module | Description |
|---|---|---|
| User Interface (Desktop App) | `main.py`, `session/sidebar_panel.py` | PySide6 desktop shell with Start/Stop recording, screenshot trigger, preview, and export controls. |
| Audio Capture | `session/audio_recorder.py` | Records 16kHz mono audio via `sounddevice`, slicing into ~15s chunks on silence/max-size boundaries. |
| Screenshot Capture | `session/screenshot_capture.py`, `session/global_overlay.py` | Captures screen regions with `mss` on manual trigger, stamps with session-relative timestamp. |
| Transcript & Context Assembler | `ai/transcript_assembler.py` | Merges transcript chunks with screenshot timestamps into a single annotated transcript. |
| Session Data Store | Filesystem (`~/DocuFlow/sessions/<timestamp>/`) | Local folder holding audio, screenshots, transcripts, and intermediate JSON — also serves as the manual-fallback resource. |
| JSON Doc Merger | `ai/doc_merger.py` | Purely deterministic merger of LLM JSON + VLM descriptions. |
| Review & Edit UI | `session/review_edit.py` | Side-by-side editable preview with original screenshot visible. |
| Markdown Preview Renderer | `session/review_edit.py` (`_PreviewBrowser`) | Live, debounced (~300ms) Markdown rendering of the edited doc. |
| Manual Editor | `session/review_edit.py` (`_StepCard`) | Inline editing with `AI`/`Edited` badges per field. |
| Export Engine (HTML/PDF/Markdown) | `export/exporter.py` | HTML with embedded base64 images; PDF via `xhtml2pdf`; Markdown via `markdownify` + `BeautifulSoup`. |
| API Gateway (TLS 1.3) | `ai/api_gateway.py` | OpenAI-compatible client wrapper with retry/backoff, 60s timeout, exponential backoff. |
| Logging & Error Handler | `main.py:setup_logging`, all modules | Rotating file handler at `~/DocuFlow/logs/docuflow.log`, max 2MB × 3 backups. |
| Settings & Config | `settings.py` | Dataclass-backed settings persisted to `~/DocuFlow/settings.json`. |
| Local File System | OS | Persists all artifacts. Provides durability for manual fallback. |

### AI / ML Components

| Component | Module | Description |
|---|---|---|
| Speech-to-Text (Whisper — local) | `ai/transcriber.py` | Local `faster-whisper` running Whisper large-v3-turbo (INT8, CPU). 100% on-device. |
| LLM Step Structurer | `ai/step_structurer.py` | Cloud LLM (default: `casperhansen/llama-3.3-70b-instruct-awq` via vLLM endpoint) that converts the transcript into structured JSON steps. |
| Vision-Language Model | `ai/screenshot_describer.py` | Cloud VLM (default: `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`) that writes titles and descriptions for each screenshot. |
| Quality Monitor | `tests/benchmark/test_benchmark.py` | Passive post-generation evaluator computing WER and ROUGE-L. Never modifies output. |

---

## AI Strategy

DocuFlow uses three distinct AI techniques, each justified by the problem it solves:

### 1. Automatic Speech Recognition (ASR) — local Whisper
**Why:** Manual typing is too slow. Cloud STT would violate the privacy constraint (audio must never leave the device).
**Implementation:** `faster-whisper` with the `deepdml/faster-whisper-large-v3-turbo-ct2` model, INT8 quantization, CPU threads tuned to physical core count.
**Threshold:** ≥95% word accuracy (NfReq4).
**Training:** None — uses the pre-trained checkpoint.

### 2. Large Language Model (LLM) — cloud
**Why:** Needed to group messy, spoken instructions into logical, well-segmented steps. The system prompt enforces an *extractive* policy: preserve the speaker's vocabulary and phrasing; only remove filler words and fix grammar/punctuation.
**Implementation:** OpenAI-compatible API (vLLM endpoint by default). Uses tool-calling (`emit_documentation`) to enforce a strict JSON schema.
**Threshold:** ≥0.70 ROUGE-L vs human reference docs (NfReq5).
**Training:** Zero-shot prompting, no fine-tuning.

### 3. Vision-Language Model (VLM) — cloud
**Why:** The user clicks UI elements they don't always name out loud. The VLM "sees" each screenshot in the context of its step and writes a grounded caption describing what's visible and what to verify.
**Implementation:** OpenAI-compatible multimodal API. Images are base64-encoded (PNG, capped at 1536px on the longest side). Up to 3 screenshots described concurrently.
**Threshold:** ≥0.70 ROUGE-L (jointly with the LLM).
**Training:** Zero-shot prompting.

---

## Privacy & Consent

DocuFlow is privacy-first by design:

- **Audio stays local.** Whisper runs entirely on-device. Raw audio is **never** sent to the cloud (NfReq8).
- **Explicit consent.** On first launch, a privacy dialog explains that transcribed text and screenshots will be sent to a cloud LLM/VLM. The app exits if consent is not given. The decision is saved to `~/DocuFlow/settings.json`.
- **TLS 1.3 everywhere.** All cloud calls go through the API Gateway with TLS 1.3 encryption enforced (NfReq9).
- **No telemetry.** DocuFlow does not phone home. The only outbound traffic is the LLM/VLM API calls you explicitly trigger.
- **Manual fallback.** If the cloud APIs fail, the raw transcript, screenshots, and any partial JSON are preserved in the session folder so you can finish the documentation by hand.

---

## Project Structure

```
docuflow/
├── main.py                          # Entry point — logging, consent, device setup, main window
├── settings.py                      # Settings dataclass + JSON persistence
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT
├── README.md                        # This file
│
├── ai/                              # AI/ML components
│   ├── __init__.py
│   ├── api_gateway.py               # OpenAI-compatible client w/ retries, TLS 1.3
│   ├── doc_merger.py                # Deterministic merger of LLM + VLM output
│   ├── languages.py                 # Supported languages and Whisper codes
│   ├── screenshot_describer.py      # VLM call per screenshot (concurrent)
│   ├── step_structurer.py           # LLM call: transcript → structured JSON
│   ├── transcriber.py               # Local Whisper (faster-whisper, INT8 CPU)
│   └── transcript_assembler.py      # Builds word timeline + screenshot markers
│
├── session/                         # UI and recording components
│   ├── __init__.py
│   ├── audio_recorder.py            # QThread recording 16kHz mono audio in chunks
│   ├── device_setup_dialog.py       # Mic picker, language, model preload dialog
│   ├── global_overlay.py            # Full-screen Ctrl+drag screenshot overlay
│   ├── mic_indicator.py             # Animated microphone level widget
│   ├── review_edit.py               # Side-by-side review/edit/export window
│   ├── screenshot_capture.py        # Simple full-screen capture utility
│   └── sidebar_panel.py             # Main recording UI (auto-collapsing sidebar)
│
├── export/                          # Export engine
│   ├── __init__.py
│   └── exporter.py                  # HTML / PDF / Markdown renderers
│
├── scripts/                         # Helper scripts
│   ├── __init__.py
│   ├── download_model.py            # Pull Whisper checkpoint from HuggingFace
│   └── generate_benchmark_steps.py  # Regenerate steps_generated.json for benchmarks
│
└── tests/
    └── benchmark/
        ├── test_benchmark.py        # WER + ROUGE-L evaluation harness
        ├── test_timing.py           # NfReq1 & NfReq2 timing tests
        └── samples/
            └── sample_001 … sample_010/
                ├── recording.wav
                ├── transcript_ground_truth.txt
                ├── transcript_whisper.txt
                ├── steps_ground_truth.json
                └── steps_generated.json
```

---

## Requirements

### Functional Requirements

| ID | Description |
|---|---|
| **Req1** | On "Start Recording", record audio from the system microphone and be ready to capture screenshots on manual command. |
| **Req2** | On manual trigger, save the screenshot with a timestamp. |
| **Req3** | On "Stop Recording", stop audio recording and save the audio file. |
| **Req4** | After recording stops, transcribe the audio locally using a speech-to-text model. |
| **Req5** | After transcription, insert screenshot markers into the transcript using timestamps. |
| **Req6** | Use an LLM to generate structured JSON steps with screenshot placeholders. |
| **Req7** | For each screenshot, use a Vision model to write a title and short description. |
| **Req8** | Merge LLM and VLM outputs into a final unified JSON file. |
| **Req9** | Display an editable Markdown preview after generation. |
| **Req10** | On "Export", save the file as HTML or PDF. |

### Non-Functional Requirements

| ID | Description |
|---|---|
| **NfReq1** | Transcribe up to 5-minute audio within **120 s** after recording stops. |
| **NfReq2** | Process 10 steps (JSON + Vision) within **90 s** during generation. |
| **NfReq3** | Preview updates within **1 s** of any edit. |
| **NfReq4** | ≥**95% word accuracy** for clear English speech. |
| **NfReq5** | ≥**0.70 ROUGE-L** vs human-written reference docs. |
| **NfReq6** | New user can record and export within **5 min**, no training. |
| **NfReq7** | Runs on **Windows 10+**. |
| **NfReq8** | Audio is processed **locally only** — never sent to the cloud. |
| **NfReq9** | All external API calls use **TLS 1.3**. |

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Windows 10+** (primary target; Linux/macOS may work but are not officially supported)
- A working microphone
- ~3 GB free disk space (for the Whisper model)
- Network access to an OpenAI-compatible LLM/VLM endpoint

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/docuflow.git
cd docuflow
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `PySide6` (Qt 6 GUI)
- `faster-whisper` (local Whisper inference)
- `sounddevice`, `numpy` (audio capture/processing)
- `mss`, `pynput` (screen capture, global hotkeys)
- `openai` (LLM/VLM client)
- `Pillow` (image preprocessing)
- `xhtml2pdf`, `markdownify`, `beautifulsoup4` (export pipeline)
- `huggingface_hub` (model download)
- `psutil` (CPU thread tuning)
- `jiwer`, `rouge-score`, `pytest` (benchmark & testing)

### Step 4 — Download the Whisper model

The Whisper checkpoint is **not** included in the repository (it's gitignored). Pull it once with:

```bash
python scripts/download_model.py
```

This downloads `deepdml/faster-whisper-large-v3-turbo-ct2` (~1.5 GB) into `models/whisper-large-v3-turbo/`. The transcriber will refuse to start if these required files are missing: `model.bin`, `config.json`, `tokenizer.json`.

---

## Usage

### Launch DocuFlow

```bash
python main.py
```

### First-run consent

On first launch, you'll see a privacy dialog explaining that audio is processed locally, and that transcribed text + screenshots will be sent to a cloud LLM/VLM via TLS 1.3. Click **Yes** to proceed (the choice is remembered).

### Device Setup dialog

1. **Microphone** — pick an input device from the dropdown. Click **Test Mic** to verify the level meter responds.
2. **Speech Recognition Model** — the Whisper large-v3-turbo model is loaded asynchronously in the background. The Start button is disabled until loading completes (typically 5–15 s on a modern CPU).
3. **Documentation Language** — choose the output language for the generated documentation (12 supported).
4. Click **Start Session →**.

### Recording a session

- A thin sidebar appears on the right edge of your screen with a session timer, a microphone level indicator, and a thumbnail of your latest screenshot.
- The sidebar **auto-expands** when you hover over it and **collapses** when you move away, so it doesn't block your work.
- **Hold Ctrl** to dim the screen and **drag-select** any region to capture a screenshot. Release Ctrl to return to normal.
- Speak your workflow aloud. The transcript preview updates as Whisper transcribes each ~15 s chunk.

### Stopping and generating

1. Click **■ Stop Session**.
2. The remaining audio is transcribed (~60 s, capped at 120 s — NfReq1).
3. A **Transcript Approval** dialog appears showing the full annotated transcript with `[SCREENSHOT: filename.png]` markers. Fix any transcription errors here, then click **Generate Documentation →**.
4. The LLM structures the transcript and the VLM describes each screenshot (≤ 90 s — NfReq2).
5. The **Review & Edit** window opens.

### Review & Edit

- Each step is a card with editable **Title**, **Instruction**, **Image Title**, and **Image Description** fields.
- Each card is badged **AI** (green) until you edit it, then **Edited** (orange). This makes it obvious what's been verified.
- The original screenshot thumbnail is shown on the card so you can directly compare the AI's caption against the actual image.
- A **Preview** tab on the right renders live Markdown (debounced at 300 ms — well within NfReq3's 1 s budget).
- An **Original transcript** tab shows the raw transcript for cross-reference.

### Export

Click any of the export buttons at the bottom of the Review window:

- **Export HTML** — single self-contained `.html` with base64-embedded images.
- **Export PDF** — rendered via `xhtml2pdf` from the same HTML.
- **Export Markdown** — `.md` file with relative image links (images stay in the session folder).

Files are saved by default into the session folder at `~/DocuFlow/sessions/<timestamp>/`.

### Where files live

```
~/DocuFlow/
├── settings.json              # Persistent settings
├── logs/
│   └── docuflow.log           # Rotating log (2MB × 3 backups)
└── sessions/
    └── 20260115_143022/       # One folder per session
        ├── full_audio.wav     # Complete recording
        ├── chunk_000_*.wav    # Individual ~15s chunks
        ├── chunk_001_*.wav
        ├── screenshot_*.png   # Captured screenshots
        ├── transcript.txt     # Final annotated transcript
        ├── steps.json         # LLM-structured steps
        ├── documentation.json # Merged final document
        └── llm_raw_output_*.txt  # (Only if LLM JSON parsing failed)
```

---

## Configuration

Settings are stored in `~/DocuFlow/settings.json` and editable through the `Settings` dataclass in `settings.py`:

| Field | Default | Description |
|---|---|---|
| `consent_given` | `False` | Whether the user has accepted the data policy. |
| `output_dir` | `~/DocuFlow/sessions` | Where session folders are created. |
| `documentation_language` | `English` | Output language for the generated docs. |
| `llm_model` | `casperhansen/llama-3.3-70b-instruct-awq` | Model ID for the step structurer. |
| `vlm_model` | `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` | Model ID for the screenshot describer. |
| `api_base_url` | `` | OpenAI-compatible endpoint base URL. |
| `api_key` | `` | API key (passed as Bearer token). |

> **Note:** To use OpenAI, Anthropic, or another provider, change `api_base_url`, `api_key`, and the model IDs. The `ApiGateway` is OpenAI-compatible and works with any provider that supports the chat completions API and tool-calling.

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Hold Ctrl** | Activate the screenshot overlay. Screen dims; drag to select a region. Release to capture. |
| **Ctrl + Alt + Q** | Emergency kill switch — immediately exits the application. |

---

## Export Formats

All three formats are rendered from the same `MergedDoc` and embed images so the exported file is self-contained (HTML/PDF) or portable with its session folder (Markdown).

### HTML
- Single `.html` file with images embedded as `data:image/...;base64,...` URIs.
- Images are downscaled to ≤1400 px wide for portability.
- Inline CSS gives a clean, GitHub-ish look.

### PDF
- Rendered from the same HTML via `xhtml2pdf` (pisa).
- Suitable for printing or sharing.

### Markdown
- Generated by `markdownify`-ing the HTML body, then normalizing whitespace.
- Image links are relative (`![alt](screenshot_xxx.png)`) so the `.md` file travels with the session folder.

---

## Benchmark & Evaluation

DocuFlow ships with a 10-workflow benchmark suite located in `tests/benchmark/samples/sample_001 … sample_010`. Each sample contains:

| File | Description |
|---|---|
| `recording.wav` | Source audio (5–10 min tutorial). |
| `transcript_ground_truth.txt` | Human-written reference transcript. |
| `transcript_whisper.txt` | Whisper's transcription of the same audio. |
| `steps_ground_truth.json` | Human-written reference step-by-step documentation. |
| `steps_generated.json` | DocuFlow's generated documentation (regeneratable). |

Samples cover diverse workflows: creating a GitHub repo, opening a PR, installing Prettier in VS Code, configuring Git, forking a repo, creating a Python venv, writing a Python script, pushing to a remote, resolving a merge conflict, and writing a Dockerfile.

### Regenerating `steps_generated.json`

If you change the LLM model, prompt, or settings and want to refresh the benchmark outputs:

```bash
python scripts/generate_benchmark_steps.py --all
# or
python scripts/generate_benchmark_steps.py --sample sample_001 --force
```

The script uses your current `Settings` and skips samples that already have a `steps_generated.json` unless you pass `--force`.

---

## Testing

### Quality benchmark (NfReq4 & NfReq5)

```bash
python tests/benchmark/test_benchmark.py
```

This computes:

- **Table 1 — Transcription Accuracy (WER):** Uses `jiwer` to compute Word Error Rate between `transcript_ground_truth.txt` and `transcript_whisper.txt`, with lowercasing, punctuation removal, and whitespace normalization. Reports accuracy as `1 - WER` with a `PERFECT / GOOD / MODERATE / BAD` rating per sample (thresholds: 0.95 / 0.85 / 0.75).
- **Table 2 — Documentation Quality (ROUGE-L):** Uses `rouge_score` to compute ROUGE-L F-measure between flattened ground-truth steps and generated steps (title + instruction concatenated). Same rating thresholds (0.70 / 0.50 / 0.30).

### Timing tests (NfReq1 & NfReq2)

```bash
pytest tests/benchmark/test_timing.py -v
```

- `test_nf_req1_transcription_speed` — picks the longest sample audio, simulates real-time chunk submission at 15 s intervals, then asserts the post-stop drain time is **≤120 s**.
- `test_nf_req2_generation_speed` — picks the sample with the most ground-truth steps, runs the full `StepStructurer` pipeline, and asserts elapsed time is **≤90 s**. Skipped automatically if the LLM API is unreachable.

---

## Scripts

### `scripts/download_model.py`
Downloads the Whisper large-v3-turbo CT2 model from HuggingFace Hub into `models/whisper-large-v3-turbo/`. Run this once after installation.

### `scripts/generate_benchmark_steps.py`
Regenerates `steps_generated.json` for one or all benchmark samples using the current LLM config. Useful for re-running the benchmark after prompt or model changes. Supports `--sample <id>`, `--all`, and `--force` flags.


---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **`Model directory not found` on startup** | Run `python scripts/download_model.py`. |
| **`Model directory is incomplete`** | Some files in `models/whisper-large-v3-turbo/` are missing. Re-run `download_model.py`. |
| **Sidebar doesn't appear** | Check `~/DocuFlow/logs/docuflow.log` for Qt errors. Make sure no other on-top window is blocking the right screen edge. |
| **Screenshot overlay doesn't activate on Ctrl** | Make sure the DocuFlow window has focus at least once after launch. The `pynput` listener runs globally, but some Linux Wayland sessions may need additional permissions. |
| **LLM/VLM API errors** | Verify `api_base_url` and `api_key` in `~/DocuFlow/settings.json`. The Gateway retries 3× with exponential backoff before failing. |
| **`ApiGatewayError: LLM returned unparseable documentation after retries`** | The LLM kept returning malformed JSON. Check `llm_raw_output_*.txt` in the session folder. Try a different `llm_model` in settings. |
| **PDF export fails** | `xhtml2pdf` can struggle with very large images or unusual fonts. Try HTML export instead; the underlying HTML is identical. |
| **Whisper is slow** | Make sure `psutil` is installed so DocuFlow can detect physical (not logical) core count. INT8 on CPU should run at roughly 0.2–0.5× real-time for clear speech. |

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

```
MIT License
Copyright (c) 2026 Davyd Pizhuk & David Radonic
```

---

## Authors

**David Radonic** — k11906471
**Davyd Pizhuk** — k12148477

---

*Just talk. We'll write the docs.*