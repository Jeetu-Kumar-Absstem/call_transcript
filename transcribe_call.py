"""
ABSSTEM AI Call Intelligence System
Path A — API Only (no local model loading)

ASR : openai/whisper-large-v3  via HuggingFace Inference API
LLM : mistralai/Mistral-7B-Instruct-v0.3 via HuggingFace Inference API

Usage:
    python transcribe_call.py "https://telephonycloud.co.in/recordings/..."
    python transcribe_call.py "https://..." --skip-llm

Setup:
    1. pip install -r requirements.txt
    2. In PowerShell: $env:HF_TOKEN="hf_xxxxxxxxxxxx"
    3. Run the script
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import requests
from huggingface_hub import InferenceClient
import soundfile as sf
import numpy as np
import torch
import torchaudio

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("absstem")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASR_MODEL  = "openai/whisper-large-v3"
LLM_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "\nHF_TOKEN is not set.\n"
            "In PowerShell run:\n"
            '  $env:HF_TOKEN="hf_your_token_here"\n'
            "Then run the script again."
        )
    return token


# ---------------------------------------------------------------------------
# Step 1 — Download WAV
# ---------------------------------------------------------------------------

def download_audio(url: str, dest_path: str) -> None:
    print("\nStep 1: Downloading audio ...")
    logger.info(f"URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} — could not download audio.\n"
            "Check the TelephonyCloud URL is correct."
        )

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  Downloaded — {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Step 2 — Inspect audio using soundfile
# ---------------------------------------------------------------------------

def inspect_audio(path: str) -> dict:
    info     = sf.info(path)
    size_mb  = os.path.getsize(path) / (1024 * 1024)
    duration = info.duration
    mins, secs = divmod(int(duration), 60)

    print(f"\n  Audio info:")
    print(f"    Duration    : {mins:02d}:{secs:02d}")
    print(f"    Sample rate : {info.samplerate:,} Hz")
    print(f"    Channels    : {info.channels}")
    print(f"    Format      : {info.format}")
    print(f"    Size        : {size_mb:.2f} MB")

    if duration < 3:
        logger.warning("Audio is very short — check the URL returns a real call recording.")

    return {
        "duration_sec" : duration,
        "sample_rate"  : info.samplerate,
        "num_channels" : info.channels,
        "size_mb"      : size_mb,
    }


# ---------------------------------------------------------------------------
# Step 3 — Preprocess: mono + 16 kHz using soundfile + numpy
# No torchaudio.info needed — just load and resample
# ---------------------------------------------------------------------------

def preprocess_audio(src_path: str, dest_path: str) -> str:
    print("\nStep 2: Preprocessing audio ...")

    # Read audio using soundfile
    data, sr = sf.read(src_path, always_2d=True)  # shape: (frames, channels)

    # Stereo -> mono (average channels)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
        logger.info("Converted stereo -> mono")
    else:
        data = data[:, 0]

    # Resample to 16000 Hz if needed
    if sr != 16000:
        # Use torchaudio resampler
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        tensor = resampler(tensor)
        data = tensor.squeeze(0).numpy()
        logger.info(f"Resampled {sr} Hz -> 16000 Hz")

    # Save as 16-bit PCM WAV
    sf.write(dest_path, data, 16000, subtype="PCM_16")
    print("  Audio ready (mono, 16 kHz)")
    return dest_path


# ---------------------------------------------------------------------------
# Step 4 — ASR via HuggingFace API (Whisper large-v3)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, hf_token: str) -> str:
    print(f"\nStep 3: Transcribing via HuggingFace InferenceClient ...")
    print(f"  Model : {ASR_MODEL}")
    print(f"  Note  : First call may take 1-2 min if model is cold-starting")

    try:
        client = InferenceClient(
            provider="auto",
            api_key=hf_token,
        )
        result = client.automatic_speech_recognition(
            audio_path,
            model=ASR_MODEL,
        )
        transcript = result.text.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err:
            raise RuntimeError(
                "HF API returned 401 Unauthorized.\n"
                "Your token may be wrong or expired.\n"
                "Check: https://huggingface.co/settings/tokens"
            )
        if "503" in err:
            raise RuntimeError(
                "HF API returned 503 — model is loading on their servers.\n"
                "Wait 30 seconds and try again."
            )
        raise RuntimeError(f"ASR request failed: {e}")

    print("  Transcription received")
    return transcript


# ---------------------------------------------------------------------------
# Step 5 — LLM Extraction via HuggingFace API (Mistral)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an AI assistant for ABSSTEM Technologies, which sells nitrogen plants, oxygen plants, and gas generation systems.

Below is a raw transcript of a sales call. It may be in Hindi, English, or Hinglish (mixed).

Extract structured information and return ONLY a valid JSON object. No explanation outside the JSON.
If a field is not mentioned, set it to null.

Transcript:
{transcript}

Return exactly this JSON:
{{
  "customer": {{
    "name": null,
    "company": null,
    "location": null
  }},
  "requirement": {{
    "gas": null,
    "plant_type": null,
    "application": null,
    "capacity": null,
    "capacity_unit": null,
    "purity": null,
    "pressure": null
  }},
  "existing_setup": {{
    "current_source": null,
    "consumption": null
  }},
  "commercial": {{
    "budget": null,
    "timeline": null,
    "purchase_intent": null
  }},
  "follow_up": {{
    "required": false,
    "next_action": null
  }},
  "summary": null
}}"""


def extract_information(transcript: str, hf_token: str) -> dict:
    print(f"\nStep 4: Extracting information via LLM API ...")
    print(f"  Model : {LLM_MODEL}")

    try:
        client = InferenceClient(
            provider="auto",
            api_key=hf_token,
        )
        response = client.text_generation(
            prompt=EXTRACTION_PROMPT.format(transcript=transcript),
            model=LLM_MODEL,
            max_new_tokens=600,
            temperature=0.1,
        )
        raw_text = response if isinstance(response, str) else str(response)
    except Exception as e:
        err = str(e)
        if LLM_MODEL not in err and ("not available" in err.lower() or "not supported" in err.lower() or "no provider" in err.lower()):
            logger.warning(
                f"LLM model '{LLM_MODEL}' is not available through any provider. "
                f"Original error: {e}"
            )
        else:
            logger.warning(f"LLM API request failed: {e}")
        return {}

    try:
        start  = raw_text.find("{")
        end    = raw_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON in LLM response")
        extracted = json.loads(raw_text[start:end])
        print("  Extraction complete")
        return extracted
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse LLM JSON: {e}")
        logger.warning(f"Raw LLM output:\n{raw_text[:400]}")
        return {}


# ---------------------------------------------------------------------------
# Step 6 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(transcript: str, extracted: dict) -> None:
    print("\n" + "=" * 50)
    print("  SAVING OUTPUT")
    print("=" * 50)

    txt_path = "output_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"  Transcript : {os.path.abspath(txt_path)}")

    if extracted:
        json_path = "output_extracted.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        print(f"  Extracted  : {os.path.abspath(json_path)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(wav_url: str, skip_llm: bool = False) -> None:
    print("\n" + "=" * 50)
    print("  ABSSTEM CALL TRANSCRIPTION  —  Path A (API)")
    print("=" * 50)

    hf_token = get_hf_token()

    with tempfile.TemporaryDirectory(prefix="absstem_") as tmpdir:
        raw_wav       = os.path.join(tmpdir, "original.wav")
        processed_wav = os.path.join(tmpdir, "processed.wav")

        # 1. Download
        download_audio(wav_url, raw_wav)

        # 2. Inspect
        meta = inspect_audio(raw_wav)

        # 3. Preprocess
        preprocess_audio(raw_wav, processed_wav)

        # 4. Transcribe
        transcript = transcribe_audio(processed_wav, hf_token)

        # 5. Print transcript
        print("\n" + "=" * 50)
        print("  RAW TRANSCRIPT")
        print("=" * 50)
        print()
        print(transcript if transcript else "[No speech detected]")
        print()

        # 6. LLM extraction
        extracted = {}
        if not skip_llm and transcript:
            extracted = extract_information(transcript, hf_token)
            if extracted:
                print("\n" + "=" * 50)
                print("  EXTRACTED INFORMATION")
                print("=" * 50)
                print(json.dumps(extracted, indent=2, ensure_ascii=False))

        # 7. Save
        save_outputs(transcript, extracted)

    print("\nDone.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ABSSTEM — Transcribe TelephonyCloud call via HuggingFace API"
    )
    parser.add_argument("url", help="TelephonyCloud WAV URL")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only transcribe, skip LLM extraction"
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.url, skip_llm=args.skip_llm)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)


if __name__ == "__main__":
    main()