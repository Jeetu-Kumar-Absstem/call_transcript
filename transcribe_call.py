"""
ABSSTEM AI Call Intelligence System
Path A — API Only (no local model loading)

ASR  : openai/whisper-large-v3  via HuggingFace Inference API
LLM  : mistralai/Mistral-7B-Instruct-v0.3 via HuggingFace Inference API

Usage:
    python transcribe_call.py "https://telephonycloud.co.in/recordings/..."

Setup:
    1. pip install -r requirements.txt
    2. Get HuggingFace token from https://huggingface.co/settings/tokens
    3. Set environment variable:
         Windows : set HF_TOKEN=hf_xxxxxxxxxxxx
         Linux   : export HF_TOKEN=hf_xxxxxxxxxxxx
    4. Run the script
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import requests
import torchaudio
import torch

# Load .env file automatically if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will fall back to system env variable

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
HF_API_URL = "https://api-inference.huggingface.co/models"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "\n\nHF_TOKEN environment variable is not set.\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Create a token with READ permission\n"
            "3. Then run:\n"
            "     Windows : set HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "     Linux   : export HF_TOKEN=hf_xxxxxxxxxxxx\n"
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
            f"Check that the TelephonyCloud URL is correct and accessible."
        )

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  ✓ Downloaded — {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Step 2 — Inspect audio
# ---------------------------------------------------------------------------

def inspect_audio(path: str) -> dict:
    info       = torchaudio.info(path)
    size_mb    = os.path.getsize(path) / (1024 * 1024)
    duration   = info.num_frames / info.sample_rate
    mins, secs = divmod(int(duration), 60)

    print(f"\n  Audio info:")
    print(f"    Duration    : {mins:02d}:{secs:02d}")
    print(f"    Sample rate : {info.sample_rate:,} Hz")
    print(f"    Channels    : {info.num_channels}")
    print(f"    Size        : {size_mb:.2f} MB")

    if duration < 3:
        logger.warning("Audio is very short — check the URL returns a real call recording.")

    return {
        "duration_sec" : duration,
        "sample_rate"  : info.sample_rate,
        "num_channels" : info.num_channels,
        "size_mb"      : size_mb,
    }


# ---------------------------------------------------------------------------
# Step 3 — Preprocess: mono + 16 kHz
# (runs on your CPU, lightweight — no GPU needed)
# ---------------------------------------------------------------------------

def preprocess_audio(src_path: str, dest_path: str) -> str:
    print("\nStep 2: Preprocessing audio ...")

    waveform, sr = torchaudio.load(src_path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        logger.info("Converted stereo → mono")

    # Resample to 16 kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform  = resampler(waveform)
        logger.info(f"Resampled {sr} Hz → 16000 Hz")

    torchaudio.save(dest_path, waveform, 16000)
    print("  ✓ Audio ready (mono, 16 kHz)")
    return dest_path


# ---------------------------------------------------------------------------
# Step 4 — ASR via HuggingFace API (Whisper large-v3)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, hf_token: str) -> str:
    print(f"\nStep 3: Transcribing via HuggingFace API ({ASR_MODEL}) ...")
    print("  (this may take 1–3 minutes depending on call length)")

    url     = f"{HF_API_URL}/{ASR_MODEL}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Send with language hint for Hindi + return timestamps disabled
    params = {
        "wait_for_model": True,          # wait if model is loading, don't fail immediately
        "language": "hi",               # hint: Hindi (Whisper auto-detects but this helps)
        "task": "transcribe",           # transcribe, not translate
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=audio_bytes,
            params=params,
            timeout=300,                # 5 min timeout for long calls
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Request timed out. The call audio may be very long.\n"
            "Try a shorter test clip first."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")

    if response.status_code == 401:
        raise RuntimeError(
            "HuggingFace API returned 401 Unauthorized.\n"
            "Check your HF_TOKEN is correct and has READ permission."
        )
    if response.status_code == 503:
        raise RuntimeError(
            "HuggingFace API returned 503 — model is loading on their servers.\n"
            "Wait 30 seconds and try again. This happens on first call of the day."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace ASR API error {response.status_code}:\n{response.text}"
        )

    result = response.json()

    # Whisper API returns {"text": "..."}
    if isinstance(result, dict) and "text" in result:
        transcript = result["text"].strip()
    elif isinstance(result, list) and len(result) > 0:
        # Some versions return a list of chunks
        transcript = " ".join(chunk.get("text", "") for chunk in result).strip()
    else:
        raise RuntimeError(f"Unexpected API response format:\n{result}")

    print("  ✓ Transcription received")
    return transcript


# ---------------------------------------------------------------------------
# Step 5 — LLM Extraction via HuggingFace API (Mistral)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an AI assistant for ABSSTEM Technologies, a company that sells nitrogen plants, oxygen plants, and gas generation systems.

Below is a raw transcript of a sales call. The transcript may be in Hindi, English, or Hinglish (mixed Hindi-English).

Your job is to extract structured information from this transcript and return ONLY a valid JSON object. Do not add any explanation or text outside the JSON.

If a field is not mentioned in the transcript, set it to null.

Transcript:
{transcript}

Return this exact JSON structure:
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
    print(f"\nStep 4: Extracting information via HuggingFace API ({LLM_MODEL}) ...")

    url     = f"{HF_API_URL}/{LLM_MODEL}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type" : "application/json",
    }

    prompt = EXTRACTION_PROMPT.format(transcript=transcript)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens"  : 600,
            "temperature"     : 0.1,    # low temp = more deterministic JSON
            "return_full_text": False,  # return only new tokens, not the prompt
        },
        "options": {
            "wait_for_model": True,
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM API request failed: {e}")

    if response.status_code != 200:
        logger.warning(
            f"LLM API returned {response.status_code} — skipping extraction.\n"
            f"Raw response: {response.text[:300]}"
        )
        return {}

    result = response.json()

    # Mistral returns [{"generated_text": "..."}]
    if isinstance(result, list) and len(result) > 0:
        raw_text = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        raw_text = result.get("generated_text", "")
    else:
        logger.warning(f"Unexpected LLM response format: {result}")
        return {}

    # Parse JSON from the response
    try:
        # Find the JSON block in the response
        start = raw_text.find("{")
        end   = raw_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in LLM response")
        json_str    = raw_text[start:end]
        extracted   = json.loads(json_str)
        print("  ✓ Information extracted")
        return extracted
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse LLM JSON output: {e}")
        logger.warning(f"Raw LLM output was:\n{raw_text[:500]}")
        return {}


# ---------------------------------------------------------------------------
# Step 6 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(transcript: str, extracted: dict, base_name: str = "output") -> None:
    # Save raw transcript
    txt_path = f"{base_name}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"\n  Transcript saved : {os.path.abspath(txt_path)}")

    # Save extracted JSON
    if extracted:
        json_path = f"{base_name}_extracted.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        print(f"  Extracted JSON   : {os.path.abspath(json_path)}")


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
        inspect_audio(raw_wav)

        # 3. Preprocess
        preprocess_audio(raw_wav, processed_wav)

        # 4. Transcribe via API
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
        print("\n" + "=" * 50)
        print("  SAVING OUTPUT")
        print("=" * 50)
        save_outputs(transcript, extracted)

    print("\n✓ Done.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ABSSTEM V1 — Transcribe TelephonyCloud call via HuggingFace API"
    )
    parser.add_argument(
        "url",
        help="TelephonyCloud WAV URL"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only transcribe, skip LLM extraction step"
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