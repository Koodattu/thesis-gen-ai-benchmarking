import os
import time
import gc
import json
import threading
import psutil
from openai import OpenAI
from datetime import datetime
import wave
import torch
from pydantic import BaseModel
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

os.makedirs("audio/english", exist_ok=True)
os.makedirs("audio/finnish", exist_ok=True)
os.makedirs("references/english", exist_ok=True)
os.makedirs("references/finnish", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Try to import pynvml for GPU VRAM monitoring (optional)
import pynvml

# Import the two Whisper implementations
import whisper  # official OpenAI Whisper package
from faster_whisper import WhisperModel as FasterWhisperModel, BatchedInferencePipeline

import logging

# -----------------------------------------------------------------------------
# Logger Setup: all events, memory usage, and status changes go into a single file.
# -----------------------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    # Remove any previously attached handlers.
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# We'll initialize this later in main() after creating the results directory.
logger = None

# Helper: format bytes to gigabytes.
def format_bytes(num_bytes):
    gb = num_bytes / (1024**3)
    return f"{gb:.2f} GB"

# -----------------------------------------------------------------------------
# Memory Logger Thread
# -----------------------------------------------------------------------------
class MemoryLogger(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.running = True
        self.status = "unloaded"  # status: unloaded, loaded, processing
        pynvml.nvmlInit()

    def run(self):
        global logger
        while self.running:
            timestamp = datetime.now().isoformat()
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss  # in bytes
            gpu_usage = None
            if pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_usage = gpu_mem.used
                except Exception:
                    gpu_usage = None
            # Log a single line that includes memory usage in a human-friendly format and current status.
            logger.info(f"Memory Usage: RAM={format_bytes(ram_usage)}, GPU={format_bytes(gpu_usage) if gpu_usage is not None else 'N/A'} | Status: {self.status}")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if pynvml:
            pynvml.nvmlShutdown()

# -----------------------------------------------------------------------------
# Accuracy Evaluation via ChatGPT API
# -----------------------------------------------------------------------------
def evaluate_accuracy(reference_text, output_text, task="transcription"):
    """
    Calls the ChatGPT API (gpt-4o-mini) to compare output with reference.
    Returns a numeric score (0-100) if possible.
    """
    class Accuracy(BaseModel):
        accuracy: float

    prompt = (f"Compare the following {task} output to the reference text. "
              "Provide an accuracy score between 0 (worst) and 100 (perfect match).\n\n"
              f"Reference:\n{reference_text}\n\nOutput:\n{output_text}\n\nScore:")
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in text similarity evaluation."},
                {"role": "user", "content": prompt}
            ],
            response_format=Accuracy,
            max_tokens=10,
            temperature=0.0
        )
        score = response.choices[0].message.parsed.accuracy
    except Exception as e:
        print("Accuracy evaluation error:", e)
        score = None
    return score

# -----------------------------------------------------------------------------
# Benchmark Test Functions for each method
# -----------------------------------------------------------------------------
def test_local_whisper(model_size, audio_file, task="transcribe", language="en"):
    """
    Uses the local whisper package.
    model_size: e.g. "tiny", "base", "small", "medium", "large", "turbo"
    task: "transcribe" (default) or "translate"
    Returns: (output_text, latency_seconds, model_load_time)
    """
    logger.info(f"Loading local Whisper model '{model_size}' on device cuda for language '{language}'")
    start_load = time.time()
    model = whisper.load_model(model_size, device="cuda")
    load_time = time.time() - start_load
    logger.info(f"Model loaded in {load_time:.2f} seconds")

    # Warmup call
    model.transcribe(audio_file, language=language, temperature=0.0)
    
    logger.info(f"Starting transcription on file '{os.path.basename(audio_file)}' (language: {language})")
    start_time = time.time()
    if task == "transcribe":
        result = model.transcribe(audio_file, language=language, temperature=0.0)
    elif task == "translate":
        result = model.transcribe(audio_file, task="translate", language=language, temperature=0.0)
    else:
        result = {}
    latency = time.time() - start_time
    output_text = result.get("text", "").strip()
    logger.info(f"Finished transcription. Latency: {latency:.2f} seconds. Output length: {len(output_text)} characters")

    # Unload model
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Unloaded local Whisper model '{model_size}'")
    return output_text, latency, load_time

def test_faster_whisper(model_size, audio_file, task="transcribe", language="en"):
    """
    Uses the faster-whisper package.
    task: "transcribe" or "translate"
    Returns: (output_text, latency_seconds, model_load_time)
    """
    logger.info(f"Loading Faster-Whisper model '{model_size}' on device cuda for language '{language}'")
    start_load = time.time()
    model = FasterWhisperModel(model_size, device="cuda", compute_type="float16")
    load_time = time.time() - start_load
    logger.info(f"Faster-Whisper model loaded in {load_time:.2f} seconds")

    # Warmup
    list(model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0))
    
    logger.info(f"Starting Faster-Whisper transcription on file '{os.path.basename(audio_file)}'")
    start_time = time.time()
    if task == "transcribe":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0)
    elif task == "translate":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, task="translate", temperature=0.0)
    else:
        segments = []
    latency = time.time() - start_time
    output_text = " ".join([seg.text for seg in segments]).strip() if segments else ""
    logger.info(f"Finished Faster-Whisper transcription. Latency: {latency:.2f} seconds. Output length: {len(output_text)} characters")

    # Unload model
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Unloaded Faster-Whisper model '{model_size}'")
    return output_text, latency, load_time

def test_faster_whisper_batched(model_size, audio_file, task="transcribe", language="en"):
    """
    Uses the faster-whisper package with batching.
    task: "transcribe" or "translate"
    Returns: (output_text, latency_seconds, model_load_time)
    """
    logger.info(f"Loading Faster-Whisper batched model '{model_size}' on device cuda for language '{language}'")
    start_load = time.time()
    model = FasterWhisperModel(model_size, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model)
    load_time = time.time() - start_load
    logger.info(f"Batched model loaded in {load_time:.2f} seconds")

    # Warmup
    list(model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0))
    
    logger.info(f"Starting batched Faster-Whisper transcription on file '{os.path.basename(audio_file)}'")
    start_time = time.time()
    if task == "transcribe":
        segments, _ = batched_model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0, batch_size=16)
    elif task == "translate":
        segments, _ = batched_model.transcribe(audio_file, beam_size=5, language=language, task="translate", temperature=0.0, batch_size=16)
    else:
        segments = []
    latency = time.time() - start_time
    output_text = " ".join([seg.text for seg in segments]).strip() if segments else ""
    logger.info(f"Finished batched Faster-Whisper transcription. Latency: {latency:.2f} seconds. Output length: {len(output_text)} characters")

    # Unload model and batched model
    del model
    del batched_model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Unloaded batched Faster-Whisper model '{model_size}'")
    return output_text, latency, load_time

def test_rest_api(audio_file, task="transcribe"):
    """
    Uses the OpenAI REST API.
    task: "transcribe" or "translate"
    Returns: (output_text, latency_seconds)
    """
    logger.info(f"Starting REST API transcription on file '{os.path.basename(audio_file)}'")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.time()
    with open(audio_file, "rb") as f:
        if task == "transcribe":
            result = openai_client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="json"
            )
        elif task == "translate":
            result = openai_client.audio.translations.create(
                model="whisper-1", file=f, response_format="json"
            )
        else:
            result = {}
    latency = time.time() - start_time
    output_text = result.text.strip()
    logger.info(f"Finished REST API transcription. Latency: {latency:.2f} seconds. Output length: {len(output_text)} characters")
    return output_text, latency

# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------
def run_benchmark(method, model_size, audio_file, task, lang="en", n_runs=3):
    """
    Runs the test n_runs times and averages the latency.
    method: "local", "faster", "fasterbatched", or "rest"
    For "local", "faster" and "fasterbatched", model_size (e.g., "tiny") is used; for "rest", it is ignored.
    Returns a dict with:
      - avg_latency
      - list of latencies
      - output_text (from the last run)
      - model_load_time (if applicable)
    """
    latencies = []
    load_time = None
    output_text = ""
    language = lang[:2].lower()
    for i in range(n_runs):
        if method == "local":
            out, lat, ltime = test_local_whisper(model_size, audio_file, task, language)
            load_time = ltime
        elif method == "faster":
            out, lat, ltime = test_faster_whisper(model_size, audio_file, task, language)
            load_time = ltime
        elif method == "fasterbatched":
            out, lat, ltime = test_faster_whisper_batched(model_size, audio_file, task, language)
            load_time = ltime
        elif method == "rest":
            out, lat = test_rest_api(audio_file, task)
        else:
            out, lat = "", 0
        latencies.append(lat)
        output_text = out
    avg_latency = sum(latencies) / len(latencies)
    return {
        "avg_latency": avg_latency,
        "latencies": latencies,
        "output_text": output_text,
        "load_time": load_time
    }

# -----------------------------------------------------------------------------
# Audio duration retrieval
# -----------------------------------------------------------------------------
def get_audio_duration(file_path):
    """
    Returns the duration of the audio file (in seconds).
    If the file is a WAV file, uses the built-in wave module.
    Otherwise, if pydub is available, uses AudioSegment.
    If neither works, returns None.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".wav":
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration
        elif AudioSegment is not None:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        else:
            logger.error(f"Cannot determine duration for {file_path} (unsupported format and no pydub available)")
            return None
    except Exception as e:
        logger.error(f"Error reading duration for {file_path}: {e}")
        return None

# -----------------------------------------------------------------------------
# Main Benchmarking Script
# -----------------------------------------------------------------------------
def main():
    global logger
    # Define directories (modify as needed)
    audio_dir = "./audio"
    ref_dir = "./references"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logger to write to a single log file in the results directory.
    logger = setup_logger(os.path.join(results_dir, "benchmark.log"))
    logger.info("Benchmarking started.")

    # Start memory logger thread
    mem_logger = MemoryLogger(interval=1.0)
    mem_logger.start()

    # Define languages and determine tasks:
    # For English: only transcription; for Finnish: both transcription and translation.
    languages = ["english", "finnish"]
    methods = ["local", "faster", "fasterbatched", "rest"]
    model_sizes = ["tiny", "base", "small", "medium", "large", "turbo", "large-v3"]

    summary = []  # to collect all benchmark results

    for lang in languages:
        audio_path = os.path.join(audio_dir, lang)
        ref_path = os.path.join(ref_dir, lang)
        for filename in os.listdir(audio_path):
            if filename.lower().endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(audio_path, filename)
                duration_seconds = get_audio_duration(file_path)
                if duration_seconds is None:
                    logger.error(f"Skipping {filename} due to duration extraction error.")
                    continue
                logger.info(f"Processing file '{filename}' with duration {duration_seconds:.2f} seconds.")

                # Determine tasks: English files get only transcription; Finnish get both.
                tasks = ["transcribe"] if lang == "english" else ["transcribe", "translate"]
                for task in tasks:
                    for method in methods:
                        if method in ["local", "faster", "fasterbatched"]:
                            for model_size in model_sizes:
                                logger.info(f"Method '{method}' (model: {model_size}) on '{filename}' for task '{task}'")
                                result = run_benchmark(method, model_size, file_path, task, lang, n_runs=3)
                                result_filename = f"{lang}_{filename}_{task}_{method}_{model_size}.json"
                                with open(os.path.join(results_dir, result_filename), "w") as rf:
                                    json.dump(result, rf, indent=2)
                                summary.append({
                                    "language": lang,
                                    "file": filename,
                                    "task": task,
                                    "method": method,
                                    "model_size": model_size,
                                    "avg_latency": result["avg_latency"],
                                    "load_time": result["load_time"],
                                    "duration": duration_seconds
                                })
                        elif method == "rest":
                            logger.info(f"Method '{method}' on '{filename}' for task '{task}'")
                            result = run_benchmark(method, None, file_path, task, n_runs=3)
                            result_filename = f"{lang}_{filename}_{task}_{method}.json"
                            with open(os.path.join(results_dir, result_filename), "w") as rf:
                                json.dump(result, rf, indent=2)
                            summary.append({
                                "language": lang,
                                "file": filename,
                                "task": task,
                                "method": method,
                                "model_size": "whisper-1",
                                "avg_latency": result["avg_latency"],
                                "load_time": None,
                                "duration": duration_seconds
                            })

    # Save summary before accuracy evaluation
    with open(os.path.join(results_dir, "summary.json"), "w") as sf:
        json.dump(summary, sf, indent=2)
    logger.info("Saved summary.json.")

    # Evaluate accuracy (if reference files exist)
    for entry in summary:
        file_base, _ = os.path.splitext(entry["file"])
        task = entry["task"]
        lang = entry["language"]
        ref_filename = f"{file_base}_{task}.txt"
        ref_file = os.path.join(ref_dir, lang, ref_filename)
        if os.path.exists(ref_file):
            with open(ref_file, "r", encoding="utf-8") as rf:
                reference_text = rf.read()
            result_filename = f"{lang}_{entry['file']}_{task}_{entry['method']}_{entry.get('model_size', 'whisper-1')}.json"
            result_filepath = os.path.join(results_dir, result_filename)
            if os.path.exists(result_filepath):
                with open(result_filepath, "r", encoding="utf-8") as resf:
                    result_data = json.load(resf)
                output_text = result_data.get("output_text", "")
                accuracy = evaluate_accuracy(reference_text, output_text, task)
                entry["accuracy"] = accuracy
                logger.info(f"Evaluated accuracy for {entry['file']} ({task}) using method {entry['method']} {entry.get('model_size', 'whisper-1')}: {accuracy}")
            else:
                entry["accuracy"] = None
        else:
            entry["accuracy"] = None

    with open(os.path.join(results_dir, "summary_with_accuracy.json"), "w") as sf:
        json.dump(summary, sf, indent=2)
    logger.info("Saved summary_with_accuracy.json.")

    # Stop memory logger and wait for thread to finish
    mem_logger.stop()
    mem_logger.join()
    logger.info("Benchmarking completed. See the 'results' directory for details.")

if __name__ == "__main__":
    main()
