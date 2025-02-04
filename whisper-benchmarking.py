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

# Try to import pynvml for GPU VRAM monitoring (optional)
import pynvml

# Import the two Whisper implementations
import whisper  # official OpenAI Whisper package
from faster_whisper import WhisperModel as FasterWhisperModel, BatchedInferencePipeline

import logging

# -----------------------------------------------------------------------------
# Logger Setup: all events, memory usage, and status changes go into a single file and to console.
# -----------------------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

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
            logger.info(f"Memory Usage: RAM={format_bytes(ram_usage)}, GPU={format_bytes(gpu_usage) if gpu_usage is not None else 'N/A'}")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if pynvml:
            pynvml.nvmlShutdown()

# -----------------------------------------------------------------------------
# Accuracy Evaluation via ChatGPT API
# -----------------------------------------------------------------------------
def evaluate_accuracy(reference_text, output_text, task="transcription"):
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
# New helper functions that use pre-loaded models (no load/unload per file)
# -----------------------------------------------------------------------------
def transcribe_local(model, audio_file, task="transcribe", language="en"):
    logger.info(f"Transcribing '{os.path.basename(audio_file)}' with local Whisper model")
    start_time = time.time()
    if task == "transcribe":
        result = model.transcribe(audio_file, language=language, temperature=0.0)
    elif task == "translate":
        result = model.transcribe(audio_file, task="translate", language=language, temperature=0.0)
    else:
        result = {}
    latency = time.time() - start_time
    output_text = result.get("text", "").strip()
    return output_text, latency

def transcribe_faster(model, audio_file, task="transcribe", language="en"):
    logger.info(f"Transcribing '{os.path.basename(audio_file)}' with Faster-Whisper model")
    start_time = time.time()
    if task == "transcribe":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0)
    elif task == "translate":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, task="translate", temperature=0.0)
    else:
        segments = []
    latency = time.time() - start_time
    output_text = " ".join([seg.text for seg in segments]).strip() if segments else ""
    return output_text, latency

def transcribe_faster_batched(model, audio_file, task="transcribe", language="en", batch_size=16):
    logger.info(f"Transcribing '{os.path.basename(audio_file)}' with batched Faster-Whisper model")
    start_time = time.time()
    if task == "transcribe":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, temperature=0.0, batch_size=batch_size)
    elif task == "translate":
        segments, _ = model.transcribe(audio_file, beam_size=5, language=language, task="translate", temperature=0.0, batch_size=batch_size)
    else:
        segments = []
    latency = time.time() - start_time
    output_text = " ".join([seg.text for seg in segments]).strip() if segments else ""
    return output_text, latency

# -----------------------------------------------------------------------------
# For REST API, we continue as before.
# -----------------------------------------------------------------------------
def test_rest_api(audio_file, task="transcribe"):
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
# New runner functions that use a pre-loaded model
# -----------------------------------------------------------------------------
def run_benchmark_with_model(method, model, audio_file, task, lang="en", n_runs=3, batch_size=16):
    latencies = []
    output_text = ""
    for i in range(n_runs):
        if method == "local":
            out, lat = transcribe_local(model, audio_file, task, language=lang)
        elif method == "faster":
            out, lat = transcribe_faster(model, audio_file, task, language=lang)
        elif method == "fasterbatched":
            out, lat = transcribe_faster_batched(model, audio_file, task, language=lang, batch_size=batch_size)
        else:
            out, lat = "", 0
        latencies.append(lat)
        output_text = out
    avg_latency = sum(latencies) / len(latencies)
    return {
        "avg_latency": avg_latency,
        "latencies": latencies,
        "output_text": output_text
    }

# -----------------------------------------------------------------------------
# Audio duration retrieval
# -----------------------------------------------------------------------------
def get_audio_duration(file_path):
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
    audio_dir = "./audio"
    ref_dir = "./references"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a subfolder for detailed result files.
    details_dir = os.path.join(results_dir, "details")
    os.makedirs(details_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(results_dir, "benchmark.log"))
    logger.info("Benchmarking started.")
    
    mem_logger = MemoryLogger(interval=1.0)
    mem_logger.start()
    
    languages = ["english", "finnish"]
    model_sizes = ["tiny"]#, "base", "small", "medium", "large", "turbo", "large-v3"]
    
    summary = []
    
    # Process REST API separately (model loading not needed).
    for lang in languages:
        audio_path = os.path.join(audio_dir, lang)
        for filename in os.listdir(audio_path):
            if filename.lower().endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(audio_path, filename)
                duration_seconds = get_audio_duration(file_path)
                if duration_seconds is None:
                    logger.error(f"Skipping {filename} due to duration extraction error.")
                    continue
                logger.info(f"Processing file '{filename}' (lang: {lang}) with duration {duration_seconds:.2f} seconds using REST API.")
                tasks = ["transcribe"] if lang == "english" else ["transcribe", "translate"]
                for task in tasks:
                    logger.info(f"Method 'rest' on '{filename}' for task '{task}'")
                    result = run_benchmark_with_model("rest", None, file_path, task, lang, n_runs=3)
                    result_filename = f"{lang}_{filename}_{task}_rest_whisper-1.json"
                    with open(os.path.join(details_dir, result_filename), "w") as rf:
                        json.dump(result, rf, indent=2)
                    summary.append({
                        "language": lang,
                        "file": filename,
                        "task": task,
                        "method": "rest",
                        "model_size": "whisper-1",
                        "avg_latency": result["avg_latency"],
                        "load_time": None,
                        "duration": duration_seconds
                    })
    
    # For methods that use a model, iterate over model sizes.
    for method in ["local", "faster", "fasterbatched"]:
        for model_size in model_sizes:
            logger.info(f"Loading model for method '{method}' with model size '{model_size}'")
            if method == "local":
                model = whisper.load_model(model_size, device="cuda")
            elif method == "faster":
                model = FasterWhisperModel(model_size, device="cuda", compute_type="float16")
            elif method == "fasterbatched":
                base_model = FasterWhisperModel(model_size, device="cuda", compute_type="float16")
                model = BatchedInferencePipeline(base_model)
            else:
                continue  # Should not happen

            # Process all audio files for each language using this model.
            for lang in languages:
                audio_path = os.path.join(audio_dir, lang)
                for filename in os.listdir(audio_path):
                    if filename.lower().endswith((".wav", ".mp3", ".m4a")):
                        file_path = os.path.join(audio_path, filename)
                        duration_seconds = get_audio_duration(file_path)
                        if duration_seconds is None:
                            logger.error(f"Skipping {filename} due to duration extraction error.")
                            continue
                        logger.info(f"Processing file '{filename}' (lang: {lang}) with duration {duration_seconds:.2f} seconds using method '{method}', model '{model_size}'")
                        tasks = ["transcribe"] if lang == "english" else ["transcribe", "translate"]
                        for task in tasks:
                            logger.info(f"Method '{method}' (model: {model_size}) on '{filename}' for task '{task}'")
                            result = run_benchmark_with_model(method, model, file_path, task, lang[:2].lower(), n_runs=3)
                            result_filename = f"{lang}_{filename}_{task}_{method}_{model_size}.json"
                            with open(os.path.join(details_dir, result_filename), "w") as rf:
                                json.dump(result, rf, indent=2)
                            summary.append({
                                "language": lang,
                                "file": filename,
                                "task": task,
                                "method": method,
                                "model_size": model_size,
                                "avg_latency": result["avg_latency"],
                                "load_time": None,  # load time is recorded once per model load
                                "duration": duration_seconds
                            })
            # Unload the model once all files for this model size are processed.
            del model
            if method == "fasterbatched":
                del base_model
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model for method '{method}' with model size '{model_size}'")
    
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
            if entry["method"] == "rest":
                result_filename = f"{lang}_{entry['file']}_{task}_rest_whisper-1.json"
            else:
                result_filename = f"{lang}_{entry['file']}_{task}_{entry['method']}_{entry.get('model_size', 'whisper-1')}.json"
            result_filepath = os.path.join(details_dir, result_filename)
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
    
    mem_logger.stop()
    mem_logger.join()
    logger.info("Benchmarking completed. See the 'results' directory for details.")

if __name__ == "__main__":
    main()
