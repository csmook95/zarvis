import gradio as gr
import sounddevice as sd
import numpy as np
import whisperx
import tempfile
import threading
import queue
import time
from scipy.io.wavfile import write
import torch

# 3초 버퍼 단위 (샘플링 16kHz)
BUFFER_DURATION = 3
SAMPLE_RATE = 16000

audio_queue = queue.Queue()
transcribed_text = ""

# WhisperX 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = whisperx.load_model("medium", device=device, compute_type=compute_type)


def record_audio_loop():
	global transcribed_text

	def callback(indata, frames, time_info, status):
		audio_queue.put(indata.copy())

	with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
		while True:
			frames = []
			try:
				# 3초 동안 오디오 수집
				for _ in range(int(SAMPLE_RATE * BUFFER_DURATION / 1024)):
					frames.append(audio_queue.get(timeout=5))
				audio_data = np.concatenate(frames, axis=0)

				# WAV 파일로 저장
				with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
					write(tmpfile.name, SAMPLE_RATE, audio_data)
					result = model.transcribe(tmpfile.name)
					text = result["text"]
					if text.strip():
						transcribed_text += text + "\n"

			except queue.Empty:
				continue


# 백그라운드 쓰레드로 마이크 스트리밍 시작
threading.Thread(target=record_audio_loop, daemon=True).start()


def get_latest_transcription():
	return transcribed_text


with gr.Blocks() as demo:
	gr.Markdown("### 🎤 WhisperX 실시간 음성 인식 (3초 간격)")
	output = gr.Textbox(label="실시간 인식 결과", lines=10)

	# 주기적으로 결과 업데이트 (1초 간격)
	demo.load(get_latest_transcription, outputs=output, every=1)

demo.launch()
