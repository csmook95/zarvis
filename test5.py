import sounddevice as sd
import torch
import whisperx
import numpy as np


# 마이크 설정
fs = 16000  # 샘플링 레이트
duration = 10  # 초 단위 버퍼 (실시간 처리 단위)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 8


def record_and_diarize():
	print("🎤 마이크에서 입력을 받고 있습니다 (Ctrl+C로 중지)")
	try:
		while True:
			# 오디오 녹음
			print(f"⏺️ {duration}초간 녹음 중...")
			audio = sd.rec(int(duration * fs), samplerate=fs,
                            channels=1, dtype='float32')
			sd.wait()
			waveform = audio.flatten()

			model = whisperx.load_model(
				"tiny", device, compute_type=compute_type)
			result = model.transcribe(waveform, batch_size=batch_size,language="ko")

			print(result["segments"])
			detected_language = result["language"]
			print(f"감지된 언어: {detected_language}")
	except KeyboardInterrupt:
		print("\n✅ 중지되었습니다.")


record_and_diarize()
