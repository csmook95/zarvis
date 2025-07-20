from datetime import timedelta
import queue
from pyannote.audio import Pipeline
import sounddevice as sd
import numpy as np
import torch
import os

# Hugging Face 토큰을 환경 변수에서 불러옵니다.
# 이는 코드에 민감한 정보를 직접 노출하지 않는 안전한 방법입니다.
HF_TOKEN = os.environ["HF_TOKEN"]
if not HF_TOKEN:
	raise ValueError(
		"Hugging Face 인증 토큰이 필요합니다. 'HF_TOKEN' 환경 변수를 설정해주세요."
	)

# Hugging Face 토큰은 코드에 직접 작성하는 대신,
# 터미널에서 `huggingface-cli login` 명령어로 미리 로그인해두는 것이 안전합니다.
pipeline = Pipeline.from_pretrained(
	"pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

# 녹음 버퍼 설정
q = queue.Queue()
samplerate = 16000
duration = 3  # seconds


def audio_callback(indata, frames, time, status):
	q.put(indata.copy())

# 시간 포맷 함수


def format_time(seconds):
	return str(timedelta(seconds=round(seconds, 1)))[:-3]


# 실시간 마이크 녹음
with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
	print("🎙️ 실시간 화자 분리 시작 (Ctrl+C 종료)")
	while True:
		# 3초 분량 수집
		frames = []
		for _ in range(int((duration * samplerate) / 1024)):
			frames.append(q.get())
		audio_chunk = np.concatenate(frames, axis=0)

		# # 3초간 녹음 버퍼 수집 (1024는 일반적인 프레임 크기 가정)
		# audio_chunk = np.concatenate(
		# 	[q.get() for _ in range(int(duration * samplerate / 1024))])

		# 오디오 데이터를 파일로 저장하는 대신 메모리에서 직접 처리
		# pyannote 파이프라인에 맞는 형태로 데이터 변환
		waveform = torch.from_numpy(
			audio_chunk.astype(np.float32).T)
		diarization_input = {"waveform": waveform, "sample_rate": samplerate}

		# diarization 수행
		diarization = pipeline(diarization_input)
		# 콘솔 자막 출력
		print("\n--- 🗣️ 실시간 자막 ---")
		for turn, _, speaker in diarization.itertracks(yield_label=True):
			print(f"[{format_time(turn.start)} - {format_time(turn.end)}] {speaker}: ...")
		# # 결과 출력
		# for turn, _, speaker in diarization.itertracks(yield_label=True):
		# 	print(f"[{turn.start:.1f}s ~ {turn.end:.1f}s] → {speaker}")
