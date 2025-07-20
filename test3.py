import os
import sounddevice as sd
import numpy as np
import queue
import tempfile
import soundfile as sf
from pyannote.audio import Pipeline
import torch
import whisper  # 또는 faster_whisper
from datetime import timedelta


# Hugging Face 토큰을 환경 변수에서 불러옵니다.
# 이는 코드에 민감한 정보를 직접 노출하지 않는 안전한 방법입니다.
HF_TOKEN = os.environ["HF_TOKEN"]
if not HF_TOKEN:
    raise ValueError(
        "Hugging Face 인증 토큰이 필요합니다. 'HF_TOKEN' 환경 변수를 설정해주세요."
    )
# Hugging Face Token 필요
diarization_pipeline = Pipeline.from_pretrained(
	"pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

# Whisper 모델 로딩 (base or small 추천)
stt_model = whisper.load_model("base")  # 또는 faster-whisper 사용

# 마이크 설정
samplerate = 16000
block_duration = 3  # 초
q = queue.Queue()

# 콜백


def callback(indata, frames, time, status):
	q.put(indata.copy())

# 시간 포맷


def format_time(seconds):
	return str(timedelta(seconds=round(seconds, 1)))[:-3]


print("🎙️ 실시간 화자 분리 + 음성 인식 자막 시작 (Ctrl+C 종료)")

with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
	try:
		while True:
			# 3초 버퍼 수집
			frames = []
			for _ in range(int((block_duration * samplerate) / 1024)):
				frames.append(q.get())
			audio_block = np.concatenate(frames, axis=0)

			waveform = torch.from_numpy(
				audio_block.astype(np.float32).T)
			diarization_input = {"waveform": waveform, "sample_rate": samplerate}

			# (1) 화자 분리
			diarization = diarization_pipeline(diarization_input)

			# (2) 음성 인식
			transcription = stt_model.transcribe(audio_block, batch_size=16)

			# (3) 매핑 및 자막 출력
			print("\n--- 🗣️ 실시간 자막 ---")
			for turn, _, speaker in diarization.itertracks(yield_label=True):
				seg_start = turn.start
				seg_end = turn.end

				# 이 구간에 포함되는 STT 텍스트 추출
				seg_texts = [
					segment["text"].strip()
					for segment in transcription["segments"]
					if segment["start"] >= seg_start and segment["end"] <= seg_end
				]

				full_text = " ".join(seg_texts) if seg_texts else "(음성 없음)"
				print(
					f"[{format_time(seg_start)} - {format_time(seg_end)}] {speaker}: {full_text}")

	except KeyboardInterrupt:
		print("🛑 종료됨.")
