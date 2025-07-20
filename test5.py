import sounddevice as sd
import torch
import whisperx
import numpy as np
import os


# 마이크 설정
SAMPLE_RATE = 16000  # 샘플링 레이트
DURATION = 3  # 초 단위 버퍼 (실시간 처리 단위)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 8

# Hugging Face 토큰을 환경 변수에서 불러옵니다.
# 이는 코드에 민감한 정보를 직접 노출하지 않는 안전한 방법입니다.
HF_TOKEN = os.environ["HF_TOKEN"]
if not HF_TOKEN:
	raise ValueError(
		"Hugging Face 인증 토큰이 필요합니다. 'HF_TOKEN' 환경 변수를 설정해주세요."
	)


def record_and_diarize():
	print("🎤 마이크에서 입력을 받고 있습니다 (Ctrl+C로 중지)")
	try:
		while True:
			# 오디오 녹음
			print(f"⏺️ {DURATION }초간 녹음 중...")
			audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                            channels=1, dtype='float32')
			sd.wait()
			waveform = audio.flatten()

			# 1. 음성 전사
			print("1. 음성 전사 중...")
			model = whisperx.load_model(
				"tiny", device, compute_type=compute_type)
			result = model.transcribe(waveform, batch_size=batch_size, language="ko")

			detected_language = result["language"]
			print(result)
			print(f"감지된 언어: {detected_language}")

			# 2. 단어 레벨 정렬
			print("2. 단어 레벨 정렬 중...")
			model_a, metadata = whisperx.load_align_model(
				language_code=detected_language, device=device)
			result = whisperx.align(
				result["segments"], model_a, metadata, waveform, device, return_char_alignments=False)

			print("3. 화자 구분 중...")
			diarize_model = whisperx.diarize.DiarizationPipeline(
				use_auth_token=HF_TOKEN, device=device)

			# 화자 수 파라미터 설정
			diarize_kwargs = {
				"min_speakers": 1,
				"max_speakers": 3
			}

			diarize_segments = diarize_model(waveform, **diarize_kwargs)
			result = whisperx.assign_word_speakers(
				diarize_segments, result)

			print(result)
			# for segment in result["segments"]:
			# 	speaker = segment.get("speaker", "Unknown")
			# 	start = segment["start"]
			# 	end = segment["end"]
			# 	text = segment["text"]

			# 	print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")

	except KeyboardInterrupt:
		print("\n✅ 중지되었습니다.")


record_and_diarize()
