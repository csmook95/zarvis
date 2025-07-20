import sounddevice as sd
from pyannote.audio import Pipeline
import torch
import os


# Hugging Face 토큰을 환경 변수에서 불러옵니다.
# 이는 코드에 민감한 정보를 직접 노출하지 않는 안전한 방법입니다.
HF_TOKEN = os.environ["HF_TOKEN"]
if not HF_TOKEN:
    raise ValueError(
        "Hugging Face 인증 토큰이 필요합니다. 'HF_TOKEN' 환경 변수를 설정해주세요."
    )

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# diarization pipeline 로딩 및 장치 할당
print("Diarization 파이프라인 로딩 중...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
pipeline.to(device)
print("파이프라인 로딩 완료.")

# 마이크 설정
SAMPLE_RATE = 16000  # 샘플링 레이트
DURATION = 10  # 초 단위 버퍼 (실시간 처리 단위)


def record_and_diarize():
    """
    마이크로부터 오디오를 녹음하고 실시간으로 화자를 분리합니다.
    임시 파일 생성 없이 메모리에서 직접 처리하여 효율성을 높였습니다.
    """
    print("🎤 마이크에서 입력을 받고 있습니다 (Ctrl+C로 중지)")
    try:
        while True:
            # 오디오 녹음 (float32 타입으로 변경하여 모델 호환성 및 정밀도 향상)
            print(f"⏺️ {DURATION}초간 녹음 중...")
            audio_np = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                              channels=1, dtype='float32')
            sd.wait()

            # pyannote 파이프라인 입력 형식에 맞게 데이터 변환
            # (samples, channels) -> (channels, samples)
            audio_torch = torch.from_numpy(audio_np.T).to(device)
            diarization_input = {"waveform": audio_torch,
                                 "sample_rate": SAMPLE_RATE}

            # diarization 수행 (메모리에서 직접 처리)
            print(f"🔍 화자 분리 중...")
            diarization = pipeline(diarization_input)

            # 결과 출력
            print("--- 화자 분리 결과 ---")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"[{turn.start:.2f}s - {turn.end:.2f}s] Speaker {speaker}")

    except KeyboardInterrupt:
        print("\n✅ 중지되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    record_and_diarize()
