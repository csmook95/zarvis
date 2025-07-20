import whisperx
import torch
import json
import os


class WhisperXTranscriber:
    def __init__(self, hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.hf_token = hf_token

    def transcribe_with_speakers(self, audio_file, batch_size=16, min_speakers=None, max_speakers=None):
        """
        음성 파일을 전사하고 화자를 구분합니다.
        """
        try:
            # 1. 음성 전사
            print("1. 음성 전사 중...")
            audio = whisperx.load_audio(audio_file)
            model = whisperx.load_model(
                "large-v2", self.device, compute_type=self.compute_type)
            result = model.transcribe(audio, batch_size=batch_size)

            detected_language = result["language"]
            print(f"감지된 언어: {detected_language}")

            # 2. 단어 레벨 정렬
            print("2. 단어 레벨 정렬 중...")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language, device=self.device)
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            # 3. 화자 구분 (토큰이 있는 경우)
            if self.hf_token:
                print("3. 화자 구분 중...")
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=self.hf_token, device=self.device)

                # 화자 수 파라미터 설정
                diarize_kwargs = {}
                if min_speakers:
                    diarize_kwargs['min_speakers'] = min_speakers
                if max_speakers:
                    diarize_kwargs['max_speakers'] = max_speakers

                diarize_segments = diarize_model(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(
                    diarize_segments, result)

            return result

        except Exception as e:
            print(f"오류 발생: {e}")
            return None

    def save_results(self, result, output_file):
        """결과를 파일로 저장"""
        if result:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"결과가 {output_file}에 저장되었습니다.")

    def format_transcript(self, result):
        """읽기 쉬운 형태로 포맷팅"""
        if not result:
            return "전사 결과가 없습니다."

        formatted_lines = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            formatted_lines.append(
                f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")

        return "\n".join(formatted_lines)


# 사용 예제
if __name__ == "__main__":
    # Hugging Face 토큰 설정 (화자 구분용)
    # HF_TOKEN = os.getenv("HF_TOKEN")  # 환경변수에서 가져오기
    HF_TOKEN = os.environ["HF_TOKEN"]

    # 전사기 초기화
    transcriber = WhisperXTranscriber(hf_token=HF_TOKEN)

    # 음성 파일 전사
    audio_file = "audiofiles/katiesteve.wav"
    result = transcriber.transcribe_with_speakers(
        audio_file,
        batch_size=16,
        min_speakers=1,
        max_speakers=3
    )

    # 결과 출력
    if result:
        print("\n=== 전사 결과 ===")
        formatted_text = transcriber.format_transcript(result)
        print(formatted_text)

        # 파일로 저장
        transcriber.save_results(result, "transcript_result.json")

        with open("transcript_formatted.txt", "w", encoding="utf-8") as f:
            f.write(formatted_text)

        print("\n결과가 파일로 저장되었습니다.")
