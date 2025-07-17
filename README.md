# Zarvis 🎙️

## 🚀 시작하기

로컬 컴퓨터에서 프로젝트를 설정하고 실행하려면 다음 지침을 따르세요.

### 전제 조건

*   Python 3.8 이상

### 🛠️ 설치 및 설정

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/csmook95/zarvis.git
    cd zarvis
    ```

2.  **가상 환경 생성 및 활성화:**

    *   가상 환경 생성:
        ```bash
        python -m venv .venv
        ```

    *   가상 환경 활성화:
        *   **Windows:**
            ```powershell
            .\.venv\Scripts\Activate.ps1
            ```
        *   **macOS / Linux:**
            ```bash
            source .venv/bin/activate
            ```

3.  **의존성 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ 사용법

애플리케이션을 시작하려면 다음 명령을 실행하세요.

```bash
python main.py
```

그런 다음 웹 브라우저를 열고 Gradio에서 제공하는 로컬 URL(일반적으로 `http://127.0.0.1:7860`)로 이동합니다.