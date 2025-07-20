import gradio as gr

with gr.Blocks() as demo:
	def change_audio(audio_path):
		return audio_path

	# input_audio = gr.Audio(sources="microphone", type="filepath", label="마이크 입력")
	input_audio = gr.Audio(type="filepath")
	output_text = gr.Textbox(
		label="음성 인식 결과", placeholder="여기에 음성 인식 결과가 표시됩니다.", interactive=False)

	input_audio.change(change_audio, inputs=[input_audio], outputs=[output_text])

demo.launch()
