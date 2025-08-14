# frontend/gradio_app.py
import gradio as gr
# (Optional) replicate the Streamlit logic here for a quick Gradio demo.
def analyze_file(file, out_lang="en", plain_english=True):
    return "Use Streamlit for full UI."

iface = gr.Interface(fn=analyze_file, inputs=[gr.File(), gr.Dropdown(["en","es","pt"])], outputs="text")
if __name__ == "__main__":
    iface.launch()
