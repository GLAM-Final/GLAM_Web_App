import os

# Limit thread overhead for memory-constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to save RAM

import warnings
# Suppress specific Starlette/Gradio deprecation warnings regarding HTTP status codes
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
from ui import create_ui, css_styles

app = create_ui()

if __name__ == "__main__":
    host = "0.0.0.0"
    
    app.launch(
        share=False,
        server_name=host,
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
        css=css_styles
    )
