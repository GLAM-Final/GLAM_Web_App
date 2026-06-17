import os
import warnings

# Suppress specific Starlette/Gradio deprecation warnings regarding HTTP status codes
warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*"
)

import gradio as gr
from ui import create_ui, css_styles

# Initialize your clean UI layout
app = create_ui()

if __name__ == "__main__":
    # Running on Render if PORT is provided
    is_render = "PORT" in os.environ

    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        share=False,
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(),
        css=css_styles
    )
