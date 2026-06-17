import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*"
)

import gradio as gr
from ui import create_ui, css_styles

# Initialize UI
app = create_ui()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        css=css_styles
    )
