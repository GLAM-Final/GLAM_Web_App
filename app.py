import os

# Limit thread overhead for memory-constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend to save RAM

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
    # Render automatically provides PORT
    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
