import os
import time
import tempfile
import traceback
import numpy as np
import soundfile as sf
import torch
import librosa
import gradio as gr

# --- Pipeline & Model Imports ---
# (Kept intact to connect with your backend framework logic)
from interface import (
    load_patient_registry, save_patient_registry, separate_audio, monitoring_table_rows, _copy_audio_to_storage,
    run_end_to_end, search_history_records, search_reasoning_records, _initialize_models_for_live_processing,
    process_audio_chunk_for_separation, infer_on_separated_chunk, _live_sr,
    LIVE_PROCESSING_WINDOW_SECONDS, LIVE_OVERLAP_SECONDS, resolve_patient_names, save_audio_file, predict_sources
)
from gnn import EnhancedPatientStateManager, ClinicalAlertSystem 


# ==========================================
# 1. STYLING CONFIGURATION
# ==========================================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            return f.read()
    return ""

css_styles = load_css()


# ==========================================
# 2. CORE BUSINESS & UTILITY LOGIC
# ==========================================
def register_patients(ref_audio_1, ref_audio_2, ref_audio_3, patient_name_1, patient_name_2, patient_name_3):
    raw_names = [patient_name_1, patient_name_2, patient_name_3]
    audio_paths = [ref_audio_1, ref_audio_2, ref_audio_3]
    patient_entries = []
    for idx, (name, audio_path) in enumerate(zip(raw_names, audio_paths), start=1):
        entry = {
            "patient_id": f"patient_{idx}",
            "name": name.strip() if name else f"Patient {idx}",
            "reference_audio": str(audio_path) if audio_path else None,
        }
        patient_entries.append(entry)

    save_patient_registry(patient_entries)

    registered = [f"Patient {idx}: {entry['name']}" for idx, entry in enumerate(patient_entries, start=1)]
    message = (
        "Patients registered successfully. Reference audio files are saved for each patient.\n"
        + "\n".join(registered)
        + "\nSaved to pipeline_results/patient_registry.json"
    )
    choices = get_registered_patient_choices()
    selected_values = [choices[i + 1] if i + 1 < len(choices) else "Unassigned" for i in range(6)]
    return (
        message,
        gr.update(choices=choices, value=selected_values[0]),
        gr.update(choices=choices, value=selected_values[1]),
        gr.update(choices=choices, value=selected_values[2]),
        gr.update(choices=choices, value=selected_values[0]),
        gr.update(choices=choices, value=selected_values[1]),
        gr.update(choices=choices, value=selected_values[2]),
    )


def get_registered_patient_choices(default_count=3):
    registry = load_patient_registry()
    names = [entry.get("name") or f"Patient {idx + 1}" for idx, entry in enumerate(registry)]
    names = list(dict.fromkeys(names))
    if not names:
        names = [f"Patient {i}" for i in range(1, default_count + 1)]
    return ["Unassigned"] + names


def normalize_live_patient_names(selected_names):
    normalized = []
    for name in selected_names:
        if name and name != "Unassigned":
            normalized.append(name)
        else:
            normalized.append(None)
    return normalized


def predict(mix_audio, p1, p2, p3): # p1, p2, p3 are patient names
    selected_names = [n if n != "Unassigned" else None for n in [p1, p2, p3]]

    # Retrieve reference audio paths for selected patients from the registry
    registry = load_patient_registry()
    reference_audio_paths = [None, None, None]
    for i, name in enumerate(selected_names):
        if name:
            for entry in registry:
                if entry.get("name") == name:
                    reference_audio_paths[i] = entry.get("local_reference_audio") or entry.get("reference_audio")
                    break
    try:
        outputs, reasoning_summaries, history_record = run_end_to_end(
            mix_audio, 
            patient_names=selected_names,
            reference_audio_paths=reference_audio_paths # Pass reference audio paths
        )
    except Exception as exc:
        error_message = f"Error during pipeline: {exc}"
        error_trace = traceback.format_exc()
        return [None] * 6 + [error_message] + [[]] + [error_trace]

    monitor_rows = monitoring_table_rows()
    history_status = f"Separation, reasoning, and monitoring complete. History record saved at {history_record.get('timestamp')}"
    message = f"Full pipeline complete. Separated sources generated, reasoning updated, and history stored. {len(reasoning_summaries)} reasoning summaries available."
    return outputs + [message] + [monitor_rows] + [history_status]


def refresh_monitoring():
    rows = monitoring_table_rows()
    if not rows:
        return [], "No reasoning summary available. Run the pipeline and make sure pipeline_results/reasoning_summary.json exists."
    return rows, f"Loaded {len(rows)} patient states from reasoning summary."


def search_history(query):
    if not query:
        return [], "Enter a patient name or ID to search history."
    rows = search_reasoning_records(query)
    if not rows:
        return [], f"No clinical findings found matching '{query}'."
    return rows, f"Found {len(rows)} matching clinical record(s)."


# ==========================================
# 3. LIVE STREAMING CONTROLLERS
# ==========================================
def start_live_monitoring_session(selected_patient_1, selected_patient_2, selected_patient_3):
    global _live_processor, _live_wav2vec_model, _live_gnn_model, _live_device
    _live_processor, _live_wav2vec_model, _live_gnn_model, _live_device = _initialize_models_for_live_processing()
    selected_names = normalize_live_patient_names([selected_patient_1, selected_patient_2, selected_patient_3])
    managers = [
        EnhancedPatientStateManager(),
        EnhancedPatientStateManager(),
        EnhancedPatientStateManager(),
    ]
    empty_audio = np.array([], dtype=np.float32)
    empty_buffers = [np.array([], dtype=np.float32) for _ in range(3)]
    empty_table = []
    status_names = [name for name in selected_names if name is not None]
    if status_names:
        status = f"Live monitoring initialized for: {', '.join(status_names)}. Click microphone to start."
    else:
        status = "No patients selected. Select at least one patient to monitor."
    return (
        empty_audio,
        empty_buffers,
        managers,
        [0.0, 0.0, 0.0],
        empty_table,
        selected_names,
        status,
        gr.update(value=None, interactive=True),
        gr.update(interactive=False),
    )


def process_live_audio_stream(audio_chunk, live_audio_buffer, live_separated_buffers, live_patient_managers, live_current_timestamps, live_patient_names):
    if audio_chunk is None:
        return live_audio_buffer, live_separated_buffers, live_patient_managers, live_current_timestamps, [], "No audio received."

    sr, np_audio = audio_chunk
    if sr is None or np_audio is None:
        return live_audio_buffer, live_separated_buffers, live_patient_managers, live_current_timestamps, [], "Invalid audio chunk received."

    mono = np.mean(np_audio, axis=-1) if np_audio.ndim > 1 else np_audio
    target_sr = _live_sr
    if sr != target_sr:
        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    mono = mono.astype(np.float32)
    
    current_audio_base = live_audio_buffer if live_audio_buffer is not None else np.array([], dtype=np.float32)
    current_audio = np.concatenate([current_audio_base, mono]) if current_audio_base.size > 0 else mono
    max_buffer = int(LIVE_PROCESSING_WINDOW_SECONDS * target_sr)
    if current_audio.size > max_buffer:
        current_audio = current_audio[-max_buffer:]

    window_samples = int(LIVE_PROCESSING_WINDOW_SECONDS * target_sr)
    overlap_samples = int(LIVE_OVERLAP_SECONDS * target_sr)
    step = window_samples - overlap_samples

    new_buffers = [buf.copy() for buf in live_separated_buffers] if live_separated_buffers else [np.array([], dtype=np.float32) for _ in range(3)]
    new_timestamps = list(live_current_timestamps)
    audios_out = [None, None, None]

    active_slots = [i for i, name in enumerate(live_patient_names) if name is not None] if live_patient_names else []
    if current_audio.size >= window_samples and active_slots:
        chunk = current_audio[-window_samples:]
        prediction, _ = predict_sources(torch.from_numpy(chunk).unsqueeze(0), _live_sr)
        separated = prediction.cpu().numpy()
        for i in range(min(3, separated.shape[0])):
            selected_name = live_patient_names[i] if i < len(live_patient_names) else None
            if selected_name is None:
                new_buffers[i] = np.array([], dtype=np.float32)
                audios_out[i] = None
                continue
            separated_i = separated[i].astype(np.float32)
            hop = step if step > 0 else window_samples
            if separated_i.size > hop:
                new_buffers[i] = separated_i[-hop:]
            else:
                new_buffers[i] = separated_i
            new_timestamps[i] = new_timestamps[i] + hop / target_sr if new_timestamps[i] > 0 else hop / target_sr

            # Performance Note: Disk I/O (sf.write/_copy_audio_to_storage) removed to reduce live latency
            try:
                audios_out[i] = (target_sr, separated_i)
            except Exception:
                audios_out[i] = None

    rows = []
    for i, manager in enumerate(live_patient_managers):
        selected_name = live_patient_names[i] if i < len(live_patient_names) else None
        if selected_name is None or manager is None:
            continue
        separated_chunk = new_buffers[i]
        timestamp = new_timestamps[i] if new_timestamps[i] > 0 else 0.0
        if separated_chunk.size >= target_sr:
            state = infer_on_separated_chunk(
                separated_chunk,
                _live_gnn_model,
                _live_processor,
                _live_wav2vec_model,
                _live_device,
                manager,
                f"live_patient_{i+1}",
                timestamp,
            )
            rows.append([
                selected_name,
                state.get("overall_state"),
                manager.patient_data.get(f"live_patient_{i+1}", {}).get("wheeze_ema"),
                manager.patient_data.get(f"live_patient_{i+1}", {}).get("crackle_ema"),
                state.get("breathing_rate_mean"),
                state.get("comment", ""),
            ])

    return current_audio, new_buffers, live_patient_managers, new_timestamps, rows, "Processing live audio...", audios_out[0], audios_out[1], audios_out[2], live_patient_names


def stop_live_monitoring_session():
    empty_audio = np.array([], dtype=np.float32)
    empty_buffers = [np.array([], dtype=np.float32) for _ in range(3)]
    cleared_managers = [None, None, None]
    cleared_timestamps = [0.0, 0.0, 0.0]
    return empty_audio, empty_buffers, cleared_managers, cleared_timestamps, [], "Live monitoring stopped.", gr.update(value=None, interactive=False), gr.update(interactive=True), None, None, None


# ==========================================
# 4. INTERFACE BUILDING METHOD
# ==========================================
def create_ui():
    # Pass structural embedded CSS string variable safely inside Blocks
    with gr.Blocks() as demo:
        gr.HTML("<div class='header-box'><h1>Patient Monitoring System</h1></div>")

        with gr.Row():
            # Sidebar Menu
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### Navigation")
                btn_register = gr.Button("Register Patients", variant="secondary", elem_classes="sidebar-btn")
                btn_separation = gr.Button("Audio Separation", variant="secondary", elem_classes="sidebar-btn")
                btn_live_mon = gr.Button("Live Monitoring", variant="secondary", elem_classes="sidebar-btn") 
                btn_history = gr.Button("View History", variant="secondary", elem_classes="sidebar-btn")

            # Content Area
            with gr.Column(scale=4):
                live_audio_buffer_state = gr.State(value=None)
                live_separated_buffers_state = gr.State(value=[])
                live_patient_managers_state = gr.State(value=[None, None, None])
                live_current_timestamps_state = gr.State(value=[0.0, 0.0, 0.0])
                live_patient_names_state = gr.State(value=[])

                # Registration Page
                with gr.Column(visible=True) as reg_page:
                    gr.Markdown("### Patient Registration")
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### Patient 1") # This is a Markdown header, not a textbox
                            patient_name_1 = gr.Textbox(label="Name", placeholder="Enter name", elem_classes="vibrant-status")
                            ref_audio_1 = gr.Audio(label="Ref Audio", type="filepath")
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### Patient 2") # This is a Markdown header, not a textbox
                            patient_name_2 = gr.Textbox(label="Name", placeholder="Enter name", elem_classes="vibrant-status")
                            ref_audio_2 = gr.Audio(label="Ref Audio", type="filepath")
                        with gr.Column(variant="panel"):
                            gr.Markdown("#### Patient 3") # This is a Markdown header, not a textbox
                            patient_name_3 = gr.Textbox(label="Name", placeholder="Enter name", elem_classes="vibrant-status")
                            ref_audio_3 = gr.Audio(label="Ref Audio", type="filepath")
                                    
                    register_btn = gr.Button("Submit Registration", variant="primary", size="lg")
                    register_status = gr.Textbox(label="Status", interactive=False, elem_classes="vibrant-status")

                # Separation Page
                with gr.Column(visible=False) as sep_page:
                    gr.Markdown("### Source Separation & Inference")
                    with gr.Row():
                        with gr.Column(scale=2, variant="panel"):
                            mix_audio = gr.Audio(label="Upload Mixture (Multiple Patients)", type="filepath")
                            gr.Markdown("#### Patient Assignment")
                            with gr.Row():
                                sep_p1 = gr.Dropdown(label="Source 1", choices=get_registered_patient_choices(), value="Unassigned")
                                sep_p2 = gr.Dropdown(label="Source 2", choices=get_registered_patient_choices(), value="Unassigned")
                                sep_p3 = gr.Dropdown(label="Source 3", choices=get_registered_patient_choices(), value="Unassigned")
                            submit_btn = gr.Button("Run Separation Pipeline", variant="primary")
                        with gr.Column(scale=1):
                            status_text = gr.Textbox(label="Process Status", interactive=False, elem_classes="vibrant-status")
                            history_status_text = gr.Textbox(label="History Logging", interactive=False, elem_classes="vibrant-status")

                    gr.Markdown("#### Separated Patient Data")
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            out_audio_1 = gr.Audio(label="Patient 1 Audio")
                            out_wave_1 = gr.Image(label="Waveform 1")
                        with gr.Column(variant="panel"):
                            out_audio_2 = gr.Audio(label="Patient 2 Audio")
                            out_wave_2 = gr.Image(label="Waveform 2")
                        with gr.Column(variant="panel"):
                            out_audio_3 = gr.Audio(label="Patient 3 Audio")
                            out_wave_3 = gr.Image(label="Waveform 3")

                    gr.Markdown("#### Immediate Findings")
                    monitor_table_small = gr.Dataframe(
                        headers=["Patient Name", "overall_state", "mean_wheeze_prob", "mean_crackle_prob", "breathing_rate_mean", "comment"],
                        interactive=False,
                    )

                # History Page
                with gr.Column(visible=False) as history_page:
                    gr.Markdown("### Patient History Search")
                    with gr.Row():
                        search_query = gr.Textbox(label="Search by Patient Name or Audio ID", placeholder="Enter name...", elem_classes="vibrant-status")
                        search_button = gr.Button("Search History", variant="primary")
                                    
                    history_results = gr.Dataframe(
                        headers=["Patient Name", "overall_state", "mean_wheeze_prob", "mean_crackle_prob", "breathing_rate_mean", "comment"],
                        interactive=False,
                    )
                    history_msg = gr.Textbox(label="Search Results", interactive=False, elem_classes="vibrant-status")

                # Live Monitoring Page
                with gr.Column(visible=False) as live_mon_page:
                    gr.Markdown("## Live Audio Monitoring")

                    # Start/Stop buttons at the very top alone
                    with gr.Row():
                        start_live_btn = gr.Button("Start Live Monitoring", variant="primary")
                        stop_live_btn = gr.Button("Stop Live Monitoring", variant="stop")
                    
                    # Monitor Slots row - Stretching along one line
                    with gr.Row():
                        select_patient_1 = gr.Dropdown(
                            label="Monitor Slot 1",
                            choices=get_registered_patient_choices(),
                            value="Unassigned",
                            interactive=True,
                        )
                        select_patient_2 = gr.Dropdown(
                            label="Monitor Slot 2",
                            choices=get_registered_patient_choices(),
                            value="Unassigned",
                            interactive=True,
                        )
                        select_patient_3 = gr.Dropdown(
                            label="Monitor Slot 3",
                            choices=get_registered_patient_choices(),
                            value="Unassigned",
                            interactive=True,
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            live_mic_input = gr.Audio(
                                sources=["microphone"],
                                streaming=True,
                                label="Live Microphone Input",
                                interactive=False,
                            )
                        with gr.Column(scale=2):
                            gr.Markdown("#### Live Separated Sources")
                            with gr.Row():
                                live_out_audio_1 = gr.Audio(label="Live Patient 1", interactive=False)
                                live_out_audio_2 = gr.Audio(label="Live Patient 2", interactive=False)
                                live_out_audio_3 = gr.Audio(label="Live Patient 3", interactive=False)

                    gr.Markdown("#### Live Findings")
                    live_monitor_table = gr.Dataframe(
                        headers=["Patient Name", "overall_state", "mean_wheeze_prob", "mean_crackle_prob", "breathing_rate_mean", "comment"],
                        interactive=False,
                    )
                    live_monitor_status = gr.Textbox(label="Live Status", interactive=False, elem_classes="vibrant-status")

        # --- Tab Routing Mechanics ---
        def nav_reg(): return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        def nav_sep(): 
            choices = get_registered_patient_choices()
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
        def nav_his(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        def nav_live(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        btn_register.click(nav_reg, outputs=[reg_page, sep_page, history_page, live_mon_page], queue=False)
        btn_separation.click(nav_sep, outputs=[reg_page, sep_page, history_page, live_mon_page, sep_p1, sep_p2, sep_p3], queue=False)
        btn_history.click(nav_his, outputs=[reg_page, sep_page, history_page, live_mon_page], queue=False)
        btn_live_mon.click(nav_live, outputs=[reg_page, sep_page, history_page, live_mon_page], queue=False)

        # --- Interactive Trigger Bindings ---
        register_btn.click(
            fn=register_patients,
            inputs=[ref_audio_1, ref_audio_2, ref_audio_3, patient_name_1, patient_name_2, patient_name_3],
            outputs=[register_status, select_patient_1, select_patient_2, select_patient_3, sep_p1, sep_p2, sep_p3],
        )

        submit_btn.click(
            fn=predict,
            inputs=[mix_audio, sep_p1, sep_p2, sep_p3],
            outputs=[out_audio_1, out_audio_2, out_audio_3, out_wave_1, out_wave_2, out_wave_3, status_text, monitor_table_small, history_status_text],
        )

        search_button.click(
            fn=search_history,
            inputs=[search_query],
            outputs=[history_results, history_msg],
        )

        start_live_btn.click(
            fn=start_live_monitoring_session,
            inputs=[select_patient_1, select_patient_2, select_patient_3],
            outputs=[
                live_audio_buffer_state,
                live_separated_buffers_state,
                live_patient_managers_state,
                live_current_timestamps_state,
                live_monitor_table,
                live_patient_names_state,
                live_monitor_status,
                live_mic_input, 
                start_live_btn, 
            ],
            queue=False, 
        )

        live_mic_input.stream(
            fn=process_live_audio_stream,
            inputs=[
                live_mic_input,
                live_audio_buffer_state,
                live_separated_buffers_state,
                live_patient_managers_state,
                live_current_timestamps_state,
                live_patient_names_state,
            ],
            outputs=[live_audio_buffer_state, live_separated_buffers_state, live_patient_managers_state, live_current_timestamps_state, live_monitor_table, live_monitor_status, live_out_audio_1, live_out_audio_2, live_out_audio_3, live_patient_names_state],
        )

        stop_live_btn.click(
            fn=stop_live_monitoring_session,
            inputs=[],
            outputs=[live_audio_buffer_state, live_separated_buffers_state, live_patient_managers_state, live_current_timestamps_state, live_monitor_table, live_monitor_status, live_mic_input, start_live_btn, live_out_audio_1, live_out_audio_2, live_out_audio_3],
            queue=False,
        )

    return demo


# ==========================================
# 5. EXECUTION CONTAINER LIFECYCLE
# ==========================================
app = create_ui()
print("UI created")

if __name__ == "__main__":
    print("Launching Gradio...")

    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        theme=gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate"
        ),
        css=css_styles
    )
