import os
import random
import threading
import time

import gradio as gr
import torch
from diffusers import LTXImageToVideoPipeline, LTXPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Lightricks/LTX-Video"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cached pipelines. The image-to-video pipeline is built from the text pipeline's
# components so the model weights are shared (no extra VRAM / no second download).
_txt_pipe: LTXPipeline | None = None
_img_pipe: LTXImageToVideoPipeline | None = None

# Set by the Cancel button; checked from the diffusers step callback to abort a run.
_cancel = threading.Event()


class _Cancelled(Exception):
    """Raised from the step callback when the user requests cancellation."""


def _place(pipe, cpu_offload: bool):
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)
    return pipe


def load_text_pipe(cpu_offload: bool) -> LTXPipeline:
    global _txt_pipe
    if _txt_pipe is None:
        print("Loading LTX-Video text-to-video model (first run may take several minutes)...")
        _txt_pipe = _place(
            LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16),
            cpu_offload,
        )
    return _txt_pipe


def load_image_pipe(cpu_offload: bool) -> LTXImageToVideoPipeline:
    global _img_pipe
    if _img_pipe is None:
        if _txt_pipe is not None:
            # Reuse the already-loaded weights — shares VRAM with the text pipeline.
            _img_pipe = LTXImageToVideoPipeline(**_txt_pipe.components)
        else:
            print("Loading LTX-Video image-to-video model (first run may take several minutes)...")
            _img_pipe = _place(
                LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16),
                cpu_offload,
            )
    return _img_pipe


def _step_callback(pipe, step, timestep, callback_kwargs):
    if _cancel.is_set():
        raise _Cancelled()
    return callback_kwargs


def _resolve_seed(seed: int) -> int:
    if seed is None or int(seed) < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _write_sidecar(video_path: str, info: dict):
    txt_path = os.path.splitext(video_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def _format_history(history: list) -> list:
    """Turn the history state into rows for the Dataframe (newest first)."""
    rows = []
    for i, h in enumerate(reversed(history), 1):
        prompt = h["prompt"]
        if len(prompt) > 60:
            prompt = prompt[:57] + "..."
        rows.append([i, h["time"], h["mode"], prompt, h["seed"], h["resolution"], h["frames"]])
    return rows


RESOLUTION_PRESETS = {
    "512×288  (16:9 fast)": (512, 288),
    "704×480  (standard)": (704, 480),
    "768×512  (16:9 quality)": (768, 512),
}

DURATION_PRESETS = {
    "~2 sec  (49 frames)": 49,
    "~4 sec  (97 frames)": 97,
    "~7 sec  (161 frames)": 161,
}

HISTORY_HEADERS = ["#", "Time", "Mode", "Prompt", "Seed", "Resolution", "Frames"]


def _run(mode: str, image, prompt, negative_prompt, width, height, num_frames,
         steps, fps, seed, cpu_offload, history, progress=gr.Progress(track_tqdm=True)):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    if mode == "i2v" and image is None:
        raise gr.Error("Please upload an input image for image-to-video.")

    _cancel.clear()
    used_seed = _resolve_seed(seed)
    generator = torch.Generator(device=DEVICE).manual_seed(used_seed)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=_step_callback,
    )

    try:
        if mode == "i2v":
            model = load_image_pipe(cpu_offload)
            result = model(image=image, **call_kwargs)
        else:
            model = load_text_pipe(cpu_offload)
            result = model(**call_kwargs)
    except _Cancelled:
        raise gr.Error("Generation cancelled.")

    frames = result.frames[0]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    resolution = f"{width}×{height}"
    filename = f"{stamp}_{mode}_seed{used_seed}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)
    export_to_video(frames, video_path, fps=fps)

    info = {
        "mode": "image-to-video" if mode == "i2v" else "text-to-video",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "num_frames": num_frames,
        "steps": steps,
        "fps": fps,
        "seed": used_seed,
        "model": MODEL_ID,
    }
    _write_sidecar(video_path, info)

    history = history + [{
        "time": time.strftime("%H:%M:%S"),
        "mode": "i2v" if mode == "i2v" else "t2v",
        "prompt": prompt,
        "seed": used_seed,
        "resolution": resolution,
        "frames": num_frames,
        "path": video_path,
    }]

    seed_msg = f"Seed used: {used_seed}  •  saved to {filename} (+ .txt with all settings)"
    return video_path, seed_msg, history, _format_history(history)


def request_cancel():
    _cancel.set()
    return "Cancelling after the current step..."


def make_t2v_handler():
    def handler(prompt, negative_prompt, resolution, duration, steps, fps, seed,
                cpu_offload, history, progress=gr.Progress(track_tqdm=True)):
        width, height = RESOLUTION_PRESETS[resolution]
        num_frames = DURATION_PRESETS[duration]
        return _run("t2v", None, prompt, negative_prompt, width, height, num_frames,
                    steps, fps, seed, cpu_offload, history, progress)
    return handler


def make_i2v_handler():
    def handler(image, prompt, negative_prompt, resolution, duration, steps, fps, seed,
                cpu_offload, history, progress=gr.Progress(track_tqdm=True)):
        width, height = RESOLUTION_PRESETS[resolution]
        num_frames = DURATION_PRESETS[duration]
        return _run("i2v", image, prompt, negative_prompt, width, height, num_frames,
                    steps, fps, seed, cpu_offload, history, progress)
    return handler


def _settings_controls():
    """Shared resolution / duration / steps / fps / seed / offload controls."""
    with gr.Row():
        resolution_dd = gr.Dropdown(
            label="Resolution", choices=list(RESOLUTION_PRESETS.keys()),
            value="704×480  (standard)",
        )
        duration_dd = gr.Dropdown(
            label="Duration", choices=list(DURATION_PRESETS.keys()),
            value="~2 sec  (49 frames)",
        )
    with gr.Row():
        steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=60, value=40, step=5)
        fps_slider = gr.Slider(label="Output FPS", minimum=8, maximum=30, value=24, step=1)
    with gr.Row():
        seed_box = gr.Number(label="Seed  (-1 = random)", value=-1, precision=0)
        offload_check = gr.Checkbox(label="CPU Offload  (use if < 12 GB VRAM)", value=False)
    return resolution_dd, duration_dd, steps_slider, fps_slider, seed_box, offload_check


with gr.Blocks(title="RTX 5070 Video Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# RTX 5070 Video Generator
**Text-to-video** and **image-to-video** generation powered by
[LTX-Video](https://huggingface.co/Lightricks/LTX-Video) running on your GPU.
> First generation will download the model (~8 GB) and may take a few minutes.
"""
    )

    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Text → Video"):
                    t2v_prompt = gr.Textbox(
                        label="Prompt", lines=3,
                        placeholder="A cinematic shot of a woman walking through neon-lit city "
                                    "streets at night, slow motion",
                    )
                    t2v_negative = gr.Textbox(
                        label="Negative Prompt", lines=2,
                        value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    )
                    (t2v_res, t2v_dur, t2v_steps, t2v_fps,
                     t2v_seed, t2v_offload) = _settings_controls()
                    with gr.Row():
                        t2v_btn = gr.Button("Generate Video", variant="primary", size="lg")
                        t2v_cancel = gr.Button("Cancel", variant="stop", size="lg")

                with gr.TabItem("Image → Video"):
                    i2v_image = gr.Image(label="Input Image", type="pil")
                    i2v_prompt = gr.Textbox(
                        label="Prompt", lines=3,
                        placeholder="The camera slowly zooms in as gentle waves ripple across the water",
                    )
                    i2v_negative = gr.Textbox(
                        label="Negative Prompt", lines=2,
                        value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    )
                    (i2v_res, i2v_dur, i2v_steps, i2v_fps,
                     i2v_seed, i2v_offload) = _settings_controls()
                    with gr.Row():
                        i2v_btn = gr.Button("Animate Image", variant="primary", size="lg")
                        i2v_cancel = gr.Button("Cancel", variant="stop", size="lg")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Generated Video", height=480)
            seed_info = gr.Textbox(label="Run info", interactive=False)

    gr.Markdown("### Generation history (this session)")
    history_table = gr.Dataframe(
        headers=HISTORY_HEADERS, interactive=False, wrap=True,
        col_count=(len(HISTORY_HEADERS), "fixed"),
    )

    t2v_event = t2v_btn.click(
        fn=make_t2v_handler(),
        inputs=[t2v_prompt, t2v_negative, t2v_res, t2v_dur, t2v_steps, t2v_fps,
                t2v_seed, t2v_offload, history_state],
        outputs=[video_out, seed_info, history_state, history_table],
    )
    t2v_cancel.click(fn=request_cancel, outputs=seed_info, queue=False, cancels=[t2v_event])

    i2v_event = i2v_btn.click(
        fn=make_i2v_handler(),
        inputs=[i2v_image, i2v_prompt, i2v_negative, i2v_res, i2v_dur, i2v_steps, i2v_fps,
                i2v_seed, i2v_offload, history_state],
        outputs=[video_out, seed_info, history_state, history_table],
    )
    i2v_cancel.click(fn=request_cancel, outputs=seed_info, queue=False, cancels=[i2v_event])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Generation will be extremely slow on CPU.")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
