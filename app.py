import os
import tempfile

import gradio as gr
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

os.makedirs("outputs", exist_ok=True)

pipe: LTXPipeline | None = None


def load_model(cpu_offload: bool) -> LTXPipeline:
    global pipe
    if pipe is None:
        print("Loading LTX-Video model (first run may take several minutes)...")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        )
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    return pipe


def generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    fps: int,
    seed: int,
    cpu_offload: bool,
) -> str:
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    model = load_model(cpu_offload)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    result = model(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        generator=generator,
    )

    frames = result.frames[0]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="outputs")
    tmp.close()
    export_to_video(frames, tmp.name, fps=fps)
    return tmp.name


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


def on_generate(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    duration: str,
    steps: int,
    fps: int,
    seed: int,
    cpu_offload: bool,
) -> str:
    width, height = RESOLUTION_PRESETS[resolution]
    num_frames = DURATION_PRESETS[duration]
    return generate(prompt, negative_prompt, width, height, num_frames, steps, fps, seed, cpu_offload)


with gr.Blocks(title="RTX 5070 Video Generator") as demo:
    gr.Markdown(
        """
# RTX 5070 Video Generator
**Text-to-video** generation powered by [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) running on your GPU.
> First generation will download the model (~8 GB) and may take a few minutes.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="A cinematic shot of a woman walking through neon-lit city streets at night, slow motion",
                lines=3,
            )
            negative_box = gr.Textbox(
                label="Negative Prompt",
                value="worst quality, inconsistent motion, blurry, jittery, distorted",
                lines=2,
            )

            with gr.Row():
                resolution_dd = gr.Dropdown(
                    label="Resolution",
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value="704×480  (standard)",
                )
                duration_dd = gr.Dropdown(
                    label="Duration",
                    choices=list(DURATION_PRESETS.keys()),
                    value="~2 sec  (49 frames)",
                )

            with gr.Row():
                steps_slider = gr.Slider(
                    label="Inference Steps", minimum=10, maximum=60, value=40, step=5
                )
                fps_slider = gr.Slider(
                    label="Output FPS", minimum=8, maximum=30, value=24, step=1
                )

            with gr.Row():
                seed_box = gr.Number(label="Seed  (-1 = random)", value=-1, precision=0)
                offload_check = gr.Checkbox(
                    label="CPU Offload  (use if < 12 GB VRAM)", value=False
                )

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Generated Video", height=480)

    generate_btn.click(
        fn=on_generate,
        inputs=[
            prompt_box,
            negative_box,
            resolution_dd,
            duration_dd,
            steps_slider,
            fps_slider,
            seed_box,
            offload_check,
        ],
        outputs=video_out,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Generation will be extremely slow on CPU.")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
