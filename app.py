import gradio as gr
from tempfile import NamedTemporaryFile
import time
import torch
from pydub import AudioSegment

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load the model (modify this to your specific model and its path)
MODEL = MusicGen.get_pretrained('facebook/musicgen-small')
MODEL.set_generation_params(duration=8)


def apply_audio_effects(audio_path, treble, bass):
    sound = AudioSegment.from_wav(audio_path)
    
    # Apply treble and bass effects
    sound = sound.low_pass_filter(bass)
    sound = sound.high_pass_filter(treble)
    
    # Save the modified audio back
    modified_audio_path = audio_path.replace(".wav", "_modified.wav")
    sound.export(modified_audio_path, format="wav")
    
    return modified_audio_path


def _do_predictions(texts, melodies, duration, treble=3000, bass=300, progress=False, gradio_progress=None, **gen_kwargs):
    if melodies:
        MODEL.set_generation_params(duration=duration)
        processed_melody = torch.tensor(melodies).unsqueeze(0)
        outputs = MODEL.generate_with_chroma(processed_melody, **gen_kwargs)
    else:
        MODEL.set_generation_params(duration=duration)
        outputs = MODEL.generate(texts, **gen_kwargs)

    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            # Apply audio effects
            modified_audio_path = apply_audio_effects(file.name, treble, bass)
            pending_videos.append(modified_audio_path)
            out_wavs.append(modified_audio_path)
    
    return pending_videos, out_wavs


def predict_full(model, text, melody, duration, topk, topp, temperature, cfg_coef, treble, bass, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], [melody], duration, treble=treble, bass=bass, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)
    
    return videos[0], wavs[0]


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            Customize your music generation with additional audio effects.
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                      "facebook/musicgen-large"], label="Model", value="facebook/musicgen-large", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Row():
                    treble = gr.Slider(minimum=1000, maximum=10000, value=3000, label="Treble", interactive=True)
                    bass = gr.Slider(minimum=20, maximum=500, value=300, label="Bass", interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_full, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef, treble, bass],
                     outputs=[output, audio_output])

        interface.launch(**launch_kwargs)


def load_model(model_name):
    global MODEL
    MODEL = MusicGen.get_pretrained(model_name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the MusicGen Gradio app.")
    parser.add_argument('--share', action='store_true', help="Whether to share the Gradio app publicly.")
    args = parser.parse_args()
    launch_kwargs = {"share": args.share}
    ui_full(launch_kwargs)


if __name__ == "__main__":
    main()
