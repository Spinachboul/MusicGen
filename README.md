# MusicGen Gradio App

This repository contains a Gradio-based web application for generating music using the Facebook MusicGen model, with additional options to apply treble and bass effects to the generated audio.

## Web User Interface

![WhatsApp Image 2024-06-27 at 16 35 19_79d2f085](https://github.com/Spinachboul/MusicGen/assets/105979087/5c178199-c06c-4c96-ac91-8c941f0288ed)


## Features

- **Text-Based Music Generation:** Generate music by providing descriptive text.
- **Melody Conditioning:** Optionally condition the generated music on a melody file.
- **Audio Effects:** Apply treble and bass effects to the generated audio.
- **Interactive Interface:** User-friendly interface built with Gradio.

## Requirements

- Python 3.8 or higher
- `gradio`
- `torch`
- `pydub`
- `audiocraft`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/musicgen-gradio-app.git
    cd musicgen-gradio-app
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Gradio app:
    ```bash
    python app.py
    ```

2. To share the app publicly, use the `--share` flag:
    ```bash
    python app.py --share
    ```

## File Structure

- `app.py`: Main script containing the Gradio app logic.
- `requirements.txt`: List of required Python packages.

## Gradio App Interface

### Input

- **Input Text:** Descriptive text for generating music.
- **Condition on a melody (optional):** Upload a melody file.
- **Model:** Select the MusicGen model variant (`facebook/musicgen-melody`, `facebook/musicgen-medium`, `facebook/musicgen-small`, `facebook/musicgen-large`).
- **Duration:** Duration of the generated music in seconds.
- **Top-k:** Top-k sampling value.
- **Top-p:** Top-p sampling value.
- **Temperature:** Sampling temperature.
- **Classifier Free Guidance:** CFG coefficient.
- **Treble:** Treble effect value.
- **Bass:** Bass effect value.

### Output

- **Generated Music:** Video preview of the generated music.
- **Generated Music (wav):** Downloadable wav file of the generated music.

## Functions

### `apply_audio_effects`

Applies treble and bass effects to the generated audio.

**Parameters:**
- `audio_path`: Path to the audio file.
- `treble`: Treble effect value.
- `bass`: Bass effect value.

**Returns:**
- Path to the modified audio file.

### `_do_predictions`

Handles the music generation and audio processing.

**Parameters:**
- `texts`: List of input texts.
- `melodies`: List of input melodies.
- `duration`: Duration of the generated music.
- `treble`: Treble effect value.
- `bass`: Bass effect value.
- `progress`: Progress indicator.
- `gradio_progress`: Gradio progress object.
- `gen_kwargs`: Additional generation parameters.

**Returns:**
- List of paths to the generated and modified audio files.

### `predict_full`

Main prediction function called by the Gradio interface.

**Parameters:**
- `model`: Selected model name.
- `text`: Input text for music generation.
- `melody`: Uploaded melody file.
- `duration`: Duration of the generated music.
- `topk`: Top-k sampling value.
- `topp`: Top-p sampling value.
- `temperature`: Sampling temperature.
- `cfg_coef`: CFG coefficient.
- `treble`: Treble effect value.
- `bass`: Bass effect value.
- `progress`: Gradio progress object.

**Returns:**
- Path to the generated video.
- Path to the generated audio file.

### `ui_full`

Defines the Gradio user interface.

**Parameters:**
- `launch_kwargs`: Arguments for launching the Gradio interface.

### `load_model`

Loads the specified MusicGen model.

**Parameters:**
- `model_name`: Name of the model to load.

### `main`

Main function to run the Gradio app.

## Acknowledgments

- [Facebook MusicGen](https://github.com/facebookresearch/audiocraft) for providing the pretrained models and codebase.
