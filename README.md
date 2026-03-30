# AI Room Designer

A simple Python interface for generating redesigned interior room concepts from an uploaded empty room image.

## Overview

`app.py` uses Stable Diffusion and ControlNet to transform empty room photos into styled interior design variations. The UI is built with `ipywidgets` and is intended to run in an interactive Python environment such as Google Colab or Jupyter.

## Features

- Upload an empty room image (`.jpg`, `.jpeg`, `.png`)
- Choose room type: Living Room, Bedroom, Kitchen
- Select design style: Modern, Minimal, Luxury, Scandinavian
- Control redesign strength and number of variations
- Generates multiple interior design outputs with a Canny edge-conditioned ControlNet pipeline
- Produces a comparison image and downloads a ZIP of generated results

## Requirements

- Python 3.10+
- `torch`
- `diffusers[torch]`
- `accelerate`
- `transformers`
- `safetensors`
- `opencv-python`
- `pillow`
- `ipywidgets`
- `google.colab` (if running in Google Colab)

Install dependencies with:

```bash
pip install diffusers[torch] accelerate transformers safetensors opencv-python pillow ipywidgets
```

## Usage

1. Open `app.py` in a Colab notebook or Jupyter environment.
2. Run the script cell by cell.
3. Upload an empty room image using the widget.
4. Select the room type, style, redesign strength, and variation count.
5. Click **Generate Interior Designs**.
6. Download the generated `AI_Room_Designs.zip` file.

## Notes

- The script uses CUDA if available; otherwise it falls back to CPU.
- Model loading may take several minutes on first run.
- The generated images are saved in `results/` and the final comparison image is saved as `AI_Room_Design_Comparison.png`.

## File Structure

- `app.py` — main script for generating redesigned room images
- `Output/` — optional output folder (existing workspace folder)

## License

Use this project responsibly and follow the licensing terms of the underlying model weights and libraries.
