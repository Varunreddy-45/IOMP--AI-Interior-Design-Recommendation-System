0# =========================================
# PRO AI ROOM DESIGNER (FIXED & WORKING)
# =========================================

# -------- INSTALL DEPENDENCIES --------
!pip install diffusers[torch] accelerate transformers safetensors opencv-python pillow ipywidgets --quiet

# -------- IMPORTS --------
import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
from google.colab import files
import os, zipfile

# -------- GPU CHECK --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------- LOAD MODELS --------
print("⏳ Loading Stable Diffusion + ControlNet (first time takes 2–3 min)...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.to(device)

print("✅ Model loaded successfully")

# -------- UTILITIES --------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def resize_image(img, size=768):
    w, h = img.size
    scale = min(size / w, size / h, 1.0)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def generate_canny(image):
    gray = np.array(image.convert("L"))
    edges = cv2.Canny(gray, 80, 160)
    return Image.fromarray(edges)

def get_prompt(room_type, style):
    return (
        f"complete interior redesign of an empty {room_type.lower()}, "
        f"{style.lower()} interior style, "
        "add realistic furniture, sofa, beds, tables, cabinets, decor, "
        "change wall colors, change flooring, "
        "professional interior design, "
        "high-end furniture, correct scale, "
        "realistic lighting, interior photography"
    )

NEGATIVE_PROMPT = (
    "distorted perspective, wrong scale, oversized furniture, "
    "floating objects, blurry, cartoon, painting, low quality"
)

# -------- UI WIDGETS --------
upload_btn = widgets.FileUpload(
    accept=".jpg,.jpeg,.png",
    multiple=False,
    description="Upload Empty Room Image"
)

room_dropdown = widgets.Dropdown(
    options=["Living Room", "Bedroom", "Kitchen"],
    value="Living Room",
    description="Room Type:"
)

style_dropdown = widgets.Dropdown(
    options=["Modern", "Minimal", "Luxury", "Scandinavian"],
    value="Modern",
    description="Style:"
)

strength_slider = widgets.FloatSlider(
    value=0.80,        # 🔥 STRONG REDESIGN
    min=0.60,
    max=0.90,
    step=0.05,
    description="Redesign Strength:"
)

variation_slider = widgets.IntSlider(
    value=3,
    min=1,
    max=5,
    step=1,
    description="Variations:"
)

generate_btn = widgets.Button(
    description="🎨 Generate Interior Designs",
    button_style="success"
)

output = widgets.Output()

display(upload_btn)
display(room_dropdown, style_dropdown, strength_slider, variation_slider, generate_btn, output)

# -------- GENERATION LOGIC --------
def on_generate(b):
    output.clear_output()

    if not upload_btn.value:
        with output:
            print("❌ Please upload an empty room image")
        return

    filename = list(upload_btn.value.keys())[0]
    content = upload_btn.value[filename]["content"]
    with open(filename, "wb") as f:
        f.write(content)

    original = load_image(filename)
    base_image = resize_image(original, 768)
    canny = generate_canny(base_image)

    room = room_dropdown.value
    style = style_dropdown.value
    strength = strength_slider.value
    variations = variation_slider.value

    os.makedirs("results", exist_ok=True)
    images = []

    with output:
        print("⏳ Generating redesigned interiors (this WILL change the room)...")

        for i in range(variations):
            print(f"→ Variation {i+1}/{variations}")

            with torch.autocast("cuda"):
                result = pipe(
                    prompt=get_prompt(room, style),
                    negative_prompt=NEGATIVE_PROMPT,
                    image=base_image,
                    control_image=canny,
                    strength=strength,
                    guidance_scale=8.5,
                    num_inference_steps=35,
                    controlnet_conditioning_scale=0.55  # 🔥 KEY FIX
                ).images[0]

            path = f"results/design_{i+1}.png"
            result.save(path)
            images.append(result)

        # ----- SIDE-BY-SIDE DISPLAY -----
        combined = Image.new("RGB", (base_image.width * (variations + 1), base_image.height))
        combined.paste(base_image, (0, 0))

        for idx, img in enumerate(images):
            combined.paste(img, ((idx + 1) * base_image.width, 0))

        display(combined)

        combined.save("AI_Room_Design_Comparison.png")

        # ----- ZIP DOWNLOAD -----
        zip_name = "AI_Room_Designs.zip"
        with zipfile.ZipFile(zip_name, "w") as zipf:
            zipf.write("AI_Room_Design_Comparison.png")
            for i in range(variations):
                zipf.write(f"results/design_{i+1}.png")

        print("✅ Interior redesign completed!")
        files.download(zip_name)

generate_btn.on_click(on_generate)
