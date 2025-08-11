from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from rembg import remove
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# Initialize FashionCLIP model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Define clothing categories
CLOTHING_CATEGORIES = [
    "dress", "t-shirt", "pants", "jeans", "skirt", "blouse", 
    "sweater", "jacket", "coat", "shorts", "suit", "shoes",
    "boots", "sneakers", "sandals", "hat", "scarf", "bag"
]

IMG_DIR = Path("data/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

def classify_image(image):
    # Convert the image format if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Prepare text inputs for all categories
    text_inputs = [f"a photo of a {category}" for category in CLOTHING_CATEGORIES]
    
    # Process inputs
    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Get model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get the most likely category
    max_prob_idx = probs[0].argmax().item()
    category = CLOTHING_CATEGORIES[max_prob_idx]
    confidence = probs[0][max_prob_idx].item()
    
    return f"{category} ({confidence:.2%} confident)"

def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = IMG_DIR / f"item_{timestamp}.png"
    image.save(save_path)

def process_image(image):
    no_bg = remove(image)
    no_bg_pil = Image.fromarray(no_bg)
    white_bg = Image.new('RGBA', no_bg_pil.size, 'WHITE')
    processed = Image.alpha_composite(white_bg, no_bg_pil)    
    return processed


def process(image):
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    processed_image = process_image(image_np)
    save_image(processed_image)

    category = classify_image(processed_image)
    return processed_image, category

def main():
    with gr.Blocks() as app:
        gr.Markdown("# Digital Wardrobe")
        
        with gr.Tabs() as tabs:
            # Wardrobe Management Tab
            with gr.Tab("Add to Wardrobe"):
                with gr.Column():
                    input_image = gr.Image(label="Upload Image")
                    process_btn = gr.Button("Process, Classify and Upload")
                    category_output = gr.Textbox(label="Detected Category")
                
                process_btn.click(
                    fn=process,
                    inputs=[input_image],
                    outputs=[input_image, category_output]
                )
    
    app.launch()

if __name__ == "__main__":
    main()
