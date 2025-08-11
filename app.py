from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from rembg import remove
import gradio as gr
import torch
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
    return processed_image

def get_wardrobe_embeddings():
    """Load all wardrobe images and compute their embeddings"""
    wardrobe_images = []
    image_paths = []
    
    for img_path in IMG_DIR.glob("*.png"):
        image = Image.open(img_path)
        wardrobe_images.append(image)
        image_paths.append(img_path)
    
    if not wardrobe_images:
        return None, None, None
        
    # Process all images in a batch
    inputs = processor(
        images=wardrobe_images,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    return image_features, wardrobe_images, image_paths

def suggest_outfits(prompt, top_k=3):
    """Find the best matching clothes for a given prompt"""
    image_features, wardrobe_images, image_paths = get_wardrobe_embeddings()
    
    if image_features is None:
        return [], "No images found in wardrobe"
        
    # Process the text prompt
    text_inputs = processor(
        text=[prompt],
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Calculate similarity scores
    similarity = torch.nn.functional.cosine_similarity(
        text_features, image_features, dim=1
    )
    
    # Get top-k matches
    top_k_indices = similarity.argsort(descending=True)[:top_k]
    top_k_images = [wardrobe_images[idx] for idx in top_k_indices]
    top_k_paths = [image_paths[idx] for idx in top_k_indices]
    top_k_scores = [similarity[idx].item() for idx in top_k_indices]

    return top_k_images, f"Found {len(top_k_images)} matching items"

def main():
    with gr.Blocks() as app:
        gr.Markdown("# 👕 Digital Wardrobe ✨")
        
        with gr.Tabs() as tabs:
            # Wardrobe Management Tab
            with gr.Tab("📸 Add to Wardrobe"):
                with gr.Column():
                    input_image = gr.Image(label="📷 Upload Your Item")
                    process_btn = gr.Button("✨ Process, Classify and Upload", variant="primary")
                    # category_output = gr.Textbox(label="Detected Category")
                
                process_btn.click(
                    fn=process,
                    inputs=[input_image],
                    outputs=[input_image]
                )
            
            # Outfit Suggestions Tab
            with gr.Tab("🎨 Outfit Suggestions"):
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="💭 What kind of outfit are you looking for?",
                        placeholder="E.g., formal wear for a business meeting, casual weekend outfit, party dress...",
                        scale=2
                    )
                    suggest_btn = gr.Button("🔍 Find Perfect Outfits", variant="primary")
                    gallery_output = gr.Gallery(
                        label="👗 Suggested Items",
                        show_label=True,
                        elem_id="gallery",
                        columns=[3],
                        height="auto"
                    )
                    suggestion_output = gr.Textbox(label="📝 Results")
                
                suggest_btn.click(
                    fn=suggest_outfits,
                    inputs=[prompt_input],
                    outputs=[gallery_output, suggestion_output]
                )
    
    app.launch()

if __name__ == "__main__":
    main()
