import gradio as gr
from PIL import Image
import torch
from torchvision import transforms, models
from rembg import remove
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Constants
ORIGINAL_IMG_DIR = Path("data/images/original")
PROCESSED_IMG_DIR = Path("data/images/processed")

# Ensure directories exist
ORIGINAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMG_DIR.mkdir(parents=True, exist_ok=True)

def classify_image(image):
    # Placeholder for classification - we'll implement this later
    # For now, return a dummy category
    return {"Tops": 0.8, "Bottoms": 0.1, "Dresses": 0.1}

def process_image(image):
    if image is None:
        return None, None
    
    # Convert from numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Remove background
    processed = remove(image)
    
    # Convert processed image to numpy array for display
    if isinstance(processed, Image.Image):
        processed = np.array(processed)
    
    # Classify
    category = classify_image(processed)
    
    return processed, category

def save_to_wardrobe(image, category):
    if image is None:
        return "No image to save!"
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"item_{timestamp}.png"
    
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Save processed image
    image.save(PROCESSED_IMG_DIR / filename)
    
    return f"Saved to wardrobe as {filename}"

def main():
    with gr.Blocks() as app:
        gr.Markdown("# Digital Wardrobe")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image")
                process_btn = gr.Button("Process Image")
            
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                category_output = gr.Label(label="Category")
                save_btn = gr.Button("Add to Wardrobe")
        
        # Handle image processing
        process_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, category_output]
        )
        
        # Handle saving to wardrobe
        save_btn.click(
            fn=save_to_wardrobe,
            inputs=[output_image, category_output],
            outputs=[gr.Text(label="Status")]
        )
    
    app.launch()

if __name__ == "__main__":
    main()
