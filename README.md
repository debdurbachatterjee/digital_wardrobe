# 👕 Digital Wardrobe 👚

A smart digital wardrobe application that helps you manage your clothing items and get AI-powered outfit suggestions. This application uses computer vision and natural language processing to classify clothing items and provide personalized outfit recommendations.

## ✨ Features

- Upload and process clothing items
- Automatic background removal
- AI-powered clothing classification
- Natural language outfit suggestions

## 🛠️ Technologies Used

- [Gradio](https://www.gradio.app/) - Web interface framework
- [Transformers](https://huggingface.co/transformers/) - Machine learning library
- [FashionCLIP from Hugging Face](https://huggingface.co/patrickjohncyh/fashion-clip) - AI model for fashion understanding
- [Rembg](https://github.com/danielgatis/rembg) - Background removal tool

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/debdurbachatterjee/digital_wardrobe
cd digital_wardrobe
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📖 Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://127.0.0.1:7860)

3. Use the application:
   - **Add to Wardrobe**: Upload photos of your clothing items. The app will automatically remove backgrounds and save them.
   - **Outfit Suggestions**: Describe the type of outfit you're looking for (e.g., "formal wear for a business meeting") and get personalized suggestions from your wardrobe.

## 📂 Project Structure

```
digital_wardrobe/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── data/
    ├── images/         # Stored wardrobe images
```

## 🔜 Next Steps

- [ ] Enhance image processing pipeline with diffusion model to generate studio-quality outfit photos from raw uploads (remove occlusions, clean up backgrounds, standardize lighting, scale, center, crop etc)
