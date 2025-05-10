import gradio as gr
from model_architecture import ImageCaptionGenerationWithAttention
from transformers import BartForConditionalGeneration, BartTokenizer, ViTModel, ViTImageProcessor
import torch
from PIL import Image
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
HF_TOKEN = os.getenv('hf_token')


class GenerateCaptions:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        vit_model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", token=HF_TOKEN).to(self.device)
        bart_model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-base").to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = ImageCaptionGenerationWithAttention(
            vit_model, bart_model, self.tokenizer)
        self.model.load_state_dict(torch.load(
            'image_captioning_model_state_dict.pt', map_location=self.device))
        self.model.eval()

    def generate_caption(self, frame, max_length=50, num_beams=5):
        try:
            image_pixel_values = self.processor(
                frame, return_tensors="pt").pixel_values
            generated_caption_ids = self.model.generate(
                image_pixel_values, max_length, num_beams)
            return self.tokenizer.decode(generated_caption_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(e)
            print(traceback.format_exc())


gc = GenerateCaptions()

demo = gr.Interface(
    fn=gc.generate_caption,
    inputs=gr.Image(type='pil'),
    outputs="text",
    title="Image Caption Generation",
    examples=['Image.jpg', 'Image 2.jpg'],
    submit_btn='Generate Caption',
    flagging_mode='never'
)


demo.launch()
