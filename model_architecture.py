import torch
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn


class ImageCaptionGenerationWithAttention(nn.Module):
    def __init__(self, vit_model, bart_model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vit = vit_model
        self.bart = bart_model
        self.visual_projection = nn.Linear(
            vit_model.config.hidden_size, bart_model.config.d_model)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        vit_outputs = self.vit(pixel_values)
        if isinstance(vit_outputs, tuple):
            last_hidden_state = vit_outputs[0]
        else:
            last_hidden_state = vit_outputs.last_hidden_state

        visual_features = self.visual_projection(last_hidden_state)

        if input_ids is not None:
            decoder_outputs = self.bart(
                labels=input_ids,
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=visual_features),
                return_dict=True
            )
            return decoder_outputs
        else:
            return visual_features

    def generate(self, pixel_values, max_length=50, num_beams=5, early_stopping=True):
        self.eval()
        with torch.no_grad():
            vit_outputs = self.vit(pixel_values)
            if isinstance(vit_outputs, tuple):
                last_hidden_state = vit_outputs[0]
            else:
                last_hidden_state = vit_outputs.last_hidden_state
            visual_features = self.visual_projection(last_hidden_state)
            generated_ids = self.bart.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=visual_features),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False
            )
            return generated_ids
