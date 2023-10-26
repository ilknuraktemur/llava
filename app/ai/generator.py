import textwrap
from io import BytesIO

import requests
import torch
from typing import Any, Callable, Mapping, Tuple
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
import logging

disable_torch_init()

class LLaVa:
    
    """This module will handle the inference pipeline for the pretrained LLaVa model"""
     
    def __init__(
        self,
        model_n: str = "4bit/llava-v1.5-13b-3GB",
      
    ): 
        self.model_n = model_n
        
        try:
            model_name = get_model_name_from_path(model_n)
            self.tokenizer, self.model ,self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_n, model_base=None, model_name=model_name, load_4bit=True
            )           
            logging.info("model loaded!")

        except Exception:
            raise RuntimeError(
                f"Model is deprecated; Could not load model: {self.model_n}"
            )
            
    def read_input(self, form: Mapping[str, Any]) -> Tuple[str, str]:
        """Read prompt and other arguments from the form."""

        prompt = form.get("prompt", None)
        url=form.get("url")

        if not prompt:
            raise ValueError("prompt did not set in request")
        return prompt, url      
            
    def load_image(self,image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image= Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def create_prompt(self,prompt: str):
        CONV_MODE = "llava_v1"
        conv = conv_templates[CONV_MODE].copy()
        roles = conv.roles
        prompt = DEFAULT_IMAGE_TOKEN +"\n" + prompt
        conv.append_message(roles[0],prompt)
        conv.append_message(roles[1],None)
        return conv.get_prompt(),conv
    
    def process_image(self,image):
        args = {"image_aspect_ratio:" "pad"}
        image_tensor = process_images([image],self.image_processor, args)
        return image_tensor.to(self.model.device, dtype=torch.float16)
    
    def ask_image(self,form: Mapping[str, Any]) -> Callable: 
        (prompt,url) = self.read_input(form)
        image=self.load_image(url)
        image_tensor = self.process_image(image)
        prompt, conv= self.create_prompt(prompt)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            keywords=[stop_str], tokenizer=self.tokenizer, input_ids=input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.01,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            return self.tokenizer.decode(
                output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
    
print("defining model")
llava = LLaVa()
            


