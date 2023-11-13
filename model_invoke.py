from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import torch
import json

model = hf_hub_download(repo_id="Remilistrasza/epiCRealism", filename="epicrealism_naturalSinRC1VAE.safetensors")
pipe = StableDiffusionPipeline.from_single_file(model,
                                                torch_dtype=torch.float16, 
                                                safety_checker=None)
pipe = pipe.to("cuda")
pipe.safety_checker = None

def invoke(input_text):
    try:
        input_json = json.loads(input_text)
        prompt = input_json['prompt']
        negative_prompt = input_json.get('negative_prompt', "")
        steps = int(input_json.get('steps', 25))
    except:
        prompt = input_text
        negative_prompt = ""
        steps = 50
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]
    image.save("generated_image.png")
    return "generated_image.png"
