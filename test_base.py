import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

torch_dtype = torch.bfloat16
device = "cuda"

model_id = "Freepik/flux.1-lite-8B"
transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

# Load the pipe
pipe = FluxPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch_dtype,
).to(device)

pipe.text_encoder = pipe.text_encoder.cpu()
pipe.text_encoder_2 = pipe.text_encoder_2.cpu()

# Inference
prompt = "An astronaut floating in space, but instead of stars, there are massive, glowing jellyfish drifting through the void. Their tentacles ripple with shifting, colorful lights, creating a mesmerizing display in the darkness of space. The astronaut moves between them, touching their delicate, translucent bodies, which react to touch with bursts of light. The scene feels dreamlike, as if the astronaut has entered a cosmic undersea world."

guidance_scale = 3.5
n_steps = 28
seed = 42

prompt_embeds = torch.load("prompt_embeds.pt").to(device)
pooled_prompt_embeds = torch.load("pooled_prompt_embeds.pt").to(device)

with torch.inference_mode():
    image = pipe(
        # prompt=prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        height=512,
        width=512,
    ).images[0]
image.save("output.png")