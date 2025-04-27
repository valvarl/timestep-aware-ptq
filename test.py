import time
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch
from tqdm import tqdm

torch_dtype = torch.bfloat16
device = "cuda"

model_id = "Freepik/flux.1-lite-8B"
transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
for block in (transformer.transformer_blocks + transformer.single_transformer_blocks):
    block.attn.processor.initialize_fake_quantization(block.attn)
# transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, 
#                                      skip_modules_pattern=["weight_scale", "weight_zero_oint", "act_stats", "act_qparams", "r_s", "t"])
print(transformer.transformer_blocks[0])

# Load the pipe
pipe = FluxPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch_dtype,
).to(device)

guidance_scale = 3.5
# n_steps = 28
n_steps = 15
seed = 42
calib_steps = 1
inference_steps = 1

# prompt_embeds = torch.load("prompt_embeds.pt").to(device)
# pooled_prompt_embeds = torch.load("pooled_prompt_embeds.pt").to(device)

with open("calib_data.txt", encoding="utf-8") as f:
    prompts = f.readlines()

for i, prompt in enumerate(prompts[:calib_steps]):
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            # prompt_embeds=prompt_embeds,
            # pooled_prompt_embeds=pooled_prompt_embeds,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        ).images[0]
    image.save(f"output_{i}.png")

for block in tqdm(transformer.transformer_blocks + transformer.single_transformer_blocks):
    block.attn.processor.freeze(block.attn)
    torch.cuda.synchronize()

for i, prompt in enumerate(prompts[calib_steps - 1: calib_steps - 1 + inference_steps]):
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            # prompt_embeds=prompt_embeds,
            # pooled_prompt_embeds=pooled_prompt_embeds,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        ).images[0]
    image.save(f"output_{calib_steps + i}.png")
