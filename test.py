import time
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch
from tqdm import tqdm

torch_dtype = torch.bfloat16
device = "cuda"

model_id = "/home/vwx1285019/flux.1-lite-8B"
transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16, device=device)
for block in (transformer.transformer_blocks + transformer.single_transformer_blocks):
    block.attn.processor.initialize_fake_quantization(block.attn)
# transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, 
#                                      skip_modules_pattern=["weight_scale", "weight_zero_oint", "act_stats", "act_qparams", "r_s", "t"])
print(transformer.transformer_blocks[0])
transformer.eval()

# Load the pipe
pipe = FluxPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch_dtype,
)
pipe.enable_model_cpu_offload()
guidance_scale = 3.5
n_steps = 28
seed = 42
calib_steps = 11

with open("calib_data.txt", encoding="utf-8") as f:
    prompts = f.readlines()

for j in range(5):
    for i, prompt in enumerate(prompts[:calib_steps]):
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
            ).images[0]
        image.save(f"./output/output_{j}_{i}.png")

    with torch.no_grad():
        for block in tqdm(transformer.transformer_blocks + transformer.single_transformer_blocks):
            block.attn.processor.freeze(block.attn)
            torch.cuda.synchronize()
