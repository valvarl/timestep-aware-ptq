import torch
import torch.nn.functional as F
from diffusers.models.fake_quant_linear import FakeQuantLinear

x = torch.load("x.pt")
w = torch.load("weight.pt")
b = torch.load("bias.pt")

print(x.dtype, w.dtype, b.dtype)

fq = FakeQuantLinear(3072, 3072, bias=False).cuda()
fq.weight.data.copy_(w.data)
fq.bias.data.copy_(b.data)

# fq.weight.data.copy_(torch.randn(10, 3072, dtype=torch.bfloat16).cuda())

bf16_out = fq(x, 0)

fq.freeze()

quant_out = fq(x, 0)

print(bf16_out[0, :10, :10] - quant_out[0, :10, :10])

# print(bf16_out[0, :, 259])

# print(quant_out[0, :, 259])

# print(bf16_out[0, :, 0])

# print(quant_out[0, :, 0])

mse = F.mse_loss(bf16_out.float(), quant_out.float())
print(f"MSE: {mse.item():.6f}")
