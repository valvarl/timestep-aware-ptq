import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantLinear(nn.Module):
    """
    A linear layer that, in 'calibration' mode, collects per-timestep, per-channel
    activation min/max.  After calling `freeze()` it:
      1) computes the joint timestep-channel scaling vector r_s
      2) reparameterizes weight: W'[:,d] = W[:,d] * r_s[d], then quantizes it
      3) for each timestamp t, computes qparams of X^t / r_s
    On inference you call forward(x, timestamp), and it applies exactly Eq.(3)-(5).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype=torch.float16,
                 R_trunc: float = None):
        super().__init__()
        self.in_features = in_features
        self.dtype = dtype
        self.R_trunc = R_trunc

        # full-precision weight & bias
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=self.dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=self.dtype)
        ) if bias else None

        # weight quantization buffers
        self.register_buffer('weight_scale',      torch.ones(out_features, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, dtype=torch.int32))
        self.register_buffer('weight_int',        torch.zeros_like(self.weight, dtype=torch.int8))

        self.act_stats   = {}  # t -> {'min': Tensor[in_features], 'max': Tensor[in_features]}
        self.act_qparams = {}  # t -> {'scale': Tensor[in_features], 'zero_point': Tensor[in_features]}

        # the joint reparameterization vector r_s[d]
        self.register_buffer('r_s', torch.ones(in_features, dtype=torch.float32))

        # start in calibration mode
        self.collecting = True

    def forward(self, x: torch.Tensor, timestamp: int) -> torch.Tensor:
        """
        If collecting: record per-channel min/max for this timestamp.
        Else: apply joint reparam + fake quant with timestamp qparams.
        """
        if self.collecting:
            # per-channel stats over batch dim
            x_min = x.min(dim=0).values  # [in_features]
            x_max = x.max(dim=0).values  # [in_features]
            stats = self.act_stats.setdefault(timestamp, {'min': x_min.clone(),
                                                          'max': x_max.clone()})
            stats['min'].clamp_max_(x_min)
            stats['max'].clamp_min_(x_max)
            return F.linear(x, self.weight, self.bias)

        # Inference path
        if timestamp not in self.act_qparams:
            raise KeyError(f"No qparams for timestamp={timestamp}. Did you freeze()?")

        qmin, qmax = -128, 127
        qparams = self.act_qparams[timestamp]
        scale_a = qparams['scale']       # Tensor of shape [in_features]
        zp_a    = qparams['zero_point']  # Tensor of shape [in_features]


        # 1) quantize the re-parameterized activation
        x_rep = x / self.r_s                   # [B, in_features]
        x_int8 = torch.clamp((x_rep / scale_a).round() + zp_a,
                             qmin, qmax).to(torch.int8)  # [B, in_features]
        
        # 2) dequantize activations per-channel
        #    (x_int8.float() - zp_a) is [B, in_features] minus [in_features] → [B, in_features]
        x_deq = (x_int8.float() - zp_a) * scale_a        # [B, in_features]

        # 3) dequant weights per-channel
        #    weight_zero_point: [out_features], weight_scale: [out_features]
        w_deq = (self.weight_int.float() - self.weight_zero_point[:, None]) \
                * self.weight_scale[:, None]              # [out_features, in_features]
        
        # 4) linear + bias
        y = F.linear(x_deq, w_deq, bias=(self.bias.float() if self.bias is not None else None))

        # 5) cast back
        return y.to(self.dtype)

    @torch.no_grad()
    def freeze(self):
        """
        1) Compute joint r_s from all timestamps (§5, Eq (5))
        2) Reparameterize and quantize weight W' = W * r_s
        3) For each timestamp, compute qparams of X^t / r_s
        """
        qmin, qmax = -128, 127
        all_ts = sorted(self.act_stats.keys())
        # 1) build per-timestep r_t[d]
        #    and accumulate numerator & denominator for r_s
        numer = torch.zeros(self.in_features, dtype=torch.float32, device=self.weight.device)
        denom = torch.zeros(self.in_features, dtype=torch.float32, device=self.weight.device)

        for t in all_ts:
            stats = self.act_stats[t]
            max_t = stats['max']  # Tensor[in_features]
            # s_tar^t = min_d max_t[d]
            s_tar = max_t.min()
            # r_t[d] = max_t[d] / s_tar
            r_t = max_t / s_tar
            # accumulate numerator & denom
            numer += r_t * max_t
            denom += max_t

        # joint r_s
        r_s = numer / denom
        # clamp if R_trunc given
        if self.R_trunc is not None:
            r_s = torch.clamp(r_s, max=self.R_trunc)
        self.r_s.copy_(r_s)

        # 2) reparameterize & quantize weight
        #    W' = W * r_s
        W_rep = self.weight.float() * r_s                  # [out, in]
        w_min = W_rep.min(dim=1).values                    # [out]
        w_max = W_rep.max(dim=1).values                    # [out]
        scale_w = (w_max - w_min) / (qmax - qmin)          # [out]
        scale_w = torch.where(scale_w>0, scale_w, 1e-8)
        zp_w    = torch.clamp((qmin - w_min/scale_w).round(),
                              qmin, qmax).to(torch.int32)  # [out]
        
        # quantize each row
        W_q = torch.clamp((W_rep/scale_w[:,None]).round()
                          + zp_w[:,None], qmin, qmax).to(torch.int8)

        self.weight_scale      = scale_w
        self.weight_zero_point = zp_w
        self.weight_int        = W_q

        # 3) compute per-timestamp activation qparams on X^t / r_s
        for t in all_ts:
            stats = self.act_stats[t]
            min_rep = stats['min'] / r_s
            max_rep = stats['max'] / r_s

            scale_a = (max_rep - min_rep) / (qmax - qmin)
            scale_a = torch.where(scale_a > 0, scale_a, torch.tensor(1e-8, device=scale_a.device))
            zp_a = torch.clamp((qmin - min_rep/scale_a).round(), qmin, qmax).to(torch.int32)

            self.act_qparams[t] = {
                'scale':      scale_a,
                'zero_point': zp_a
            }

        # switch to inference
        self.collecting = False

    def get_activation_qparams(self, timestamp: int):
        """Return (scale, zero_point) for a given timestamp after freeze()."""
        if timestamp not in self.act_qparams:
            raise KeyError(f"No activation qparams for timestamp {timestamp}.")
        p = self.act_qparams[timestamp]
        return p['scale'], p['zero_point']

torch.manual_seed(35)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель с двумя квантованными линейными блоками
class QuantizedModel(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.fc1 = FakeQuantLinear(10, 20, dtype=dtype)
        self.relu = nn.ReLU()
        self.fc2 = FakeQuantLinear(20, 5, dtype=dtype)

    def forward(self, x, timestamp: int) -> torch.Tensor:
        x = self.fc1(x, timestamp)
        x = self.relu(x)
        x = self.fc2(x, timestamp)
        return x
    
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Создаем две версии модели: fp16 и квантизованную
model_fp16 = SimpleModel().to(device, dtype=torch.float16).eval()
model_quant = QuantizedModel().to(device).eval()

# Копируем веса из оригинальной модели
with torch.no_grad():
    model_quant.fc1.weight.copy_(model_fp16.fc1.weight)
    model_quant.fc1.bias.copy_(model_fp16.fc1.bias)
    model_quant.fc2.weight.copy_(model_fp16.fc2.weight)
    model_quant.fc2.bias.copy_(model_fp16.fc2.bias)

# Сбор статистики: 10000 примеров для каждого timestep
with torch.no_grad():
    for _ in range(10000):
        sample = (torch.randn(1, 10, dtype=torch.float16, device=device) * 2) + 1.5
        if torch.rand(1).item() < 0.05:
            noise = torch.randn(1, device=device, dtype=torch.float16) * 50
            sample[:, 2] += noise
            noise = torch.randn(1, device=device, dtype=torch.float16) * 50
            sample[:, 7] += noise
        model_quant(sample, 0)

    for _ in range(10000):
        sample = (torch.randn(1, 10, dtype=torch.float16, device=device) * 6) - 3
        if torch.rand(1).item() < 0.05:
            noise = torch.randn(1, device=device, dtype=torch.float16) * 80
            sample[:, 1] += noise
            noise = torch.randn(1, device=device, dtype=torch.float16) * 80
            sample[:, 5] += noise
        model_quant(sample, 1)

model_quant.fc1.freeze()
model_quant.fc2.freeze()

# Проверка на тестовом батче
test_input_0 = (torch.randn(100, 10, dtype=torch.float16, device=device) * 2) + 1.5
test_input_1 = (torch.randn(100, 10, dtype=torch.float16, device=device) * 6) - 3

# В тесте тоже добавляем редкие выбросы
with torch.no_grad():
    mask = torch.rand(100, device=device) < 0.05
    noise = torch.randn(100, device=device, dtype=torch.float16) * 50
    test_input_0[:, 2] = torch.where(mask, test_input_0[:, 2] + noise, test_input_0[:, 2])
    noise = torch.randn(100, device=device, dtype=torch.float16) * 50
    test_input_0[:, 7] = torch.where(mask, test_input_0[:, 7] + noise, test_input_0[:, 7])

    mask = torch.rand(100, device=device) < 0.05
    noise = torch.randn(100, device=device, dtype=torch.float16) * 80
    test_input_1[:, 1] = torch.where(mask, test_input_1[:, 1] + noise, test_input_1[:, 1])
    noise = torch.randn(100, device=device, dtype=torch.float16) * 80
    test_input_1[:, 5] = torch.where(mask, test_input_1[:, 5] + noise, test_input_1[:, 5])

print(test_input_0.min(), test_input_0.max())
print(test_input_1.min(), test_input_1.max())

with torch.no_grad():
    output_fp16_0 = model_fp16(test_input_0)
    output_quant_0 = model_quant(test_input_0, 0)
    output_fp16_1 = model_fp16(test_input_1)
    output_quant_1 = model_quant(test_input_1, 1)

# Оценка ошибки
mse = F.mse_loss(output_fp16_0.float(), output_quant_0.float())
print(f"MSE между FP16 и FakeQuantized для timestep 0: {mse.item():.6f}")
mse = F.mse_loss(output_fp16_1.float(), output_quant_1.float())
print(f"MSE между FP16 и FakeQuantized для timestep 1: {mse.item():.6f}")

print("Квантизационные параметры для timestep 0:", model_quant.fc1.get_activation_qparams(0))
print("Квантизационные параметры для timestep 1:", model_quant.fc1.get_activation_qparams(1))