import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantLinear(nn.Module):
    """
    A linear layer that, in 'calibration' mode, collects activation
    min/max separately for each integer timestamp.  After calling
    `freeze()`, it:
      - computes per-tensor quant params for the weight
      - quantizes the weight to int8
      - for each seen timestamp, computes act_scale & act_zero_point
    During inference, you must pass the same timestamp to forward,
    and it will use that timestamp’s qparams to fake-quantize the input.
    """
    def __init__(self, in_features, out_features, bias: bool = True, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        # full-precision weight & bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype))
        self.bias   = nn.Parameter(torch.zeros(out_features, dtype=self.dtype)) if bias else None

        # weight quantization buffers
        self.register_buffer('weight_scale',      torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.tensor(0,   dtype=torch.int32))
        self.register_buffer('weight_int',        torch.zeros_like(self.weight, dtype=torch.int8))

        # per-timestamp activation stats: t → {'min': Tensor, 'max': Tensor}
        self.act_stats: dict[int, dict[str, torch.Tensor]] = {}

        # after freeze: per-timestamp qparams: t → {'scale': Tensor, 'zero_point': Tensor}
        self.act_qparams: dict[int, dict[str, torch.Tensor]] = {}

        # start in calibration mode
        self.collecting = True

    def forward(self, x: torch.Tensor, timestamp: int) -> torch.Tensor:
        if self.collecting:
            # calibration: update min/max for this timestamp
            x_min, x_max = x.min().detach(), x.max().detach()
            if timestamp not in self.act_stats:
                # first time we see this timestamp
                self.act_stats[timestamp] = {
                    'min': x_min.clone(),
                    'max': x_max.clone()
                }
            else:
                stats = self.act_stats[timestamp]
                stats['min'] = torch.min(stats['min'], x_min)
                stats['max'] = torch.max(stats['max'], x_max)
            return F.linear(x, self.weight, self.bias)

        # inference: use the qparams for this timestamp
        if timestamp not in self.act_qparams:
            raise KeyError(f"No quant params for timestamp {timestamp}. "
                           "Did you forget to call freeze()?")

        qmin, qmax = -128, 127
        scale_a = self.act_qparams[timestamp]['scale']
        zp_a    = self.act_qparams[timestamp]['zero_point']

        # 1) quantize activations
        x_int8 = torch.clamp((x / scale_a).round() + zp_a, qmin, qmax).to(torch.int8)

        # 2) de-quantize inputs & weights in fp32 to do F.linear
        x_cent = x_int8.float() - float(zp_a)
        w_cent = self.weight_int.float() - float(self.weight_zero_point)

        # 3) fake-quantized matmul + rescale + bias
        y = F.linear(x_cent, w_cent, bias=None)
        y = y * (scale_a * self.weight_scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def freeze(self):
        """
        Compute weight qparams & quantize weight, then
        compute and store activation qparams for each timestamp.
        """
        qmin, qmax = -128, 127

        # --- weight quant ---
        w_fp = self.weight.float()
        w_min, w_max = w_fp.min(), w_fp.max()
        scale_w = (w_max - w_min) / (qmax - qmin)
        scale_w = torch.where(scale_w > 0, scale_w, torch.tensor(1e-8))
        zp_w = torch.clamp((qmin - w_min/scale_w).round(), qmin, qmax).to(torch.int32)

        self.weight_scale      = scale_w
        self.weight_zero_point = zp_w
        self.weight_int = torch.clamp((w_fp/scale_w).round() + zp_w,
                                     qmin, qmax).to(torch.int8)

        # --- activation quant for each timestamp ---
        for t, stats in self.act_stats.items():
            min_a = stats['min']
            max_a = stats['max']
            scale_a = (max_a - min_a) / (qmax - qmin)
            scale_a = torch.where(scale_a > 0, scale_a, torch.tensor(1e-8))
            zp_a = torch.clamp((qmin - min_a/scale_a).round(), qmin, qmax).to(torch.int32)

            self.act_qparams[t] = {
                'scale':      scale_a,
                'zero_point': zp_a
            }

        # switch to inference mode
        self.collecting = False

    def get_activation_qparams(self, timestamp: int):
        """
        After freeze(), returns (scale, zero_point) for a given timestamp.
        """
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