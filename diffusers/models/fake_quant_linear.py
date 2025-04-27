import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantLinear(nn.Module):
    """
    A linear layer that, in 'calibration' mode, collects per-timestep, per-channel
    activation min/max.  After calling `freeze()` it:
      1) computes the joint timestep-channel scaling vector r_s
      2) reparameterizes weight: W'[:,d] = W[:,d] * r_s[d], then quantizes it
      3) for each timestep t, computes qparams of X^t / r_s
    On inference you call forward(x, timestep), and it applies exactly Eq.(3)-(5).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype=torch.bfloat16,
                 R_trunc: float = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.R_trunc = R_trunc

        # full-precision weight & bias
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=self.dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=self.dtype)
        ) if bias is not None else None

        # weight quantization buffers
        self.register_buffer('weight_scale',      torch.ones(out_features, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, dtype=torch.int32))

        self.act_stats   = {}  # t -> {'min': Tensor[in_features], 'max': Tensor[in_features]}
        self.act_qparams = {}  # t -> {'scale': Tensor[in_features], 'zero_point': Tensor[in_features]}

        # the joint reparameterization vector r_s[d]
        self.register_buffer('r_s', torch.ones(in_features, dtype=torch.float32))

        # start in calibration mode
        self.collecting = True

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        If collecting: record per-channel min/max for this timestep.
        Else: apply joint reparam + fake quant with timestep qparams.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep[0].item()
        if self.collecting:
            # per-channel stats over batch dim
            # attn.to_q(hidden_states) is [B, S, (N * D)]
            x_min = x.view(-1, x.size(-1)).min(dim=0).values  # [in_features]
            x_max = x.view(-1, x.size(-1)).max(dim=0).values  # [in_features]
            stats = self.act_stats.setdefault(timestep, {
                'min': x_min.clone(),
                'max': x_max.clone()
            })
            stats['min'] = torch.min(stats['min'], x_min)
            stats['max'] = torch.max(stats['max'], x_max)
            # torch.save(x, "x.pt")
            # torch.save(self.weight, "weight.pt")
            # torch.save(self.bias, "bias.pt")
            # exit()
            return F.linear(x, self.weight, self.bias)

        # Inference path
        if isinstance(timestep, torch.Tensor):
            timestep = timestep[0].item()
        if timestep not in self.act_qparams:
            raise KeyError(f"No qparams for timestep={timestep}. Did you freeze()?")

        qmin, qmax = -128, 127
        qparams = self.act_qparams[timestep]
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
        # y = F.linear(x_deq, w_deq, bias=(self.bias.float() if self.bias is not None else None))
        y = F.linear(x_deq.to(torch.bfloat16), w_deq.to(torch.bfloat16), bias=self.bias)

        # 5) cast back
        return y.to(self.dtype)

    @torch.no_grad()
    def freeze(self):
        """
        1) Compute joint r_s from all timesteps (§5, Eq (5))
        2) Reparameterize and quantize weight W' = W * r_s
        3) For each timestep, compute qparams of X^t / r_s
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
            # # s_tar^t = min_d max_t[d]
            # s_tar = max_t.min()

            # print("s_tar", s_tar)

            # # r_t[d] = max_t[d] / s_tar
            # r_t = max_t / s_tar
            # print("r_t", r_t)

            min_t = stats['min']
            # per-channel "peak" magnitude
            peak_t = torch.max(min_t.abs(), max_t.abs())    # always ≥ 0
            # peak_t = torch.max(max_t - min_t)
            s_tar = peak_t.kthvalue(int(0.05*len(peak_t))).values.clamp(min=1e-6)   # still ≥ 0

            print("s_tar", s_tar)

            r_t    = peak_t / s_tar                          # ≥ 0 for all d

            print("r_t", r_t)

            # accumulate numerator & denom
            numer += r_t * max_t
            denom += max_t

        # joint r_s
        r_s = numer / denom
        r_s = r_s.clamp(min=0.1, max=(self.R_trunc or 10.0))
        self.r_s.copy_(r_s)

        # r_s.copy_(self.r_s)
        print("R_S", r_s)

        # 2) reparameterize & quantize weight
        #    W' = W * r_s
        self.register_buffer('weight_int', torch.zeros_like(self.weight, dtype=torch.int8))
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

        # 3) compute per-timestep activation qparams on X^t / r_s
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
        del self.weight

    def get_activation_qparams(self, timestep: int):
        """Return (scale, zero_point) for a given timestep after freeze()."""
        if timestep not in self.act_qparams:
            raise KeyError(f"No activation qparams for timestep {timestep}.")
        p = self.act_qparams[timestep]
        return p['scale'], p['zero_point']

    def __repr__(self):
        return f"FakeQuantLinear(in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, dtype={self.dtype}, R_trunc={self.R_trunc})"