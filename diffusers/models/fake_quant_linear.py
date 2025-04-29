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
        self.register_buffer('weight_int',        torch.zeros_like(self.weight, dtype=torch.int8))

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
            # print(timestep)
        
        if self.collecting:
            # per-channel stats over batch dim
            # attn.to_q(hidden_states) is [B, S, (N * D)]
            x_min = x.view(-1, x.size(-1)).min(dim=0).values  # [in_features]
            x_max = x.view(-1, x.size(-1)).max(dim=0).values  # [in_features]

            if timestep not in self.act_stats:
                self.act_stats[timestep] = {
                    'min': x_min.clone(),
                    'max': x_max.clone()
                }
            else:
                s = self.act_stats[timestep]
                s['min'] = torch.min(s['min'], x_min)
                s['max'] = torch.max(s['max'], x_max)

            # stats = self.act_stats.setdefault(timestep, {'min': x_min.clone(),
            #                                               'max': x_max.clone()})
            # stats['min'].clamp_max_(x_min)
            # stats['max'].clamp_min_(x_max)
            # torch.save(x, "x.pt")
            # torch.save(self.weight, "weight.pt")
            # torch.save(self.bias, "bias.pt")
            # exit()
            return F.linear(x, self.weight, self.bias)

        # Inference path
        if timestep not in self.act_qparams:
            raise KeyError(f"No qparams for timestep={timestep}. Did you freeze()?")

        qmin, qmax = -128, 127
        qparams = self.act_qparams[timestep]
        scale_a = qparams['scale']       # Tensor of shape [in_features]
        zp_a    = qparams['zero_point']  # Tensor of shape [in_features]

        # print("SCALE", scale_a)
        # print("ZERO", zp_a)


        # 1) quantize the re-parameterized activation
        x_rep = x / self.r_s                   # [B, in_features]
        x_int8 = torch.clamp((x_rep / scale_a).round() + zp_a,
                             qmin, qmax).to(torch.int8)  # [B, in_features]
        
        # 2) dequantize activations per-channel
        #    (x_int8.float() - zp_a) is [B, in_features] minus [in_features] → [B, in_features]
        x_deq = (x_int8.float() - zp_a) * scale_a        # [B, in_features]
        # x_deq = x

        # 3) dequant weights per-channel
        #    weight_zero_point: [out_features], weight_scale: [out_features]
        w_deq = (self.weight_int.float() - self.weight_zero_point[:, None]) \
                * self.weight_scale[:, None]              # [out_features, in_features]
        
        x_min = x_deq.view(-1, x.size(-1)).min(dim=0).values  # [in_features]
        x_max = x_deq.view(-1, x.size(-1)).max(dim=0).values  # [in_features]
        if timestep not in self.act_stats:
            self.act_stats[timestep] = {
                'min': x_min.clone(),
                'max': x_max.clone()
            }
        else:
            s = self.act_stats[timestep]
            s['min'] = torch.min(s['min'], x_min)
            s['max'] = torch.max(s['max'], x_max)
        
        # 4) linear + bias
        # y = F.linear(x_deq, w_deq, bias=(self.bias.float() if self.bias is not None else None))
        y = F.linear(x_deq.bfloat16(), w_deq.bfloat16(), bias=self.bias)

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
            min_t = stats['min']  # Tensor[in_features]
            
            # s_tar^t = min_d max_t[d]
            peaks_t = torch.max(max_t.abs(), min_t.abs())
            s_tar = peaks_t.min()

            # print("s_tar", s_tar)

            # r_t[d] = max_t[d] / s_tar
            r_t = peaks_t / s_tar
            # accumulate numerator & denom
            numer += (r_t * peaks_t).cpu()
            denom += (peaks_t).cpu()

            # print("numer", numer)
            # print("denom", denom)

        # joint r_s
        r_s = numer / denom
        # clamp if R_trunc given
        if self.R_trunc is not None:
            r_s = torch.clamp(r_s, max=self.R_trunc)
        r_s = torch.clamp(r_s, min=1, max=100.0)
        self.r_s.data.copy_(r_s)

        # r_s.data.copy_(self.r_s.data)
        # self.r_s.data.copy_(r_s.data.abs())
        # r_s.data.copy_(self.r_s.data)

        # print("r_s", r_s)
        # print("self.r_s", self.r_s)


        # 2) reparameterize & quantize weight
        #    W' = W * r_s
        W_rep = self.weight.float() * r_s                  # [out, in]
        w_min = W_rep.min(dim=1).values                    # [out]
        w_max = W_rep.max(dim=1).values                    # [out]
        scale_w = (w_max - w_min) / (qmax - qmin)          # [out]
        # scale_w = torch.where(scale_w>0, scale_w, 1e-8)
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
            min_rep = stats['min'] / r_s.cuda()
            max_rep = stats['max'] / r_s.cuda()

            scale_a = (max_rep - min_rep) / (qmax - qmin)
            # scale_a = torch.where(scale_a > 0, scale_a, torch.tensor(1e-8, device=scale_a.device))
            zp_a = torch.clamp((qmin - min_rep/scale_a).round(), qmin, qmax).to(torch.int32)

            self.act_qparams[t] = {
                'scale':      scale_a,
                'zero_point': zp_a
            }

        # switch to inference
        self.collecting = False
