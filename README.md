## FakeQuantLinear

**FakeQuantLinear** is a PyTorch `nn.Module` that implements post-training, fake quantization for linear (or equivalently convolutional) layers with timestep-channel joint reparameterization, as described in the paper [TCAQ-DM: Timestep-Channel Adaptive Quantization for Diffusion Models](https://arxiv.org/abs/2412.16700).

### Key Features

- **Per-timestep, per-channel calibration**
  - During calibration mode, activation statistics (min and max) are collected separately for each integer timestep and each input channel.

- **Timestep-Channel Joint Reparameterization**
  - Aggregates per-timestep channel scales into a single joint scaling vector `r_s`, aligning activation ranges across channels while preserving overall model behavior.  
  - Implements Eqs. (3)–(5) from the TCAQ-DM paper, shifting dynamic range from activations to weights.

- **Per-Channel Weight Quantization**
  - Reparameterized weights are quantized per output channel (one scale and zero-point per output feature) for higher fidelity.

- **Fake-Quantized Inference**
  - Inference mode applies joint reparameterization, per-channel activation quantization, and per-channel weight dequantization to produce float outputs with minimal overhead.

### Usage

1. **Initialize**
   ```python
   from fake_quant import FakeQuantLinear

   fq = FakeQuantLinear(
       in_features=512,
       out_features=256,
       bias=True,
       dtype=torch.float16,
       R_trunc=10.0
   )
   ```

2. **Calibration**
   ```python
   # For each timestep t in your calibration dataset:
   for t, batch in enumerate(cal_loader):
       x = batch['input']  # shape [batch_size, in_features]
       _ = fq(x, timestamp=t)
   ```

3. **Freeze Quantization**
   ```python
   fq.freeze()
   ```

4. **Inference**
   ```python
   # For each input x at timestep t:
   y = fq(x, timestamp=t)  # returns float16 output
   ```

5. **Inspect QParams (Optional)**
   ```python
   scale, zp = fq.get_activation_qparams(timestamp=t)
   ```

### Experiment Details

For our experiments on the Freepik/flux.1-lite-8B diffusion model, we integrated **FakeQuantLinear** into to_q, to_k, to_v projections in attention, calibrating over 15 timesteps and truncating the joint scaling vector with `R_trunc=10.0`.

### Reference

> **TCAQ-DM**: Timestep-Channel Adaptive Quantization for Diffusion Models, arXiv: [2412.16700](https://arxiv.org/abs/2412.16700)

Preprint available at: https://arxiv.org/abs/2412.16700

---

Feel free to open issues or contribute via pull requests!

