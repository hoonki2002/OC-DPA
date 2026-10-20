"""Self-check of the simulated array (ocdpa/analog.py, analog_matmul): is the quality path the path the hardware takes?

With ideal cells (no variation, no IR-drop) the bit-serial path must

  (1) equal the product of the dequantised planes with the 8-bit quantised input, whatever the converter does, because
      an 8-bit ADC resolves the integer column sum of a 128-row, 1-bit read exactly (up to saturation of the extreme level);
  (2) be independent of the memory-bounding split of the tokens (TOKEN_CHUNK);

and the earlier idealisation (serial=False) must equal the full-precision product when the converter is ideal.

  python scripts/08_check_analog_path.py
"""
import torch

import ocdpa.analog as A
from ocdpa.analog import analog_matmul, quantize_input, GS, KMAX, NoiseModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
rows, J, k, T = 32, 4, 4, 300
B = (torch.randint(0, 2, (rows * J, GS, KMAX)) * 2 - 1).to(torch.int8).to(dev)
alpha = (torch.rand(rows * J, KMAX, device=dev) * 0.1).sort(dim=1, descending=True).values
beta = torch.randn(rows * J, device=dev) * 0.01
x = torch.randn(T, J * GS, device=dev) * 0.7
x[:, 5] *= 30
W = (beta.view(rows, J, 1) + (B.view(rows, J, GS, KMAX)[..., :k].float() * alpha.view(rows, J, 1, KMAX)[..., :k]).sum(-1)).reshape(rows, J * GS)
ideal = dict(onoffratio=1e9)

for per_group in (True, False):
    q, s = quantize_input(x, J, per_group=per_group)
    ref = (q * s).reshape(T, -1) @ W.T
    for bits in (None, 8):
        out = analog_matmul(x, B, alpha, beta, rows, J, k, adc_bits=bits, serial=True, per_group_input=per_group, **ideal)
        print(f"scale per {'group' if per_group else 'token'}, ADC {bits}: max |bit-serial - quantised-input product| = {float((out - ref).abs().max()):.2e}")

A.TOKEN_CHUNK = 64
chunked = analog_matmul(x, B, alpha, beta, rows, J, k, noise=NoiseModel("mixed", 0.01, 0.02, "none"), key=1)
A.TOKEN_CHUNK = 10 ** 9
whole = analog_matmul(x, B, alpha, beta, rows, J, k, noise=NoiseModel("mixed", 0.01, 0.02, "none"), key=1)
print(f"token chunking changes the output by {float((chunked - whole).abs().max()):.2e}")

full = analog_matmul(x, B, alpha, beta, rows, J, k, adc_bits=None, serial=False, **ideal)
print(f"full-precision inputs, ideal converter vs product: {float((full - x @ W.T).abs().max()):.2e}")
