import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

batch_size = 1
seq_len = 1024
hidden_dim = 3072

torch.manual_seed(0)
tensor = torch.load("x.pt").detach().cpu().float()

tensor_abs = tensor.abs()

subsample_tokens = 50
subsample_channels = 50

token_indices = torch.linspace(0, seq_len - 1, subsample_tokens).long()
channel_indices = torch.linspace(0, hidden_dim - 1, subsample_channels).long()

tensor_subsample = tensor_abs[0, token_indices][:, channel_indices]  # [tokens, channels]

# Теперь строим график с поменянными осями
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

channel_grid, token_grid = torch.meshgrid(channel_indices, token_indices, indexing='ij')
values = tensor_subsample.T  # Транспонируем значения!

ax.plot_surface(channel_grid.numpy(), token_grid.numpy(), values.numpy(), cmap='viridis')

ax.set_xlabel('Channel index')  # Теперь X — каналы
ax.set_ylabel('Token index')    # Теперь Y — токены
ax.set_zlabel('Absolute Value')
ax.set_title('Distribution of Absolute Values across Channels and Tokens')

plt.show()