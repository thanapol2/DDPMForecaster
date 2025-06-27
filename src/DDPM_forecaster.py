import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DDPMForecaster(nn.Module):
    def __init__(self, num_sensors, context_len, forecast_len, hidden_dim=128, T = 100):
        super().__init__()
        self.num_sensors = num_sensors
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.T = T
        self.fc = nn.Sequential(
            nn.Linear(num_sensors * (context_len + forecast_len) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensors * forecast_len)
        )

    def forward(self, y_noisy, context, t):
        B = y_noisy.shape[0]
        t = t.float().unsqueeze(1) / self.T
        x = torch.cat([
            y_noisy.view(B, -1),
            context.view(B, -1),
            t
        ], dim=1)
        out = self.fc(x).view(B, self.num_sensors, self.forecast_len)
        return out  # (B, M, forecast_len), predicted noise

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def q_sample(y, n, alpha_hats, device):
    noise = torch.randn_like(y)
    alpha_hat = alpha_hats[n].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
    y_noisy = torch.sqrt(alpha_hat) * y + torch.sqrt(1 - alpha_hat) * noise
    return y_noisy, noise

def p_sample(model, context, forecast_shape, alpha_hats, device, num_steps=100):
    B, M, forecast_len = forecast_shape
    y = torch.randn(forecast_shape, device=device)
    for n in reversed(range(num_steps)):
        t = torch.full((B,), n, device=device, dtype=torch.long)
        eps_pred = model(y, context, t)
        alpha_hat = alpha_hats[n]
        alpha = torch.sqrt(alpha_hat)
        beta = torch.sqrt(1 - alpha_hat)
        if n > 0:
            z = torch.randn_like(y)
        else:
            z = torch.zeros_like(y)
        y = (1/alpha) * (y - beta * eps_pred) + beta * z
    return y  # (B, M, forecast_len)


def train_ddpm_forecaster(model, optimizer, X, Y, T, alpha_hats,
                                  epochs=20, batch_size=32, device='cpu'):
    N = X.shape[0]
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        batch_losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            context = torch.tensor(X[idx], dtype=torch.float32).to(device)   # (B, M, input_len)
            target = torch.tensor(Y[idx], dtype=torch.float32).to(device)    # (B, M, forecast_len)
            n = torch.randint(0, T, (context.shape[0],), device=device)
            y_noisy, noise = q_sample(target, n, alpha_hats, device)
            noise_pred = model(y_noisy, context, n)
            loss = ((noise - noise_pred) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(np.mean(batch_losses))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
    return losses


def ddpm_mc_sample(model, context, alpha_hats, device, forecast_len, num_samples=100, num_steps=100):
    """
    Monte Carlo sampling for DDPM-based probabilistic forecasting.

    Args:
        model:       Trained DDPM forecaster.
        context:     (1, M, context_len) torch tensor (batch size = 1).
        alpha_hats:  Diffusion schedule cumulative alphas (on device).
        device:      torch device.
        forecast_len: Forecasting horizon (H).
        num_samples: Number of stochastic forecast samples.
        num_steps:   Number of diffusion steps.

    Returns:
        samples: np.ndarray, shape (num_samples, M, forecast_len)
    """
    model.eval()  # Ensure deterministic eval mode (no dropout, BN, etc.)
    M = context.shape[1]
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            y_pred = p_sample(model, context, (1, M, forecast_len), alpha_hats, device, num_steps=num_steps)
            preds.append(y_pred.squeeze(0).cpu().numpy())  # (M, forecast_len)
    return np.stack(preds, axis=0)  # (num_samples, M, forecast_len)
