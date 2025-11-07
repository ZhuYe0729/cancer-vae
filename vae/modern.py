import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# ======================= CONFIG =======================
mb_size = 64
Z_dim = 100
h_dim = 128
lr = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================= DATA =========================
transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)


# ======================= MODEL ========================
class VAE(nn.Module):
    def __init__(self, X_dim=784, h_dim=128, Z_dim=100):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(X_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, Z_dim)
        self.fc_logvar = nn.Linear(h_dim, Z_dim)

        # Decoder
        self.fc2 = nn.Linear(Z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, X_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
bce = nn.BCELoss(reduction="sum")


# ======================= LOSS =========================
def vae_loss(x_recon, x, mu, log_var):
    recon_loss = bce(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl_loss) / x.size(0)


# ======================= TRAINING =======================
os.makedirs("out", exist_ok=True)

for epoch in range(epochs):
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)

        x_recon, mu, log_var = model(x)
        loss = vae_loss(x_recon, x, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

    # Save sample images per epoch
    z = torch.randn(16, Z_dim).to(device)
    samples = model.decode(z).cpu().detach()
    samples = samples.view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    plt.savefig(f"out/epoch_{epoch+1}.png", bbox_inches="tight")
    plt.close()

print("Training finished!")
