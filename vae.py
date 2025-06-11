import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

def get_rank_weights(outputs, k):
    outputs_argsort = np.argsort(np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (k * (1 + ranks))

def reduce_weight_variance(weights: np.array, data: np.array):
    weights_new = []
    data_new = []
    for w, d in zip(weights, data):
        if w == 0.0:
            continue
        while w > 1:
            weights_new.append(1.0)
            data_new.append(d)
            w -= 1
        weights_new.append(w)
        data_new.append(d)

    return np.array(weights_new), np.array(data_new)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.fc_mu = None
        self.fc_logvar = None
        self.latent_dim = 32

    def build(self, input_dim):

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, self.latent_dim)
        self.fc_logvar = nn.Linear(64, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self._reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(self, X):
        input_dim = X.shape[1]
        self.build(input_dim)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        criterion = nn.MSELoss(reduction='none')
        best_loss = torch.tensor(1e100)
        count = 0

        for _ in range(100):
            for xb, xb in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = self(xb)
                recon_loss = criterion(recon, xb).mean(dim=1)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                elbo = recon_loss + kl_div
                elbo.backward()
                optimizer.step()
            if elbo < best_loss:
                count = 0
                best_loss = elbo
            else:
                count += 1
            if count > 5:
                break
            

    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.encoder(X_tensor)
            mu = self.fc_mu(encoded)
        return mu.numpy()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class WeightedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.fc_mu = None
        self.fc_logvar = None
        self.weights_k = 1e-3
        self.latent_dim = 32

    def build(self, input_dim):

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, self.latent_dim)
        self.fc_logvar = nn.Linear(64, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self._reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def _weights(self, X, y, optim_direc):
        if optim_direc == "min":
            c = -1
        if optim_direc == "max":
            c = 1
        else:
            c = 1
        ranked_weights = get_rank_weights(c * y, self.weights_k)
        normed_weights = ranked_weights / np.mean(ranked_weights)
        weights, X = reduce_weight_variance(normed_weights, X)
        return weights, X

    def fit(self, X, y, optim_direc=None):
        perc = np.percentile(y, 50)
        percentile_cutoff = y >= perc
        X = X[percentile_cutoff]
        y = y[percentile_cutoff]
        weights, X = self._weights(X, y, optim_direc)

        input_dim = X.shape[1]
        self.build(input_dim)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, weights)
        sampler = WeightedRandomSampler(weights, len(X))
        dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, shuffle=False)
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        criterion = nn.MSELoss(reduction='none')
        best_loss = torch.tensor(1e100)
        count = 0

        for _ in range(100):
            for xb, weights_b in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = self(xb)
                recon_loss = criterion(recon, xb).mean(dim=1)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                elbo = recon_loss + kl_div
                loss = torch.mean(elbo * weights_b)
                loss.backward()
                optimizer.step()
            if loss < best_loss:
                count = 0
                best_loss = loss
            else:
                count += 1
            if count > 5:
                break

    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.encoder(X_tensor)
            mu = self.fc_mu(encoded)
        return mu.numpy()

    def fit_transform(self, X, y, optim_direc=None):
        self.fit(X, y, optim_direc)
        return self.transform(X)