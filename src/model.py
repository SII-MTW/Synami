import torch
import torch.nn as nn

class PureVAE(nn.Module):
    """
    Pure Variational Autoencoder for extracting latent representations of peptides.
    """
    def __init__(self, input_dim, latent_dim=64):
        super(PureVAE, self).__init__()
        
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to allow backpropagation through stochastic nodes.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar