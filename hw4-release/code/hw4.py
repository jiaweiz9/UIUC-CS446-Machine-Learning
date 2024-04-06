import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(50, latent_dim)
        self.logstd2_layer = nn.Linear(50, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lrate)


    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        self.encoder[0].weight.data = We1
        self.encoder[0].bias.data = be1
        self.encoder[2].weight.data = We2
        self.encoder[2].bias.data = be2
        self.encoder[4].weight.data = We3
        self.encoder[4].bias.data = be3
        self.mu_layer.weight.data = Wmu
        self.mu_layer.bias.data = bmu
        self.logstd2_layer.weight.data = Wstd
        self.logstd2_layer.bias.data = bstd

        self.decoder[0].weight.data = Wd1
        self.decoder[0].bias.data = bd1
        self.decoder[2].weight.data = Wd2
        self.decoder[2].bias.data = bd2
        self.decoder[4].weight.data = Wd3
        self.decoder[4].bias.data = bd3

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """

        y = self.encoder(x)
        mean = self.mu_layer(y)
        stddev_p = self.logstd2_layer(y)
        z = torch.distributions.Normal(mean, torch.exp(stddev_p/2)).rsample()
        xhat = self.decoder(z)
        return y, mean, stddev_p, z, xhat

    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        y, mean, stddev_p, z, xhat = self.forward(x)
        L_rec = self.loss_fn(x, xhat).mean()
        L_kl = (- 1 / 2 * (torch.sum(stddev_p - mean ** 2 - torch.exp(stddev_p) + 1, dim=1))).mean()
        L = L_rec + self.lam * L_kl
        
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        return L_rec.item(), L_kl.item(), L.item()


def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    losses_rec = []
    losses_kl = []
    losses = []
    for i in range(n_iter):
        L_rec, L_kl, L = net.step(X)
        print("Iteration: ", i, "Loss: ", L)
        losses_rec.append(L_rec)
        losses_kl.append(L_kl)
        losses.append(L)

    _, _, _, _, Xhat = net.forward(X)
    
    gen_latent = torch.distributions.Normal(torch.zeros(net.latent_dim), torch.ones(net.latent_dim)).sample((400,))
    gen_samples = net.decoder(gen_latent)
    return losses_rec, losses_kl, losses, Xhat.detach().numpy(), gen_samples.detach().numpy()
