import hw4
import hw4_utils
import torch.nn as nn


def main():

    # initialize parameters
    lr = 0.01
    latent_dim = 6
    lam = 0.1
    loss_fn = nn.MSELoss()

    # initialize model
    vae = hw4.VAE(lam=lam, lrate=lr, latent_dim=latent_dim, loss_fn=loss_fn)

    # generate data
    X = hw4_utils.generate_data()

    # fit the model to the data
    loss_rec, loss_kl, loss_total, Xhat, gen_samples = hw4.fit(vae, X, n_iter=10)


if __name__ == "__main__":
    main()
