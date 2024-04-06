import hw4
import hw4_utils
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

def main():

    # initialize parameters
    lr = 0.01
    latent_dim = 6
    lam = 5e-5
    loss_fn = nn.MSELoss()

    # initialize model
    vae = hw4.VAE(lam=lam, lrate=lr, latent_dim=latent_dim, loss_fn=loss_fn)

    # generate data
    X = hw4_utils.generate_data()

    # fit the model to the data
    loss_rec, loss_kl, loss_total, Xhat, gen_samples = hw4.fit(vae, X, n_iter=8000)

    torch.save(vae.cpu().state_dict(), 'vae.pb')

    # plot the training set fitting results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, c='g')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], s=10, c='b')
    plt.title('Scatter plot of data points')
    plt.show()

    # plot the loss changes
    loss_kl = [l * lam for l in loss_kl]
    plt.figure(figsize=(10, 6))
    plt.plot(loss_rec, color='r', label='Reconstruction Loss')
    plt.plot(loss_kl, color='g', label='KL Divergence Loss')
    plt.plot(loss_total, color='b', label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss values over iterations')
    plt.legend()
    plt.show()

    # plot the generated samples
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, c='g')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], s=10, c='b')
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], s=10, c='r')
    plt.title('Scatter plot of data points')
    plt.legend(['Data', 'Reconstructed', 'Generated'])
    plt.show()


if __name__ == "__main__":
    main()
