import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from siren_utils import get_cameraman_tensor, get_coords, model_results, get_coords_no_norm


ACTIVATIONS = {
    "relu": torch.relu,
    "sin": torch.sin,
    "tanh": torch.tanh
}

class SingleLayer(nn.Module):
    def __init__(self, in_features, out_features, activation, 
                 bias, is_first, init_weights=True):
        super().__init__()
        # TODO: create your single linear layer 
        # with the provided input features, output features, and bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.in_features = torch.tensor(in_features)

        # self.torch_activation will contain the appropriate activation function that you should use
        if activation is None:
            self.torch_activation = nn.Identity() # no-op
        elif not activation in ACTIVATIONS:
            raise ValueError("Invalid activation")
        else:
            self.torch_activation = ACTIVATIONS[activation]
        # NOTE: when activation is sin omega is 30.0, otherwise 1.0
        self.omega = 30.0 if activation == "sin" else 1.0
        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # TODO: initialize the weights of your linear layer 
            # - First layer params should be initialized in: 
            #     UNIFORM(-1/input_features, 1/input_features)
            # - Every other layer params should be initialized in: 
            #     UNIFORM(-\sqrt{6/input_features}/omega, \sqrt{6/input_features}/omega)
            
            if self.is_first:
                self.linear.weight.uniform_(-1/self.in_features, 1/self.in_features)
            else:
                self.linear.weight.uniform_(-torch.sqrt(6 / self.in_features) / self.omega, 
                                            torch.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        # TODO: pass the input through your linear layer, multiply by omega, then apply activation
        output = self.linear(input) * self.omega
        return self.torch_activation(output)

# We've implemented the model for you - you need to implement SingleLayer above
# We use 7 hidden_layer and 32 hidden_features in Siren 
#   - you do not need to experiment with different architectures, but you may.
class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, activation, init_weights=True):
        super().__init__()

        self.net = []
        # first layer
        self.net.append(SingleLayer(in_features, hidden_features, activation,
                                    bias=True, is_first=True, init_weights=init_weights))
        # hidden layers
        for i in range(hidden_layers):
            self.net.append(SingleLayer(hidden_features, hidden_features, activation,
                                        bias=True, is_first=False, init_weights=init_weights))
        # output layer - NOTE: activation is None
        self.net.append(SingleLayer(hidden_features, out_features, activation=None, 
                                    bias=False, is_first=False, init_weights=init_weights))
        # combine as sequential
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # the input to this model is a batch of (x,y) pixel coordinates
        return self.net(coords)

class MyDataset(Dataset):
    def __init__(self, sidelength) -> None:
        super().__init__()
        self.sidelength = sidelength
        self.cameraman_img = get_cameraman_tensor(sidelength)
        # self.coords = get_coords(sidelength)
        self.coords = get_coords_no_norm(sidelength)
        # TODO: we recommend printing the shapes of this data (coords and img) 
        #       to get a feel for what you're working with
        print("shape:", self.coords.shape, self.cameraman_img.shape) # shape: torch.Size([65536, 2]) torch.Size([65536, 1])


    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # TODO: return the model input (coords) and output (pixel) corresponding to idx
        # raise NotImplementedError
        return self.coords[idx], self.cameraman_img[idx]
    
def train(total_epochs, batch_size, activation, hidden_size=32, hidden_layer=7, k=4):
    # TODO(1): finish the implementation of the MyDataset class
    dataset = MyDataset(sidelength=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # TODO(2): implement SingleLayer class which is used by the Siren model
    siren_model = Siren(in_features=2, out_features=1, 
                        hidden_features=hidden_size, hidden_layers=hidden_layer, activation=activation,
                        init_weights=True)
    
    # TODO(3): set the learning rate for your optimizer
    learning_rate=10 ** (-k) # 1.0 is usually too large, a common setting is 10^{-k} for k=2, 3, or 4
    # TODO: try other optimizers such as torch.optim.SGD
    optim = torch.optim.Adam(lr=learning_rate, params=siren_model.parameters())
    # optim = torch.optim.SGD(lr=learning_rate, params=siren_model.parameters())
    # scheudler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.9)
    
    # TODO(4): implement the gradient descent train loop
    losses = [] # Track losses to make plot at end
    for epoch in range(total_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # a. TODO: pass inputs (pixel coords) through mode
            # batch.to("cuda")
            model_output = siren_model(batch[0])
            # b. TODO: compute loss (mean squared error - L2) between:
            #   model outputs (predicted pixel values) and labels (true pixels values)
            loss = ((model_output - batch[1])**2).mean()

            # loop should end with...
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() # NOTE: .item() very important!
        epoch_loss /= len(dataloader)
        print(f"Epoch: {epoch}, loss: {epoch_loss:4.5f}", end="\r")
        losses.append(epoch_loss)
        # scheudler.step()

    # example for saving model
    torch.save(siren_model.state_dict(), f"siren_model.p")
    
    # Example code for visualizing results
    # To debug you may want to modify this to be in its own function and use a saved model...
    # You can also save the plots with plt.savefig(path)
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    model_output, grad, lap = model_results(siren_model)
    ax[0].imshow(model_output, cmap="gray")
    ax[0].set_title("Model Output")
    ax[1].imshow(grad, cmap="gray")
    ax[1].set_title("Gradient")
    ax[2].imshow(lap, cmap="gray")
    ax[2].set_title("Laplacian")
    # TODO: in order to really see how your loss is updating you may want to change the axis scale...
    #       ...or skip the first few values
    ax[3].plot(losses)
    ax[3].set_yscale('log')
    ax[3].set_title("Last 10 avg:" + str(sum(losses[-10:])/10))
    plt.savefig(f"images/siren_{activation}_e{total_epochs}_b{batch_size}_lr10-{k}_decay_nonorm.png")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Siren model.')
    parser.add_argument('-e', '--total_epochs', required=True, type=int)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-a', '--activation', required=True, choices=ACTIVATIONS.keys())
    parser.add_argument('-k', '--k', required=False, type=int, default=4)
    args = parser.parse_args()
    
    train(args.total_epochs, args.batch_size, args.activation, k=args.k)