import torch
import torch.nn as nn
import numpy as np

# geometry generalization layer
class GeGeLayer(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.channels = in_shape[0]
        self.in_size = np.prod(in_shape[1:])
        self.out_shape = list(out_shape)
        self.out_size = np.prod(self.out_shape)
        assert self.out_size>=self.in_size, "out_size {} ({}) less than in_size {} ({})".format(self.out_size, self.out_shape, self.in_size, self.in_shape)
        self.padding = self.out_size - self.in_size
        self.hidden = None

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, x):
        # flatten all but first 2 dims (batch, channels)
        x = torch.reshape(x, (x.size()[0], x.size()[1], self.in_size))
        # add fake cells at the end to match output size
        x = nn.functional.pad(x, pad = (0, self.padding))
        # apply hidden layers to get score vector
        score = self.hidden(x)
        # sort inputs by score
        _, indices = torch.sort(score)
        x = torch.take(x, indices)
        # reshape to new geometry
        x = torch.reshape(x, [x.size()[0], x.size()[1]] + self.out_shape)
        # get indices to reverse sort later
        _, reverse_indices = torch.sort(indices)
        return x, reverse_indices

    # starting from new geometry, restore original order, size, shape
    def restore(self, x, indices):
        restore_shape = [x.size()[0], x.size()[1]] + self.in_shape[1:]
        x = torch.reshape(x, (x.size()[0], x.size()[1], self.out_size))
        x = torch.narrow(x, x.dim()-1, 0, self.in_size)
        x = torch.take(x, indices)
        x = torch.reshape(x, restore_shape)
        return x

# wrap existing NN in GeGe layer
class GeGeWrapper(nn.Module):
    def __init__(self, geom_layer, model):
        super().__init__()
        self.geom_layer = geom_layer
        self.model = model

    def forward(self, x):
        x, reverse_indices = self.geom_layer(x)
        x = self.model(x)
        x = self.geom_layer.restore(x, reverse_indices)
        return x

def make_GeGeModel(model, in_shape, out_shape, hidden_layer_sizes, hidden_act):
    gege_layer = GeGeLayer(in_shape, out_shape)
    gege_hidden = [
        nn.Linear(gege_layer.out_size, hidden_layer_sizes[0]),
        hidden_act()
    ]
    for counter in range(len(hidden_layer_sizes)-1):
        gege_hidden.extend([
            nn.Linear(hidden_layer_sizes[counter], hidden_layer_sizes[counter+1]),
            hidden_act()
        ])
    gege_hidden.append(nn.Linear(hidden_layer_sizes[-1], gege_layer.out_size))
    hidden_model = nn.Sequential(*gege_hidden)
    gege_layer.set_hidden(hidden_model)
    return GeGeWrapper(gege_layer, model)
