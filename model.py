import torch
import struct
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def visualize_latent_space_pca_3d(model, data):
    model.eval()
    with torch.no_grad():
        _, z = model(data.x, data.edge_index)

    z = z.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], c=labels, cmap='tab10', alpha=0.7, s=40)

    ax.set_title('3D Latent Space Visualization with PCA')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    fig.colorbar(scatter, ax=ax, label='Class Label')
    plt.tight_layout()
    plt.show()


class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='mean')  # mean aggregation
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x_transformed = self.linear(x)
        out = self.propagate(edge_index, x=x_transformed)
        out = F.relu(out)

        # Add back the node’s own transformed representation
        out = out + x_transformed

        return out

    def message(self, x_j):
        return x_j  # raw messages

    def update(self, aggr_out):
        return aggr_out


class GCN(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(GCN, self).__init__()
        assert len(in_dims) == len(out_dims)
        self.num_layers = len(in_dims)
        self.layers = nn.ModuleList([
            GCNLayer(in_d, out_d) for in_d, out_d in zip(in_dims, out_dims)
        ])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x



class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dim, latent_dim, decoder_hidden_dim):
        super(GraphAutoEncoder, self).__init__()

        # GCN encoder: compress node features using graph topology
        self.encoder = GCN(
            in_dims=[input_dim, gcn_hidden_dim],
            out_dims=[gcn_hidden_dim, latent_dim]
        )

        # MLP decoder: reconstruct features from latent space
        self.decoder = SimpleNN(latent_dim, decoder_hidden_dim, input_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)  # Encode with GCN
        x_hat = self.decoder(z)          # Decode with MLP
        return x_hat, z

    def export(self, path_to_file):
        with open(path_to_file, 'wb') as f:
            encoder_dims = [(layer.linear.in_features, layer.linear.out_features) for layer in self.encoder.layers]

            num_layers = len(encoder_dims)
            print(num_layers)
            print(encoder_dims)
            f.write(struct.pack('i', num_layers))

            # Write all dimensions (in_dim, out_dim) for each layer
            for in_dim, out_dim in encoder_dims:
                f.write(struct.pack('ii', in_dim, out_dim))

            # Write encoder weights and biases
            for layer in self.encoder.layers:
                        
                weight = layer.linear.weight.t().data.cpu().numpy().astype(np.float32) # Transpose from [out dim, in dim] to [in, out]
                bias = layer.linear.bias.data.cpu().numpy().astype(np.float32)
                f.write(weight.tobytes())  # shape: [out_dim, in_dim]
                f.write(bias.tobytes())    # shape: [out_dim]

