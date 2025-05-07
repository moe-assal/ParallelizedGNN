from model import *
from exporter import export_pyg_to_bin
from torch_geometric.utils import to_undirected, remove_isolated_nodes

FILE_NAME = "model.bin"

# Load Cora
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]  # Cora has a single graph

data.edge_index = to_undirected(data.edge_index)

# Create a mask that identifies non-isolated nodes
non_isolated_mask = data.x.sum(dim=1) != 0  # Nodes with features (non-isolated)

# Filter the edge index based on the mask
edge_index = data.edge_index[:, non_isolated_mask[data.edge_index[0]]]  # Apply mask to edge_index

# If edge_attr exists, filter it as well
if data.edge_attr is not None:
    data.edge_attr = data.edge_attr[non_isolated_mask[data.edge_index[0]]]

# After removing isolated nodes, update node features and edge indices
data.x = data.x[non_isolated_mask]
data.edge_index = edge_index

# Now the graph is undirected with isolated nodes removed

print("Exporting graph...")
export_pyg_to_bin(data, "graph.bin")

input_dim = dataset.num_node_features
model = GraphAutoEncoder(
    input_dim=input_dim,
    gcn_hidden_dim=200,
    latent_dim=400,
    decoder_hidden_dim=512
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training...")
model.train()
for epoch in range(1, 40):
    optimizer.zero_grad()
    x_hat, z = model(data.x, data.edge_index)
    loss = F.mse_loss(x_hat, data.x)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

print("Model finished training")

print("Exporting to " + FILE_NAME + " ...")

model.export(FILE_NAME)

print("Visualizing...")

visualize_latent_space_pca_3d(model, data)

