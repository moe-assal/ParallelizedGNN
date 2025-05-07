import torch
import numpy as np

def export_pyg_to_bin(data, filepath):
    """
    Export a PyTorch Geometric Data object to a custom binary format.
    
    Parameters:
        data (torch_geometric.data.Data): PyG graph
        filepath (str): path to save binary file (e.g., 'graph.bin')
    """

    assert hasattr(data, 'x') and data.x is not None, "Missing node features (data.x)"
    assert hasattr(data, 'edge_index') and data.edge_index is not None, "Missing edge index"

    node_features = data.x.cpu().numpy().astype(np.float32)         # [N, d]
    edge_index = data.edge_index.cpu().numpy().astype(np.int32).T
    srcs = edge_index[:, 0].copy()
    dsts = edge_index[:, 1].copy()

    N, d = node_features.shape
    E = edge_index.shape[0]

    with open(filepath, 'wb') as f:
        f.write(np.array([N, d, E], dtype=np.int32).tobytes())
        f.write(node_features.tobytes())
        f.write(srcs.astype(np.int32).tobytes())
        f.write(dsts.astype(np.int32).tobytes())
    
    print(f"Exported graph to {filepath}: N={N}, E={E}, d={d}")

