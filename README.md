# ParallelizedGNN
Created to fulfill the requirement of the Parallel Programming course at LAU. In collaboration with Zeina Mershad and Raghad Hmede.

This project implements a **Graph AutoEncoder (GAE)** trained using PyTorch Geometric in Python, with a custom **CUDA inference backend** for high-performance evaluation on GPUs.

It consists of two main components:

- **Python (PyTorch Geometric)**: Constructs and trains a GCN-based autoencoder on the Cora graph dataset, and exports the graph and model to a binary format.
- **CUDA Backend**: A custom CUDA C++ implementation that loads the exported model and graph and performs inference in a highly parallelized fashion.

## Project Goals

- Enable fast and portable inference of GCN-based models using CUDA kernels.
- Demonstrate how Graph Neural Networks (GNNs) can be serialized and executed outside of Python.

## Python Component

### Features

- Graph pre-processing: removes isolated nodes and enforces undirected structure.
- GCN-based autoencoder model with residual connections.
- Trains on the Cora dataset using MSE loss between original and reconstructed node features.
- PCA-based 3D visualization of latent embeddings.
- Export to custom binary format compatible with CUDA backend.

### Exported Binary Format (`graph.bin`, `model.bin`)

**Graph (graph.bin):**

```
[int32] N (number of nodes)
[int32] d (feature dim)
[int32] E (number of edges)
[float32] Node features (N × d)
[int32] Edge sources (E)
[int32] Edge destinations (E)
```

**Model (model.bin):**

```
[int32] Number of GCN layers
[int32 × 2] (in_dim, out_dim) per layer
[float32] Weights for each layer (out_dim × in_dim)
[float32] Biases for each layer (out_dim)
```

## CUDA Inference Backend

The CUDA backend loads the exported graph and model and performs efficient forward propagation using custom kernels.

### Expected Kernel Behavior

- Parallelized sparse message passing and degree computation using edge list.
- Matrix multiplication using shared memory for latent projections.
- ReLU activation + residuals between layers.
- Final output is a latent representation `z` used for downstream tasks or visualization.

### Run Instructions

```bash
pip install torch torchvision torch_geometric scikit-learn matplotlib
python main.py
nvcc -o gnn_cuda gnn_cuda.cu
./gnn_cuda
```

## Visualization

After training, a 3D PCA projection of the latent space is displayed:

- Each point is a node.
- Colors indicate class labels from the original dataset.

This helps interpret the structure of the learned embeddings.

## License

MIT License

## Acknowledgements
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- Cora dataset by [McCallum et al. (2000)](https://people.cs.umass.edu/~mccallum/papers/nips01-normbias.pdf)
