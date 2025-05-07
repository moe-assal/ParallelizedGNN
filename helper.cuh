#pragma once
#include <cuda_runtime.h>

#define TILE_SIZE 32
// Standart tiling. However, I transpose tileB so that memory access is coalesced
__global__ void MatrixMultiply(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];  // Transposed

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE;

        if (row < M && tiled_k + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tiled_k + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tiled_k + threadIdx.y < K)
            tileB[threadIdx.x][threadIdx.y] = B[(tiled_k + threadIdx.y) * N + col];
        else
            tileB[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[threadIdx.x][i];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


__global__ void MatrixVecAddition(const float* M, const float* b, float* R, int n, int m) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < m) {
        R[row * m + col] = M[row * m + col] + b[col];
    }
}


__global__ void MatrixAddInPlace(float* A, const float* B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) A[idx] += B[idx];
}


__global__ void ReLUMatrix(float* __restrict__ M, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
    	
        M[idx] = fmaxf(0.0f, M[idx]);
    }
}


// I exploit the fact that srcs array is sorted
// Limitations: No isolated nodes (with degree zero) and graph has to be undirected


// This function computes the first occurence of a node. So for instance: [0 0 0 1 1 1 2] -> [0 3 6]
__global__ void computeLastOccurrences(const int* __restrict__ arr,
                                       int* __restrict__ last_occurrence,
                                       int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements) {
        int current_value = arr[tid];
        int next_value = (tid < num_elements - 1) ? arr[tid + 1] : -1;

        if (current_value != next_value) {
            last_occurrence[current_value] = tid;
        }
    }
}


__global__ void differencesFromLast(const int* __restrict__ last,
                                    int* __restrict__ result,
                                    int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        int end = last[tid];
        int start = (tid == 0) ? 0 : (last[tid - 1] + 1);
        result[tid] = end - start + 1;
    }
}


__host__ void ComputeDegrees(const int* __restrict__ srcs,
                             int* __restrict__ deg,
                             int num_edges,
                             int num_nodes) {
    int *d_last_occurrence;

    cudaMalloc(&d_last_occurrence, num_nodes * sizeof(int));
    
    int blockSize = 256;
    int numBlocksEdges = (num_edges + blockSize - 1) / blockSize;
    int numBlocksNodes = (num_nodes + blockSize - 1) / blockSize;

    computeLastOccurrences<<<numBlocksEdges, blockSize>>>(srcs, d_last_occurrence, num_edges);
    differencesFromLast<<<numBlocksNodes, blockSize>>>(d_last_occurrence, deg, num_nodes);


    cudaFree(d_last_occurrence);
}


// Expand kernel
__global__ void expand(const float* __restrict__ M, const int* __restrict__ D,
                                 float* __restrict__ R, int m, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * d;
    if (tid < total) {
        int row = tid / d;
        int col = tid % d;
        int src_row = D[row];
        R[tid] = M[src_row * d + col];
    }
}


// Reduce kernel
__global__ void reduce(const float* __restrict__ R,
                       const int* __restrict__ End,
                       const int* __restrict__ deg,
                       float* __restrict__ K,
                       int n, int d) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int col = tid; col < d; col += blockDim.x) {
        if (row < n) {
            int start = (row == 0) ? 0 : End[row - 1] + 1;
            int end = End[row];

            float sum = 0.0f;
            for (int i = start; i <= end; ++i) {
                sum += R[i * d + col];
            }

            K[row * d + col] = sum / deg[row];
        }
    }
}


// Host message passing function
__host__ void MessagePropagate(const float* X, const int* srcs, const int* dsts, const int* deg,
                                 float* result, int num_nodes, int num_edges, int d) {
    int* occrs;
    float* messages;

    cudaMalloc(&occrs, num_nodes * sizeof(int));
    cudaMalloc(&messages, num_edges * d * sizeof(float));

    // Compute last occurrences of each node
    int blockSize = 256;
    int numBlocksEdges = (num_edges + blockSize - 1) / blockSize;
    computeLastOccurrences<<<numBlocksEdges, blockSize>>>(srcs, occrs, num_edges);
	
    // Expand the destination node features
    int numThreadsExpand = 256;
    int totalExpand = num_edges * d;
    int numBlocksExpand = (totalExpand + numThreadsExpand - 1) / numThreadsExpand;
    expand<<<numBlocksExpand, numThreadsExpand>>>(X, dsts, messages, num_edges, d);
	
	
    // Reduce to accumulate features
    int threadsPerBlockReduce = 256;
    int blocksPerGridReduce = num_nodes;
    reduce<<<blocksPerGridReduce, threadsPerBlockReduce>>>(messages, occrs, deg, result, num_nodes, d);
    
    // Cleanup
    cudaFree(occrs);
    cudaFree(messages);
}

