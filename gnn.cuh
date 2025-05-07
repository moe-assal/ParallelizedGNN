#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "helper_optimized.cuh"


typedef struct {
    int in_dim;
    int out_dim;

    float* W;   // [in_dim x out_dim]
    float* b;   // [out_dim]

    float* out_features; // [num_nodes x out_dim]

    void allocate(int in_d, int out_d, int num_nodes) {
        in_dim = in_d;
        out_dim = out_d;

        cudaMalloc(&W, sizeof(float) * in_d * out_d);
        cudaMalloc(&b, sizeof(float) * out_d);
        cudaMalloc(&out_features, sizeof(float) * num_nodes * out_d);
    }

    void free_() {
        cudaFree(W);
        cudaFree(b);
        cudaFree(out_features);
    }

    void load_weights(FILE *file) {
        // Load weights
        float *weight_buffer = (float*)malloc(sizeof(float) * in_dim * out_dim);
        size_t weights_read = fread(weight_buffer, sizeof(float), in_dim * out_dim, file);
        cudaMemcpy(W, weight_buffer, sizeof(float) * in_dim * out_dim, cudaMemcpyHostToDevice);
        
        
        free(weight_buffer);
        // Load bias
        float *bias_buffer = (float*)malloc(sizeof(float) * out_dim);
        size_t biases_read = fread(bias_buffer, sizeof(float), out_dim, file);
        cudaMemcpy(b, bias_buffer, sizeof(float) * out_dim, cudaMemcpyHostToDevice);
        free(bias_buffer);
    }

} LinearLayer;


typedef struct {
    LinearLayer* layers;  // Array of layers
    int num_layers;

    void forward_pass(const float* X, const int* srcs, const int* dsts, int num_nodes, int num_edges, int* deg) {
		const float * input_features;
        float * aggregated;
        
        // Iterate over layers
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
			
			// Initialize input array
			if (layer_idx > 0) input_features = layers[layer_idx - 1].out_features;
			else input_features = X;

			// Linear Layer			
			dim3 gridDimMM((layers[layer_idx].out_dim + 15) / 16, (num_nodes + 15) / 16);
			dim3 blockDimMM(16, 16);
			
			MatrixMultiply<<<gridDimMM, blockDimMM>>>(input_features, layers[layer_idx].W, layers[layer_idx].out_features, num_nodes, layers[layer_idx].in_dim, layers[layer_idx].out_dim);
            
			MatrixVecAddition<<<gridDimMM, blockDimMM>>>(layers[layer_idx].out_features, layers[layer_idx].b, layers[layer_idx].out_features, num_nodes, layers[layer_idx].out_dim);

            // Message Propagation with Mean Aggregation and Symmetric Normalization
            cudaMalloc(&aggregated, sizeof(float) * num_nodes * layers[layer_idx].out_dim);
			cudaMemset(aggregated, 0, sizeof(float) * num_nodes * layers[layer_idx].out_dim);
			
			MessagePropagate(layers[layer_idx].out_features, srcs, dsts, deg, aggregated, num_nodes, num_edges, layers[layer_idx].out_dim);

			// Apply ReLU
            ReLUMatrix<<<(num_nodes * layers[layer_idx].out_dim + 255) / 256, 256>>>(aggregated, num_nodes, layers[layer_idx].out_dim);
			
			// Add self loop            
        	MatrixAddInPlace<<<(num_nodes * layers[layer_idx].out_dim + 255) / 256, 256>>>(layers[layer_idx].out_features, aggregated, num_nodes, layers[layer_idx].out_dim);

			cudaFree(aggregated);
        }
    }


    // Allocate memory for all layers
    void allocate_layers(int num_layers, int* in_dims, int* out_dims, int num_nodes) {
        this->num_layers = num_layers;
        layers = (LinearLayer*)malloc(num_layers * sizeof(LinearLayer));
        for (int i = 0; i < num_layers; ++i) {
            layers[i].allocate(in_dims[i], out_dims[i], num_nodes);
        }
    }

    // Free memory for all layers
    void free_layers() {
        for (int i = 0; i < num_layers; ++i) {
            layers[i].free_();
        }
        free(layers);
    }
	
	void load_weights(FILE* file) {
    	for (int i = 0; i < num_layers; ++i) {
    	    layers[i].load_weights(file);
    	}
	}

    void load_from_file(const char* path_to_file, int num_nodes) {
        FILE* file = fopen(path_to_file, "rb");
        if (!file) {
            fprintf(stderr, "Error opening model file: %s\n", path_to_file);
            exit(EXIT_FAILURE);
        }

        // Read total number of layers
        fread(&num_layers, sizeof(int), 1, file);

        // Read all layer dimensions
        int* in_dims = (int*)malloc(sizeof(int) * num_layers);
        int* out_dims = (int*)malloc(sizeof(int) * num_layers);
        for (int i = 0; i < num_layers; ++i) {
            fread(&in_dims[i], sizeof(int), 1, file);
            fread(&out_dims[i], sizeof(int), 1, file);
        }

        // Allocate layers
        allocate_layers(num_layers, in_dims, out_dims, num_nodes);
        free(in_dims);
        free(out_dims);

        // Load weights from file
        load_weights(file);

        fclose(file);
    }

} GCN;

