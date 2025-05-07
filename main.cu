#include <stdio.h>
#include <cuda_runtime.h>
#include "gnn_optimized.cuh"
#include "loader_optimized.cuh"

void print_int_array(const int* d_array, int n, const char* label = "Array") {
    int* h_array = (int*)malloc(sizeof(int) * n);
    cudaMemcpy(h_array, d_array, sizeof(int) * n, cudaMemcpyDeviceToHost);

    printf("%s:\n", label);
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    free(h_array);
}


void print_float_array(const float* d_array, int n, const char* label = "Array") {
    float* h_array = (float*)malloc(sizeof(float) * n);
    cudaMemcpy(h_array, d_array, sizeof(float) * n, cudaMemcpyDeviceToHost);

    printf("%s:\n", label);
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_array[i]);
    }
    printf("\n");

    free(h_array);
}



int main() {
    // Load graph
    Graph* h_graph = load_graph_cpu("graph.bin");
    Graph* d_graph = copy_graph_to_gpu(h_graph);
	d_graph->compute_degrees();

    printf("Graph loaded: %d nodes, %d features, %d edges\n", h_graph->N, h_graph->d, h_graph->E);

    // Load model
    GCN gcn;
    gcn.load_from_file("model.bin", d_graph->N);
    printf("Model loaded: %d layers\n", gcn.num_layers);
	

    gcn.forward_pass(
        d_graph->node_features,
        d_graph->srcs,
        d_graph->dsts,
        d_graph->N,
        d_graph->E,
        d_graph->degrees
    );

    float* output = (float*)malloc(sizeof(float) * h_graph->N * gcn.layers[gcn.num_layers - 1].out_dim);
    cudaMemcpy(output, gcn.layers[gcn.num_layers - 1].out_features,
               sizeof(float) * h_graph->N * gcn.layers[gcn.num_layers - 1].out_dim,
               cudaMemcpyDeviceToHost);
	printf("First 5 node outputs from last GCN layer:\n");
    for (int i = 0; i < 5 && i < h_graph->N; ++i) {
        printf("Node %d: ", i);
        for (int j = 0; j < gcn.layers[gcn.num_layers - 1].out_dim; ++j) {
            printf("%.4f ", output[i * gcn.layers[gcn.num_layers - 1].out_dim + j]);
        }
        printf("\n");
    }
	
    // Free everything
    free(output);
    free_graph(h_graph, 0);
    free_graph(d_graph, 1);
    gcn.free_layers();

    return 0;
}

