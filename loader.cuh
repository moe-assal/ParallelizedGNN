#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_optimized.cuh"

typedef struct {
    int N;
    int d;
    int E;

    float* node_features;  // [N * d]
    int* srcs;       // [E]
    int* dsts; // [E]
   	int* degrees;
    
   	// computes on gpu
   	void compute_degrees() {   		 		
		cudaMalloc(&degrees, sizeof(int) * N);
		cudaMemset(degrees, 0, sizeof(int) * N);
		ComputeDegrees(srcs, degrees, E, N);
	}

} Graph;


Graph* load_graph_cpu(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if(!f){
    	printf("Error opening file %s\n", filename);
    	exit(1);
    }
    Graph* g = (Graph*)malloc(sizeof(Graph));

    fread(&g->N, sizeof(int), 1, f);
    fread(&g->d, sizeof(int), 1, f);
    fread(&g->E, sizeof(int), 1, f);

    size_t n_bytes = g->N * g->d * sizeof(float);
    size_t e_bytes = g->E * sizeof(int);


    g->node_features = (float*)malloc(n_bytes);
    g->srcs = (int*)malloc(e_bytes);
    g->dsts = (int*)malloc(e_bytes);

    fread(g->node_features, sizeof(float), g->N * g->d, f);
    fread(g->srcs, sizeof(int), g->E, f);
	fread(g->dsts, sizeof(int), g->E, f);
	
    fclose(f);
    return g;
}

Graph* copy_graph_to_gpu(const Graph* h) {
    Graph* d = (Graph*)malloc(sizeof(Graph));
    *d = *h;  // copy metadata

    size_t n_bytes = h->N * h->d * sizeof(float);
    size_t e_bytes = h->E * sizeof(int);

    cudaMalloc(&d->node_features, n_bytes);
    cudaMalloc(&d->srcs, e_bytes);
	cudaMalloc(&d->dsts, e_bytes);

    cudaMemcpy(d->node_features, h->node_features, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d->srcs, h->srcs, e_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d->dsts, h->dsts, e_bytes, cudaMemcpyHostToDevice);

    return d;
}

void free_graph(Graph* g, int on_gpu) {
    if (!g) return;
    if (on_gpu) {
        cudaFree(g->node_features);
        cudaFree(g->dsts);
        cudaFree(g->srcs);
        cudaFree(g->degrees);
    } else {
       	free(g->degrees);
        free(g->node_features);
        free(g->dsts);
        free(g->srcs);
    }
    free(g);
}

