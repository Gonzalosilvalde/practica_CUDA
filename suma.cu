#include <stdio.h>
#include <curand_kernel.h>

#define N 50

// Función kernel para inicializar la matriz con valores aleatorios
__global__ void initializeMatrix(float *matrix, unsigned int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = i * N + j;
    
    // Inicializa el generador de números aleatorios para cada hilo
    curandState_t state;
    curand_init(seed, index, 0, &state);
    
    // Asegurarse de no exceder los límites de la matriz
    if (i < N && j < N) {
        matrix[index] = curand_uniform(&state) * 100; // Asigna un valor aleatorio entre 0 y 99
    }
}

__global__ void sumMatrix(float *matrix, float *result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = i * N + j;

    if (i < N && j < N) {
        atomicAdd(result, matrix[index]);
    }
}

int main() {
    float *matrix;
    size_t size = N * N * sizeof(float);
    
    // Aloja memoria en el dispositivo CUDA para la matriz
    cudaMalloc(&matrix, size);
    
    // Define las dimensiones del grid y del bloque
    dim3 blockSize(16, 16); // Bloque de 16x16 hilos
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    // Define una semilla para los números aleatorios
    unsigned int seed = time(NULL);
    
    // Llama al kernel para inicializar la matriz con valores aleatorios
    initializeMatrix<<<gridSize, blockSize>>>(matrix, seed);
    
    // Espera a que todos los threads finalicen
    cudaDeviceSynchronize();

    // Aloja memoria en el dispositivo CUDA para el resultado
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    
    // Inicializa el resultado en 0 en el dispositivo CUDA
    cudaMemset(d_result, 0, sizeof(float));
    
    // Llama al kernel para sumar la matriz y almacenar el resultado en d_result
    sumMatrix<<<gridSize, blockSize>>>(matrix, d_result);
    
    // Copia el resultado desde el dispositivo al host
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Muestra el resultado
    printf("Suma de la matriz: %f\n", h_result);
    
    // Libera la memoria de la matriz y el resultado en el dispositivo CUDA
    cudaFree(matrix);
    cudaFree(d_result);
    
    return 0;
}
