#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <random>

#include<cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace std;

// Kernel

__global__ void AddBias(float* Y, const float* B, int filas, int columnas) {

    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila >= filas || columna >= columnas) {
        return;
    }

    int i = fila + columna * filas;
    Y[i] = 1.0 / (1.0 + expf(-(Y[i] + B[fila])));
    
}

__global__ void GradienteOculto(float* dz, const float* A, int filas, int columnas) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila >= filas || columna >= columnas) {
        return;
    }

    int i = fila + columna * filas;
    dz[i] = dz[i] * A[i] * (1.0 - A[i]);
}

__global__ void GradienteSalida(float* Y, const float* B, float* dz, int filas, int columnas) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila >= filas || columna >= columnas) {
        return;
    }

    int i = fila + columna * filas;
    dz[i] = Y[i] - B[i];
}

__global__ void ActualizarBias(float* dz, float* db, int filas, int columnas) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= filas) {
        return;
    }

    float d = 0.0;
    for (int i = 0; i < columnas; i++) {
        d = d + dz[fila + i * filas];
    }
    db[fila] = d / columnas;
}

__global__ void SGD(float* W, const float* dw, int n, float tasa) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    W[i] = W[i] - tasa * dw[i];
}

__global__ void Capas() {

}

struct MLP {
    int B = 0;
    //int L = 0;

    vector<int> Tamanos = { 512,256,128,10 };
    int L = Tamanos.size();

    vector<float*> d_w;
    vector<float*> d_b;
    vector<float*> d_A;

    vector<float*> d_dA;
    vector<float*> d_dw;
    vector<float*> d_db;

    float* d_T = 0;

    float tasa_aprendizaje = 0.01f;

    cublasHandle_t CUDA;

    

    MLP(int b) {
        B = b;

        mt19937 gen(0);
        normal_distribution<float> dist(0.0f, 0.01f);

        d_w.resize(L);
        d_b.resize(L);
        d_A.resize(L + 1);

        d_dw.resize(L);
        d_db.resize(L);
        d_dA.resize(L + 1);

        cublasCreate(&CUDA);

        cudaMalloc(&d_A[0], 3072 * B * sizeof(float));

        for (int i = 0; i < L; i++) {
            int input = 0;
            if (i == 0) {
                input = 3072;
            }
            else {
                input = Tamanos[i - 1];
            }

            int output = Tamanos[i];

            cudaMalloc(&d_w[i], input * output * sizeof(float));
            cudaMalloc(&d_b[i], output * sizeof(float));
            cudaMalloc(&d_A[i + 1], B * output * sizeof(float));

            cudaMalloc(&d_dw[i], input * output * sizeof(float));
            cudaMalloc(&d_db[i], output * sizeof(float));

            cudaMalloc(&d_dA[i], input * B * sizeof(float));
            cudaMalloc(&d_dA[i + 1], B * output * sizeof(float));

            vector<float> h_w(input * output);
            vector<float> h_b(output);

            for (auto& peso : h_w) {
                peso = dist(gen);
            }

            fill(h_b.begin(), h_b.end(), 0.0f);

            cudaMemcpy(d_w[i], h_w.data(), input * output * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b[i], h_b.data(), output * sizeof(float), cudaMemcpyHostToDevice);

        }

        int salida = Tamanos.back();
        cudaMalloc(&d_T, salida * B * sizeof(float));
    }

    void Target(const vector<int>& ETIQUETAS, int B) {
        int salida = Tamanos.back();
        vector<float> h_T(salida * B, 0.0);

        for (int i = 0; i < B; i++) {
            int j = ETIQUETAS[i];

            if (j >= 0 && j < salida) {
                h_T[j + i * salida] = 1.0;
            }
        }

        cudaMemcpy(d_T, h_T.data(), salida * B * sizeof(float), cudaMemcpyHostToDevice);
    }

    void forward(int batch) {
        float alpha = 1.0;
        float beta = 0.0;
        int input = 0;
        int output = 0;

        dim3 bloque(16, 16);

        for (int i = 0; i < L; i++) {

            if (i == 0) {
                input = 3072;
            }
            else {
                input = Tamanos[i - 1];
            }

            output = Tamanos[i];

            cublasSgemm(CUDA, CUBLAS_OP_T, CUBLAS_OP_N, output, batch, input, &alpha,
                d_w[i], input,
                d_A[i], input,
                &beta, d_A[i + 1], output);

            dim3 GRID((batch + 15) / 16, (output + 15) / 16);
            AddBias << < GRID, bloque >> > (d_A[i + 1], d_b[i], output, batch);
        }
    }

    void Batch(vector<vector<float>>& IMAGENES, vector<int>& INDICES, int B) {
        vector<float> h_x(3072 * B);

        for (int i = 0; i < B; i++) {
            auto& IMAGEN = IMAGENES[INDICES[i]];

            for (int j = 0; j < 3072; j++) {
                h_x[j + i * 3072] = IMAGEN[j];
            }
        }

        cudaMemcpy(d_A[0], h_x.data(), 3072 * B * sizeof(float), cudaMemcpyHostToDevice);
    }

    void Output(vector<float>& host, int batch) {
        int dim = Tamanos.back();
        host.resize(dim * batch);
        cudaMemcpy(host.data(), d_A[L], dim * batch * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void backward(int B) {
        float alpha = 1.0f;
        float beta = 0.0f;


        int outL = Tamanos.back();

        int threads = 0;
        int bloques = 0;

        dim3 block2D(16, 16);
        dim3 grid2D((B + 15) / 16, (outL + 15) / 16);

        GradienteSalida << <grid2D, block2D >> > (d_A[L], d_T, d_dA[L], outL, B);
        cudaDeviceSynchronize();

        for (int i = L - 1; i >= 0; --i) {
            int salida = Tamanos[i];
            int entrada = 0;
            if (i == 0) {
                entrada = 3072;
            }
            else {
                entrada = Tamanos[i - 1];
            }

            cublasSgemm(CUDA, CUBLAS_OP_N, CUBLAS_OP_T, entrada, salida, B, &alpha,
                d_A[i], entrada,
                d_dA[i + 1], salida,
                &beta, d_dw[i], entrada);

            threads = 128;
            bloques = (salida + threads - 1) / threads;
            ActualizarBias << <bloques, threads >> > (d_dA[i + 1], d_db[i], salida, B);

            if (i > 0) {
                cublasSgemm(CUDA, CUBLAS_OP_N, CUBLAS_OP_N, entrada, B, salida, &alpha,
                    d_w[i], entrada,
                    d_dA[i + 1], salida,
                    &beta, d_dA[i], entrada);

                dim3 gridHidden((B + 15) / 16, (entrada + 15) / 16);
                GradienteOculto << <gridHidden, block2D >> > (d_dA[i], d_A[i], entrada, B);
            }

            int nW = entrada * salida;
            int nB = salida;

            int bW = (nW + 255) / 256;
            int bB = (nB + 255) / 256;

            SGD << <bW, 256 >> > (d_w[i], d_dw[i], nW, tasa_aprendizaje);
            SGD << <bB, 256 >> > (d_b[i], d_db[i], nB, tasa_aprendizaje);

            cudaDeviceSynchronize();
        }
    }
};


void load(string archivo, vector<vector<float>>& IMAGENES, vector<int>& ETIQUETAS) {

    ifstream f(archivo, ios::binary);

    for (int i = 0; i < 10000; i++) {

        unsigned char etiqueta;
        unsigned char buffer[3072];

        f.read((char*)&etiqueta, 1);
        f.read((char*)buffer, 3072);

        vector<float> IMAGEN(3072);

        for (int j = 0; j < 3072; j++) {
            float v = buffer[j] / 255.0;
            IMAGEN[j] = (v - 0.5) * 2.0;
        }

        IMAGENES.push_back(move(IMAGEN));
        ETIQUETAS.push_back(etiqueta);
    }
}

int main()
{
    const int B = 64;
    const int Epocas = 100;

    vector<vector<float>> IMAGENES;
    vector<vector<float>> IMAGENES_TESTING;
    vector<int> ETIQUETAS;
    vector<int> ETIQUETAS_TESTING;

    int correctos = 0;
    int total = 0;

    for (int i = 1; i <= 5; ++i) {
        string archivo = "data_batch_" + to_string(i) + ".bin";
        load(archivo, IMAGENES, ETIQUETAS);

    }

    load("test_batch.bin", IMAGENES_TESTING, ETIQUETAS_TESTING);


    MLP MultiPerceptron(B);

    int Entrenamiento = IMAGENES.size();
    int Testing = IMAGENES_TESTING.size();
    int dim = MultiPerceptron.Tamanos.back();


    for (int e = 0; e < Epocas; ++e) {

        cout << "Epoca: " << (e + 1) << endl;

        correctos = 0;
        total = 0;

        for (int i = 0; i < Entrenamiento; i = i + B) {

            int batch = min(B, Entrenamiento - i);

            vector<int> INDICES(batch);
            vector<int> LABELS(batch);

            for (int j = 0; j < batch; ++j) {
                INDICES[j] = i + j;
                LABELS[j] = ETIQUETAS[i + j];
            }

            MultiPerceptron.Batch(IMAGENES, INDICES, batch);
            MultiPerceptron.Target(LABELS, batch);

            MultiPerceptron.forward(batch);
            MultiPerceptron.backward(batch);
        }

        for (int i = 0; i < Testing; i = i + B) {

            int batch = min(B, Testing - i);

            vector<int> INDICES(batch);

            for (int j = 0; j < batch; ++j) {
                INDICES[j] = i + j;

            }

            vector<float> Y;

            MultiPerceptron.Batch(IMAGENES_TESTING, INDICES, batch);
            MultiPerceptron.forward(batch);

            MultiPerceptron.Output(Y, batch);

            for (int j = 0; j < batch; ++j) {
                int prev = 0;
                float best = Y[0 + j * dim];

                for (int k = 1; k < dim; ++k) {

                    float v = Y[k + j * dim];

                    if (v > best) {

                        best = v;
                        prev = k;

                    }
                }
                int label = ETIQUETAS_TESTING[INDICES[j]];

                if (prev == label) {
                    correctos++;
                }
                total++;
            }
        }

        int test = 100.0 * correctos / max(1, total);

        cout << "Precision: " << test << "%" << endl;
    }

    return 0;
}
