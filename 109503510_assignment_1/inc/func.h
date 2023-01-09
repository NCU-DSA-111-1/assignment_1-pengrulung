#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define numTrainingSets 8
#define numInputs 3
#define numOutputs 1
#define numHiddenNodes 5
#define EPOCH 100000
//Activation function and its derivative
double sigmoid(double x);
double dSigmoid(double x);
//// Init all weights and biases between 0.0 and 1.0
double init_weight();
//shuffle to randonize 
void shuffle(int* array, size_t n);
void user_inputoutput( double(*hiddenWeights)[numHiddenNodes], double (*outputWeights)[numOutputs], double* hiddenLayerBias, double* outputLayerBias);
