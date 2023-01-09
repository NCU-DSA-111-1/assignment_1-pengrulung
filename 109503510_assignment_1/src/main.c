#include"func.h"
// Activation function and its derivative
double sigmoid(double x) {
       	return 1 / (1 + exp(-x));
}
double dSigmoid(double x) {
       	return x * (1 - x);
}
// Init all weights and biases between 0.0 and 1.0
double init_weight() {
       	return ((double)rand()) / ((double)RAND_MAX);
}
//shuffle the traning sets to randomize 
void shuffle(int* array, size_t n){
    if (n > 1){
        size_t i;
        for (i = 0; i < n - 1; i++){
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
void user_inputoutput(double (*hiddenWeights)[numHiddenNodes], double (*outputWeights)[numOutputs], double* hiddenLayerBias, double* outputLayerBias ){
    //users input and output
    double *userhiddenlayer=(double*)malloc(sizeof(double)*numInputs);
    double useroutput;
    double *userinputs=(double*)malloc(sizeof(double)*numInputs);
    printf("Input(3 bits):");
    //user input
    for(int i = 0;i <numInputs; i++){
    	scanf("%lf",&userinputs[i]);
    }
    //computation
    for (int j = 0; j < numHiddenNodes; j++) {
         double activation = hiddenLayerBias[j];
         for (int k = 0; k < numInputs; k++) {
              activation += userinputs[k] * hiddenWeights[k][j];
         }
         userhiddenlayer[j] = sigmoid(activation);
    }
    for (int j = 0; j < numOutputs; j++) {
         double activation = outputLayerBias[j];
         for (int k = 0; k < numHiddenNodes; k++) {
             activation += userhiddenlayer[k] * outputWeights[k][j];
         }
         useroutput= sigmoid(activation);
    }
    //user output
    printf("Output:%0.lf\n",round(useroutput));
}
void print_weightandbias(double(*hiddenWeights)[numHiddenNodes], double(*outputWeights)[numOutputs], double* hiddenLayerBias, double* outputLayerBias ){
    // Print hidden weights
    printf("Final Hidden Weights:\n[");
    for (int j = 0; j < numHiddenNodes; j++) {
        printf( "[ ");
        for (int k = 0; k < numInputs; k++) {
            printf("%lf\t",hiddenWeights[k][j]);
        }
        printf("] ");
    }
    printf("]\n");
    //Print hidden biases
        printf("Final Hidden Biases:\n[ ");
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%lf\t",hiddenLayerBias[j]);
    }
    printf("]\n");
    //Print output weights
    printf("Final Output Weights:\n[");
    for (int j = 0; j < numOutputs; j++) {
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("[%lf\t]",outputWeights[k][j]);
        }
    }
    printf("]\n");
    //Print output bias
        printf("Final Output Biases:\n");
    for (int j = 0; j < numOutputs; j++) {
        printf("[%lf", outputLayerBias[j]);
    }
    printf("]\n");}
//usersetiings
void user_setting(double (*training_inputs)[numInputs], double(*training_outputs)[numOutputs] ){
	for(int i=0;i<numTrainingSets;i++){
    	printf("Enter the Inputs(3 bits) for training example[%d]:",i);
    	for(int j=0;j<numInputs;j++){
    	scanf("%lf",&training_inputs[i][j]);
    	}
    	}
    	for(int i=0;i<numTrainingSets;i++){
    	printf("Enter the Desired Outputs (Labels) for training example[%d]:",i);
    	scanf("%lf",&training_outputs[i][0]);
    	}
}
int main() {
    const double lr = 0.05f;
    //learing rate
    double *hiddenLayer=(double*)malloc(sizeof(double)*numHiddenNodes);
    double *outputLayer=(double*)malloc(sizeof(double)*numOutputs);
    //hiddenbias and outputbias
    double *hiddenLayerBias=(double*)malloc(sizeof(double)*numHiddenNodes);
    double *outputLayerBias=(double*)malloc(sizeof(double)*numOutputs);
    //hiddenweight and outputweight
    double (*arr_hid_weights)[numInputs][numHiddenNodes]=malloc(sizeof *arr_hid_weights); //hiddenWeights
    double (*arr_out_weights)[numHiddenNodes][numOutputs]=malloc(sizeof *arr_out_weights);//outputWeights;
    //training data
    double (*training_inputs)[numTrainingSets][numInputs]=malloc(sizeof *training_inputs);
    double (*training_outputs)[numTrainingSets][numOutputs]=malloc(sizeof *training_outputs); 
    user_setting((*training_inputs), (*training_outputs));
    double loss = 0;
    FILE *fp;
    //initialize
    //call init_weight function
    // Init all weights and biases between 0.0 and 1.0
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            (*arr_hid_weights)[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            (*arr_out_weights)[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }
    if ((fp=fopen("tmp1.csv","a"))==NULL){
    	printf("Open file error!\n");
    }
    //initialize training sets order
    int *trainingSetOrder=(int*)malloc(sizeof(int)*numTrainingSets);
    for(int i=0;i<numTrainingSets;i++){
    	trainingSetOrder[i]=i;
    }
    // Iterate through the entire training for a number of epochs
    for (int n = 0; n < EPOCH; n++) {
	// As per SGD, shuffle the order of the training set 
        shuffle(trainingSetOrder, numTrainingSets);
        // Cycle through each of the training set elements(all
        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];
            // Forward pass
            //hiddenlayer=sigmoid(hiddenlayerbias+traininginput*hiddenlayerweight)
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += (*training_inputs)[i][k] * (*arr_hid_weights)[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            //outputlayer=sigmoid(outputlayerbias+hiddenlayer*outputweight)
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * (*arr_out_weights)[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            //calculate loss
            loss = loss + ((*training_outputs)[i][0] - outputLayer[0]) * ((*training_outputs)[i][0] - outputLayer[0]);
            // Backprop
            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double errorOutput = ((*training_outputs)[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }
            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    errorHidden += deltaOutput[k] * (*arr_out_weights)[j][k];
                }
                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
            }
            //Apply change in output weights
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    (*arr_out_weights)[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            //Apply chages in hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    (*arr_hid_weights)[k][j] += (*training_inputs)[i][k] * deltaHidden[j] * lr;
                }
            }
        }
        //print loss to file
            if((n%10)==9||n==0){
            	fprintf(fp,"%d,%lf\n",n+1,loss/(8*(n+1)));
            }
    }
    //file close
    fclose(fp);
    //print weight and bias
    print_weightandbias((*arr_hid_weights),(*arr_out_weights),hiddenLayerBias,outputLayerBias);
    //user input and output
    user_inputoutput((*arr_hid_weights),(*arr_out_weights),hiddenLayerBias,outputLayerBias);
    //free(malloc)(memory)
    free(hiddenLayer);
    free(outputLayer);
    free(hiddenLayerBias);
    free(outputLayerBias);
    free(*arr_hid_weights); 
    free(*arr_out_weights);
    free(trainingSetOrder);
    return 0;
}
