# Neural Network learn XOR gate in C

Data Structure Assignment1

109503510_通訊三_龍芃如

## STRUCTURE

* Input 3 bits

* 1 hidden layer with 5 nodes

* 1 output layer with 1 node

* Output 1 bit

trained with stochastic gradient descent (SGD) using backprop as a gradient computing technique

## MATH Function
1. Activation function: 1 / (1 + exp(-x)
2. Derivative function: x * (1 — x)
3. Init all weights and biases between 0.0 and 1.0: ((double)rand())/((double)RAND_MAX)

-------------------------------------
# Getting Started!



## Compile & Run

```sh
# Compile
cd 109503510_assignment_1
gcc -o main src/main.c inc/func.h -lm
# Run
./main
# Input training data
Enter the Inputs(3 bits) for training example[0]:0 0 0
Enter the Inputs(3 bits) for training example[1]:0 0 1
Enter the Inputs(3 bits) for training example[2]:0 1 0
Enter the Inputs(3 bits) for training example[3]:0 1 1
Enter the Inputs(3 bits) for training example[4]:1 0 0
Enter the Inputs(3 bits) for training example[5]:1 0 1
Enter the Inputs(3 bits) for training example[6]:1 1 0
Enter the Inputs(3 bits) for training example[7]:1 1 1

# Input the desired output for training data 
Enter the Desired Outputs (Labels) for training example[0]:0 
Enter the Desired Outputs (Labels) for training example[1]:1
Enter the Desired Outputs (Labels) for training example[2]:1
Enter the Desired Outputs (Labels) for training example[3]:0
Enter the Desired Outputs (Labels) for training example[4]:1
Enter the Desired Outputs (Labels) for training example[5]:0
Enter the Desired Outputs (Labels) for training example[6]:0
Enter the Desired Outputs (Labels) for training example[7]:1
```
## Results

```sh
# show hidden layer weight 
Final Hidden Weights:
[[ 3.609603	3.609468	3.609765	] [ 1.055680	1.041983	1.143507	] [ 1.699646	1.703810	1.614928	] [ 6.109442	6.109169	6.109746	] [ 6.219217	6.218272	6.216978	] ]
# show hidden layer bias
Final Hidden Biases:
[ -9.192276	1.885612	1.840868	-8.813599	-2.556926	]
# show output layer weight  
Final Output Weights:
[[11.590770	][-0.541753	][-1.325039	][-10.985385	][10.103797	]]
# show output layer bias 
Final Output Biases:
[-3.142519]
```
## Input & Output

```sh
#Testing
#EXAMPLE
# input(3 bits)
Input:1 0 0
# output (1 bit:0 or 1) 
Output:1
```
## Reference
https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
