# 3-Pile NIM Player Trained via ANN
    Isaac Lindland CS 362 lab 3

## Overview
    
Here we build and train a neural network to play moves in a 3-pile game on nim    
1. Building training data with labels, inputs and expected outputs via the xor algorithm
2. Initialize NN as a set of weight matrices with random weights
3. Feed in labeled data, use back-propagation to adjust weights

Currently only achieves roughly ~95 % accuracy in prediction of the correct move
 
### Requires

* python 3.6+
* numpy 
* scikitlearn

### Usage

To play against the trained neural net, run  

    python Game.py 
in the terminal.
To train a new network, run 
    
    python NN_2.py

There are several trained networks in the files training*.npy, with various architectures. 

### Includes

* Game.py - The game to play against the trained NN
* NN_2.py - The class containing the neural net code. To save/load use the .save and .load methods on the nn. Save will create a '.npy' file which contains the weights used for the ANN.
* testing_data.py - The nim algorithm which is used to produce the supervised data. The data is stored in the '.npy' files "X.npy" and "Y.npy"

### Outcome

I ran the testing with an alpha of roughly 0.1 to 0.5 with my best results. Accuracy was roughly ~95% at about 200,000 iterations, which took about 20 minutes to train with a batch size of 200. This NN had an architecture of 18,70,70,9. I found increasing the network length didn't have a great deal of benefit in increasing accuracy.