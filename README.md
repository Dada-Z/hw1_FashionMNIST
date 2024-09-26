### Three program files
1. main.py: Code that runs the main loop of training the TensorFlow models
- measure loss with cross-entropy after mapping the class labels to one-hot vectors
- use Adam to optimize your parameters
- For each training run, you will use at least two sets of hyperparameters
- choose at least one regularizer and evaluate your system's performance with and without it
- you will perform at least 2×2×2=8 training runs

2. model.py: TensorFlow code that defines the network
- At least 2 architectures (different number and sizes of the layers)
- you must use at least one hidden layer and you must use fully connected layers with ReLU
- or a variation (not convolutional nodes) for all hidden nodes
- softmax for the output layer
 
3. util.py: Helper functions (e.g., for loading the data, small repetitive functions)
- fashion MNIST data
- access to both training data and testing data
- to get validation data from training data, you should partition it into (1) a single training set and a single validation set, or (2) k subsets for k-fold cross validation.

### Submission File
Report + Model
