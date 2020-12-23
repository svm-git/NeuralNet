# NeuralNet

Library of building blocks for configuring, training, applying and persisting deep neural networks. The library provides implementation for many common types of neuron layers and a set of primitives to define a network architecture. The library also offers compile time verification of compatibility between adjacent layers in the network.

## Neural Networks

At a high level, in NeuralNet a neural network comprises:
- Interconnected *layers* of neurons.
- The *architecture*, which defines the logical structure of the network. The topology defines layers with their parameters and describes the connections between those layers.
- The *model*, which contains information about layer structures and all the data associated with the layers and about weights and biases to be optimized.
- The *loss* function, which is used to minimize the difference between the network output and the ground truth value at the training stage. It is not used at the prediction stage.

To start using a neural network you need to define a network architecture by using *make_network* helper function:

    auto network = neural_network::make_network(layer_1, layer_2, ..., layer_n);

The *make_network* function takes a variable number of layers as parameters, and constructs a *network* object that connects together all layers. To define and configure individual layers, use various *make_\*_layer* helper functions. Each *layer* is essentially a transformation function, that accepts input data in a form of a tensor, and produces another tensor as output.

A *tensor* is a multidimensional array of numbers, which is defined by its *metrics*. To define tensor, start with defining the rank and the number of dimensions for each rank:

    typedef neural_network::algebra::metrics<4> _4;         // rank-1 tensor (or vector) with 4 elements
    typedef neural_network::algebra::metrics<10, 5> _10x5;  // rank-2 tensor (or matrix) with 10 x 5 elements
    
    _4::tensor_type inputVector;
    _10x5::tensor_type inputMatrix;

The NeuralNet library allows you to define tensors with arbitrary number of ranks and dimensions within each rank, and many layers support tensors of arbitrary ranks. However, some of the layers have restrictions on the ranks of input and output tensors that are allowed.

When multiple layers are connected to each other using the *make_network* function, the output data from a layer is passed as input into the next layer. The type of input tensor of the very first layer defines the type of the input tensor for the entire network, and the type of output tensor of the very last layer defines the network output tensor. The library automatically verifies that output tensor of a hidden layer has the same rank and dimensions as the input tensor of the next layer in the network. If a compatibility problem is detected, it results in a compilation error with the additional information that explains the problem.

For example, a simple network that accepts a rank 1 tensor with 10 elements, consists of two fully connected layers with 5 and 4 neurons and uses logistic activation function between the layers can be defined as following:

    typedef neural_network::algebra::metrics<4> _4;
    typedef neural_network::algebra::metrics<5> _5;
    typedef neural_network::algebra::metrics<10> _10;

    auto network = neural_network::make_network(
        neural_network::make_fully_connected_layer<_10, _5>(),
        neural_network::make_logistic_activation_layer<_5>(),
        neural_network::make_fully_connected_layer<_5, _4>(),
        neural_network::make_logistic_activation_layer<_4>());

To train the network you need to use its *train* member function repeatedly using the training data set.  

    network.train(input, truth, loss, rate);

where *input* is the input tensor; *truth* is the desired output tensor value for the given input tensor; *loss* is the target loss function that is minimized during the training; and *rate* is the learning rate to apply.

The NeuralNet library uses backprogapation method to train the network and adjust weights of the inner layers. Each invocation of the *train* method represents a single application of backpropagation algorithm, where input tensor is processed by the network, a loss function gradient is computed, and layer weights are updated to reduce the loss.

To use the network for prediction you need to use its *process* member function:

    auto result = network.process(input);

where *input* is the input tensor and *result* is the resulting tensor.

## Layers

The NeuralNet library supports these layers:

- Fully connected layer
- Activation layers
  - ReLU activation layer
  - Logistic activation layer
- Max Pooling layers
- Convolution layers
- Service layers
  - Reshape layer
- Loss layers
  - Squared error loss layer
