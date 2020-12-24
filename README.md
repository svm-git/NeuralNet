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

To use the network for prediction you should use its *process* member function:

    auto result = network.process(input);

where *input* is the input tensor and *result* is the resulting tensor.

## Layers

The NeuralNet library supports these layers:

- Fully connected layer
- Activation layers
  - ReLU activation layer
  - Logistic activation layer
- Max pooling layers
- Convolution layers
- Service layers
  - Reshape layer
- Loss layers
  - Squared error loss layer

### Fully Connected Layer

Fully connected layer computes an inner product of a weighted sum of inputs plus bias for each element of the output tensor. The fully connected layer supports only a rank-1 tensors for input and output, and it is best to use a reshape layer to transform tensors of a higher rank.

To create a new instance of a fully connected layer, use *make_fully_connected_layer* helper function. You can also customize the initial weights and regularization parameter of the layer. This example creates a fully connected layer with 10 input and 5 output neurons and initializes its weights and bias with a uniformly distributed random values in range -0.5..0.5, and configures regularization to 0.00003.

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distr(-0.5, 0.5);

    auto random_values = [&distr, &gen]() { return distr(gen); };

    typedef neural_network::algebra::metrics<10> _Input;
    typedef neural_network::algebra::metrics<5> _Output;
    
    auto layer = neural_network::make_fully_connected_layer<_Input, _Output>(
        random_values, 0.00003);

### ReLU Activation Layer

ReLU activation layer applies Rectifier Linear Unit (ReLU) function to all elements of the input tensor, and produces the output tensor that has the same rank and dimensions. ReLU activation layer supports tensors of any rank and dimensions. This example create a ReLU activation layer for a rank-3 tensor with 15 x 15 x 3 elements.

    typedef neural_network::algebra::metrics<15, 15, 3> _Input;
    
    auto layer = neural_network::make_relu_activation_layer<_Input>();
    
### Logistic Activation Layer

Logistic activation layer applies logistic function f(x) = 1 / (1 + exp(-x)) to all elements of the input tensor, and produces the output tensor that has the same rank and dimensions. Logistic activation layer supports tensors of any rank and dimensions. This example create a logistic activation layer for a rank-2 tensor with 10 x 7 elements.

    typedef neural_network::algebra::metrics<10, 7> _Input;
    
    auto layer = neural_network::make_logistic_activation_layer<_Input>();

### Max Pooling Layers

The NeuralNet library supports two types of max pooling layers.

A layer that reduces the rank of the input tensor by selecting the largest element within all subtensors of the smaller rank. For example, given a 3 x 10 x 4 input tensor, the layer computes the 10 x 4 output tensor by selecting the largest of the elements with the same indices in the 3 subtensors with 10 x 4 elements. This type of layer is best used together with network ensembles to combine the output of several networks into a single tensor. This layer supports input tensors of all ranks and dimensions.

To create this layer, use *make_max_pooling_layer* helper function without any parameters:

    typedef neural_network::algebra::metrics<4, 3, 2> _Input;

    auto layer = neural_network::make_max_pooling_layer<_Input>();

A downsampling layer that selects a maximum element within the given core, and applies the core repeatedly by shifting it by the given stride. For example, given an input tensor of 11 x 11 elements, a core of 3 x 3, and stride of 2 x 2, the layer produces an output tensor with 5 x 5 elements. This layer supports only tensors with ranks 1, 2, and 3, and requires that rank of core and stride parameters is the same as the rank of the input tensor.

The layer also verifies that when a given core and stride parameters are applied, then all elements of the input tensors are used, and there is no padding is needed. In other words, for each corresponding dimension of the input, core, and stride metrics this equation is true: *input_size - core_size = n * stride_size* for some integer value *n* > 0.

To create this layer, use *make_max_pooling_layer* helper function and specify the core and stride as template parameters:

    typedef neural_network::algebra::metrics<7, 8> _Input;
    typedef neural_network::algebra::metrics<3, 2> _Core;
    typedef neural_network::algebra::metrics<2, 2> _Stride;

    auto layer = neural_network::make_max_pooling_layer<_Input, _Core, _Stride>();

### Convolution layers

Convolution layer is another type of downsampling layers which applies multiple convolution kernels of a given size with a given stride. The layer supports only tensors with ranks 1, 2, and 3, and requires that rank of core and stride parameters is the same as the rank of the input tensor. The layer produces an output tensor with a rank that is input tensor rank + 1, and has as many dimensions in the first rank as there are kernels. For example, a convolution tensor that is applied to a rank-2 input tensor with 19 x 19 elements, with a 3 x 3 core and 2 x 1 stride and 5 kernels produces an output tensor with 5 x 9 x 17 elements.

The layer also verifies that when a given core and stride parameters are applied, then all elements of the input tensors are used, and there is no padding is needed. In other words, for each corresponding dimension of the input, core, and stride metrics this equation is true: *input_size - core_size = n * stride_size* for some integer value *n* > 0.

To create a convolution layer, use *make_convolution_layer* helper function, and specify core, stride and the number of kernels as template parameters. This example create an convolution layer that can be applied to a rank-3 tensor with 11 x 11 x 3 elements, with a 3 x 3 x 3 core, 2 x 2 x 1 stride, and 7 kernels. The kernel weights are initialized with uniformly distributed random values in the -0.5..0.5 range. The layer produces a rank-4 output tensor with 7 x 5 x 5 x 2 elements.

    typedef neural_network::algebra::metrics<11, 11, 3> _Input;
    typedef neural_network::algebra::metrics<3, 3, 2> _Core;
    typedef neural_network::algebra::metrics<2, 2, 1> _Stride;
    const size_t _Kernels = 7;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distr(-0.5, 0.5);

    auto random_values = [&distr, &gen]() { return distr(gen); };
    
    auto layer = neural_network::make_convolution_layer<_Input, _Core, _Stride, _Kernels>(random_values);

### Reshape Layer

Reshape layer is a utility layer that changes the rank and dimensions of an input tensor without loosing the data. To create a reshape layer, use *make_reshape_layer* helper function, and specify the input and output metrics. The layer verifies that the total number of elements in the output tensor is exactly the same as the total number of elements in the input tensor. For example, a rank-3 with 10 x 5 x 3 elements can be reshaped into a rank-2 tensor with 25 x 6 elements, or a rank-1 tensor with 150 elements.

    typedef neural_network::algebra::metrics<3, 2, 1> _Input;
    typedef neural_network::algebra::metrics<6> _Output;

    auto layer = neural_network::make_reshape_layer<_Input, _Output>();
