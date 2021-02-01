# NeuralNet

Library of building blocks for configuring, training, applying and persisting deep neural networks. The library provides implementation for many common types of neuron layers and a set of primitives to define a network architecture. The library also offers compile time verification of compatibility between adjacent layers in the network.

## Topics

- [Neural Networks](#neural-networks)
- [Layers](#layers)
- [Network Ensebles](#network-ensebles)
- [Model Serialization](#model-serialization)
- [Hardware Acceleration](#hardware-acceleration)

## Neural Networks

At a high level, in NeuralNet library a neural network comprises of:
- Interconnected *layers* of neurons.
- The *architecture*, which defines the logical structure of the network. The topology defines layers with their parameters and describes the connections between those layers.
- The *model*, which contains information about layer structures and all the data associated with the layers and about weights and biases to be optimized.
- The *loss* function, which is used to minimize the difference between the network output and the ground truth value at the training stage. It is not used at the prediction stage.

To start using a neural network you should define a network architecture by using *neural_network::make_network* helper function:

    auto network = neural_network::make_network(layer_1, layer_2, ..., layer_n);

The *make_network* function takes a variable number of layers as parameters, and constructs a *network* object that connects together all layers. To define and configure individual layers, use various *neural_network::make_\*_layer* helper functions. Each *layer* is essentially a transformation function, that accepts input data in a form of a tensor, and produces another tensor as output.

A *tensor* is a multidimensional array of numbers, which is defined by its *metrics*. To define tensor, start with defining the rank and the number of dimensions for each rank:

    typedef neural_network::algebra::metrics<4> m4;         // rank-1 tensor (or vector) with 4 elements
    typedef neural_network::algebra::metrics<10, 5> m10x5;  // rank-2 tensor (or matrix) with 10 x 5 elements
    
    m4::tensor_type inputVector;
    m10x5::tensor_type inputMatrix;

The NeuralNet library allows you to define tensors with arbitrary number of ranks and dimensions within each rank, and many layers support tensors of arbitrary ranks. However, some of the layers have restrictions on the ranks of input and output tensors that are allowed.

When multiple layers are connected to each other using the *neural_network::make_network* function, the output data from a layer is passed as input into the next layer. The type of input tensor of the very first layer defines the type of the input tensor for the entire network, and the type of output tensor of the very last layer defines the network output tensor. The library automatically verifies that output tensor of a hidden layer has the same rank and dimensions as the input tensor of the next layer in the network. If a compatibility problem is detected, it results in a compilation error with the additional information that explains the problem.

For example, a simple network that accepts a rank-1 tensor with 10 elements, consists of two fully connected layers with 5 and 4 neurons and uses logistic activation function between the layers can be defined as following:

    typedef neural_network::algebra::metrics<4> m4;
    typedef neural_network::algebra::metrics<5> m5;
    typedef neural_network::algebra::metrics<10> m10;

    auto network = neural_network::make_network(
        neural_network::make_fully_connected_layer<m10, m5>(),
        neural_network::make_logistic_activation_layer<m5>(),
        neural_network::make_fully_connected_layer<m5, m4>(),
        neural_network::make_logistic_activation_layer<m4>());

To train the network you need to use its *train* member function repeatedly using the training data set.  

    network.train(input, truth, loss, rate);

where *input* is the input tensor; *truth* is the desired output tensor value for the given input tensor; *loss* is the target loss function that is minimized during the training; and *rate* is the learning rate to apply.

The NeuralNet library uses backprogapation method to train the network and adjust weights of the inner layers. Each invocation of the *train* method represents a single application of backpropagation algorithm, where input tensor is processed by the network, a loss function gradient is computed, and layer weights are updated to reduce the loss.

To use the network for prediction you should use its *process* member function:

    auto result = network.process(input);

where *input* is the input tensor and *result* is the resulting tensor.

## Layers

The NeuralNet library supports these layers:

- [Fully connected layer](#fully-connected-layer)
- Activation layers
  - [ReLU activation layer](#relu-activation-layer)
  - [Logistic activation layer](#logistic-activation-layer)
- [Max pooling layers](#max-pooling-layers)
- [Convolution layers](#convolution-layers)
- Service layers
  - [Reshape layer](#reshape-layer)
- Loss functions
  - [Squared error loss](#squared-error-loss)

### Fully Connected Layer

Fully connected layer computes an inner product of a weighted sum of inputs plus bias for each element of the output tensor.

To create a new instance of a fully connected layer, use *neural_network::make_fully_connected_layer* helper function. You can also customize the initial weights and regularization parameter of the layer. This example creates a fully connected layer with 10 input and 5 output neurons and initializes its weights and bias with a uniformly distributed random values in range -0.5..0.5, and configures regularization to 0.00003.

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-0.5, 0.5);

    auto random_values = [&distr, &gen]() { return distr(gen); };

    typedef neural_network::algebra::metrics<10> Input;
    typedef neural_network::algebra::metrics<5> Output;
    
    auto layer = neural_network::make_fully_connected_layer<Input, Output>(
        random_values, 0.00003f);

### ReLU Activation Layer

ReLU activation layer applies Rectifier Linear Unit (ReLU) function to all elements of the input tensor, and produces the output tensor that has the same rank and dimensions. ReLU activation layer supports tensors of any rank and dimensions. This example creates a ReLU activation layer for a rank-3 tensor with 15 x 15 x 3 elements using *neural_network::make_relu_activation_layer* helper function.

    typedef neural_network::algebra::metrics<15, 15, 3> Input;
    
    auto layer = neural_network::make_relu_activation_layer<Input>();
    
### Logistic Activation Layer

Logistic activation layer applies logistic function f(x) = 1 / (1 + exp(-x)) to all elements of the input tensor, and produces the output tensor that has the same rank and dimensions. Logistic activation layer supports tensors of any rank and dimensions. This example creates a logistic activation layer for a rank-2 tensor with 10 x 7 elements using *neural_network::make_logistic_activation_layer* helper function.

    typedef neural_network::algebra::metrics<10, 7> Input;
    
    auto layer = neural_network::make_logistic_activation_layer<Input>();

### Max Pooling Layers

The NeuralNet library supports two types of max pooling layers.

A layer that reduces the rank of the input tensor by selecting the largest element within all subtensors of the smaller rank. For example, given a 3 x 10 x 4 input tensor, the layer computes the 10 x 4 output tensor by selecting the largest of the elements with the same indices in the 3 subtensors with 10 x 4 elements. This type of layer is best used together with [network ensembles](#network-ensebles) to combine the output of several networks into a single tensor. This layer supports input tensors of all ranks and dimensions.

To create this layer, use *neural_network::make_max_pooling_layer* helper function without any parameters:

    typedef neural_network::algebra::metrics<4, 3, 2> Input;

    auto layer = neural_network::make_max_pooling_layer<Input>();

A downsampling layer that selects a maximum element within the given core, and applies the core repeatedly by shifting it by the given stride. For example, given an input tensor of 11 x 11 elements, a core of 3 x 3, and stride of 2 x 2, the layer produces an output tensor with 5 x 5 elements. This layer supports only tensors with ranks 1, 2, and 3, and requires that rank of core and stride parameters is the same as the rank of the input tensor.

The layer also verifies that when a given core and stride parameters are applied, then all elements of the input tensors are used, and there is no padding is needed. In other words, for each corresponding dimension of the input, core, and stride metrics this equation is true: *input_size - core_size = n * stride_size* for some integer value *n* > 0.

To create this layer, use *neural_network::make_max_pooling_layer* helper function and specify the core and stride as template parameters:

    typedef neural_network::algebra::metrics<7, 8> Input;
    typedef neural_network::algebra::metrics<3, 2> Core;
    typedef neural_network::algebra::metrics<2, 2> Stride;

    auto layer = neural_network::make_max_pooling_layer<Input, Core, Stride>();

### Convolution layers

Convolution layer is another type of downsampling layers which applies multiple convolution kernels of a given size with a given stride. The layer supports only tensors with ranks 1, 2, and 3, and requires that rank of core and stride parameters is the same as the rank of the input tensor. The layer produces an output tensor with a rank that is input tensor rank + 1, and has as many dimensions in the first rank as there are kernels. For example, a convolution tensor that is applied to a rank-2 input tensor with 19 x 19 elements, with a 3 x 3 core and 2 x 1 stride and 5 kernels produces an output tensor with 5 x 9 x 17 elements.

The layer also verifies that when a given core and stride parameters are applied, then all elements of the input tensors are used, and there is no padding is needed. In other words, for each corresponding dimension of the input, core, and stride metrics this equation is true: *input_size - core_size = n * stride_size* for some integer value *n* > 0.

To create a convolution layer, use *neural_network::make_convolution_layer* helper function, and specify core, stride and the number of kernels as template parameters. This example create an convolution layer that can be applied to a rank-3 tensor with 11 x 11 x 3 elements, with a 3 x 3 x 3 core, 2 x 2 x 1 stride, and 7 kernels. The kernel weights are initialized with uniformly distributed random values in the -0.5..0.5 range. The layer produces a rank-4 output tensor with 7 x 5 x 5 x 2 elements.

    typedef neural_network::algebra::metrics<11, 11, 3> Input;
    typedef neural_network::algebra::metrics<3, 3, 2> Core;
    typedef neural_network::algebra::metrics<2, 2, 1> Stride;
    const size_t Kernels = 7;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

    auto random_values = [&distr, &gen]() { return distr(gen); };
    
    auto layer = neural_network::make_convolution_layer<Input, Core, Stride, Kernels>(random_values);

### Reshape Layer

Reshape layer is a utility layer that changes the rank and dimensions of an input tensor without loosing the data. To create a reshape layer, use *neural_network::make_reshape_layer* helper function, and specify the input and output metrics. The layer verifies that the total number of elements in the output tensor is exactly the same as the total number of elements in the input tensor. For example, a rank-3 with 10 x 5 x 3 elements can be reshaped into a rank-2 tensor with 25 x 6 elements, or a rank-1 tensor with 150 elements.

    typedef neural_network::algebra::metrics<3, 2, 1> Input;
    typedef neural_network::algebra::metrics<6> Output;

    auto layer = neural_network::make_reshape_layer<Input, Output>();

Reshape layer can be used as an adapter layer between different layers.

### Squared Error Loss

Squared error loss function is used during the training stage and it is not used in prediction stage. The squared error loss function can be applied to a pair of tensors that represent network output value and the expected, or ground truth, value and computes the loss value l(x, g) = sum((x - g) ^ 2).

    typedef neural_network::algebra::metrics<2, 2> Output;
	
    neural_network::squared_error_loss<Output> loss;

    network.train(input, truth, loss, rate);

## Network Ensebles

Several neural networks which have identical input and output can be configured, traned and used in parallel by combining them into a *network ensemble*. The network ensemble can be formed by using a *neural_network::make_ensemble* helper function, which takes a variable number of networks as parameters.

    auto ensemble = neural_network::make_ensemble(network_1, network_2, ..., network_n);

The library automatically verifies that input and output tensors of each network in the ensemble have the same rank and dimensions. If a compatibility problem is detected, it results in a compilation error with the additional information that explains the problem.

The resulting enseble is a network that takes an input tensor that is identical to the input tensors for each network in the ensemble, and passes it into each of the networks. The output of the network ensemble is a tensor that is 1 rank higher than the output tensor of the networks in the ensemble, and has as much dimensions as number of networks. For example, an ensemble of 3 networks, each taking a rank-2 input tensor with 10 x 5 elements and producing a rank-1 tensor with 5 elements, results in an ensemble that takes a rank-2 tensor with 10 x 5 elements as input, and produces a rank-2 tensor with 3 x 5 elements as output.

The network ensemble is compatible with a layer interface, and can be used as a complex layer in a more sophisticated networks. The network ensemble layer can also be used in conjunction with a [max pooling layer](#max-pooling-layers) without a core.

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

    auto random_values = [&distr, &gen]() { return distr(gen); };

    typedef neural_network::algebra::metrics<4> m4;
    typedef neural_network::algebra::metrics<5> m5;
    typedef neural_network::algebra::metrics<7> m7;
    typedef neural_network::algebra::metrics<8> m8;
    typedef neural_network::algebra::metrics<2, 2> m2x2;
    typedef neural_network::algebra::metrics<4, 2> m4x2;
    typedef neural_network::algebra::metrics<3, 2, 2> m3x2x2;
	
    auto net = neural_network::make_network(
		
        // Dense layer with ReLU activation
        neural_network::make_network(
            neural_network::make_reshape_layer<m4x2, m8>(),
            neural_network::make_fully_connected_layer<m8, m8>(
                random_values, 0.00005f),
            neural_network::make_relu_activation_layer<m8>(),
            neural_network::make_reshape_layer<m8, m4x2>()
        ),

        // Ensemble of 3 networks
        neural_network::make_ensemble(

            // Network with a single dense layer
            neural_network::make_network(
                neural_network::make_reshape_layer<m4x2, m8>(),
                neural_network::make_fully_connected_layer<m8, m4>(
                    random_values, 0.00003f),
                neural_network::make_relu_activation_layer<m4>(),
                neural_network::make_reshape_layer<m4, m2x2>()
            ),

            // Network with two dense layers
            neural_network::make_network(
                neural_network::make_reshape_layer<m4x2, m8>(),
                neural_network::make_fully_connected_layer<m8, m5>(
                    random_values, 0.00001f),
                neural_network::make_relu_activation_layer<m5>(),
                neural_network::make_fully_connected_layer<m5, m4>(
                    random_values, 0.00002f),
                neural_network::make_relu_activation_layer<m4>(),
                neural_network::make_reshape_layer<m4, m2x2>()
            ),

            // Network with three dense layers
            neural_network::make_network(
                neural_network::make_reshape_layer<m4x2, m8>(),
                neural_network::make_fully_connected_layer<m8, m7>(
                    random_values, 0.00001f),
                neural_network::make_relu_activation_layer<m7>(),
                neural_network::make_fully_connected_layer<m7, m5>(
                    random_values, 0.00002f),
                neural_network::make_relu_activation_layer<m5>(),
                neural_network::make_fully_connected_layer<m5, m4>(
                    random_values, 0.00003f),
                neural_network::make_relu_activation_layer<m4>(),
                neural_network::make_reshape_layer<m4, m2x2>()
            )
        ),

        // Combine the ensemble output
        neural_network::make_max_pooling_layer<m3x2x2>(),
		
        // Final dense layer with Logistic activation
        neural_network::make_network(
            neural_network::make_reshape_layer<m2x2, m4>(),
            neural_network::make_fully_connected_layer<m4, m4>(
                random_values, 0.00003f),
            neural_network::make_logistic_activation_layer<m4>()
        )
    );

## Model Serialization

Trained models can be written into an output stream to save the network weights and parameters, and read from an input stream to initialize the network with the weights of a previously trained network.

To save a model into an output stream, use *neural_network::serialization::write* helper function.

    std::ostream output(...);

    neural_network::serialization::write(output, network);
    
To load a trained model into a network, use *neural_network::serialization::read* helper function.

    std::istream input(...);

    neural_network::serialization::read(input, network);
    
To get the size of a model for a network, use *neural_network::serialization::model_size* helper function.

    size_t modelSize = neural_network::serialization::model_size(network);

## Hardware Acceleration

The NeuralNet library allows you to utilize specialized hardware, such as GPU of FPGA, while training networks or using the trained networks for predictions. To enable this optional feature, define *NEURAL_NET_ENABLE_OPEN_CL* before including any of the NeuralNet headers:

    #define NEURAL_NET_ENABLE_OPEN_CL
    
    #include <ai.h>

The NeuralNet library uses [Boost.Compute](https://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/index.html) library to interact with the OpenCL-enabled devices. Therefore, when you enable the hardware acceleration mode, please make sure that Boost.Compute headers are available in your include directories list and you are linking with the proper OpenCL library for your system. For more details, please follow instructions from the [Compilation and Usage](https://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/getting_started.html#boost_compute.getting_started.compilation_and_usage) section of Boost.Compute library documentation.

To train the network using the available hardware, please use its *train* method and pass an instance of the command queue:

    ::boost::compute::device device = ::boost::compute::system::default_device();
    ::boost::compute::context context(device);
    ::boost::compute::command_queue queue(context, device);
    
    network.train(input, truth, loss, rate, queue);

To use the network for prediction please use its *process* member function and pass an instance of the command queue:

    auto result = network.process(input, queue);

Please note that for smaller tensors it may be more efficient to execute the computations on the main system device rather than scheduling the execution on an OpenCL-enabled devices. In the cases like these the library automatically decides at the compile time which implementation to use.
