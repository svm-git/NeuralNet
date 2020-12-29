/*

Copyright (c) 2020 svm-git

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "stdafx.h"

#include <random>
#include <sstream>

#include "unittest.h"
#include "training.h"
#include "serializationtest.h"

#include "..\src\ai.h"

void test_ensemble()
{
	scenario sc("Test for neural_network::network_ensemble class");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	typedef neural_network::algebra::metrics<4> _4;
	typedef neural_network::algebra::metrics<5> _5;
	typedef neural_network::algebra::metrics<7> _7;
	typedef neural_network::algebra::metrics<2, 2> _2x2;
	typedef neural_network::algebra::metrics<4, 2> _4x2;
	typedef neural_network::algebra::metrics<3, 2, 2> _3x2x2;
	
	auto net = neural_network::make_network(
		
		// Dense layer with ReLU activation
		neural_network::make_network(

			neural_network::make_fully_connected_layer<_4x2, _4x2>(
				random_values, 0.00005),

			neural_network::make_relu_activation_layer<_4x2>()
		),

		// Ensemble of 3 networks
		neural_network::make_ensemble(

			// Network with a single dense layer
			neural_network::make_network(

				neural_network::make_fully_connected_layer<_4x2, _2x2>(
					random_values, 0.00003),

				neural_network::make_relu_activation_layer<_2x2>()
			),

			// Network with two dense layers
			neural_network::make_network(

				neural_network::make_fully_connected_layer<_4x2, _5>(
					random_values, 0.00001),

				neural_network::make_relu_activation_layer<_5>(),

				neural_network::make_fully_connected_layer<_5, _2x2>(
					random_values, 0.00002),

				neural_network::make_relu_activation_layer<_2x2>()
			),

			// Network with three dense layers
			neural_network::make_network(

				neural_network::make_fully_connected_layer<_4x2, _7>(
					random_values, 0.00001),

				neural_network::make_relu_activation_layer<_7>(),

				neural_network::make_fully_connected_layer<_7, _5>(
					random_values, 0.00002),

				neural_network::make_relu_activation_layer<_5>(),

				neural_network::make_fully_connected_layer<_5, _2x2>(
					random_values, 0.00003),

				neural_network::make_relu_activation_layer<_2x2>()
			)
		),

		// Combine the ensemble output
		neural_network::make_max_pooling_layer<_3x2x2>(),
		
		// Final dense layer with Logistic activation
		neural_network::make_network(

			neural_network::make_fully_connected_layer<_2x2, _4>(
				random_values, 0.00003),

			neural_network::make_logistic_activation_layer<_4>()
		)
	);
	
	_4x2::tensor_type input(random_values);
	_4::tensor_type truth;

	truth(0) = 1.0;

	neural_network::squared_error_loss<_4> loss;

	double initialLoss = 0.0, finalLoss = 0.0;

	train_test_network(net, input, truth, loss, initialLoss, finalLoss);

	test::assert(finalLoss < initialLoss, "Training did not improve the network.");

	test_layer_serialization("Network Ensemble Serialization Tests", net);

	sc.pass();
}
