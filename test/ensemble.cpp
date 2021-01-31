/*

Copyright (c) 2020-2021 svm-git

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

#include "opencltest.h"

void test_ensemble()
{
	scenario sc("Test for neural_network::network_ensemble class");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

	auto random_values = [&distr, &gen]() { return distr(gen); };
	
	{
		test::verbose("C++ Network Ensemble Training Tests");

		typedef neural_network::algebra::metrics<4> m4;
		typedef neural_network::algebra::metrics<5> m5;
		typedef neural_network::algebra::metrics<7> m7;
		typedef neural_network::algebra::metrics<2, 2> m2x2;
		typedef neural_network::algebra::metrics<4, 2> m4x2;
		typedef neural_network::algebra::metrics<3, 2, 2> m3x2x2;
	
		auto net = neural_network::make_network(
		
			// Dense layer with ReLU activation
			neural_network::make_network(

				neural_network::make_fully_connected_layer<m4x2, m4x2>(
					random_values, 0.00005f),

				neural_network::make_relu_activation_layer<m4x2>()
			),

			// Ensemble of 3 networks
			neural_network::make_ensemble(

				// Network with a single dense layer
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m4x2, m2x2>(
						random_values, 0.00003f),

					neural_network::make_relu_activation_layer<m2x2>()
				),

				// Network with two dense layers
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m4x2, m5>(
						random_values, 0.00001f),

					neural_network::make_relu_activation_layer<m5>(),

					neural_network::make_fully_connected_layer<m5, m2x2>(
						random_values, 0.00002f),

					neural_network::make_relu_activation_layer<m2x2>()
				),

				// Network with three dense layers
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m4x2, m7>(
						random_values, 0.00001f),

					neural_network::make_relu_activation_layer<m7>(),

					neural_network::make_fully_connected_layer<m7, m5>(
						random_values, 0.00002f),

					neural_network::make_relu_activation_layer<m5>(),

					neural_network::make_fully_connected_layer<m5, m2x2>(
						random_values, 0.00003f),

					neural_network::make_relu_activation_layer<m2x2>()
				)
			),

			// Combine the ensemble output
			neural_network::make_max_pooling_layer<m3x2x2>(),
		
			// Final dense layer with Logistic activation
			neural_network::make_network(

				neural_network::make_fully_connected_layer<m2x2, m4>(
					random_values, 0.00003f),

				neural_network::make_logistic_activation_layer<m4>()
			)
		);
	
		m4x2::tensor_type input(random_values);
		m4::tensor_type truth;

		truth(0) = 1.0f;

		neural_network::squared_error_loss<m4> loss;

		float initialLoss = 0.0f, finalLoss = 0.0f;

		train_test_network(net, input, truth, loss, initialLoss, finalLoss);

		test::check_true(finalLoss < initialLoss, "Training did not improve the network.");

		test_layer_serialization("Network Ensemble Serialization Tests", net);
	}

	{
		test::verbose("OpenCL Network Ensemble Training Tests");

		auto device = find_test_device();
		::boost::compute::context context(device);
		::boost::compute::command_queue queue(context, device);

		typedef neural_network::algebra::metrics<8> m8;
		typedef neural_network::algebra::metrics<29> m29;
		typedef neural_network::algebra::metrics<32> m32;
		typedef neural_network::algebra::metrics<37> m37;
		typedef neural_network::algebra::metrics<4, 4> m4x4;
		typedef neural_network::algebra::metrics<8, 8> m8x8;
		typedef neural_network::algebra::metrics<3, 8, 8> m3x8x8;

		auto openclNet = neural_network::make_network(
		
			// Dense layer with ReLU activation
			neural_network::make_network(

				neural_network::make_fully_connected_layer<m8x8, m8x8>(
					random_values, 0.00005f),

				neural_network::make_relu_activation_layer<m8x8>()
			),

			// Ensemble of 3 networks
			neural_network::make_ensemble(

				// Network with a single dense layer
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m8x8, m8x8>(
						random_values, 0.00003f),

					neural_network::make_relu_activation_layer<m8x8>()
				),

				// Network with two dense layers
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m8x8, m37>(
						random_values, 0.00001f),

					neural_network::make_relu_activation_layer<m37>(),

					neural_network::make_fully_connected_layer<m37, m8x8>(
						random_values, 0.00002f),

					neural_network::make_relu_activation_layer<m8x8>()
				),

				// Network with three dense layers
				neural_network::make_network(

					neural_network::make_fully_connected_layer<m8x8, m37>(
						random_values, 0.00001f),

					neural_network::make_relu_activation_layer<m37>(),

					neural_network::make_fully_connected_layer<m37, m29>(
						random_values, 0.00002f),

					neural_network::make_relu_activation_layer<m29>(),

					neural_network::make_fully_connected_layer<m29, m8x8>(
						random_values, 0.00003f),

					neural_network::make_relu_activation_layer<m8x8>()
				)
			),

			// Combine the ensemble output
			neural_network::make_max_pooling_layer<m3x8x8>(),
		
			// Final dense layer with Logistic activation
			neural_network::make_network(

				neural_network::make_fully_connected_layer<m8x8, m8>(
					random_values, 0.00003f),

				neural_network::make_logistic_activation_layer<m8>()
			)
		);

		m8x8::tensor_type input(random_values);
		m8::tensor_type truth;

		truth(0) = 1.0f;

		neural_network::squared_error_loss<m8> loss;

		float initialLoss = 0.0f, finalLoss = 0.0f;

		train_test_network_on_device(
			openclNet,
			input,
			truth,
			loss,
			0.17f,
			initialLoss,
			finalLoss,
			queue);

		test::check_true(finalLoss < initialLoss, "Training did not improve the network.");
	}

	sc.pass();
}
