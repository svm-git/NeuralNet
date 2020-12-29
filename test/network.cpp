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

#pragma once

#include "stdafx.h"

#include <random>
#include <sstream>

#include "unittest.h"
#include "training.h"
#include "serializationtest.h"

#include "..\src\ai.h"

void test_network()
{
	scenario sc("Test for neural_network::network class");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	typedef neural_network::algebra::metrics<4> _4;
	typedef neural_network::algebra::metrics<5> _5;
	typedef neural_network::algebra::metrics<5, 2> _5x2;
	typedef neural_network::algebra::metrics<2, 2> _2x2;

	auto net = neural_network::make_network(

		neural_network::make_fully_connected_layer<_5x2, _5>(
			random_values, 0.00003),

		neural_network::make_relu_activation_layer<_5>(),

		neural_network::make_fully_connected_layer<_5, _2x2>(
			random_values, 0.00005),

		neural_network::make_logistic_activation_layer<_2x2>()
	);

	_5x2::tensor_type input(random_values);
	_2x2::tensor_type truth;

	truth(0, 0) = 1.0;

	neural_network::squared_error_loss<_2x2> loss;

	double initialLoss = 0.0, finalLoss = 0.0;

	train_test_network(net, input, truth, loss, initialLoss, finalLoss);

	test::assert(finalLoss < initialLoss, "Training did not improve the network.");

	test_layer_serialization("Network Serialization Tests", net);

	sc.pass();
}