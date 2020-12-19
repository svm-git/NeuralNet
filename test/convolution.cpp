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

#include "unittest.h"
#include "training.h"
#include "..\src\convolution.h"

void test_convolution()
{
	scenario sc("Test for neural_network::convolution_layer class");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::verbose("1D Convolution Tests");

		typedef neural_network::algebra::metrics<2> _2;
		typedef neural_network::algebra::metrics<3> _3;
		typedef neural_network::algebra::metrics<9> _9;

		auto layer = neural_network::make_convolution_layer<_9, _3, _2, 4>(random_values);

		_9::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001);
	}

	{
		test::verbose("2D Convolution Tests");

		typedef neural_network::algebra::metrics<2, 2> _2x2;
		typedef neural_network::algebra::metrics<10, 10> _10x10;

		auto layer = neural_network::make_convolution_layer<_10x10, _2x2, _2x2, 3>(random_values);

		_10x10::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001);
	}

	{
		test::verbose("3D Convolution Tests");

		typedef neural_network::algebra::metrics<2, 2, 1> _2x2x2;
		typedef neural_network::algebra::metrics<3, 3, 2> _3x3x2;
		typedef neural_network::algebra::metrics<11, 11, 3> _11x11x3;
		typedef neural_network::algebra::metrics<7, 5, 5> _7x4x4;

		auto layer = neural_network::make_convolution_layer<_11x11x3, _3x3x2, _2x2x2, 7>(random_values);

		_11x11x3::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001);
	}

	sc.pass();
}