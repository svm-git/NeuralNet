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
#include "..\src\pooling.h"

void test_pooling()
{
	scenario sc("Test for neural_network::*_pooling layers");

	const double minRnd = 0.5;
	const double maxRnd = 1.5;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minRnd, maxRnd);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::verbose("1D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4> _4;

		auto layer = neural_network::make_max_pooling_layer<_4>();

		_4::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::assert(tmp.size<0>() == 1, "Invalid size of 1D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1);
	}

	{
		test::verbose("2D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4, 3> _4x3;

		auto layer = neural_network::make_max_pooling_layer<_4x3>();

		_4x3::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::assert(tmp.size<0>() == 3, "Invalid size of 2D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1);
	}

	{
		test::verbose("3D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4, 3, 2> _4x3x2;

		auto layer = neural_network::make_max_pooling_layer<_4x3x2>();

		_4x3x2::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::assert(tmp.size<0>() == 3, "Invalid size of 3D max pooling output tensor.");
		test::assert(tmp.size<1>() == 2, "Invalid size of 3D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1);
	}

	{
		test::verbose("4D Max Pooling Tests");

		typedef neural_network::algebra::metrics<5, 4, 3, 2> _5x4x3x2;

		auto layer = neural_network::make_max_pooling_layer<_5x4x3x2>();

		_5x4x3x2::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::assert(tmp.size<0>() == 4, "Invalid size of 4D max pooling output tensor.");
		test::assert(tmp.size<1>() == 3, "Invalid size of 4D max pooling output tensor.");
		test::assert(tmp.size<2>() == 2, "Invalid size of 4D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1);
	}

	sc.pass();
}