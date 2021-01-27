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

#include "unittest.h"

#include "..\src\loss.h"

#include "opencltest.h"

template <typename Function>
void test_loss_on_device(
	::boost::compute::command_queue& queue)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	typename Function cppFunction;
	typename Function openclFunction;

	typename Function::tensor_type input(random_values);
	typename Function::tensor_type truth(random_values);

	test::check_true(
		cppFunction.compute(input, truth) == openclFunction.compute(input, truth, queue),
		"Unexpected mismatch between C++ and OpenCL results.");

	check_tensors_3d(
		cppFunction.compute_gradient(input, truth),
		openclFunction.compute_gradient(input, truth, queue));
}

void test_loss()
{
	scenario sc("Test for neural_network::*_loss classes");

	auto device = find_test_device();
	::boost::compute::context context(device);
	::boost::compute::command_queue queue(context, device);

	{
		test::verbose("OpenCL Squared Error Loss function Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
		typedef neural_network::algebra::metrics<30, 20, 10> m30x20x10;

		test_loss_on_device<neural_network::squared_error_loss<m3x2x1>>(queue);
		test_loss_on_device<neural_network::squared_error_loss<m30x20x10>>(queue);
	}

	sc.pass();
}
