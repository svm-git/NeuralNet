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
#include "serializationtest.h"

#include "..\src\connected.h"

#include "opencltest.h"

template <typename Layer>
void test_dense_layer_on_device(
	::boost::compute::command_queue& queue)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	const unsigned long seedValue = 123;

	gen.seed(seedValue);
	typename Layer cppLayer(random_values);

	gen.seed(seedValue);
	typename Layer openclLayer(random_values);

	typename Layer::input input(random_values);
	typename Layer::output gradient(random_values);

	check_tensors_2d(
		cppLayer.process(input),
		openclLayer.process(input, queue));

	check_tensors_3d(
		cppLayer.compute_gradient(gradient),
		openclLayer.compute_gradient(gradient, queue));

	cppLayer.update_weights(0.001f);
	openclLayer.update_weights(0.001f, queue);

	check_tensors_2d(
		cppLayer.process(input),
		openclLayer.process(input, queue));
}

void test_connected()
{
	scenario sc("Test for neural_network::fully_connected_layer");

	typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
	typedef neural_network::algebra::metrics<5, 4> m5x4;

	m5x4::tensor_type tmp;
	auto layer = neural_network::make_fully_connected_layer<m5x4, m3x2x1>();
	
	m3x2x1::tensor_type ret = layer.process(tmp);
	layer.compute_gradient(ret);
	layer.update_weights(0.9f);

	test_layer_serialization("Fully Connected Layer Serialization Tests", layer);

	{
		auto device = find_test_device();
		::boost::compute::context context(device);
		::boost::compute::command_queue queue(context, device);

		{
			test::verbose("OpenCL Fully Connected Layer Tests");

			typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
			typedef neural_network::algebra::metrics<30, 20, 10> m30x20x10;

			test_dense_layer_on_device<neural_network::fully_connected<m3x2x1, m5x4>>(queue);
			test_dense_layer_on_device<neural_network::fully_connected<m30x20x10, m5x4>>(queue);
		}
	}

	sc.pass();
}
