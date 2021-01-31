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

#include "..\src\convolution.h"

#include "opencltest.h"

template <typename Layer, typename Process, typename Gradient, typename Weights>
void test_convolution_layer_on_device(
	Process process,
	Gradient gradient,
	Weights weights)
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
	typename Layer::output grad(random_values);

	process(cppLayer, openclLayer, input);

	gradient(cppLayer, openclLayer, grad);

	weights(cppLayer, openclLayer, input);
}

template <typename Layer>
void test_1d_convolution_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_convolution_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
		{
			check_tensors_2d(
				cppLayer.process(input),
				openclLayer.process(input, queue));
		},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
		{
			check_tensors_1d(
				cppLayer.compute_gradient(gradient),
				openclLayer.compute_gradient(gradient, queue));
		},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
		{
			cppLayer.update_weights(0.001f);
			openclLayer.update_weights(0.001f, queue);

			check_tensors_2d(
				cppLayer.process(input),
				openclLayer.process(input, queue));
		});
}

template <typename Layer>
void test_2d_convolution_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_convolution_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_3d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_2d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		cppLayer.update_weights(0.001f);
		openclLayer.update_weights(0.001f, queue);

		check_tensors_3d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	});
}

template <typename Layer>
void test_3d_convolution_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_convolution_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_4d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_3d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		cppLayer.update_weights(0.001f);
		openclLayer.update_weights(0.001f, queue);

		check_tensors_4d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	});
}

void test_convolution()
{
	scenario sc("Test for neural_network::convolution_layer class");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::verbose("1D Convolution Tests");

		typedef neural_network::algebra::metrics<2> m2;
		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::algebra::metrics<9> m9;

		auto layer = neural_network::make_convolution_layer<m9, m3, m2, 4>(random_values);

		m9::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001f);

		test_layer_serialization("1D Convolution Layer Serialization Tests", layer);
	}

	{
		test::verbose("2D Convolution Tests");

		typedef neural_network::algebra::metrics<2, 2> m2x2;
		typedef neural_network::algebra::metrics<10, 10> m10x10;

		auto layer = neural_network::make_convolution_layer<m10x10, m2x2, m2x2, 3>(random_values);

		m10x10::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001f);

		test_layer_serialization("2D Convolution Layer Serialization Tests", layer);
	}

	{
		test::verbose("3D Convolution Tests");

		typedef neural_network::algebra::metrics<2, 2, 1> m2x2x2;
		typedef neural_network::algebra::metrics<3, 3, 2> m3x3x2;
		typedef neural_network::algebra::metrics<11, 11, 3> m11x11x3;
		typedef neural_network::algebra::metrics<7, 5, 5> m7x4x4;

		auto layer = neural_network::make_convolution_layer<m11x11x3, m3x3x2, m2x2x2, 7>(random_values);

		m11x11x3::tensor_type input(random_values);
		auto output = layer.process(input);
		auto grad = layer.compute_gradient(output);
		layer.update_weights(0.001f);

		test_layer_serialization("3D Convolution Layer Serialization Tests", layer);
	}

	{
		auto context = find_test_device_context();
		::boost::compute::command_queue queue(context, context.get_device());

		{
			test::verbose("OpenCL 1D Convolution Layer Tests");

			typedef neural_network::algebra::metrics<2> m2;
			typedef neural_network::algebra::metrics<3> m3;
			typedef neural_network::algebra::metrics<9> m9;

			test_1d_convolution_layer_on_device<neural_network::convolution<m9, m3, m2, 4>>(queue);
			test_1d_convolution_layer_on_device<neural_network::convolution<m9, m3, m2, 96>>(queue);
		}

		{
			test::verbose("OpenCL 2D Convolution Layer Tests");

			typedef neural_network::algebra::metrics<2, 2> m2x2;
			typedef neural_network::algebra::metrics<10, 10> m10x10;

			test_2d_convolution_layer_on_device<neural_network::convolution<m10x10, m2x2, m2x2, 4>>(queue);
			test_2d_convolution_layer_on_device<neural_network::convolution<m10x10, m2x2, m2x2, 36>>(queue);
		}

		{
			test::verbose("OpenCL 3D Convolution Layer Tests");

			typedef neural_network::algebra::metrics<2, 2, 1> m2x2x2;
			typedef neural_network::algebra::metrics<3, 3, 2> m3x3x2;
			typedef neural_network::algebra::metrics<11, 11, 3> m11x11x3;
			typedef neural_network::algebra::metrics<7, 5, 5> m7x4x4;

			test_3d_convolution_layer_on_device<neural_network::convolution<m11x11x3, m3x3x2, m2x2x2, 7>>(queue);
			test_3d_convolution_layer_on_device<neural_network::convolution<m11x11x3, m3x3x2, m2x2x2, 19>>(queue);
		}
	}

	sc.pass();
}