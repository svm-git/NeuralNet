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

#include "..\src\pooling.h"

#include "opencltest.h"

template <typename Layer, typename Process, typename Gradient>
void test_pooling_layer_on_device(
	Process process,
	Gradient gradient)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5f, 0.5f);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	const unsigned long seedValue = 123;

	typename Layer cppLayer;
	typename Layer openclLayer;

	gen.seed(seedValue);
	typename Layer::input input(random_values);
	typename Layer::output grad(random_values);

	process(cppLayer, openclLayer, input);

	gradient(cppLayer, openclLayer, grad);
}

template <typename Layer>
void test_1d_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
		{
			check_tensors_1d(
				cppLayer.process(input),
				openclLayer.process(input, queue));
		},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
		{
			check_tensors_1d(
				cppLayer.compute_gradient(gradient),
				openclLayer.compute_gradient(gradient, queue));
		});
}

template <typename Layer>
void test_2d_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
		{
			check_tensors_2d(
				cppLayer.process(input),
				openclLayer.process(input, queue));
		},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
		{
			check_tensors_2d(
				cppLayer.compute_gradient(gradient),
				openclLayer.compute_gradient(gradient, queue));
		});
}

template <typename Layer>
void test_3d_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
		{
			check_tensors_3d(
				cppLayer.process(input),
				openclLayer.process(input, queue));
		},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
		{
			check_tensors_3d(
				cppLayer.compute_gradient(gradient),
				openclLayer.compute_gradient(gradient, queue));
		});
}

template <typename Layer>
void test_1d_generic_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_1d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_1d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	});
}

template <typename Layer>
void test_2d_generic_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_1d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_2d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	});
}

template <typename Layer>
void test_3d_generic_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_2d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_3d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	});
}

template <typename Layer>
void test_4d_generic_pooling_layer_on_device(
	::boost::compute::command_queue& queue)
{
	test_pooling_layer_on_device<Layer>(
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::input& input)
	{
		check_tensors_3d(
			cppLayer.process(input),
			openclLayer.process(input, queue));
	},
		[&queue](Layer& cppLayer, Layer& openclLayer, const typename Layer::output& gradient)
	{
		check_tensors_4d(
			cppLayer.compute_gradient(gradient),
			openclLayer.compute_gradient(gradient, queue));
	});
}

void test_pooling()
{
	scenario sc("Test for neural_network::*_pooling layers");

	const float minRnd = 0.5;
	const float maxRnd = 1.5;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(minRnd, maxRnd);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::verbose("1D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4> m4;

		auto layer = neural_network::make_max_pooling_layer<m4>();

		m4::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 1, "Invalid size of 1D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("1D Max Pooling Layer Serialization Tests", layer);
	}

	{
		test::verbose("2D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4, 3> m4x3;

		auto layer = neural_network::make_max_pooling_layer<m4x3>();

		m4x3::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 3, "Invalid size of 2D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("2D Max Pooling Layer Serialization Tests", layer);
	}

	{
		test::verbose("3D Max Pooling Tests");

		typedef neural_network::algebra::metrics<4, 3, 2> m4x3x2;

		auto layer = neural_network::make_max_pooling_layer<m4x3x2>();

		m4x3x2::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 3, "Invalid size of 3D max pooling output tensor.");
		test::check_true(tmp.size<1>() == 2, "Invalid size of 3D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("3D Max Pooling Layer Serialization Tests", layer);
	}

	{
		test::verbose("4D Max Pooling Tests");

		typedef neural_network::algebra::metrics<5, 4, 3, 2> m5x4x3x2;

		auto layer = neural_network::make_max_pooling_layer<m5x4x3x2>();

		m5x4x3x2::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 4, "Invalid size of 4D max pooling output tensor.");
		test::check_true(tmp.size<1>() == 3, "Invalid size of 4D max pooling output tensor.");
		test::check_true(tmp.size<2>() == 2, "Invalid size of 4D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("4D Max Pooling Layer Serialization Tests", layer);
	}

	{
		test::verbose("1D Max Pooling With Core Tests");

		typedef neural_network::algebra::metrics<1> m1;
		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::algebra::metrics<4> m4;

		auto layer = neural_network::make_max_pooling_layer<m4, m3, m1>();

		m4::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 2, "Invalid size of 1D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("1D Max Pooling With Core Layer Serialization Tests", layer);
	}

	{
		test::verbose("2D Max Pooling With Core Tests");

		typedef neural_network::algebra::metrics<2, 2> m2x2;
		typedef neural_network::algebra::metrics<3, 2> m3x2;
		typedef neural_network::algebra::metrics<7, 8> m7x8;

		auto layer = neural_network::make_max_pooling_layer<m7x8, m3x2, m2x2>();

		m7x8::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 3, "Invalid size of 2D max pooling output tensor.");
		test::check_true(tmp.size<1>() == 4, "Invalid size of 2D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("2D Max Pooling With Core Layer Serialization Tests", layer);
	}

	{
		test::verbose("3D Max Pooling With Core Tests");

		typedef neural_network::algebra::metrics<2, 2, 1> m2x2x1;
		typedef neural_network::algebra::metrics<3, 3, 3> m3x3x3;
		typedef neural_network::algebra::metrics<19, 19, 3> m19x19x3;

		auto layer = neural_network::make_max_pooling_layer<m19x19x3, m3x3x3, m2x2x1>();

		m19x19x3::tensor_type input(random_values);

		auto tmp = layer.process(input);

		test::check_true(tmp.size<0>() == 9, "Invalid size of 3D max pooling output tensor.");
		test::check_true(tmp.size<1>() == 9, "Invalid size of 3D max pooling output tensor.");
		test::check_true(tmp.size<2>() == 1, "Invalid size of 3D max pooling output tensor.");

		layer.compute_gradient(tmp);
		layer.update_weights(0.1f);

		test_layer_serialization("3D Max Pooling With Core Layer Serialization Tests", layer);
	}

	{
		auto context = find_test_device_context();
		::boost::compute::command_queue queue(context, context.get_device());

		{
			test::verbose("OpenCL 1D Pooling Layer Tests");

			typedef neural_network::algebra::metrics<4> m4;
			typedef neural_network::algebra::metrics<40> m40;

			test_1d_generic_pooling_layer_on_device<neural_network::max_pooling<m4>>(queue);
			test_1d_generic_pooling_layer_on_device<neural_network::max_pooling<m40>>(queue);
		}

		{
			test::verbose("OpenCL 2D Pooling Layer Tests");

			typedef neural_network::algebra::metrics<4, 3> m4x3;
			typedef neural_network::algebra::metrics<10, 33> m10x33;

			test_2d_generic_pooling_layer_on_device<neural_network::max_pooling<m4x3>>(queue);
			test_2d_generic_pooling_layer_on_device<neural_network::max_pooling<m10x33>>(queue);
		}

		{
			test::verbose("OpenCL 3D Pooling Layer Tests");

			typedef neural_network::algebra::metrics<4, 3, 2> m4x3x2;
			typedef neural_network::algebra::metrics<4, 8, 6> m4x8x6;

			test_3d_generic_pooling_layer_on_device<neural_network::max_pooling<m4x3x2>>(queue);
			test_3d_generic_pooling_layer_on_device<neural_network::max_pooling<m4x8x6>>(queue);
		}

		{
			test::verbose("OpenCL 4D Pooling Layer Tests");

			typedef neural_network::algebra::metrics<5, 4, 3, 2> m5x4x3x2;
			typedef neural_network::algebra::metrics<5, 8, 4, 2> m5x8x4x2;

			test_4d_generic_pooling_layer_on_device<neural_network::max_pooling<m5x4x3x2>>(queue);
			test_4d_generic_pooling_layer_on_device<neural_network::max_pooling<m5x8x4x2>>(queue);
		}

		{
			test::verbose("OpenCL 1D Pooling With Core Layer Tests");

			typedef neural_network::algebra::metrics<1> m1;
			typedef neural_network::algebra::metrics<3> m3;
			typedef neural_network::algebra::metrics<4> m4;
			typedef neural_network::algebra::metrics<48> m48;

			test_1d_pooling_layer_on_device<neural_network::max_pooling_with_core<m4, m3, m1>>(queue);
			test_1d_pooling_layer_on_device<neural_network::max_pooling_with_core<m48, m3, m1>>(queue);
		}

		{
			test::verbose("OpenCL 2D Pooling With Core Layer Tests");

			typedef neural_network::algebra::metrics<2, 2> m2x2;
			typedef neural_network::algebra::metrics<3, 2> m3x2;
			typedef neural_network::algebra::metrics<7, 8> m7x8;
			typedef neural_network::algebra::metrics<17, 18> m17x18;

			test_2d_pooling_layer_on_device<neural_network::max_pooling_with_core<m7x8, m3x2, m2x2>>(queue);
			test_2d_pooling_layer_on_device<neural_network::max_pooling_with_core<m17x18, m3x2, m2x2>>(queue);
		}

		{
			test::verbose("OpenCL 3D Pooling With Core Layer Tests");

			typedef neural_network::algebra::metrics<2, 2, 1> m2x2x1;
			typedef neural_network::algebra::metrics<3, 3, 3> m3x3x3;
			typedef neural_network::algebra::metrics<9, 9, 3> m9x9x3;
			typedef neural_network::algebra::metrics<19, 19, 4> m19x19x3;

			auto layer = neural_network::make_max_pooling_layer<m19x19x3, m3x3x3, m2x2x1>();

			test_3d_pooling_layer_on_device<neural_network::max_pooling_with_core<m9x9x3, m3x3x3, m2x2x1>>(queue);
			test_3d_pooling_layer_on_device<neural_network::max_pooling_with_core<m19x19x3, m3x3x3, m2x2x1>>(queue);
		}
	}

	sc.pass();
}