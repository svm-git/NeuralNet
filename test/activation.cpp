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

#include "..\src\activation.h"

#include "opencltest.h"

template <typename Layer>
void test_activation_layer_on_device(
	::boost::compute::command_queue& queue)
{
	typename Layer::input input;

	for (size_t x = 0; x < input.size<0>(); ++x)
	{
		for (size_t y = 0; y < input.size<1>(); ++y)
		{
			for (size_t z = 0; z < input.size<2>(); ++z)
			{
				input(x, y, z) = ((x + y + z) & 1) ? 0.5f : -0.5f;
			}
		}
	}

	typename Layer cppLayer;
	typename Layer openclLayer;

	check_tensors_3d(
		cppLayer.process(input),
		openclLayer.process(input, queue));

	check_tensors_3d(
		cppLayer.compute_gradient(input),
		openclLayer.compute_gradient(input, queue));
}

void test_activation()
{
	scenario sc("Test for neural_network::*_activation classes");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::verbose("1D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::relu_activation<m3> relu_1d;

		static_assert(std::is_same<relu_1d::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU input type.");
		static_assert(std::is_same<relu_1d::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU output type.");

		m3::tensor_type tmp(random_values);
		relu_1d relu = neural_network::make_relu_activation_layer<m3>();

		relu.process(tmp);
		relu.compute_gradient(tmp);

		test_layer_serialization("1D RELU Activation Layer Serialization Tests", relu);
	}

	{
		test::verbose("2D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3, 2> m3x2;
		typedef neural_network::relu_activation<m3x2> relu_2d;

		static_assert(std::is_same<relu_2d::input, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-RELU input type.");
		static_assert(std::is_same<relu_2d::output, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-RELU output type.");

		m3x2::tensor_type tmp(random_values);
		relu_2d relu = neural_network::make_relu_activation_layer<m3x2>();

		relu.process(tmp);
		relu.compute_gradient(tmp);

		test_layer_serialization("2D RELU Activation Layer Serialization Tests", relu);
	}

	{
		test::verbose("3D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
		typedef neural_network::relu_activation<m3x2x1> relu_3d;

		static_assert(std::is_same<relu_3d::input, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-RELU input type.");
		static_assert(std::is_same<relu_3d::output, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-RELU output type.");

		m3x2x1::tensor_type tmp(random_values);
		relu_3d relu = neural_network::make_relu_activation_layer<m3x2x1>();

		relu.process(tmp);
		relu.compute_gradient(tmp);

		test_layer_serialization("3D RELU Activation Layer Serialization Tests", relu);
	}

	{
		test::verbose("1D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::logistic_activation<m3> logistic_1d;

		static_assert(std::is_same<logistic_1d::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Logistic input type.");
		static_assert(std::is_same<logistic_1d::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Logistic output type.");

		m3::tensor_type tmp(random_values);
		logistic_1d logistic = neural_network::make_logistic_activation_layer<m3>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);

		test_layer_serialization("1D Logistic Activation Layer Serialization Tests", logistic);
	}

	{
		test::verbose("2D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3, 2> m3x2;
		typedef neural_network::logistic_activation<m3x2> logistic_2d;

		static_assert(std::is_same<logistic_2d::input, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Logistic input type.");
		static_assert(std::is_same<logistic_2d::output, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Logistic output type.");

		m3x2::tensor_type tmp(random_values);
		logistic_2d logistic = neural_network::make_logistic_activation_layer<m3x2>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);

		test_layer_serialization("2D Logistic Activation Layer Serialization Tests", logistic);
	}

	{
		test::verbose("3D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
		typedef neural_network::logistic_activation<m3x2x1> logistic_3d;

		static_assert(std::is_same<logistic_3d::input, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Logistic input type.");
		static_assert(std::is_same<logistic_3d::output, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Logistic output type.");

		m3x2x1::tensor_type tmp(random_values);
		logistic_3d logistic = neural_network::make_logistic_activation_layer<m3x2x1>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);

		test_layer_serialization("3D Logistic Activation Layer Serialization Tests", logistic);
	}

	{
		test::verbose("1D Tanh Activation Tests");

		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::tanh_activation<m3> tanh_1d;

		static_assert(std::is_same<tanh_1d::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Tanh input type.");
		static_assert(std::is_same<tanh_1d::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Tanh output type.");

		m3::tensor_type tmp(random_values);
		tanh_1d layer = neural_network::make_tanh_activation_layer<m3>();

		layer.process(tmp);
		layer.compute_gradient(tmp);

		test_layer_serialization("1D Tanh Activation Layer Serialization Tests", layer);
	}

	{
		test::verbose("2D Tanh Activation Tests");

		typedef neural_network::algebra::metrics<3, 2> m3x2;
		typedef neural_network::tanh_activation<m3x2> tanh_2d;

		static_assert(std::is_same<tanh_2d::input, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Tanh input type.");
		static_assert(std::is_same<tanh_2d::output, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Tanh output type.");

		m3x2::tensor_type tmp(random_values);
		tanh_2d layer = neural_network::make_tanh_activation_layer<m3x2>();

		layer.process(tmp);
		layer.compute_gradient(tmp);

		test_layer_serialization("2D Tanh Activation Layer Serialization Tests", layer);
	}

	{
		test::verbose("3D Tanh Activation Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
		typedef neural_network::tanh_activation<m3x2x1> tanh_3d;

		static_assert(std::is_same<tanh_3d::input, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Tanh input type.");
		static_assert(std::is_same<tanh_3d::output, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Tanh output type.");

		m3x2x1::tensor_type tmp(random_values);
		tanh_3d layer = neural_network::make_tanh_activation_layer<m3x2x1>();

		layer.process(tmp);
		layer.compute_gradient(tmp);

		test_layer_serialization("3D Tanh Activation Layer Serialization Tests", layer);
	}

	{
		auto context = find_test_device_context();
		::boost::compute::command_queue queue(context, context.get_device());

		{
			test::verbose("OpenCL ReLU Activation Tests");

			typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
			typedef neural_network::algebra::metrics<300, 20, 10> m300x20x10;

			test_activation_layer_on_device<neural_network::relu_activation<m3x2x1>>(queue);
			test_activation_layer_on_device<neural_network::relu_activation<m300x20x10>>(queue);
		}

		{
			test::verbose("OpenCL Logistic Activation Tests");

			typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
			typedef neural_network::algebra::metrics<300, 20, 10> m300x20x10;

			test_activation_layer_on_device<neural_network::logistic_activation<m3x2x1>>(queue);
			test_activation_layer_on_device<neural_network::logistic_activation<m300x20x10>>(queue);
		}

		{
			test::verbose("OpenCL Tanh Activation Tests");

			typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
			typedef neural_network::algebra::metrics<300, 20, 10> m300x20x10;

			test_activation_layer_on_device<neural_network::tanh_activation<m3x2x1>>(queue);
			test_activation_layer_on_device<neural_network::tanh_activation<m300x20x10>>(queue);
		}
	}

	sc.pass();
}
