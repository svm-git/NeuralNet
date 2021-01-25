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

#include "unittest.h"
#include "serializationtest.h"

#include "..\src\activation.h"

template <typename Tensor>
void check_tensors_3d(
	const Tensor& expected,
	const Tensor& actual)
{
	for (size_t x = 0; x < expected.size<0>(); ++x)
	{
		for (size_t y = 0; y < expected.size<1>(); ++y)
		{
			for (size_t z = 0; z < expected.size<2>(); ++z)
			{
				test::check_true(expected(x, y, z) == actual(x, y, z), "Unexpected mismatch between C++ and OpenCL results.");
			}
		}
	}
}

::boost::compute::device find_device(
	int deviceType)
{
	for (auto device : ::boost::compute::system::devices())
	{
		if (device.type() & deviceType)
			return device;
	}

	return ::boost::compute::system::default_device();
}

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

	{
		test::verbose("1D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3> m3;
		typedef neural_network::relu_activation<m3> relu_1d;

		static_assert(std::is_same<relu_1d::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU input type.");
		static_assert(std::is_same<relu_1d::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU output type.");

		m3::tensor_type tmp;
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

		m3x2::tensor_type tmp;
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

		m3x2x1::tensor_type tmp;
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

		m3::tensor_type tmp;
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

		m3x2::tensor_type tmp;
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

		m3x2x1::tensor_type tmp;
		logistic_3d logistic = neural_network::make_logistic_activation_layer<m3x2x1>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);

		test_layer_serialization("3D Logistic Activation Layer Serialization Tests", logistic);
	}

	{
		auto device = find_device(::boost::compute::device::cpu);
		::boost::compute::context context(device);
		::boost::compute::command_queue queue(context, device);

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
	}

	sc.pass();
}
