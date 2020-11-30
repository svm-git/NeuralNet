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
#include "..\src\activation.h"

void test_activation()
{
	scenario sc("Test for neural_network::*_activation classes");

	{
		test::verbose("1D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3> _3;
		typedef neural_network::relu_activation<_3> _1d_relu;

		static_assert(std::is_same<_1d_relu::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU input type.");
		static_assert(std::is_same<_1d_relu::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-RELU output type.");

		_3::tensor_type tmp;
		_1d_relu relu = neural_network::make_relu_activation_layer<_3>();

		relu.process(tmp);
		relu.compute_gradient(tmp);
	}

	{
		test::verbose("2D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3, 2> _3_x_2;
		typedef neural_network::relu_activation<_3_x_2> _2d_relu;

		static_assert(std::is_same<_2d_relu::input, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-RELU input type.");
		static_assert(std::is_same<_2d_relu::output, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-RELU output type.");

		_3_x_2::tensor_type tmp;
		_2d_relu relu = neural_network::make_relu_activation_layer<_3_x_2>();

		relu.process(tmp);
		relu.compute_gradient(tmp);
	}

	{
		test::verbose("3D RELU Activation Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> _3_x_2_x_1;
		typedef neural_network::relu_activation<_3_x_2_x_1> _3d_relu;

		static_assert(std::is_same<_3d_relu::input, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-RELU input type.");
		static_assert(std::is_same<_3d_relu::output, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-RELU output type.");

		_3_x_2_x_1::tensor_type tmp;
		_3d_relu relu = neural_network::make_relu_activation_layer<_3_x_2_x_1>();

		relu.process(tmp);
		relu.compute_gradient(tmp);
	}

	{
		test::verbose("1D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3> _3;
		typedef neural_network::logistic_activation<_3> _1d_logistic;

		static_assert(std::is_same<_1d_logistic::input, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Logistic input type.");
		static_assert(std::is_same<_1d_logistic::output, neural_network::algebra::tensor<3>>::value, "Invalid 1D-Logistic output type.");

		_3::tensor_type tmp;
		_1d_logistic logistic = neural_network::make_logistic_activation_layer<_3>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);
	}

	{
		test::verbose("2D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3, 2> _3_x_2;
		typedef neural_network::logistic_activation<_3_x_2> _2d_logistic;

		static_assert(std::is_same<_2d_logistic::input, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Logistic input type.");
		static_assert(std::is_same<_2d_logistic::output, neural_network::algebra::tensor<3, 2>>::value, "Invalid 2D-Logistic output type.");

		_3_x_2::tensor_type tmp;
		_2d_logistic logistic = neural_network::make_logistic_activation_layer<_3_x_2>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);
	}

	{
		test::verbose("3D Logistic Activation Tests");

		typedef neural_network::algebra::metrics<3, 2, 1> _3_x_2_x_1;
		typedef neural_network::logistic_activation<_3_x_2_x_1> _3d_logistic;

		static_assert(std::is_same<_3d_logistic::input, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Logistic input type.");
		static_assert(std::is_same<_3d_logistic::output, neural_network::algebra::tensor<3, 2, 1>>::value, "Invalid 3D-Logistic output type.");

		_3_x_2_x_1::tensor_type tmp;
		_3d_logistic logistic = neural_network::make_logistic_activation_layer<_3_x_2_x_1>();

		logistic.process(tmp);
		logistic.compute_gradient(tmp);
	}

	sc.pass();
}
