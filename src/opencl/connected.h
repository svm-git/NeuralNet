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

#pragma once

#include "layer_kernels.h"

namespace neural_network {
namespace opencl {
namespace detail {

	struct fully_connected
	{
		template <typename Input, typename Output, typename Weights>
		static void process(
			const typename Input& input,
			const typename Weights& weights,
			const typename Output& bias,
			typename Output& output,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto inputView = input.get_device_view(context);
			auto weightsView = weights.get_device_view(context);
			auto biasView = bias.get_device_view(context);
			auto outputView = output.get_device_view(context);

			layer_kernels::execute_fully_connected_kernel(
				inputView,
				weightsView,
				biasView,
				outputView,
				Output::data_size,
				Input::data_size,
				program,
				kernelName,
				queue);
		}

		template <typename Input, typename Output, typename Weights>
		static void compute_gradient(
			const typename Input& input,
			const typename Weights& weights,
			const typename Output& gradient,
			typename Input& resultGradient,
			typename Weights& weightsGradient,
			typename Output& biasGradient,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto inputView = input.get_device_view(context);
			auto weightsView = weights.get_device_view(context);
			auto gradientView = gradient.get_device_view(context);
			auto resultGradientView = resultGradient.get_device_view(context);
			auto weightsGradientView = weightsGradient.get_device_view(context);
			auto biasGradientView = biasGradient.get_device_view(context);

			layer_kernels::execute_fully_connected_gradient_kernel(
				inputView,
				weightsView,
				gradientView,
				resultGradientView,
				weightsGradientView,
				biasGradientView,
				Output::data_size,
				Input::data_size,
				program,
				kernelName,
				queue);
		}

		template <typename Weights, typename Bias>
		static void update_weights(
			const typename Weights& weightsGradient,
			typename Weights& weights,
			const typename Bias& biasGradient,
			typename Bias& bias,
			float rate,
			float regularization,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto weightsGradientView = weightsGradient.get_device_view(context);
			auto weightsView = weights.get_device_view(context);
			auto biasGradientView = biasGradient.get_device_view(context);
			auto biasView = bias.get_device_view(context);

			layer_kernels::execute_fully_connected_update_weights_kernel(
				weightsGradientView,
				weightsView,
				biasGradientView,
				biasView,
				Weights::data_size,
				Bias::data_size,
				rate,
				regularization,
				program,
				kernelName,
				queue);
		}
	};
}
}
}
