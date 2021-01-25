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

#pragma warning (push)
#pragma warning (disable: 4512)

#include <boost/compute/container/mapped_view.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/utility/source.hpp>

#pragma warning (pop)

namespace neural_network {
namespace opencl {
namespace detail {

	struct layer_kernels
	{
		enum { block_size = 512 };

		static const size_t get_block_count(
			const size_t size)
		{
			return ((size + block_size - 1) / block_size);
		}

		static ::boost::compute::program make_program(
			const ::boost::compute::context& context)
		{
			auto cache = ::boost::compute::program_cache::get_global_cache(context);
			std::string cacheKey = "neural_net_layer_kernels";
			::boost::optional<::boost::compute::program> program = cache->get(cacheKey);

			if (!program)
			{
				std::string source = BOOST_COMPUTE_STRINGIZE_SOURCE(

					__kernel void neural_net_relu_kernel(
						__global const float * vIn,
						__global float * vOut,
						int length)
					{
						int pos = get_global_id(0) * BLOCK_SIZE;
						int end = pos + BLOCK_SIZE;
						if (length < end)
						{
							end = length;
						}

						for (; pos < end; ++pos)
						{
							float val = vIn[pos];
							vOut[pos] = (val > 0.0f) ? val : 0.0f;
						}
					}
			
					__kernel void neural_net_relu_gradient_kernel(
						__global const float * vIn,
						__global const float * vGrad,
						__global float * vOut,
						int length)
					{
						int pos = get_global_id(0) * BLOCK_SIZE;
						int end = pos + BLOCK_SIZE;
						if (length < end)
						{
							end = length;
						}

						for (; pos < end; ++pos)
						{
							float val = vIn[pos];
							vOut[pos] = (val > 0.0f) ? vGrad[pos] : 0.0f;
						}
					}

					__kernel void neural_net_logistic_kernel(
						__global const float * vIn,
						__global float * vOut,
						int length)
					{
						int pos = get_global_id(0) * BLOCK_SIZE;
						int end = pos + BLOCK_SIZE;
						if (length < end)
						{
							end = length;
						}

						for (; pos < end; ++pos)
						{
							float x = vIn[pos];
							if (x > 0.0f)
							{
								vOut[pos] = 1.0f / (1.0f + exp(-x));
							}
							else
							{
								x = exp(x);
								vOut[pos] = x / (1.0f + x);
							}
						}
					}

					__kernel void neural_net_logistic_gradient_kernel(
						__global const float * vIn,
						__global const float * vGrad,
						__global float * vOut,
						int length)
					{
						int pos = get_global_id(0) * BLOCK_SIZE;
						int end = pos + BLOCK_SIZE;
						if (length < end)
						{
							end = length;
						}

						for (; pos < end; ++pos)
						{
							float x = vIn[pos];
							if (x > 0.0f)
							{
								x = 1.0f / (1.0f + exp(-x));
							}
							else
							{
								x = exp(x);
								x = x / (1.0f + x);
							}

							vOut[pos] = vGrad[pos] * x * (1.0f - x);
						}
					}
				);

				std::stringstream options;
				options << "-DBLOCK_SIZE=" << block_size;

				program = ::boost::compute::program::build_with_source(source, context, options.str());

				cache->insert(cacheKey, *program);
			}

			return *program;
		}

		static void execute_activation_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t dataSize,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, resultView.get_buffer());
			kernel.set_arg(2, static_cast<int>(dataSize));

			queue.enqueue_1d_range_kernel(
				kernel,
				0,
				get_block_count(dataSize),
				0);

			queue.finish();
		}

		static void execute_activation_gradient_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& gradientView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t dataSize,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, gradientView.get_buffer());
			kernel.set_arg(2, resultView.get_buffer());
			kernel.set_arg(3, static_cast<int>(dataSize));

			queue.enqueue_1d_range_kernel(
				kernel,
				0,
				get_block_count(dataSize),
				0);

			queue.finish();
		}

		static inline std::string get_relu_kernel_name()
		{
			return "neural_net_relu_kernel";
		}

		static inline std::string get_relu_gradient_kernel_name()
		{
			return "neural_net_relu_gradient_kernel";
		}

		static inline std::string get_logistic_kernel_name()
		{
			return "neural_net_logistic_kernel";
		}

		static inline std::string get_logistic_gradient_kernel_name()
		{
			return "neural_net_logistic_gradient_kernel";
		}
	};

}
}
}
