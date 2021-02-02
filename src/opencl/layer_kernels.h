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
		enum 
		{ 
			block_size = 128,
			min_matrix_size = 1024,
			min_pooling_size = 32
		};

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
						__global const float * vOut,
						__global const float * vGrad,
						__global float * vRes,
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
							vRes[pos] = (vOut[pos] > 0.0f) ? vGrad[pos] : 0.0f;
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
						__global const float * vOut,
						__global const float * vGrad,
						__global float * vRes,
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
							float f = vOut[pos];
							vRes[pos] = vGrad[pos] * f * (1.0f - f);
						}
					}

					__kernel void neural_net_tanh_kernel(
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
							vOut[pos] = tanh(vIn[pos]);
						}
					}

					__kernel void neural_net_tanh_gradient_kernel(
						__global const float * vOut,
						__global const float * vGrad,
						__global float * vRes,
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
							float f = vOut[pos];
							vRes[pos] = vGrad[pos] * (1.0f - f * f);
						}
					}

					__kernel void neural_net_fully_connected_kernel(
						__global const float * vIn,
						__global const float * mWeights,
						__global const float * vBias,
						__global float * vResult,
						int rows,
						int cols)
					{
						int row = get_global_id(0);
						int off = row * cols;

						float sum = 0.0f;
						for (int col = 0; col < cols; ++col)
						{
							sum += mWeights[off + col] * vIn[col];
						}

						vResult[row] = sum + vBias[row];
					}

					__kernel void neural_net_fully_connected_gradient_kernel(
						__global const float * vIn,
						__global const float * mWeights,
						__global const float * vGradient,
						__global float * vResult,
						__global float * mWeightsGradient,
						__global float * vBiasGradient,
						int rows,
						int cols)
					{
						int col = get_global_id(0);
						int off = col;

						float sum = 0.0f;
						float inVal = vIn[col];

						for (int row = 0; row < rows; ++row)
						{
							float gVal = vGradient[row];

							sum += mWeights[off] * gVal;
							mWeightsGradient[off] = inVal * gVal;
							off += cols;

							if (col == 0)
							{
								vBiasGradient[row] = gVal;
							}
						}

						vResult[col] = sum;
					}

					__kernel void neural_net_update_weights_kernel(
						__global const float * gradient,
						__global float * weights,
						float rate,
						float regularization,
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
							weights[pos] += (gradient[pos] + regularization * weights[pos]) * rate;
						}
					}

					__kernel void neural_net_squared_error_loss_gradient_kernel(
						__global const float * vResult,
						__global const float * vTruth,
						__global float * vGradient,
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
							vGradient[pos] = vResult[pos] - vTruth[pos];
						}
					}

					__kernel void neural_net_1d_convolution_kernel(
						__global const float * vInput,
						__global const float * mKernels,
						__global const float * vBias,
						__global float * vResult,
						int kernelSizeX,
						int nStridesX,
						int strideSizeX)
					{
						int iKernel = get_global_id(0);

						int resultBase = iKernel * nStridesX;
						int kernelBase = iKernel * kernelSizeX;

						for (int strideX = 0; strideX < nStridesX; ++strideX)
						{
							float sum = 0.0f;

							int baseX = strideX * strideSizeX;

							for (int x = 0; x < kernelSizeX; ++x)
							{
								sum += mKernels[kernelBase + x] * vInput[baseX + x];
							}

							vResult[resultBase + strideX] = sum + vBias[iKernel];
						}
					}

					__kernel void neural_net_2d_convolution_kernel(
						__global const float * vInput,
						__global const float * mKernels,
						__global const float * vBias,
						__global float * vResult,
						int inputSizeY,
						int kernelSizeX,
						int kernelSizeY,
						int nStridesX,
						int nStridesY,
						int strideSizeX,
						int strideSizeY)
					{
						int iKernel = get_global_id(0);

						int resultBase = iKernel * nStridesX * nStridesY;
						int kernelBase = iKernel * kernelSizeX * kernelSizeY;

						for (int strideX = 0; strideX < nStridesX; ++strideX)
						{
							int resultBaseY = resultBase + (strideX * nStridesY);

							for (int strideY = 0; strideY < nStridesY; ++strideY)
							{
								float sum = 0.0f;

								int baseX = strideX * strideSizeX;
								int baseY = strideY * strideSizeY;

								for (int x = 0; x < kernelSizeX; ++x)
								{
									int kernelBaseY = kernelBase + (x * kernelSizeY);
									int inputBaseY = ((baseX + x) * inputSizeY) + baseY;

									for (int y = 0; y < kernelSizeY; ++y)
									{
										sum += mKernels[kernelBaseY + y] * vInput[inputBaseY + y];
									}
								}

								vResult[resultBaseY + strideY] = sum + vBias[iKernel];
							}
						}
					}

					__kernel void neural_net_3d_convolution_kernel(
						__global const float * vInput,
						__global const float * mKernels,
						__global const float * vBias,
						__global float * vResult,
						int inputSizeY,
						int inputSizeZ,
						int kernelSizeX,
						int kernelSizeY,
						int kernelSizeZ,
						int nStridesX,
						int nStridesY,
						int nStridesZ,
						int strideSizeX,
						int strideSizeY,
						int strideSizeZ)
					{
						int iKernel = get_global_id(0);

						int resultBase = iKernel * nStridesX * nStridesY * nStridesZ;
						int kernelBase = iKernel * kernelSizeX * kernelSizeY * kernelSizeZ;

						for (int strideX = 0; strideX < nStridesX; ++strideX)
						{
							int resultBaseY = resultBase + (strideX * nStridesY * nStridesZ);

							for (int strideY = 0; strideY < nStridesY; ++strideY)
							{
								int resultBaseZ = resultBaseY + (strideY * nStridesZ);

								for (int strideZ = 0; strideZ < nStridesZ; ++strideZ)
								{
									float sum = 0.0f;

									int baseX = strideX * strideSizeX;
									int baseY = strideY * strideSizeY;
									int baseZ = strideZ * strideSizeZ;

									for (int x = 0; x < kernelSizeX; ++x)
									{
										int kernelBaseY = kernelBase + (x * kernelSizeY * kernelSizeZ);
										int inputBaseY = (((baseX + x) * inputSizeY) + baseY) * inputSizeZ;

										for (int y = 0; y < kernelSizeY; ++y)
										{
											int kernelBaseZ = kernelBaseY + (y * kernelSizeZ);
											int inputBaseZ = inputBaseY + (y * inputSizeZ) + baseZ;

											for (int z = 0; z < kernelSizeZ; ++z)
											{
												sum += mKernels[kernelBaseZ + z] * vInput[inputBaseZ + z];
											}
										}
									}

									vResult[resultBaseZ + strideZ] = sum + vBias[iKernel];
								}
							}
						}
					}

					__kernel void neural_net_1d_max_pooling_kernel(
						__global const float * vInput,
						__global float * vResult,
						__global float * vMask,
						int coreSizeX,
						int strideSizeX)
					{
						int strideX = get_global_id(0);

						int baseX = strideX * strideSizeX;

						float max = vInput[baseX];
						int maxX = baseX;

						for (int x = 1; x < coreSizeX; ++x)
						{
							int curX = baseX + x;
							float e = vInput[curX];
							if (max < e)
							{
								max = e;
								maxX = curX;
							}
						}

						vResult[strideX] = max;
						vMask[maxX] = 1.0f;
					}

					__kernel void neural_net_2d_max_pooling_kernel(
						__global const float * vInput,
						__global float * vResult,
						__global float * vMask,
						int inputSizeY,
						int coreSizeX,
						int coreSizeY,
						int strideSizeX,
						int strideSizeY)
					{
						int strideX = get_global_id(0);
						int strideY = get_global_id(1);
						
						int baseX = strideX * strideSizeX;
						int baseY = strideY * strideSizeY;

						int maxPos = baseX * inputSizeY + baseY;
						float max = vInput[maxPos];

						for (int x = 0; x < coreSizeX; ++x)
						{
							int inputBaseY = (baseX + x) * inputSizeY + baseY;

							for (int y = 0; y < coreSizeY; ++y)
							{
								int inputY = inputBaseY + y;
								float e = vInput[inputY];
								if (max < e)
								{
									max = e;
									maxPos = inputY;
								}
							}
						}

						vResult[(strideX * get_global_size(1)) + strideY] = max;
						vMask[maxPos] = 1.0f;
					}

					__kernel void neural_net_3d_max_pooling_kernel(
						__global const float * vInput,
						__global float * vResult,
						__global float * vMask,
						int inputSizeY,
						int inputSizeZ,
						int coreSizeX,
						int coreSizeY,
						int coreSizeZ,
						int strideSizeX,
						int strideSizeY,
						int strideSizeZ)
					{
						int strideX = get_global_id(0);
						int strideY = get_global_id(1);
						int strideZ = get_global_id(2);

						int baseX = strideX * strideSizeX;
						int baseY = strideY * strideSizeY;
						int baseZ = strideZ * strideSizeZ;

						int maxPos = (baseX * inputSizeY + baseY) * inputSizeZ + baseZ;
						float max = vInput[maxPos];

						for (int x = 0; x < coreSizeX; ++x)
						{
							int inputBaseY = ((baseX + x) * inputSizeY + baseY) * inputSizeZ;

							for (int y = 0; y < coreSizeY; ++y)
							{
								int inputBaseZ = inputBaseY + (y * inputSizeZ) + baseZ;

								for (int z = 0; z < coreSizeZ; ++z)
								{ 
									int inputZ = inputBaseZ + z;
									float e = vInput[inputZ];
									if (max < e)
									{
										max = e;
										maxPos = inputZ;
									}
								}
							}
						}

						vResult[(((strideX * get_global_size(1)) + strideY) * get_global_size(2)) + strideZ] = max;
						vMask[maxPos] = 1.0f;
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
			::boost::compute::mapped_view<float>& outputView,
			::boost::compute::mapped_view<float>& gradientView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t dataSize,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, outputView.get_buffer());
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

		static inline std::string get_tanh_kernel_name()
		{
			return "neural_net_tanh_kernel";
		}

		static inline std::string get_tanh_gradient_kernel_name()
		{
			return "neural_net_tanh_gradient_kernel";
		}

		static void execute_fully_connected_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& weightsView,
			::boost::compute::mapped_view<float>& biasView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t rows,
			const size_t columns,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, weightsView.get_buffer());
			kernel.set_arg(2, biasView.get_buffer());
			kernel.set_arg(3, resultView.get_buffer());
			kernel.set_arg(4, static_cast<int>(rows));
			kernel.set_arg(5, static_cast<int>(columns));

			queue.enqueue_1d_range_kernel(kernel, 0, rows, 0);
			queue.finish();
		}

		static void execute_fully_connected_gradient_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& weightsView,
			::boost::compute::mapped_view<float>& gradientView,
			::boost::compute::mapped_view<float>& resultView,
			::boost::compute::mapped_view<float>& weightsGradientView,
			::boost::compute::mapped_view<float>& biasGradientView,
			const size_t rows,
			const size_t columns,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, weightsView.get_buffer());
			kernel.set_arg(2, gradientView.get_buffer());
			kernel.set_arg(3, resultView.get_buffer());
			kernel.set_arg(4, weightsGradientView.get_buffer());
			kernel.set_arg(5, biasGradientView.get_buffer());
			kernel.set_arg(6, static_cast<int>(rows));
			kernel.set_arg(7, static_cast<int>(columns));

			queue.enqueue_1d_range_kernel(kernel, 0, columns, 0);
			queue.finish();
		}

		static void execute_generic_update_weights_kernel(
			::boost::compute::mapped_view<float>& weightsGradientView,
			::boost::compute::mapped_view<float>& weightsView,
			::boost::compute::mapped_view<float>& biasGradientView,
			::boost::compute::mapped_view<float>& biasView,
			const size_t weightsLength,
			const size_t biasLenth,
			const float rate,
			const float regularization,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto weightsKernel = program.create_kernel(kernelName);

			weightsKernel.set_arg(0, weightsGradientView.get_buffer());
			weightsKernel.set_arg(1, weightsView.get_buffer());
			weightsKernel.set_arg(2, rate);
			weightsKernel.set_arg(3, regularization);
			weightsKernel.set_arg(4, static_cast<int>(weightsLength));

			queue.enqueue_1d_range_kernel(weightsKernel, 0, get_block_count(weightsLength), 0);

			auto biasKernel = program.create_kernel(kernelName);

			biasKernel.set_arg(0, biasGradientView.get_buffer());
			biasKernel.set_arg(1, biasView.get_buffer());
			biasKernel.set_arg(2, rate);
			biasKernel.set_arg(3, regularization);
			biasKernel.set_arg(4, static_cast<int>(biasLenth));

			queue.enqueue_1d_range_kernel(biasKernel, 0, get_block_count(biasLenth), 0);

			queue.finish();
		}

		static inline std::string get_fully_connected_kernel_name()
		{
			return "neural_net_fully_connected_kernel";
		}

		static inline std::string get_fully_connected_gradient_kernel_name()
		{
			return "neural_net_fully_connected_gradient_kernel";
		}

		static inline std::string get_update_weights_kernel_name()
		{
			return "neural_net_update_weights_kernel";
		}

		static void execute_squared_error_loss_gradient_kernel(
			::boost::compute::mapped_view<float>& resultView,
			::boost::compute::mapped_view<float>& truthView,
			::boost::compute::mapped_view<float>& gradientView,
			const size_t length,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, resultView.get_buffer());
			kernel.set_arg(1, truthView.get_buffer());
			kernel.set_arg(2, gradientView.get_buffer());
			kernel.set_arg(3, static_cast<int>(length));

			queue.enqueue_1d_range_kernel(kernel, 0, get_block_count(length), 0);
			queue.finish();
		}

		static inline std::string get_squared_error_loss_gradient_kernel_name()
		{
			return "neural_net_squared_error_loss_gradient_kernel";
		}

		static void execute_1d_convolution_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& kernelView,
			::boost::compute::mapped_view<float>& biasView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t kernels,
			const size_t kernelSizeX,
			const size_t strides,
			const size_t strideSizeX,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, kernelView.get_buffer());
			kernel.set_arg(2, biasView.get_buffer());
			kernel.set_arg(3, resultView.get_buffer());
			kernel.set_arg(4, static_cast<int>(kernelSizeX));
			kernel.set_arg(5, static_cast<int>(strides));
			kernel.set_arg(6, static_cast<int>(strideSizeX));

			queue.enqueue_1d_range_kernel(kernel, 0, kernels, 0);
			queue.finish();
		}

		static inline std::string get_1d_convolution_kernel_name()
		{
			return "neural_net_1d_convolution_kernel";
		}

		static void execute_2d_convolution_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& kernelView,
			::boost::compute::mapped_view<float>& biasView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t inputSizeY,
			const size_t kernels,
			const size_t kernelSizeX,
			const size_t kernelSizeY,
			const size_t stridesX,
			const size_t stridesY,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, kernelView.get_buffer());
			kernel.set_arg(2, biasView.get_buffer());
			kernel.set_arg(3, resultView.get_buffer());
			kernel.set_arg(4, static_cast<int>(inputSizeY));
			kernel.set_arg(5, static_cast<int>(kernelSizeX));
			kernel.set_arg(6, static_cast<int>(kernelSizeY));
			kernel.set_arg(7, static_cast<int>(stridesX));
			kernel.set_arg(8, static_cast<int>(stridesY));
			kernel.set_arg(9, static_cast<int>(strideSizeX));
			kernel.set_arg(10, static_cast<int>(strideSizeY));

			queue.enqueue_1d_range_kernel(kernel, 0, kernels, 0);
			queue.finish();
		}

		static inline std::string get_2d_convolution_kernel_name()
		{
			return "neural_net_2d_convolution_kernel";
		}

		static void execute_3d_convolution_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& kernelView,
			::boost::compute::mapped_view<float>& biasView,
			::boost::compute::mapped_view<float>& resultView,
			const size_t inputSizeY,
			const size_t inputSizeZ,
			const size_t kernels,
			const size_t kernelSizeX,
			const size_t kernelSizeY,
			const size_t kernelSizeZ,
			const size_t stridesX,
			const size_t stridesY,
			const size_t stridesZ,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const size_t strideSizeZ,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, kernelView.get_buffer());
			kernel.set_arg(2, biasView.get_buffer());
			kernel.set_arg(3, resultView.get_buffer());
			kernel.set_arg(4, static_cast<int>(inputSizeY));
			kernel.set_arg(5, static_cast<int>(inputSizeZ));
			kernel.set_arg(6, static_cast<int>(kernelSizeX));
			kernel.set_arg(7, static_cast<int>(kernelSizeY));
			kernel.set_arg(8, static_cast<int>(kernelSizeZ));
			kernel.set_arg(9, static_cast<int>(stridesX));
			kernel.set_arg(10, static_cast<int>(stridesY));
			kernel.set_arg(11, static_cast<int>(stridesZ));
			kernel.set_arg(12, static_cast<int>(strideSizeX));
			kernel.set_arg(13, static_cast<int>(strideSizeY));
			kernel.set_arg(14, static_cast<int>(strideSizeZ));

			queue.enqueue_1d_range_kernel(kernel, 0, kernels, 0);
			queue.finish();
		}

		static inline std::string get_3d_convolution_kernel_name()
		{
			return "neural_net_3d_convolution_kernel";
		}

		static void execute_1d_max_pooling_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& resultView,
			::boost::compute::mapped_view<float>& maskView,
			const size_t coreSizeX,
			const size_t stridesX,
			const size_t strideSizeX,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, resultView.get_buffer());
			kernel.set_arg(2, maskView.get_buffer());
			kernel.set_arg(3, static_cast<int>(coreSizeX));
			kernel.set_arg(4, static_cast<int>(strideSizeX));

			queue.enqueue_1d_range_kernel(
				kernel,
				0,
				stridesX,
				0);

			queue.finish();
		}

		static inline std::string get_1d_max_pooling_kernel_name()
		{
			return "neural_net_1d_max_pooling_kernel";
		}

		static void execute_2d_max_pooling_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& resultView,
			::boost::compute::mapped_view<float>& maskView,
			const size_t inputSizeY,
			const size_t coreSizeX,
			const size_t coreSizeY,
			const size_t stridesX,
			const size_t stridesY,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, resultView.get_buffer());
			kernel.set_arg(2, maskView.get_buffer());
			kernel.set_arg(3, static_cast<int>(inputSizeY));
			kernel.set_arg(4, static_cast<int>(coreSizeX));
			kernel.set_arg(5, static_cast<int>(coreSizeY));
			kernel.set_arg(6, static_cast<int>(strideSizeX));
			kernel.set_arg(7, static_cast<int>(strideSizeY));

			const size_t map_global_work_size[2] = { stridesX, stridesY };

			queue.enqueue_nd_range_kernel(
				kernel,
				2,
				0,
				map_global_work_size,
				nullptr);

			queue.finish();
		}

		static inline std::string get_2d_max_pooling_kernel_name()
		{
			return "neural_net_2d_max_pooling_kernel";
		}

		static void execute_3d_max_pooling_kernel(
			::boost::compute::mapped_view<float>& inputView,
			::boost::compute::mapped_view<float>& resultView,
			::boost::compute::mapped_view<float>& maskView,
			const size_t inputSizeY,
			const size_t inputSizeZ,
			const size_t coreSizeX,
			const size_t coreSizeY,
			const size_t coreSizeZ,
			const size_t stridesX,
			const size_t stridesY,
			const size_t stridesZ,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const size_t strideSizeZ,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			::boost::compute::command_queue& queue)
		{
			auto kernel = program.create_kernel(kernelName);

			kernel.set_arg(0, inputView.get_buffer());
			kernel.set_arg(1, resultView.get_buffer());
			kernel.set_arg(2, maskView.get_buffer());
			kernel.set_arg(3, static_cast<int>(inputSizeY));
			kernel.set_arg(4, static_cast<int>(inputSizeZ));
			kernel.set_arg(5, static_cast<int>(coreSizeX));
			kernel.set_arg(6, static_cast<int>(coreSizeY));
			kernel.set_arg(7, static_cast<int>(coreSizeZ));
			kernel.set_arg(8, static_cast<int>(strideSizeX));
			kernel.set_arg(9, static_cast<int>(strideSizeY));
			kernel.set_arg(10, static_cast<int>(strideSizeZ));

			const size_t map_global_work_size[3] = { stridesX, stridesY, stridesZ };

			queue.enqueue_nd_range_kernel(
				kernel,
				3,
				0,
				map_global_work_size,
				nullptr);

			queue.finish();
		}

		static inline std::string get_3d_max_pooling_kernel_name()
		{
			return "neural_net_3d_max_pooling_kernel";
		}

	};

}
}
}
