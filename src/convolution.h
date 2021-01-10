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

#pragma once

#include "layer.h"
#include "core.h"
#include "serialization.h"

namespace neural_network {

	template <class _Kernels, class _Bias>
	struct _convolution_kernels
	{
		typedef typename _convolution_kernels<_Kernels, _Bias> _Self;
		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::convolution_layer,
			serialization::composite_serializer<
				serialization::tensor_serializer<_Kernels>,
				serialization::tensor_serializer<_Bias>
			>
		> _serializer_impl;

		_convolution_kernels()
			: m_kernels(), m_bias()
		{}

		_convolution_kernels(
			std::function<double()> initializer)
			: m_kernels(initializer), m_bias(initializer)
		{
		}

		struct serializer
		{
			typedef _Self value;

			enum : size_t { serialized_data_size = _serializer_impl::serialized_data_size };

			static void read(
				std::istream& in,
				value& layer)
			{
				_serializer_impl::read(in, layer.m_kernels, layer.m_bias);
			}

			static void write(
				std::ostream& out,
				const value& layer)
			{
				_serializer_impl::write(out, layer.m_kernels, layer.m_bias);
			}
		};
	
		_Kernels m_kernels;
		_Bias m_bias;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	struct _1d_convolution_impl
	{
		static_assert(_Metrics::rank == 1, "Invalid metric rank for 1D convolution.");

		typedef typename _1d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;
		typedef typename _convolution_kernels<kernel_weights, bias> _Weights;
		typedef typename _Weights::serializer serializer;

		_1d_convolution_impl()
			: m_weights()
		{}

		_1d_convolution_impl(
			std::function<double()> initializer)
			: m_weights(initializer)
		{
		}

		void process(
			const input& input,
			output& result)
		{
			for (size_t kernel = 0; kernel < result.size<0>(); ++kernel)
			{
				for (size_t stride = 0; stride < result.size<1>(); ++stride)
				{
					double sum = 0.0;

					const size_t baseX = stride * algebra::detail::dimension<_Stride, 0>::size;

					for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
					{
						sum += m_weights.m_kernels(kernel, x) * input(baseX + x);
					}

					result(kernel, stride) = sum + m_weights.m_bias(kernel);
				}
			}
		}

		void compute_gradient(
			const input& in,
			const output& grad,
			input& result,
			kernel_weights& kernelGradient,
			bias& biasGradient)
		{
			result.fill(0.0);
			kernelGradient.fill(0.0);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				double sum = 0.0;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					double g = grad(kernel, x);
					sum += g;

					const size_t baseX = x * algebra::detail::dimension<_Stride, 0>::size;

					for (size_t i = 0; i < algebra::detail::dimension<_Core, 0>::size; ++i)
					{
						result(baseX + i) += g * m_weights.m_kernels(kernel, i);
						kernelGradient(kernel, i) += g * in(baseX + i);
					}
				}

				biasGradient(kernel) = sum;
			}
		}

		void update_weights(
			const kernel_weights& kernelGradient,
			const bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_weights.m_kernels.size<0>(); ++kernel)
			{
				for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
				{
					m_weights.m_kernels(kernel, x) += kernelGradient(kernel, x) * rate;
				}

				m_weights.m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

		_Weights m_weights;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	struct _2d_convolution_impl
	{
		static_assert(_Metrics::rank == 2, "Invalid metric rank for 2D convolution.");

		typedef typename _2d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;
		typedef typename _convolution_kernels<kernel_weights, bias> _Weights;
		typedef typename _Weights::serializer serializer;

		_2d_convolution_impl()
			: m_weights()
		{}

		_2d_convolution_impl(
			std::function<double()> initializer)
			: m_weights(initializer)
		{
		}

		void process(
			const input& input,
			output& result)
		{
			for (size_t kernel = 0; kernel < result.size<0>(); ++kernel)
			{
				for (size_t strideX = 0; strideX < result.size<1>(); ++strideX)
				{
					for (size_t strideY = 0; strideY < result.size<2>(); ++strideY)
					{
						double sum = 0.0;

						const size_t baseX = strideX * algebra::detail::dimension<_Stride, 0>::size;
						const size_t baseY = strideY * algebra::detail::dimension<_Stride, 1>::size;

						for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
						{
							for (size_t y = 0; y < m_weights.m_kernels.size<2>(); ++y)
							{
								sum += m_weights.m_kernels(kernel, x, y) * input(baseX + x, baseY + y);
							}
						}

						result(kernel, strideX, strideY) = sum + m_weights.m_bias(kernel);
					}
				}
			}
		}

		void compute_gradient(
			const input& in,
			const output& grad,
			input& result,
			kernel_weights& kernelGradient,
			bias& biasGradient)
		{
			result.fill(0.0);
			kernelGradient.fill(0.0);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				double sum = 0.0;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					for (size_t y = 0; y < grad.size<2>(); ++y)
					{
						double g = grad(kernel, x, y);
						sum += g;

						const size_t baseX = x * algebra::detail::dimension<_Stride, 0>::size;
						const size_t baseY = y * algebra::detail::dimension<_Stride, 1>::size;

						for (size_t i = 0; i < algebra::detail::dimension<_Core, 0>::size; ++i)
						{
							for (size_t j = 0; j < algebra::detail::dimension<_Core, 1>::size; ++j)
							{
								result(baseX + i, baseY + j) += g * m_weights.m_kernels(kernel, i, j);
								kernelGradient(kernel, i, j) += g * in(baseX + i, baseY + j);
							}
						}
					}
				}

				biasGradient(kernel) = sum;
			}
		}

		void update_weights(
			const kernel_weights& kernelGradient,
			bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_weights.m_bias.size<0>(); ++kernel)
			{
				for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
				{
					for (size_t y = 0; y < m_weights.m_kernels.size<2>(); ++y)
					{
						m_weights.m_kernels(kernel, x, y) += kernelGradient(kernel, x, y) * rate;
					}
				}

				m_weights.m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

		_Weights m_weights;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	struct _3d_convolution_impl
	{
		static_assert(_Metrics::rank == 3, "Invalid metric rank for 3D convolution.");

		typedef typename _3d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;
		typedef typename _convolution_kernels<kernel_weights, bias> _Weights;
		typedef typename _Weights::serializer serializer;

		_3d_convolution_impl()
			: m_weights()
		{}

		_3d_convolution_impl(
			std::function<double()> initializer)
			: m_weights(initializer)
		{
		}

		void process(
			const input& input,
			output& result)
		{
			for (size_t kernel = 0; kernel < result.size<0>(); ++kernel)
			{
				for (size_t strideX = 0; strideX < result.size<1>(); ++strideX)
				{
					for (size_t strideY = 0; strideY < result.size<2>(); ++strideY)
					{
						for (size_t strideZ = 0; strideZ < result.size<3>(); ++strideZ)
						{
							double sum = 0.0;

							const size_t baseX = strideX * algebra::detail::dimension<_Stride, 0>::size;
							const size_t baseY = strideY * algebra::detail::dimension<_Stride, 1>::size;
							const size_t baseZ = strideZ * algebra::detail::dimension<_Stride, 2>::size;

							for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
							{
								for (size_t y = 0; y < m_weights.m_kernels.size<2>(); ++y)
								{
									for (size_t z = 0; z < m_weights.m_kernels.size<3>(); ++z)
									{
										sum += m_weights.m_kernels(kernel, x, y, z) * input(baseX + x, baseY + y, baseZ + z);
									}
								}
							}

							result(kernel, strideX, strideY, strideZ) = sum + m_weights.m_bias(kernel);
						}
					}
				}
			}
		}

		void compute_gradient(
			const input& in,
			const output& grad,
			input& result,
			kernel_weights& kernelGradient,
			bias& biasGradient)
		{
			result.fill(0.0);
			kernelGradient.fill(0.0);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				double sum = 0.0;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					for (size_t y = 0; y < grad.size<2>(); ++y)
					{
						for (size_t z = 0; z < grad.size<3>(); ++z)
						{
							double g = grad(kernel, x, y, z);
							sum += g;

							const size_t baseX = x * algebra::detail::dimension<_Stride, 0>::size;
							const size_t baseY = y * algebra::detail::dimension<_Stride, 1>::size;
							const size_t baseZ = z * algebra::detail::dimension<_Stride, 2>::size;

							for (size_t i = 0; i < algebra::detail::dimension<_Core, 0>::size; ++i)
							{
								for (size_t j = 0; j < algebra::detail::dimension<_Core, 1>::size; ++j)
								{
									for (size_t k = 0; k < algebra::detail::dimension<_Core, 2>::size; ++k)
									{
										result(baseX + i, baseY + j, baseZ + k) += g * m_weights.m_kernels(kernel, i, j, k);
										kernelGradient(kernel, i, j, k) += g * in(baseX + i, baseY + j, baseZ + k);
									}
								}
							}
						}
					}
				}

				biasGradient(kernel) = sum;
			}
		}

		void update_weights(
			const kernel_weights& kernelGradient,
			const bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_weights.m_bias.size<0>(); ++kernel)
			{
				for (size_t x = 0; x < m_weights.m_kernels.size<1>(); ++x)
				{
					for (size_t y = 0; y < m_weights.m_kernels.size<2>(); ++y)
					{
						for (size_t z = 0; z < m_weights.m_kernels.size<3>(); ++z)
						{
							m_weights.m_kernels(kernel, x, y, z) += kernelGradient(kernel, x, y, z) * rate;
						}
					}
				}

				m_weights.m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

		_Weights m_weights;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	struct _convolution_impl
	{
		static_assert(1 <= _Metrics::rank == 1 && _Metrics::rank <= 3, "Convolution is supported only for 1D, 2D or 3D tensors.");

		typedef typename std::conditional<
			_Metrics::rank == 1,
			_1d_convolution_impl<_Metrics, _Core, _Stride, _Kernels>,
			typename std::conditional<
				_Metrics::rank == 2,
				_2d_convolution_impl<_Metrics, _Core, _Stride, _Kernels>,
				_3d_convolution_impl<_Metrics, _Core, _Stride, _Kernels>
			>::type
		>::type type;
	};

	template <class _InputMetrics, class _Core, class _Stride, const size_t _Kernels>
	class convolution 
		: public layer_base<
			_InputMetrics,
			typename _convolution_impl<_InputMetrics, _Core, _Stride, _Kernels>::type::output::metrics>
	{
	public:
		typedef typename convolution<_InputMetrics, _Core, _Stride, _Kernels> _Self;
		typedef typename _convolution_impl<_InputMetrics, _Core, _Stride, _Kernels>::type impl;
		typedef typename impl::serializer _serializer_impl;

		typedef typename layer_base<_InputMetrics, typename impl::output::metrics> _Base;

		convolution()
			: _Base(), m_impl(), m_input(), m_biasGradient()
		{}

		convolution(
			std::function<double()> initializer)
			: _Base(), m_impl(initializer), m_input(), m_biasGradient()
		{
		}

		const output& process(const input& input)
		{
			m_input = input;
			m_impl.process(input, m_output);
			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			m_impl.compute_gradient(
				m_input,
				grad,
				m_gradient,
				m_kernelGradient,
				m_biasGradient);

			return m_gradient;
		}

		void update_weights(
			const double rate)
		{
			m_impl.update_weights(
				m_kernelGradient,
				m_biasGradient,
				rate);
		}

		struct serializer
		{
			typedef _Self value;

			enum : size_t { serialized_data_size = _serializer_impl::serialized_data_size };

			static void read(
				std::istream& in,
				value& layer)
			{
				_serializer_impl::read(in, layer.m_impl.m_weights);
			}

			static void write(
				std::ostream& out,
				const value& layer)
			{
				_serializer_impl::write(out, layer.m_impl.m_weights);
			}
		};

	private:
		impl m_impl;
		input m_input;
		typename impl::bias m_biasGradient;
		typename impl::kernel_weights m_kernelGradient;
	};

	template <class _Input, class _Core, class _Stride, const size_t _Kernels, class... _Args>
	convolution<_Input, _Core, _Stride, _Kernels> make_convolution_layer(
		_Args&&... args)
	{
		typedef convolution<_Input, _Core, _Stride, _Kernels> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}
}
