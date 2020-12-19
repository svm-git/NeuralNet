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

namespace neural_network {

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	class _1d_convolution_impl
	{
	public:
		static_assert(_Metrics::rank == 1, "Invalid metric rank for 1D convolution.");

		typedef typename _1d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;

		_1d_convolution_impl()
			: m_kernels(), m_bias()
		{}

		_1d_convolution_impl(
			std::function<double()> initializer)
			: m_kernels(initializer), m_bias(initializer)
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

					const size_t baseX = stride * algebra::_dimension<_Stride, 0>::size;

					for (size_t x = 0; x < m_kernels.size<1>(); ++x)
					{
						sum += m_kernels(kernel, x) * input(baseX + x);
					}

					result(kernel, stride) = sum + m_bias(kernel);
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result,
			bias& biasGradient)
		{
			result.fill(0.0);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				double sum = 0.0;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					double g = grad(kernel, x);
					sum += g;

					const size_t baseX = x * algebra::_dimension<_Stride, 0>::size;

					for (size_t i = 0; i < algebra::_dimension<_Core, 0>::size; ++i)
					{
						result(baseX + i) += g * m_kernels(kernel, i);
					}
				}

				biasGradient(kernel) = sum;
			}
		}

		void update_weights(
			const input& in,
			const input& gradient,
			const bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_bias.size<0>(); ++kernel)
			{
				for (size_t stride = 0; stride < algebra::_dimension<_convolution, 0>::size; ++stride)
				{
					const size_t baseX = stride * algebra::_dimension<_Stride, 0>::size;

					for (size_t x = 0; x < m_kernels.size<1>(); ++x)
					{
						m_kernels(kernel, x) += in(baseX + x) * gradient(baseX + x) * rate;
					}
				}

				m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

	private:
		kernel_weights m_kernels;
		bias m_bias;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	class _2d_convolution_impl
	{
	public:
		static_assert(_Metrics::rank == 2, "Invalid metric rank for 2D convolution.");

		typedef typename _2d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;

		_2d_convolution_impl()
			: m_kernels(), m_bias()
		{}

		_2d_convolution_impl(
			std::function<double()> initializer)
			: m_kernels(initializer), m_bias(initializer)
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

						const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
						const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;

						for (size_t x = 0; x < m_kernels.size<1>(); ++x)
						{
							for (size_t y = 0; y < m_kernels.size<2>(); ++y)
							{
								sum += m_kernels(kernel, x, y) * input(baseX + x, baseY + y);
							}
						}

						result(kernel, strideX, strideY) = sum + m_bias(kernel);
					}
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result,
			bias& biasGradient)
		{
			result.fill(0.0);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				double sum = 0.0;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					for (size_t y = 0; y < grad.size<2>(); ++y)
					{
						double g = grad(kernel, x, y);
						sum += g;

						const size_t baseX = x * algebra::_dimension<_Stride, 0>::size;
						const size_t baseY = y * algebra::_dimension<_Stride, 1>::size;

						for (size_t i = 0; i < algebra::_dimension<_Core, 0>::size; ++i)
						{
							for (size_t j = 0; j < algebra::_dimension<_Core, 1>::size; ++j)
							{
								result(baseX + i, baseY + j) += g * m_kernels(kernel, i, j);
							}
						}
					}
				}

				biasGradient(kernel) = sum;
			}
		}

		void update_weights(
			const input& in,
			const input& gradient,
			bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_bias.size<0>(); ++kernel)
			{
				for (size_t strideX = 0; strideX < algebra::_dimension<_convolution, 0>::size; ++strideX)
				{
					for (size_t strideY = 0; strideY < algebra::_dimension<_convolution, 1>::size; ++strideY)
					{
						const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
						const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;

						for (size_t x = 0; x < m_kernels.size<1>(); ++x)
						{ 
							for (size_t y = 0; y < m_kernels.size<2>(); ++y)
							{
								m_kernels(kernel, x, y) += in(baseX + x, baseY + y) * gradient(baseX + x, baseY + y) * rate;
							}
						}
					}
				}

				m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

	private:
		kernel_weights m_kernels;
		bias m_bias;
	};

	template <class _Metrics, class _Core, class _Stride, const size_t _Kernels>
	class _3d_convolution_impl
	{
	public:
		static_assert(_Metrics::rank == 3, "Invalid metric rank for 3D convolution.");

		typedef typename _3d_convolution_impl<_Metrics, _Core, _Stride, _Kernels> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics _convolution;
		typedef typename _convolution::template expand<_Kernels>::type::tensor_type output;
		typedef typename _Core::template expand<_Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<_Kernels>::tensor_type bias;

		_3d_convolution_impl()
			: m_kernels(), m_bias()
		{}

		_3d_convolution_impl(
			std::function<double()> initializer)
			: m_kernels(initializer), m_bias(initializer)
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

							const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
							const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;
							const size_t baseZ = strideZ * algebra::_dimension<_Stride, 2>::size;

							for (size_t x = 0; x < m_kernels.size<1>(); ++x)
							{
								for (size_t y = 0; y < m_kernels.size<2>(); ++y)
								{
									for (size_t z = 0; z < m_kernels.size<3>(); ++z)
									{
										sum += m_kernels(kernel, x, y, z) * input(baseX + x, baseY + y, baseZ + z);
									}
								}
							}

							result(kernel, strideX, strideY, strideZ) = sum + m_bias(kernel);
						}
					}
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result,
			bias& biasGradient)
		{
			result.fill(0.0);

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

							const size_t baseX = x * algebra::_dimension<_Stride, 0>::size;
							const size_t baseY = y * algebra::_dimension<_Stride, 1>::size;
							const size_t baseZ = z * algebra::_dimension<_Stride, 2>::size;

							for (size_t i = 0; i < algebra::_dimension<_Core, 0>::size; ++i)
							{
								for (size_t j = 0; j < algebra::_dimension<_Core, 1>::size; ++j)
								{
									for (size_t k = 0; k < algebra::_dimension<_Core, 2>::size; ++k)
									{
										result(baseX + i, baseY + j, baseZ + k) += g * m_kernels(kernel, i, j, k);
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
			const input& in,
			const input& gradient,
			const bias& biasGradient,
			const double rate)
		{
			for (size_t kernel = 0; kernel < m_bias.size<0>(); ++kernel)
			{
				for (size_t strideX = 0; strideX < algebra::_dimension<_convolution, 0>::size; ++strideX)
				{
					for (size_t strideY = 0; strideY < algebra::_dimension<_convolution, 1>::size; ++strideY)
					{
						for (size_t strideZ = 0; strideZ < algebra::_dimension<_convolution, 2>::size; ++strideZ)
						{
							const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
							const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;
							const size_t baseZ = strideZ * algebra::_dimension<_Stride, 2>::size;

							for (size_t x = 0; x < m_kernels.size<1>(); ++x)
							{
								for (size_t y = 0; y < m_kernels.size<2>(); ++y)
								{
									for (size_t z = 0; z < m_kernels.size<3>(); ++z)
									{
										m_kernels(kernel, x, y, z) += in(baseX + x, baseY + y, baseZ + z) * gradient(baseX + x, baseY + y, baseZ + z) * rate;
									}
								}
							}
						}
					}
				}

				m_bias(kernel) += biasGradient(kernel) * rate;
			}
		}

	private:
		kernel_weights m_kernels;
		bias m_bias;
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

		typedef typename layer_base<_InputMetrics, typename impl::output::metrics> _Base;
		typedef typename impl::bias _Bias;

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
				grad,
				m_gradient,
				m_biasGradient);

			return m_gradient;
		}

		void update_weights(
			const double rate)
		{
			m_impl.update_weights(
				m_input,
				m_gradient,
				m_biasGradient,
				rate);
		}

	private:
		impl m_impl;
		input m_input;
		typename impl::bias m_biasGradient;
	};

	template <class _Input, class _Core, class _Stride, const size_t _Kernels, class... _Args>
	convolution<_Input, _Core, _Stride, _Kernels> make_convolution_layer(
		_Args&&... args)
	{
		typedef convolution<_Input, _Core, _Stride, _Kernels> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}
}
