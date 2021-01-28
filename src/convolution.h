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

#include "layer.h"
#include "core.h"
#include "serialization.h"

#ifdef NEURAL_NET_ENABLE_OPEN_CL

#include "opencl/layer_kernels.h"

#endif

namespace neural_network {

namespace detail {

	template <class Kernels, class Bias>
	struct convolution_kernels
	{
		typedef typename convolution_kernels<Kernels, Bias> this_type;
		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::convolution_layer,
			serialization::composite_serializer<
				serialization::tensor_serializer<Kernels>,
				serialization::tensor_serializer<Bias>
			>
		> serializer_impl_type;

		static_assert(
			std::is_same<typename Kernels::number_type, typename Bias::number_type>::value,
			"Kernel and bias tensor value types do not match.");

		typedef typename Kernels::number_type number_type;

		convolution_kernels()
			: m_kernels(), m_bias()
		{}

		convolution_kernels(
			std::function<number_type()> initializer)
			: m_kernels(initializer), m_bias(initializer)
		{
		}

		struct serializer
		{
			typedef this_type value_type;

			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value_type& layer)
			{
				serializer_impl_type::read(in, layer.m_kernels, layer.m_bias);
			}

			static void write(
				std::ostream& out,
				const value_type& layer)
			{
				serializer_impl_type::write(out, layer.m_kernels, layer.m_bias);
			}
		};
	
		Kernels m_kernels;
		Bias m_bias;
	};

	template <class Metrics, class Core, class Stride, const size_t Kernels>
	struct convolution_1d
	{
		static_assert(Metrics::rank == 1, "Invalid metric rank for 1D convolution.");

		typedef typename convolution_1d<Metrics, Core, Stride, Kernels> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics convolution_metrics;
		typedef typename convolution_metrics::template expand<Kernels>::type::tensor_type output;
		typedef typename Core::template expand<Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<Kernels>::tensor_type bias;
		typedef typename convolution_kernels<kernel_weights, bias> weights_type;
		typedef typename weights_type::serializer serializer;
		typedef typename weights_type::number_type number_type;

		convolution_1d()
			: m_weights()
		{}

		convolution_1d(
			std::function<number_type()> initializer)
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
					number_type sum = 0.0f;

					const size_t baseX = stride * algebra::detail::dimension<Stride, 0>::size;

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
			result.fill(0.0f);
			kernelGradient.fill(0.0f);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				number_type sum = 0.0f;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					number_type g = grad(kernel, x);
					sum += g;

					const size_t baseX = x * algebra::detail::dimension<Stride, 0>::size;

					for (size_t i = 0; i < algebra::detail::dimension<Core, 0>::size; ++i)
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
			const number_type rate)
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

		weights_type m_weights;
	};

	template <class Metrics, class Core, class Stride, const size_t Kernels>
	struct convolution_2d
	{
		static_assert(Metrics::rank == 2, "Invalid metric rank for 2D convolution.");

		typedef typename convolution_2d<Metrics, Core, Stride, Kernels> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics convolution_metrics;
		typedef typename convolution_metrics::template expand<Kernels>::type::tensor_type output;
		typedef typename Core::template expand<Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<Kernels>::tensor_type bias;
		typedef typename convolution_kernels<kernel_weights, bias> weights_type;
		typedef typename weights_type::serializer serializer;
		typedef typename weights_type::number_type number_type;

		convolution_2d()
			: m_weights()
		{}

		convolution_2d(
			std::function<number_type()> initializer)
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
						number_type sum = 0.0f;

						const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
						const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;

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
			result.fill(0.0f);
			kernelGradient.fill(0.0f);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				number_type sum = 0.0f;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					for (size_t y = 0; y < grad.size<2>(); ++y)
					{
						number_type g = grad(kernel, x, y);
						sum += g;

						const size_t baseX = x * algebra::detail::dimension<Stride, 0>::size;
						const size_t baseY = y * algebra::detail::dimension<Stride, 1>::size;

						for (size_t i = 0; i < algebra::detail::dimension<Core, 0>::size; ++i)
						{
							for (size_t j = 0; j < algebra::detail::dimension<Core, 1>::size; ++j)
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
			const number_type rate)
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

		weights_type m_weights;
	};

	template <class Metrics, class Core, class Stride, const size_t Kernels>
	struct convolution_3d
	{
		static_assert(Metrics::rank == 3, "Invalid metric rank for 3D convolution.");

		typedef typename convolution_3d<Metrics, Core, Stride, Kernels> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics convolution_metrics;
		typedef typename convolution_metrics::template expand<Kernels>::type::tensor_type output;
		typedef typename Core::template expand<Kernels>::type::tensor_type kernel_weights;
		typedef typename algebra::metrics<Kernels>::tensor_type bias;
		typedef typename convolution_kernels<kernel_weights, bias> weights_type;
		typedef typename weights_type::serializer serializer;
		typedef typename weights_type::number_type number_type;

		convolution_3d()
			: m_weights()
		{}

		convolution_3d(
			std::function<number_type()> initializer)
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
							number_type sum = 0.0f;

							const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
							const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;
							const size_t baseZ = strideZ * algebra::detail::dimension<Stride, 2>::size;

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
			result.fill(0.0f);
			kernelGradient.fill(0.0f);

			for (size_t kernel = 0; kernel < grad.size<0>(); ++kernel)
			{
				number_type sum = 0.0f;

				for (size_t x = 0; x < grad.size<1>(); ++x)
				{
					for (size_t y = 0; y < grad.size<2>(); ++y)
					{
						for (size_t z = 0; z < grad.size<3>(); ++z)
						{
							number_type g = grad(kernel, x, y, z);
							sum += g;

							const size_t baseX = x * algebra::detail::dimension<Stride, 0>::size;
							const size_t baseY = y * algebra::detail::dimension<Stride, 1>::size;
							const size_t baseZ = z * algebra::detail::dimension<Stride, 2>::size;

							for (size_t i = 0; i < algebra::detail::dimension<Core, 0>::size; ++i)
							{
								for (size_t j = 0; j < algebra::detail::dimension<Core, 1>::size; ++j)
								{
									for (size_t k = 0; k < algebra::detail::dimension<Core, 2>::size; ++k)
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
			const number_type rate)
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

		weights_type m_weights;
	};

	template <class Metrics, class Core, class Stride, const size_t Kernels>
	struct convolution_impl
	{
		static_assert(1 <= Metrics::rank == 1 && Metrics::rank <= 3, "Convolution is supported only for 1D, 2D or 3D tensors.");

		typedef typename std::conditional<
			Metrics::rank == 1,
			convolution_1d<Metrics, Core, Stride, Kernels>,
			typename std::conditional<
				Metrics::rank == 2,
				convolution_2d<Metrics, Core, Stride, Kernels>,
				convolution_3d<Metrics, Core, Stride, Kernels>
			>::type
		>::type type;
	};

}

template <class InputMetrics, class Core, class Stride, const size_t Kernels>
	class convolution 
		: public layer_base<
			InputMetrics,
			typename detail::convolution_impl<InputMetrics, Core, Stride, Kernels>::type::output::metrics>
	{
	public:
		typedef typename convolution<InputMetrics, Core, Stride, Kernels> this_type;
		typedef typename detail::convolution_impl<InputMetrics, Core, Stride, Kernels>::type impl;
		typedef typename impl::serializer serializer_impl_type;

		typedef typename layer_base<InputMetrics, typename impl::output::metrics> base_type;

		convolution()
			: base_type(), m_impl(), m_input(), m_biasGradient()
		{}

		convolution(
			std::function<number_type()> initializer)
			: base_type(), m_impl(initializer), m_input(), m_biasGradient()
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
			const number_type rate)
		{
			m_impl.update_weights(
				m_kernelGradient,
				m_biasGradient,
				rate);
		}

		struct serializer
		{
			typedef this_type value;

			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value& layer)
			{
				serializer_impl_type::read(in, layer.m_impl.m_weights);
			}

			static void write(
				std::ostream& out,
				const value& layer)
			{
				serializer_impl_type::write(out, layer.m_impl.m_weights);
			}
		};

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		const output& process(
			const input& input,
			::boost::compute::command_queue&)
		{
			return this->process(input);
		}

		const input& compute_gradient(
			const output& gradient,
			::boost::compute::command_queue&)
		{
			return this->compute_gradient(gradient);
		}

		void update_weights(
			const number_type rate,
			::boost::compute::command_queue&)
		{
			this->update_weights(rate);
		}

#endif

	private:
		impl m_impl;
		input m_input;
		typename impl::bias m_biasGradient;
		typename impl::kernel_weights m_kernelGradient;
	};

	template <class Input, class Core, class Stride, const size_t Kernels, class... Args>
	convolution<Input, Core, Stride, Kernels> make_convolution_layer(
		Args&&... args)
	{
		typedef convolution<Input, Core, Stride, Kernels> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}
}
