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
	
namespace detail {

	template <class Metrics>
	class scalar_max_pooling
	{
	public:
		typedef typename scalar_max_pooling<Metrics> this_type;

		typedef typename Metrics::tensor_type input;
		typedef algebra::metrics<1>::tensor_type output;

		static_assert(Metrics::rank == 1, "Invalid metric rank for scalar max pooling.");

		static_assert(
			std::is_same<typename input::number_type, typename input::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		scalar_max_pooling()
			: m_mask()
		{}
	
		void process(
			const input& input,
			output& result)
		{
			m_mask(0) = 0.0f;

			number_type max = input(0);
			size_t imax = 0;

			for (size_t i = 1; i < input.size<0>(); ++i)
			{
				m_mask(i) = 0.0f;

				auto e = input(i);
				if (max < e)
				{
					max = e;
					imax = i;
				}
			}

			m_mask(imax) = 1.0f;
			result(0) = max;
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			for (size_t i = 0; i < result.size<0>(); ++i)
			{
				result(i) = (0.0f < m_mask(i)) ? grad(0) : 0.0f;
			}
		}

	private:
		input m_mask;
	};

	template <class Metrics>
	class generic_max_pooling
	{
	public:
		typedef typename generic_max_pooling<Metrics> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename Metrics::shrink::type::tensor_type output;

		typedef typename algebra::metrics<Metrics::dimension_size, output::data_size>::tensor_type reshaped_input;
		typedef typename algebra::metrics<output::data_size>::tensor_type reshaped_output;

		static_assert(2 <= Metrics::rank, "Metric rank is too small for generic max pooling.");

		static_assert(
			std::is_same<typename input::number_type, typename input::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		generic_max_pooling()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			reshaped_input rin = input.reshape<reshaped_input::metrics>();
			reshaped_output rout = result.reshape<reshaped_output::metrics>();

			for (size_t j = 0; j < rin.size<1>(); ++j)
			{
				m_mask(0, j) = 0.0f;

				number_type max = rin(0, j);
				size_t imax = 0;

				for (size_t i = 1; i < rin.size<0>(); ++i)
				{
					m_mask(i, j) = 0.0f;

					auto e = rin(i, j);
					if (max < e)
					{
						max = e;
						imax = i;
					}
				}

				m_mask(imax, j) = 1.0f;
				rout(j) = max;
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			reshaped_input rresult = result.reshape<reshaped_input::metrics>();
			reshaped_output rgrad = grad.reshape<reshaped_output::metrics>();

			for (size_t i = 0; i < rresult.size<0>(); ++i)
			{
				for (size_t j = 0; j < rresult.size<1>(); ++j)
				{
					rresult(i, j) = (0.0f < m_mask(i, j)) ? rgrad(j) : 0.0f;
				}
			}
		}

	private:
		reshaped_input m_mask;
	};

	template <class Metrics>
	struct max_pooling_impl
	{
		typedef typename std::conditional<
			Metrics::rank == 1, 
			scalar_max_pooling<Metrics>,
			generic_max_pooling<Metrics>
		>::type type;
	};

	template <class Metrics, class Core, class Stride>
	class max_pooling_1d
	{
	public:
		static_assert(Metrics::rank == 1, "Invalid metric rank for 1D max pooling.");

		typedef typename max_pooling_1d<Metrics, Core, Stride> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics::tensor_type output;

		static_assert(
			std::is_same<typename input::number_type, typename input::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		max_pooling_1d()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0f);

			for (size_t stride = 0; stride < result.size<0>(); ++stride)
			{
				const size_t baseX = stride * algebra::detail::dimension<Stride, 0>::size;

				number_type max = input(baseX);
				size_t maxX = baseX;

				for (size_t x = 1; x < algebra::detail::dimension<Core, 0>::size; ++x)
				{
					auto e = input(baseX + x);
					if (max < e)
					{
						max = e;
						maxX = baseX + x;
					}
				}

				result(stride) = max;
				m_mask(maxX) += 1.0f;
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0f);

			for (size_t stride = 0; stride < grad.size<0>(); ++stride)
			{
				number_type g = grad(stride);

				const size_t baseX = stride * algebra::detail::dimension<Stride, 0>::size;

				for (size_t x = 0; x < algebra::detail::dimension<Core, 0>::size; ++x)
				{
					if (0.0f < m_mask(baseX + x))
					{
						result(baseX + x) += g;
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class Metrics, class Core, class Stride>
	class max_pooling_2d
	{
	public:
		static_assert(Metrics::rank == 2, "Invalid metric rank for 2D max pooling.");

		typedef typename max_pooling_2d<Metrics, Core, Stride> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics::tensor_type output;

		static_assert(
			std::is_same<typename input::number_type, typename input::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		max_pooling_2d()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0f);

			for (size_t strideX = 0; strideX < result.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < result.size<1>(); ++strideY)
				{
					const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
					const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;

					number_type max = input(baseX, baseY);
					size_t maxX = baseX;
					size_t maxY = baseY;

					for (size_t x = 0; x < algebra::detail::dimension<Core, 0>::size; ++x)
					{
						for (size_t y = 0; y < algebra::detail::dimension<Core, 1>::size; ++y)
						{
							auto e = input(baseX + x, baseY + y);
							if (max < e)
							{
								max = e;
								maxX = baseX + x;
								maxY = baseY + y;
							}
						}
					}

					result(strideX, strideY) = max;
					m_mask(maxX, maxY) += 1.0f;
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0f);

			for (size_t strideX = 0; strideX < grad.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < grad.size<1>(); ++strideY)
				{
					number_type g = grad(strideX, strideY);

					const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
					const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;

					for (size_t x = 0; x < algebra::detail::dimension<Core, 0>::size; ++x)
					{
						for (size_t y = 0; y < algebra::detail::dimension<Core, 1>::size; ++y)
						{
							if (0.0f < m_mask(baseX + x, baseY + y))
							{
								result(baseX + x, baseY + y) += g;
							}
						}
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class Metrics, class Core, class Stride>
	class max_pooling_3d
	{
	public:
		static_assert(Metrics::rank == 3, "Invalid metric rank for 3D max pooling.");

		typedef typename max_pooling_3d<Metrics, Core, Stride> this_type;

		typedef typename Metrics::tensor_type input;
		typedef typename algebra::detail::apply_core_with_stride<Metrics, Core, Stride, Metrics::rank>::metrics::tensor_type output;

		static_assert(
			std::is_same<typename input::number_type, typename input::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		max_pooling_3d()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0f);

			for (size_t strideX = 0; strideX < result.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < result.size<1>(); ++strideY)
				{
					for (size_t strideZ = 0; strideZ < result.size<2>(); ++strideZ)
					{
						const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
						const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;
						const size_t baseZ = strideZ * algebra::detail::dimension<Stride, 2>::size;

						number_type max = input(baseX, baseY, baseZ);
						size_t maxX = baseX;
						size_t maxY = baseY;
						size_t maxZ = baseZ;

						for (size_t x = 0; x < algebra::detail::dimension<Core, 0>::size; ++x)
						{
							for (size_t y = 0; y < algebra::detail::dimension<Core, 1>::size; ++y)
							{
								for (size_t z = 0; z < algebra::detail::dimension<Core, 2>::size; ++z)
								{
									auto e = input(baseX + x, baseY + y, baseZ + z);
									if (max < e)
									{
										max = e;
										maxX = baseX + x;
										maxY = baseY + y;
										maxZ = baseZ + z;
									}
								}
							}
						}

						result(strideX, strideY, strideZ) = max;
						m_mask(maxX, maxY, maxZ) += 1.0f;
					}
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0f);

			for (size_t strideX = 0; strideX < grad.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < grad.size<1>(); ++strideY)
				{
					for (size_t strideZ = 0; strideZ < grad.size<2>(); ++strideZ)
					{
						number_type g = grad(strideX, strideY, strideZ);

						const size_t baseX = strideX * algebra::detail::dimension<Stride, 0>::size;
						const size_t baseY = strideY * algebra::detail::dimension<Stride, 1>::size;
						const size_t baseZ = strideZ * algebra::detail::dimension<Stride, 2>::size;

						for (size_t x = 0; x < algebra::detail::dimension<Core, 0>::size; ++x)
						{
							for (size_t y = 0; y < algebra::detail::dimension<Core, 1>::size; ++y)
							{
								for (size_t z = 0; z < algebra::detail::dimension<Core, 2>::size; ++z)
								{
									if (0.0f < m_mask(baseX + x, baseY + y, baseZ + z))
									{
										result(baseX + x, baseY + y, baseZ + z) += g;
									}
								}
							}
						}
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class Metrics, class Core, class Stride>
	struct max_pooling_core_impl
	{
		static_assert(1 <= Metrics::rank == 1 && Metrics::rank <= 3, "Max pooling with core is supported only for 1D, 2D or 3D tensors.");

		typedef typename max_pooling_core_impl<Metrics, Core, Stride> this_type;

		typedef typename std::conditional<
			Metrics::rank == 1,
			max_pooling_1d<Metrics, Core, Stride>,
			typename std::conditional<
				Metrics::rank == 2,
				max_pooling_2d<Metrics, Core, Stride>,
				max_pooling_3d<Metrics, Core, Stride>
			>::type
		>::type type;

		template <typename _Layer>
		struct serializer
		{
			typedef typename _Layer value_type;

			typedef typename serialization::metrics_serializer<Metrics> _metrics_serializer;
			typedef typename serialization::metrics_serializer<Core> _core_serializer;
			typedef typename serialization::metrics_serializer<Stride> _stride_serializer;

			enum : size_t {
				serialized_data_size =
				_metrics_serializer::serialized_data_size
				+ _core_serializer::serialized_data_size
				+ _stride_serializer::serialized_data_size
			};

			static void read(
				std::istream& in,
				value_type&)
			{
				_metrics_serializer::read(in);
				_core_serializer::read(in);
				_stride_serializer::read(in);
			}

			static void write(
				std::ostream& out,
				const value_type&)
			{
				_metrics_serializer::write(out);
				_core_serializer::write(out);
				_stride_serializer::write(out);
			}
		};
	};

}

	template <class InputMetrics>
	class max_pooling : public layer_base<InputMetrics, typename detail::max_pooling_impl<InputMetrics>::type::output::metrics>
	{
	public:
		typedef typename max_pooling<InputMetrics> this_type;
		typedef typename detail::max_pooling_impl<InputMetrics>::type impl;

		typedef typename layer_base<InputMetrics, typename impl::output::metrics> base_type;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::max_pooling_layer,
			serialization::metrics_serializer<InputMetrics>
		> serializer_impl_type;

		max_pooling()
			: base_type(), m_impl()
		{}

		const output& process(const input& input)
		{
			m_impl.process(input, m_output);
			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			m_impl.compute_gradient(grad, m_gradient);
			return m_gradient;
		}

		void update_weights(
			const number_type)
		{}

		struct serializer
		{
			typedef this_type value_type;

			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value_type&)
			{
				serializer_impl_type::read(in);
			}

			static void write(
				std::ostream& out,
				const value_type&)
			{
				serializer_impl_type::write(out);
			}
		};

	private:
		impl m_impl;
	};
	
	template <class InputMetrics, class Core, class Stride>
	class max_pooling_with_core : public layer_base<InputMetrics, typename detail::max_pooling_core_impl<InputMetrics, Core, Stride>::type::output::metrics>
	{
	public:
		typedef typename max_pooling_with_core<InputMetrics, Core, Stride> this_type;
		typedef typename detail::max_pooling_core_impl<InputMetrics, Core, Stride>::type impl;

		typedef typename layer_base<InputMetrics, typename impl::output::metrics> base_type;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::max_pooling_with_core_layer,
			typename detail::max_pooling_core_impl<InputMetrics, Core, Stride>::template serializer<this_type>
		> serializer;

		max_pooling_with_core()
			: base_type(), m_impl()
		{}

		const output& process(const input& input)
		{
			m_impl.process(input, m_output);
			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			m_impl.compute_gradient(grad, m_gradient);
			return m_gradient;
		}

		void update_weights(
			const number_type)
		{}

	private:
		impl m_impl;
	};

	template <class Input, class... Args>
	max_pooling<Input> make_max_pooling_layer(
		Args&&... args)
	{
		typedef max_pooling<Input> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}

	template <class Input, class Core, class Stride, class... Args>
	max_pooling_with_core<Input, Core, Stride> make_max_pooling_layer(
		Args&&... args)
	{
		typedef max_pooling_with_core<Input, Core, Stride> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}
}