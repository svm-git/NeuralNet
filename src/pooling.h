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
	
	template <class _Metrics>
	class _scalar_max_pooling_impl
	{
	public:
		typedef typename _scalar_max_pooling_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef algebra::metrics<1>::tensor_type output;

		static_assert(_Metrics::rank == 1, "Invalid metric rank for scalar max pooling.");

		_scalar_max_pooling_impl()
			: m_mask()
		{}
	
		void process(
			const input& input,
			output& result)
		{
			m_mask(0) = 0.0;

			double max = input(0);
			size_t imax = 0;

			for (size_t i = 1; i < input.size<0>(); ++i)
			{
				m_mask(i) = 0.0;

				auto e = input(i);
				if (max < e)
				{
					max = e;
					imax = i;
				}
			}

			m_mask(imax) = 1.0;
			result(0) = max;
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			for (size_t i = 0; i < result.size<0>(); ++i)
			{
				result(i) = (0.0 < m_mask(i)) ? grad(0) : 0.0;
			}
		}

	private:
		input m_mask;
	};

	template <class _Metrics>
	class _generic_max_pooling_impl
	{
	public:
		typedef typename _generic_max_pooling_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename _Metrics::shrink::type::tensor_type output;

		typedef typename algebra::metrics<_Metrics::dimension_size, output::data_size>::tensor_type reshaped_input;
		typedef typename algebra::metrics<output::data_size>::tensor_type reshaped_output;

		static_assert(2 <= _Metrics::rank, "Metric rank is too small for generic max pooling.");

		_generic_max_pooling_impl()
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
				m_mask(0, j) = 0.0;

				double max = rin(0, j);
				size_t imax = 0;

				for (size_t i = 1; i < rin.size<0>(); ++i)
				{
					m_mask(i, j) = 0.0;

					auto e = rin(i, j);
					if (max < e)
					{
						max = e;
						imax = i;
					}
				}

				m_mask(imax, j) = 1.0;
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
					rresult(i, j) = (0.0 < m_mask(i, j)) ? rgrad(j) : 0.0;
				}
			}
		}

	private:
		reshaped_input m_mask;
	};

	template <class _Metrics>
	struct _max_pooling_impl
	{
		typedef typename std::conditional<
			_Metrics::rank == 1, 
			_scalar_max_pooling_impl<_Metrics>,
			_generic_max_pooling_impl<_Metrics>
		>::type type;
	};

	template <class _InputMetrics>
	class max_pooling : public layer_base<_InputMetrics, typename _max_pooling_impl<_InputMetrics>::type::output::metrics>
	{
	public:
		typedef typename max_pooling<_InputMetrics> _Self;
		typedef typename _max_pooling_impl<_InputMetrics>::type impl;

		typedef typename layer_base<_InputMetrics, typename impl::output::metrics> _Base;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::max_pooling_layer,
			serialization::metrics_serializer<_InputMetrics>
		> _serializer_impl;

		max_pooling()
			: _Base(), m_impl()
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
			const double /*rate*/)
		{}

		struct serializer
		{
			typedef _Self value;

			enum : size_t { serialized_data_size = _serializer_impl::serialized_data_size };

			static void read(
				std::istream& in,
				value&)
			{
				_serializer_impl::read(in);
			}

			static void write(
				std::ostream& out,
				const value&)
			{
				_serializer_impl::write(out);
			}
		};

	private:
		impl m_impl;
	};
	
	template <class _Input, class... _Args>
	max_pooling<_Input> make_max_pooling_layer(
		_Args&&... args)
	{
		typedef max_pooling<_Input> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}

	template <class _Metrics, class _Core, class _Stride>
	class _1d_max_pooling_impl
	{
	public:
		static_assert(_Metrics::rank == 1, "Invalid metric rank for 1D max pooling.");

		typedef typename _1d_max_pooling_impl<_Metrics, _Core, _Stride> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics::tensor_type output;

		_1d_max_pooling_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0);

			for (size_t stride = 0; stride < result.size<0>(); ++stride)
			{
				const size_t baseX = stride * algebra::_dimension<_Stride, 0>::size;

				double max = input(baseX);
				size_t maxX = baseX;

				for (size_t x = 1; x < algebra::_dimension<_Core, 0>::size; ++x)
				{
					auto e = input(baseX + x);
					if (max < e)
					{
						max = e;
						maxX = baseX + x;
					}
				}

				result(stride) = max;
				m_mask(maxX) += 1.0;
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0);

			for (size_t stride = 0; stride < grad.size<0>(); ++stride)
			{
				double g = grad(stride);

				const size_t baseX = stride * algebra::_dimension<_Stride, 0>::size;

				for (size_t x = 0; x < algebra::_dimension<_Core, 0>::size; ++x)
				{
					if (0.0 < m_mask(baseX + x))
					{
						result(baseX + x) += g;
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class _Metrics, class _Core, class _Stride>
	class _2d_max_pooling_impl
	{
	public:
		static_assert(_Metrics::rank == 2, "Invalid metric rank for 2D max pooling.");

		typedef typename _2d_max_pooling_impl<_Metrics, _Core, _Stride> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics::tensor_type output;

		_2d_max_pooling_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0);

			for (size_t strideX = 0; strideX < result.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < result.size<1>(); ++strideY)
				{
					const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
					const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;

					double max = input(baseX, baseY);
					size_t maxX = baseX;
					size_t maxY = baseY;

					for (size_t x = 0; x < algebra::_dimension<_Core, 0>::size; ++x)
					{
						for (size_t y = 0; y < algebra::_dimension<_Core, 1>::size; ++y)
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
					m_mask(maxX, maxY) += 1.0;
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0);

			for (size_t strideX = 0; strideX < grad.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < grad.size<1>(); ++strideY)
				{
					double g = grad(strideX, strideY);

					const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
					const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;

					for (size_t x = 0; x < algebra::_dimension<_Core, 0>::size; ++x)
					{
						for (size_t y = 0; y < algebra::_dimension<_Core, 1>::size; ++y)
						{
							if (0.0 < m_mask(baseX + x, baseY + y))
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

	template <class _Metrics, class _Core, class _Stride>
	class _3d_max_pooling_impl
	{
	public:
		static_assert(_Metrics::rank == 3, "Invalid metric rank for 3D max pooling.");

		typedef typename _3d_max_pooling_impl<_Metrics, _Core, _Stride> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename algebra::_apply_core_with_stride<_Metrics, _Core, _Stride, _Metrics::rank>::metrics::tensor_type output;

		_3d_max_pooling_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			m_mask.fill(0.0);

			for (size_t strideX = 0; strideX < result.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < result.size<1>(); ++strideY)
				{
					for (size_t strideZ = 0; strideZ < result.size<2>(); ++strideZ)
					{
						const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
						const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;
						const size_t baseZ = strideZ * algebra::_dimension<_Stride, 2>::size;

						double max = input(baseX, baseY, baseZ);
						size_t maxX = baseX;
						size_t maxY = baseY;
						size_t maxZ = baseZ;

						for (size_t x = 0; x < algebra::_dimension<_Core, 0>::size; ++x)
						{
							for (size_t y = 0; y < algebra::_dimension<_Core, 1>::size; ++y)
							{
								for (size_t z = 0; z < algebra::_dimension<_Core, 2>::size; ++z)
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
						m_mask(maxX, maxY, maxZ) += 1.0;
					}
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			result.fill(0.0);

			for (size_t strideX = 0; strideX < grad.size<0>(); ++strideX)
			{
				for (size_t strideY = 0; strideY < grad.size<1>(); ++strideY)
				{
					for (size_t strideZ = 0; strideZ < grad.size<2>(); ++strideZ)
					{
						double g = grad(strideX, strideY, strideZ);

						const size_t baseX = strideX * algebra::_dimension<_Stride, 0>::size;
						const size_t baseY = strideY * algebra::_dimension<_Stride, 1>::size;
						const size_t baseZ = strideZ * algebra::_dimension<_Stride, 2>::size;

						for (size_t x = 0; x < algebra::_dimension<_Core, 0>::size; ++x)
						{
							for (size_t y = 0; y < algebra::_dimension<_Core, 1>::size; ++y)
							{
								for (size_t z = 0; z < algebra::_dimension<_Core, 2>::size; ++z)
								{
									if (0.0 < m_mask(baseX + x, baseY + y, baseZ + z))
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

	template <class _Metrics, class _Core, class _Stride>
	struct _max_pooling_core_impl
	{
		static_assert(1 <= _Metrics::rank == 1 && _Metrics::rank <= 3, "Max pooling with core is supported only for 1D, 2D or 3D tensors.");

		typedef typename _max_pooling_core_impl<_Metrics, _Core, _Stride> _Self;

		typedef typename std::conditional<
			_Metrics::rank == 1,
			_1d_max_pooling_impl<_Metrics, _Core, _Stride>,
			typename std::conditional<
				_Metrics::rank == 2,
				_2d_max_pooling_impl<_Metrics, _Core, _Stride>,
				_3d_max_pooling_impl<_Metrics, _Core, _Stride>
			>::type
		>::type type;

		template <typename _Layer>
		struct serializer
		{
			typedef typename _Layer value;

			typedef typename serialization::metrics_serializer<_Metrics> _metrics_serializer;
			typedef typename serialization::metrics_serializer<_Core> _core_serializer;
			typedef typename serialization::metrics_serializer<_Stride> _stride_serializer;

			enum : size_t { serialized_data_size = 
				_metrics_serializer::serialized_data_size 
					+ _core_serializer::serialized_data_size
					+ _stride_serializer::serialized_data_size
			};

			static void read(
				std::istream& in,
				value&)
			{
				_metrics_serializer::read(in);
				_core_serializer::read(in);
				_stride_serializer::read(in);
			}

			static void write(
				std::ostream& out,
				const value&)
			{
				_metrics_serializer::write(out);
				_core_serializer::write(out);
				_stride_serializer::write(out);
			}
		};
	};

	template <class _InputMetrics, class _Core, class _Stride>
	class max_pooling_with_core : public layer_base<_InputMetrics, typename _max_pooling_core_impl<_InputMetrics, _Core, _Stride>::type::output::metrics>
	{
	public:
		typedef typename max_pooling_with_core<_InputMetrics, _Core, _Stride> _Self;
		typedef typename _max_pooling_core_impl<_InputMetrics, _Core, _Stride>::type impl;

		typedef typename layer_base<_InputMetrics, typename impl::output::metrics> _Base;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::max_pooling_with_core_layer,
			typename _max_pooling_core_impl<_InputMetrics, _Core, _Stride>::template serializer<_Self>
		> serializer;

		max_pooling_with_core()
			: _Base(), m_impl()
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
			const double /*rate*/)
		{}

	private:
		impl m_impl;
	};

	template <class _Input, class _Core, class _Stride, class... _Args>
	max_pooling_with_core<_Input, _Core, _Stride> make_max_pooling_layer(
		_Args&&... args)
	{
		typedef max_pooling_with_core<_Input, _Core, _Stride> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}
}