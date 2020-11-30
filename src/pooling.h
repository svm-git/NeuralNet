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

namespace neural_network {

	template <class _Metrics>
	class _1d_max_pooling_impl
	{
	public:
		typedef typename _1d_max_pooling_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef algebra::metrics<1>::tensor_type output;

		static_assert(_Metrics::rank == 1, "Invalid metric rank for 1D max pooing.");

		_1d_max_pooling_impl()
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
	class _2d_max_pooing_impl
	{
	public:
		typedef typename _2d_max_pooing_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename _Metrics::shrink::type::tensor_type output;

		static_assert(_Metrics::rank == 2, "Invalid metric rank for 2D max pooing.");

		_2d_max_pooing_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			for (size_t j = 0; j < input.size<1>(); ++j)
			{
				m_mask(0, j) = 0.0;

				double max = input(0, j);
				size_t imax = 0;

				for (size_t i = 1; i < input.size<0>(); ++i)
				{
					m_mask(i, j) = 0.0;

					auto e = input(i, j);
					if (max < e)
					{
						max = e;
						imax = i;
					}
				}

				m_mask(imax, j) = 1.0;
				result(j) = max;
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			for (size_t i = 0; i < result.size<0>(); ++i)
			{
				for (size_t j = 0; j < result.size<1>(); ++j)
				{
					result(i, j) = (0.0 < m_mask(i, j)) ? grad(j) : 0.0;
				}
			}
		}

	private:
		input m_mask;
	};

	template <class _Metrics>
	class _3d_max_pooing_impl
	{
	public:
		typedef typename _3d_max_pooing_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename _Metrics::shrink::type::tensor_type output;

		static_assert(_Metrics::rank == 3, "Invalid metric rank for 3D max pooing.");

		_3d_max_pooing_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			for (size_t j = 0; j < input.size<1>(); ++j)
			{
				for (size_t k = 0; k < input.size<2>(); ++k)
				{
					m_mask(0, j, k) = 0.0;

					double max = input(0, j, k);
					size_t imax = 0;

					for (size_t i = 1; i < input.size<0>(); ++i)
					{
						m_mask(i, j, k) = 0.0;

						auto e = input(i, j, k);
						if (max < e)
						{
							max = e;
							imax = i;
						}
					}

					m_mask(imax, j, k) = 1.0;
					result(j, k) = max;
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			for (size_t i = 0; i < result.size<0>(); ++i)
			{
				for (size_t j = 0; j < result.size<1>(); ++j)
				{
					for (size_t k = 0; k < result.size<2>(); ++k)
					{
						result(i, j, k) = (0.0 < m_mask(i, j, k)) ? grad(j, k) : 0.0;
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class _Metrics>
	class _4d_max_pooing_impl
	{
	public:
		typedef typename _4d_max_pooing_impl<_Metrics> _Self;

		typedef typename _Metrics::tensor_type input;
		typedef typename _Metrics::shrink::type::tensor_type output;

		static_assert(_Metrics::rank == 4, "Invalid metric rank for 4D max pooing.");

		_4d_max_pooing_impl()
			: m_mask()
		{}

		void process(
			const input& input,
			output& result)
		{
			for (size_t j = 0; j < input.size<1>(); ++j)
			{
				for (size_t k = 0; k < input.size<2>(); ++k)
				{
					for (size_t l = 0; l < input.size<3>(); ++l)
					{
						m_mask(0, j, k, l) = 0.0;

						double max = input(0, j, k, l);
						size_t imax = 0;

						for (size_t i = 1; i < input.size<0>(); ++i)
						{
							m_mask(i, j, k, l) = 0.0;

							auto e = input(i, j, k, l);
							if (max < e)
							{
								max = e;
								imax = i;
							}
						}

						m_mask(imax, j, k, l) = 1.0;
						result(j, k, l) = max;
					}
				}
			}
		}

		void compute_gradient(
			const output& grad,
			input& result)
		{
			for (size_t i = 0; i < result.size<0>(); ++i)
			{
				for (size_t j = 0; j < result.size<1>(); ++j)
				{
					for (size_t k = 0; k < result.size<2>(); ++k)
					{
						for (size_t l = 0; l < result.size<3>(); ++l)
						{
							result(i, j, k, l) = (0.0 < m_mask(i, j, k, l)) ? grad(j, k, l) : 0.0;
						}
					}
				}
			}
		}

	private:
		input m_mask;
	};

	template <class _Metrics>
	struct _max_pooling_impl
	{
		static_assert(1 <= _Metrics::rank == 1 && _Metrics::rank <= 4, "Max pooling is supported only for 1D, 2D, 3D or 4D tensors.");

		typedef typename std::conditional<
			_Metrics::rank == 1, 
			_1d_max_pooling_impl<_Metrics>,
			typename std::conditional<
				_Metrics::rank == 2, 
				_2d_max_pooing_impl<_Metrics>,
				typename std::conditional<
					_Metrics::rank == 3,
					_3d_max_pooing_impl<_Metrics>,
					_4d_max_pooing_impl<_Metrics>
				>::type
			>::type
		>::type type;
	};

	template <class _InputMetrics>
	class max_pooling : public layer_base<_InputMetrics, typename _max_pooling_impl<_InputMetrics>::type::output::metrics>
	{
	public:
		typedef typename max_pooling<_InputMetrics> _Self;
		typedef typename _max_pooling_impl<_InputMetrics>::type impl;

		typedef typename layer_base<_InputMetrics, typename impl::output::metrics> _Base;

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
			const double rate)
		{}

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
}