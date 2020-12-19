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