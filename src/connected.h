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
#include "serialization.h"

namespace neural_network {

	template <typename _InputMetrics, typename _OutputMetrics>
	class fully_connected : public layer_base<_InputMetrics, _OutputMetrics>
	{
	public:
		typedef typename fully_connected<_InputMetrics, _OutputMetrics> _Self;
		typedef typename layer_base<_InputMetrics, _OutputMetrics> _Base;

		typedef typename algebra::metrics<input::data_size> reshaped_input;
		typedef typename algebra::metrics<output::data_size> reshaped_output;

		typedef typename algebra::metrics<
			reshaped_input::data_size,
			reshaped_output::data_size>::tensor_type _Weights;
		typedef typename reshaped_output::tensor_type _Bias;
	
		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::fully_connected_layer,
			serialization::composite_serializer<
				serialization::tensor_serializer<_Weights>,
				serialization::tensor_serializer<_Bias>,
				serialization::value_serializer<double>>
		> _serializer_impl;

		fully_connected(
			const double regularization = 0.000001)
				: _Base(), m_input(), m_weights(), m_weightsGradient(), m_bias(), m_biasGradient(), m_regularization(regularization)
		{
		}

		fully_connected(
			std::function<double()> initializer,
			const double regularization = 0.000001)
				: _Base(), m_input(), m_weights(initializer), m_weightsGradient(), m_bias(initializer), m_biasGradient(), m_regularization(regularization)
		{
		}

		const output& process(const input& input)
		{
			m_input = input;

			reshaped_input::tensor_type rin = input.reshape<reshaped_input>();
			reshaped_output::tensor_type rout = m_output.reshape<reshaped_output>();

			for (size_t j = 0; j < rout.size<0>(); ++j)
			{
				double sum = 0.0;
				for (size_t i = 0; i < rin.size<0>(); ++i)
				{
					sum += m_weights(i, j) * rin(i);
				}

				rout(j) = sum + m_bias(j);
			}

			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			reshaped_input::tensor_type rin = m_input.reshape<reshaped_input>();
			reshaped_input::tensor_type rgradResult = m_gradient.reshape<reshaped_input>();
			reshaped_output::tensor_type rgrad = grad.reshape<reshaped_output>();

			for (size_t i = 0; i < rgradResult.size<0>(); ++i)
			{
				double sum = 0.0;
				for (size_t j = 0; j < rgrad.size<0>(); ++j)
				{
					sum += m_weights(i, j) * rgrad(j);
					
					m_weightsGradient(i, j) = rin(i) * rgrad(j);
				}

				rgradResult(i) = sum;
			}

			for (size_t j = 0; j < rgrad.size<0>(); ++j)
			{
				m_biasGradient(j) = rgrad(j);
			}

			return m_gradient;
		}

		void update_weights(
			const double rate)
		{
			for (size_t i = 0; i < m_weights.size<0>(); ++i)
			{
				for (size_t j = 0; j < m_weights.size<1>(); ++j)
				{
					m_weights(i, j) += (m_weightsGradient(i, j) + m_regularization * m_weights(i, j)) * rate;
				}
			}

			for (size_t j = 0; j < m_bias.size<0>(); ++j)
			{
				m_bias(j) += (m_biasGradient(j) + m_regularization * m_bias(j)) * rate;
			}
		}

		struct serializer
		{
			typedef _Self value;

			enum : size_t { serialized_data_size = _serializer_impl::serialized_data_size };

			static void read(
				std::istream& in,
				_Self& layer)
			{
				_serializer_impl::read(in, layer.m_weights, layer.m_bias, layer.m_regularization);
			}

			static void write(
				std::ostream& out,
				const _Self& layer)
			{
				_serializer_impl::write(out, layer.m_weights, layer.m_bias, layer.m_regularization);
			}
		};

	private:
		input m_input;
		_Weights m_weights;
		_Weights m_weightsGradient;
		_Bias m_bias;
		_Bias m_biasGradient;
		double m_regularization;
	};

	template <class _Input, class _Output, class... _Args>
	fully_connected<_Input, _Output> make_fully_connected_layer(
		_Args&&... args)
	{
		typedef fully_connected<_Input, _Output> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}
}