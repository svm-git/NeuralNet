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

	template <typename _Metrics>
	class activation_base : public layer_base<_Metrics, _Metrics>
	{
	public:
		typedef typename layer_base<_Metrics, _Metrics> _Base;

		activation_base() 
			: m_input(), _Base()
		{}

		void update_weights(
			const double /*rate*/)
		{}

	protected:
		input m_input;
	};

	template <typename _Metrics>
	class relu_activation : public activation_base<_Metrics>
	{
	public:
		typedef typename relu_activation<_Metrics> _Self;
		typedef typename activation_base<_Metrics> _Base;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::relu_activation_layer,
			serialization::metrics_serializer<typename _Metrics>
		> _serializer_impl;

		relu_activation()
			: _Base()
		{}

		const output& process(const input& input)
		{
			m_input = input;

			input.transform(
				m_output,
				[](const double& i)
				{
					return std::max(i, 0.0);
				});

			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			grad.transform(
				m_input,
				m_gradient,
				[](const double& g, const double& i)
				{
					return (i > 0.0) ? g : 0.0;
				});

			return m_gradient;
		}

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
	};

	template <typename _Metrics>
	class logistic_activation : public activation_base<_Metrics>
	{
	public:
		typedef typename logistic_activation<_Metrics> _Self;
		typedef typename activation_base<_Metrics> _Base;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::logistic_activation_layer,
			serialization::metrics_serializer<typename _Metrics>
		> _serializer_impl;

		logistic_activation()
			: _Base()
		{}

		const output& process(const input& input)
		{
			m_input = input;

			input.transform(
				m_output,
				[](const double& i)
			{
				return _Self::logistic(i);
			});

			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			grad.transform(
				m_input,
				m_gradient,
				[](const double& g, const double& i)
			{
				auto _f = _Self::logistic(i);
				return g * _f * (1.0 - _f);
			});

			return m_gradient;
		}

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
		static double logistic(const double& x)
		{
			return 1.0 / (1.0 + std::exp(-x));
		}
	};

	template <class _Input, class... _Args>
	logistic_activation<_Input> make_logistic_activation_layer(
		_Args&&... args)
	{
		typedef logistic_activation<_Input> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}

	template <class _Input, class... _Args>
	relu_activation<_Input> make_relu_activation_layer(
		_Args&&... args)
	{
		typedef relu_activation<_Input> _Ltype;
		return (_Ltype(std::forward<_Args>(args)...));
	}
}
