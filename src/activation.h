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

	template <typename Metrics>
	class activation_base : public layer_base<Metrics, Metrics>
	{
	public:
		typedef typename layer_base<Metrics, Metrics> base_type;

		activation_base() 
			: m_input(), base_type()
		{}

		void update_weights(
			const number_type)
		{}

	protected:
		input m_input;
	};

	template <typename Metrics>
	class relu_activation : public activation_base<Metrics>
	{
	public:
		typedef typename relu_activation<Metrics> this_type;
		typedef typename activation_base<Metrics> base_type;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::relu_activation_layer,
			serialization::metrics_serializer<typename Metrics>
		> serializer_impl_type;

		relu_activation()
			: base_type()
		{}

		const output& process(const input& input)
		{
			m_input = input;

			input.transform(
				m_output,
				[](const number_type& i)
				{
					return std::max(i, 0.0f);
				});

			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			grad.transform(
				m_input,
				m_gradient,
				[](const number_type& g, const number_type& i)
				{
					return (i > 0.0f) ? g : 0.0f;
				});

			return m_gradient;
		}

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
	};

	template <typename Metrics>
	class logistic_activation : public activation_base<Metrics>
	{
	public:
		typedef typename logistic_activation<Metrics> this_type;
		typedef typename activation_base<Metrics> base_type;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::logistic_activation_layer,
			serialization::metrics_serializer<typename Metrics>
		> serializer_impl_type;

		logistic_activation()
			: base_type()
		{}

		const output& process(const input& input)
		{
			m_input = input;

			input.transform(
				m_output,
				[](const number_type& i)
			{
				return this_type::logistic(i);
			});

			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			grad.transform(
				m_input,
				m_gradient,
				[](const number_type& g, const number_type& i)
			{
				auto f = this_type::logistic(i);
				return g * f * (1.0f - f);
			});

			return m_gradient;
		}

		struct serializer
		{
			typedef this_type value;
			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value&)
			{
				serializer_impl_type::read(in);
			}

			static void write(
				std::ostream& out,
				const value&)
			{
				serializer_impl_type::write(out);
			}
		};

	private:
		static number_type logistic(const number_type& x)
		{
			return 1.0f / (1.0f + std::exp(-x));
		}
	};

	template <class Input, class... Args>
	logistic_activation<Input> make_logistic_activation_layer(
		Args&&... args)
	{
		typedef logistic_activation<Input> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}

	template <class Input, class... Args>
	relu_activation<Input> make_relu_activation_layer(
		Args&&... args)
	{
		typedef relu_activation<Input> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}
}
