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

#include "serialization.h"

namespace neural_network {

namespace detail {

	template <class Network, class Loss>
	void train_network(
		Network& net,
		const typename Network::input& input,
		const typename Network::output& truth,
		Loss& loss,
		const typename Network::number_type rate)
	{
		net.compute_gradient(
			loss.compute_gradient(
				net.process(input),
				truth));

		net.update_weights(-std::abs(rate));
	}

}

	template <class Layer, class... Args>
	class network : protected network <Args...>
	{
	public:
		typedef typename network<Layer, Args...> this_type;
		typedef typename network<Args...> base_type;

		typedef typename Layer::input input;
		typedef typename base_type::output output;

		static_assert(std::is_same<typename Layer::output, typename base_type::input>::value, "Output of the current layer does not match input of the next layer.");

		typedef typename Layer::number_type number_type;

		network()
			: base_type(), m_layer()
		{}

		network(const Layer& layer, const Args&... args)
			: base_type(args...), m_layer(layer)
		{
		}

		const output& process(const input& input)
		{
			return base_type::process(
				m_layer.process(input));
		}

		const input& compute_gradient(const output& grad)
		{
			return m_layer.compute_gradient(
				base_type::compute_gradient(grad));
		}

		void update_weights(
			const number_type rate)
		{
			base_type::update_weights(rate);
			m_layer.update_weights(rate);
		}

		template <class Loss>
		void train(
			const typename input& input,
			const typename output& truth,
			Loss& loss,
			const number_type rate)
		{
			detail::train_network(*this, input, truth, loss, rate);
		}

		struct serializer
		{
			typedef this_type value;

			enum : size_t { 
				serialized_data_size = 
					base_type::serializer::serialized_data_size
					+ Layer::serializer::serialized_data_size
			};

			static void read(
				std::istream& in,
				value& layer)
			{
				Layer::serializer::read(in, layer.m_layer);
				base_type::serializer::read(in, layer);
			}

			static void write(
				std::ostream& out,
				const value& layer)
			{
				Layer::serializer::write(out, layer.m_layer);
				base_type::serializer::write(out, layer);
			}
		};

	private:
		Layer m_layer;
	};

	template <class Layer>
	class network<Layer>
	{
	public:
		typedef typename network<Layer> this_type;

		typedef typename Layer::input input;
		typedef typename Layer::output output;

		typedef typename Layer::number_type number_type;

		network()
			: m_layer()
		{}

		network(const Layer& layer)
			: m_layer(layer)
		{
		}

		const output& process(const input& input)
		{
			return m_layer.process(input);
		}

		const input& compute_gradient(const output& grad)
		{
			return m_layer.compute_gradient(grad);
		}

		void update_weights(
			const number_type rate)
		{
			m_layer.update_weights(rate);
		}

		template <class Loss>
		void train(
			const typename input& input,
			const typename output& truth,
			Loss& loss,
			const number_type rate)
		{
			train_network(*this, input, truth, loss, rate);
		}

		struct serializer
		{
			typedef this_type value;

			enum : size_t { serialized_data_size = Layer::serializer::serialized_data_size };

			static void read(
				std::istream& in,
				value& layer)
			{
				Layer::serializer::read(in, layer.m_layer);
			}

			static void write(
				std::ostream& out,
				const value& layer)
			{
				Layer::serializer::write(out, layer.m_layer);
			}
		};

	private:
		Layer m_layer;
	};

	template <class... Layers>
	network<Layers...> make_network(
		Layers&&... args)
	{
		typedef network<Layers...> network_type;
		return (network_type(std::forward<Layers>(args)...));
	}
}