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
#include "serialization.h"

namespace neural_network {
	
namespace detail {

	template <class Network, class Output>
	void process_and_copy_network_result(
		Network& network,
		const size_t index,
		const typename Network::input& input,
		Output& output)
	{
		typedef typename typename Network::output local_output_type;
		typedef typename algebra::metrics<local_output_type::data_size> reshaped_local_output_metrics;
		typedef typename Output::metrics::shrink::type shrink_metrics;

		typedef typename algebra::metrics<Output::dimension_size, shrink_metrics::data_size> reshaped_output_metrics;

		auto localResult = network
			.process(input)
			.reshape<reshaped_local_output_metrics>();

		auto reshaped_output = output.reshape<reshaped_output_metrics>();
		
		for (size_t i = 0; i < localResult.size<0>(); ++i)
		{
			// Reshaped tensors share the same data, therefore
			// data in 'output' tensor is updated by this loop.
			reshaped_output(index, i) = localResult(i);
		}
	}

	template <class Network, class Gradient>
	void compute_gradient_and_add_result(
		Network& network,
		const size_t index,
		const Gradient& grad,
		typename Network::output& local,
		typename Network::input& result)
	{
		typedef typename algebra::metrics<Network::output::data_size> reshaped_local_metrics;
		typedef typename Gradient::metrics::shrink::type shrink_metrics;

		typedef typename algebra::metrics<Gradient::dimension_size, shrink_metrics::data_size> reshaped_gradient_metrics;

		auto gradient = grad.reshape<reshaped_gradient_metrics>();
		auto localGradient = local.reshape<reshaped_local_metrics>();

		for (size_t i = 0; i < localGradient.size<0>(); ++i)
		{
			localGradient(i) = gradient(index, i);
		}

		// Reshaped tensors share the same data, therefore
		// data in 'local' tensor is initialized by the loop above.
		auto localResult = network.compute_gradient(local);

		// result = result + localResult
		localResult.transform(
			result,
			result,
			[](const typename Network::number_type& l, const typename Network::number_type& r)
			{
				return r + l;
			});
	}

	template <class Network, class... Args>
	class network_ensemble_impl : protected network_ensemble_impl<Args...>
	{
	public:
		typedef typename network_ensemble_impl<Network, Args...> this_type;
		typedef typename network_ensemble_impl<Args...> base_type;

		static_assert(std::is_same<typename Network::input, typename base_type::input>::value, "Network input types do not match.");
		static_assert(std::is_same<typename Network::output, typename base_type::common_output>::value, "Network output types do not match.");

		enum : size_t { ensemble_size = base_type::ensemble_size + 1 };

		typedef typename Network::input input;
		typedef typename base_type::common_output common_output;
		typedef typename base_type::common_output::metrics::template expand<ensemble_size>::type::tensor_type output;

		typedef typename input::number_type number_type;

		network_ensemble_impl()
			: m_network()
		{}

		network_ensemble_impl(const Network& n, const Args&... args)
			: base_type(args...), m_network(n)
		{}

		template <class Output>
		void process(
			const input& input,
			Output& output)
		{
			process_and_copy_network_result(
				m_network,
				this_type::ensemble_size - 1,
				input,
				output);

			base_type::process(input, output);
		}

		template <class Output, class LocalGradient, class Gradient>
		void compute_gradient(
			const Output& grad,
			LocalGradient& local,
			Gradient& result)
		{
			compute_gradient_and_add_result(
				m_network,
				this_type::ensemble_size - 1,
				grad,
				local,
				result);

			base_type::compute_gradient(grad, local, result);
		}

		void update_weights(
			const number_type rate)
		{
			m_network.update_weights(rate);

			base_type::update_weights(rate);
		}

		struct serializer
		{
			typedef this_type value_type;

			enum : size_t {
				serialized_data_size =
					base_type::serializer::serialized_data_size
					+ Network::serializer::serialized_data_size
			};

			static void read(
				std::istream& in,
				value_type& network)
			{
				Network::serializer::read(in, network.m_network);
				base_type::serializer::read(in, network);
			}

			static void write(
				std::ostream& out,
				const value_type& network)
			{
				Network::serializer::write(out, network.m_network);
				base_type::serializer::write(out, network);
			}
		};

	private:
		Network m_network;
	};

	template <class Network>
	class network_ensemble_impl<Network>
	{
	public:
		typedef typename network_ensemble_impl<Network> this_type;

		enum : size_t { ensemble_size = 1 };

		typedef typename Network::input input;
		typedef typename Network::output common_output;
		typedef typename Network::output::metrics::template expand<ensemble_size>::type::tensor_type output;

		typedef typename input::number_type number_type;

		network_ensemble_impl()
			: m_network()
		{}

		network_ensemble_impl(const Network& n)
			: m_network(n)
		{}

		template <class Output>
		void process(
			const input& input,
			Output& output)
		{
			process_and_copy_network_result(
				m_network,
				this_type::ensemble_size - 1,
				input,
				output);
		}

		template <class Output, class LocalGradient, class Gradient>
		void compute_gradient(
			const Output& grad,
			LocalGradient& local,
			Gradient& result)
		{
			compute_gradient_and_add_result(
				m_network,
				this_type::ensemble_size - 1,
				grad,
				local,
				result);
		}

		void update_weights(
			const number_type rate)
		{
			m_network.update_weights(rate);
		}

		struct serializer
		{
			typedef this_type value_type;

			enum : size_t { serialized_data_size = Network::serializer::serialized_data_size };

			static void read(
				std::istream& in,
				value_type& network)
			{
				Network::serializer::read(in, network.m_network);
			}

			static void write(
				std::ostream& out,
				const value_type& network)
			{
				Network::serializer::write(out, network.m_network);
			}
		};

	private:
		Network m_network;
	};
	
}

	template <class Network1, class Network2, class... Args>
	class network_ensemble
	{
	public:
		typedef typename network_ensemble<Network1, Network2, Args...> this_type;
		typedef typename detail::network_ensemble_impl<Network1, Network2, Args...> ensemble_type;
		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::ensemble_layer,
			typename ensemble_type::serializer
		> serializer_impl_type;

		typedef typename ensemble_type::input input;
		typedef typename ensemble_type::output output;

		typedef typename input::number_type number_type;

		network_ensemble()
			: m_ensemble(), m_output(), m_gradient(), m_local()
		{
		}

		network_ensemble(const Network1& n1, const Network2& n2, const Args&... args)
			: m_ensemble(n1, n2, args...), m_output(), m_gradient(), m_local()
		{
		}

		const output& process(const input& input)
		{
			m_ensemble.process(input, m_output);
			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			m_gradient.fill(0.0f);
			m_ensemble.compute_gradient(grad, m_local, m_gradient);
			return m_gradient;
		}

		void update_weights(
			const number_type rate)
		{
			m_ensemble.update_weights(rate);
		}

		struct serializer
		{
			typedef this_type value_type;

			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value_type& network)
			{
				serializer_impl_type::read(in, network.m_ensemble);
			}

			static void write(
				std::ostream& out,
				const value_type& network)
			{
				serializer_impl_type::write(out, network.m_ensemble);
			}
		};

	private:
		ensemble_type m_ensemble;
		output m_output;
		input m_gradient;
		typename ensemble_type::common_output m_local;
	};

	template <class... Networks>
	network_ensemble<Networks...> make_ensemble(
		Networks&&... args)
	{
		typedef network_ensemble<Networks...> network_type;
		return (network_type(std::forward<Networks>(args)...));
	}
}
