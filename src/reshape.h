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

namespace detail {

	template <typename Reshape>
	struct reshape_serializer_impl
	{
		typedef typename Reshape value_type;

		typedef typename serialization::metrics_serializer<typename Reshape::input::metrics> input_metrics_serializer_type;
		typedef typename serialization::metrics_serializer<typename Reshape::output::metrics> ouput_metrics_serializer_type;

		enum : size_t {
			serialized_data_size =
				input_metrics_serializer_type::serialized_data_size
				+ ouput_metrics_serializer_type::serialized_data_size
		};

		static void read(
			std::istream& in,
			value_type&)
		{
			input_metrics_serializer_type::read(in);
			ouput_metrics_serializer_type::read(in);
		}

		static void write(
			std::ostream& out,
			const value_type&)
		{
			input_metrics_serializer_type::write(out);
			ouput_metrics_serializer_type::write(out);
		}
	};
}

	template <typename InputMetrics, typename OutputMetrics>
	class reshape : public layer_base <InputMetrics, OutputMetrics>
	{
	public:
		typedef typename reshape<InputMetrics, OutputMetrics> this_type;
		typedef typename layer_base<InputMetrics, OutputMetrics> base_type;

		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::reshape_layer,
			detail::reshape_serializer_impl<this_type>
		> serializer;

		const output& process(const input& input)
		{
			m_output = input.reshape<output::metrics>();
			return m_output;
		}

		const input& compute_gradient(const output& grad)
		{
			m_gradient = grad.reshape<input::metrics>();
			return m_gradient;
		}

		void update_weights(
			const float)
		{}
	};

	template <class Input, class Output, class... Args>
	reshape<Input, Output> make_reshape_layer(
		Args&&... args)
	{
		typedef reshape<Input, Output> layer_type;
		return (layer_type(std::forward<Args>(args)...));
	}
}