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

#ifdef NEURAL_NET_ENABLE_OPEN_CL

#include "opencl/connected.h"

#endif

namespace neural_network {

	template <typename InputMetrics, typename OutputMetrics>
	class fully_connected : public layer_base<InputMetrics, OutputMetrics>
	{
	public:
		typedef typename fully_connected<InputMetrics, OutputMetrics> this_type;
		typedef typename layer_base<InputMetrics, OutputMetrics> base_type;

		typedef typename algebra::metrics<input::data_size> reshaped_input;
		typedef typename algebra::metrics<output::data_size> reshaped_output;

		typedef typename algebra::metrics<
			reshaped_output::data_size,
			reshaped_input::data_size>::tensor_type weights_type;
		typedef typename reshaped_output::tensor_type bias_type;
	
		typedef typename serialization::chunk_serializer<
			serialization::chunk_types::fully_connected_layer,
			serialization::composite_serializer<
				serialization::tensor_serializer<weights_type>,
				serialization::tensor_serializer<bias_type>,
				serialization::value_serializer<number_type>>
		> serializer_impl_type;

		fully_connected(
			const number_type regularization = 0.000001f)
				: base_type(), m_input(), m_weights(), m_weightsGradient(), m_bias(), m_biasGradient(), m_regularization(regularization)
#ifdef NEURAL_NET_ENABLE_OPEN_CL
				, m_kernelProgram(), m_processKernelName(), m_gradientKernelName(), m_weightsKernelName()
#endif
		{
		}

		fully_connected(
			std::function<number_type()> initializer,
			const number_type regularization = 0.000001f)
				: base_type(), m_input(), m_weights(initializer), m_weightsGradient(), m_bias(initializer), m_biasGradient(), m_regularization(regularization)
#ifdef NEURAL_NET_ENABLE_OPEN_CL
				, m_kernelProgram(), m_processKernelName(), m_gradientKernelName(), m_weightsKernelName()
#endif
		{
		}

		const output& process(const input& input)
		{
			m_input = input;

			reshaped_input::tensor_type rin = input.reshape<reshaped_input>();
			reshaped_output::tensor_type rout = m_output.reshape<reshaped_output>();

			for (size_t j = 0; j < rout.size<0>(); ++j)
			{
				number_type sum = 0.0f;
				for (size_t i = 0; i < rin.size<0>(); ++i)
				{
					sum += m_weights(j, i) * rin(i);
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
				number_type sum = 0.0f;
				for (size_t j = 0; j < rgrad.size<0>(); ++j)
				{
					sum += m_weights(j, i) * rgrad(j);

					m_weightsGradient(j, i) = rin(i) * rgrad(j);
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
			const number_type rate)
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
			typedef this_type value_type;

			enum : size_t { serialized_data_size = serializer_impl_type::serialized_data_size };

			static void read(
				std::istream& in,
				value_type& layer)
			{
				serializer_impl_type::read(in, layer.m_weights, layer.m_bias, layer.m_regularization);
			}

			static void write(
				std::ostream& out,
				const value_type& layer)
			{
				serializer_impl_type::write(out, layer.m_weights, layer.m_bias, layer.m_regularization);
			}
		};

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		const output& process(
			const input& input,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_process<weights_type::data_size>(input, queue);
		}

		const input& compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_compute_gradient<weights_type::data_size>(gradient, queue);
		}

		void update_weights(
			const number_type rate,
			::boost::compute::command_queue& queue)
		{
			this->dispatch_update_weights<weights_type::data_size>(rate, queue);
		}

	private:
		template <const size_t TensorSize>
		const output& dispatch_process(
			const input& input,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			return this->process(input);
		}

		template <const size_t TensorSize>
		const output& dispatch_process(
			const input& input,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			m_input = input;

			auto context = queue.get_context();

			initialize_opencl(context);

			reshaped_input::tensor_type rin = m_input.reshape<reshaped_input>();
			reshaped_output::tensor_type rout = m_output.reshape<reshaped_output>();

			opencl::detail::fully_connected::process(
				rin,
				m_weights,
				m_bias,
				rout,
				m_kernelProgram,
				m_processKernelName,
				context,
				queue);

			return m_output;
		}

		template <const size_t TensorSize>
		const input& dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			return this->compute_gradient(gradient);
		}

		template <const size_t TensorSize>
		const input& dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			auto context = queue.get_context();

			initialize_opencl(context);

			reshaped_input::tensor_type rin = m_input.reshape<reshaped_input>();
			reshaped_input::tensor_type rgradResult = m_gradient.reshape<reshaped_input>();
			reshaped_output::tensor_type rgrad = gradient.reshape<reshaped_output>();

			opencl::detail::fully_connected::compute_gradient(
				rin,
				m_weights,
				rgrad,
				rgradResult,
				m_weightsGradient,
				m_biasGradient,
				m_kernelProgram,
				m_gradientKernelName,
				context,
				queue);

			return m_gradient;
		}

		template <const size_t TensorSize>
		void dispatch_update_weights(
			const number_type rate,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			this->update_weights(rate);
		}

		template <const size_t TensorSize>
		void dispatch_update_weights(
			const number_type rate,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::min_matrix_size)
			>* = 0)
		{
			auto context = queue.get_context();

			initialize_opencl(context);

			opencl::detail::fully_connected::update_weights(
				m_weightsGradient,
				m_weights,
				m_biasGradient,
				m_bias,
				rate,
				m_regularization,
				m_kernelProgram,
				m_weightsKernelName,
				context,
				queue);
		}

		void initialize_opencl(
			const ::boost::compute::context& context)
		{
			if (0 == m_processKernelName.size())
			{
				m_kernelProgram = opencl::detail::layer_kernels::make_program(context);

				m_processKernelName = opencl::detail::layer_kernels::get_fully_connected_kernel_name();
				m_gradientKernelName = opencl::detail::layer_kernels::get_fully_connected_gradient_kernel_name();
				m_weightsKernelName = opencl::detail::layer_kernels::get_update_weights_kernel_name();
			}
		}

#endif

	private:
		input m_input;
		weights_type m_weights;
		weights_type m_weightsGradient;
		bias_type m_bias;
		bias_type m_biasGradient;
		number_type m_regularization;

#ifdef NEURAL_NET_ENABLE_OPEN_CL

	private:
		::boost::compute::program m_kernelProgram;
		std::string m_processKernelName;
		std::string m_gradientKernelName;
		std::string m_weightsKernelName;

#endif
	};

	template <class Input, class Output, class... Args>
	fully_connected<Input, Output> make_fully_connected_layer(
		Args&&... args)
	{
		typedef fully_connected<Input, Output> _Ltype;
		return (_Ltype(std::forward<Args>(args)...));
	}
}