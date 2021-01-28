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

#include "opencl/activation.h"

#endif

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

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		void update_weights(
			const number_type,
			::boost::compute::command_queue&)
		{}

#endif

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
#ifdef NEURAL_NET_ENABLE_OPEN_CL
			, m_kernelProgram(), m_activationKernelName(), m_gradientKernelName()
#endif
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
				m_output,
				m_gradient,
				[](const number_type& g, const number_type& o)
				{
					return (o > 0.0f) ? g : 0.0f;
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

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		const output& process(
			const input& input,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_process<input::data_size>(input, queue);
		}

		const input& compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_compute_gradient<input::data_size>(gradient, queue);
		}
	
	private:
		template <const size_t TensorSize>
		const output&  dispatch_process(
			const input& input,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			return this->process(input);
		}

		template <const size_t TensorSize>
		const output&  dispatch_process(
			const input& input,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			m_input = input;

			auto context = queue.get_context();

			initialize_opencl(context);

			opencl::detail::generic_activation::process(
				m_input,
				m_output,
				m_kernelProgram,
				m_activationKernelName,
				context,
				queue);

			return m_output;
		}

		template <const size_t TensorSize>
		const input&  dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			return this->compute_gradient(gradient);
		}

		template <const size_t TensorSize>
		const input&  dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			auto context = queue.get_context();

			initialize_opencl(context);

			opencl::detail::generic_activation::compute_gradient(
				m_output,
				gradient,
				m_gradient,
				m_kernelProgram,
				m_gradientKernelName,
				context,
				queue);

			return m_gradient;
		}

		void initialize_opencl(
			const ::boost::compute::context& context)
		{
			if (0 == m_activationKernelName.size())
			{
				m_kernelProgram = opencl::detail::layer_kernels::make_program(context);

				m_activationKernelName = opencl::detail::layer_kernels::get_relu_kernel_name();
				m_gradientKernelName = opencl::detail::layer_kernels::get_relu_gradient_kernel_name();
			}
		}

	private:
		::boost::compute::program m_kernelProgram;
		std::string m_activationKernelName;
		std::string m_gradientKernelName;
#endif
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
#ifdef NEURAL_NET_ENABLE_OPEN_CL
			, m_kernelProgram(), m_activationKernelName(), m_gradientKernelName()
#endif
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
				m_output,
				m_gradient,
				[](const number_type& g, const number_type& o)
			{
				return g * o * (1.0f - o);
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

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		const output& process(
			const input& input,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_process<input::data_size>(input, queue);
		}

		const input& compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue)
		{
			return this->dispatch_compute_gradient<input::data_size>(gradient, queue);
		}
	
	private:
		template <const size_t TensorSize>
		const output&  dispatch_process(
			const input& input,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			return this->process(input);
		}

		template <const size_t TensorSize>
		const output& dispatch_process(
			const input& input,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			m_input = input;

			auto context = queue.get_context();

			initialize_opencl(context);

			opencl::detail::generic_activation::process(
				m_input,
				m_output,
				m_kernelProgram,
				m_activationKernelName,
				context,
				queue);

			return m_output;
		}

		template <const size_t TensorSize>
		const input&  dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue&,
			std::enable_if_t<
				(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			return this->compute_gradient(gradient);
		}

		template <const size_t TensorSize>
		const input&  dispatch_compute_gradient(
			const output& gradient,
			::boost::compute::command_queue& queue,
			std::enable_if_t<
				!(TensorSize < opencl::detail::layer_kernels::block_size)
			>* = 0)
		{
			auto context = queue.get_context();

			initialize_opencl(context);

			opencl::detail::generic_activation::compute_gradient(
				m_output,
				gradient,
				m_gradient,
				m_kernelProgram,
				m_gradientKernelName,
				context,
				queue);

			return m_gradient;
		}

		void initialize_opencl(
			const ::boost::compute::context& context)
		{
			if (0 == m_activationKernelName.size())
			{
				m_kernelProgram = opencl::detail::layer_kernels::make_program(
					context);

				m_activationKernelName = opencl::detail::layer_kernels::get_logistic_kernel_name();
				m_gradientKernelName = opencl::detail::layer_kernels::get_logistic_gradient_kernel_name();
			}
		}

	private:
		::boost::compute::program m_kernelProgram;
		std::string m_activationKernelName;
		std::string m_gradientKernelName;

#endif

	private:
		static number_type logistic(const number_type& x)
		{
			if (x > 0.0f)
			{
				return (1.0f / (1.0f + std::exp(-x)));
			}
			else
			{
				number_type e = std::exp(x);
				return e / (1.0f + e);
			}
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
