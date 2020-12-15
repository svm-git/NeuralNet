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
	
	template <class _Network, class _Output>
	void _process_and_copy_network_result(
		_Network& network,
		const size_t index,
		const typename _Network::input& input,
		_Output& output)
	{
		typedef typename typename _Network::output _local_output;
		typedef typename algebra::metrics<_local_output::data_size> _reshaped_local_output;
		typedef typename _Output::metrics::shrink::type _shrink;

		typedef typename algebra::metrics<_Output::dimension_size, _shrink::data_size> _reshaped_output;

		auto localResult = network
			.process(input)
			.reshape<_reshaped_local_output>();

		auto reshaped_output = output.reshape<_reshaped_output>();
		
		for (size_t i = 0; i < localResult.size<0>(); ++i)
		{
			// Reshaped tensors share the same data, therefore
			// data in 'output' tensor is updated by this loop.
			reshaped_output(index, i) = localResult(i);
		}
	}

	template <class _Network, class _Gradient>
	void _compute_gradient_and_add_result(
		_Network& network,
		const size_t index,
		const _Gradient& grad,
		typename _Network::output& local,
		typename _Network::input& result)
	{
		typedef typename _Network::input _result;
		typedef typename _Network::output _local_gradient;
		typedef typename algebra::metrics<_local_gradient::data_size> _reshaped_local;
		typedef typename _Gradient::metrics::shrink::type _shrink;

		typedef typename algebra::metrics<_Gradient::dimension_size, _shrink::data_size> _reshaped_gradient;

		auto gradient = grad.reshape<_reshaped_gradient>();
		auto localGradient = local.reshape<_reshaped_local>();

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
			[](const double& l, const double r) { return r + l; });
	}

	template <class _N, class... _Args>
	class _network_ensemble_impl : protected _network_ensemble_impl<_Args...>
	{
	public:
		typedef typename _network_ensemble_impl<_N, _Args...> _Self;
		typedef typename _network_ensemble_impl<_Args...> _Base;

		static_assert(std::is_same<typename _N::input, typename _Base::input>::value, "Network input types do not match.");
		static_assert(std::is_same<typename _N::output, typename _Base::common_output>::value, "Network output types do not match.");

		enum : size_t { ensemble_size = _Base::ensemble_size + 1 };

		typedef typename _N::input input;
		typedef typename _Base::common_output common_output;
		typedef typename _Base::common_output::metrics::template expand<ensemble_size>::type::tensor_type output;

		_network_ensemble_impl(const _N& n, const _Args&... args)
			: _Base(args...), m_network(n)
		{
		}

		template <class _Output>
		void process(
			const input& input,
			_Output& output)
		{
			_process_and_copy_network_result(
				m_network,
				_Self::ensemble_size - 1,
				input,
				output);

			_Base::process(input, output);
		}

		template <class _Output, class _LocalGradient, class _Gradient>
		void compute_gradient(
			const _Output& grad,
			_LocalGradient& local,
			_Gradient& result)
		{
			_compute_gradient_and_add_result(
				m_network,
				_Self::ensemble_size - 1,
				grad,
				local,
				result);

			_Base::compute_gradient(grad, local, result);
		}

		void update_weights(
			const double rate)
		{
			m_network.update_weights(rate);

			_Base::update_weights(rate);
		}

	private:
		_N m_network;
	};

	template <class _N>
	class _network_ensemble_impl<_N>
	{
	public:
		typedef typename _network_ensemble_impl<_N> _Self;

		enum : size_t { ensemble_size = 1 };

		typedef typename _N::input input;
		typedef typename _N::output common_output;
		typedef typename _N::output::metrics::template expand<ensemble_size>::type::tensor_type output;

		_network_ensemble_impl(const _N& n)
			: m_network(n)
		{
		}

		template <class _Output>
		void process(
			const input& input,
			_Output& output)
		{
			_process_and_copy_network_result(
				m_network,
				_Self::ensemble_size - 1,
				input,
				output);
		}

		template <class _Output, class _LocalGradient, class _Gradient>
		void compute_gradient(
			const _Output& grad,
			_LocalGradient& local,
			_Gradient& result)
		{
			_compute_gradient_and_add_result(
				m_network,
				_Self::ensemble_size - 1,
				grad,
				local,
				result);
		}

		void update_weights(
			const double rate)
		{
			m_network.update_weights(rate);
		}

	private:
		_N m_network;
	};
	
	template <class _N1, class _N2, class... _Args>
	class network_ensemble
	{
	public:
		typedef typename network_ensemble<_N1, _N2, _Args...> _Self;
		typedef typename _network_ensemble_impl<_N1, _N2, _Args...> _Ensemble;

		typedef typename _Ensemble::input input;
		typedef typename _Ensemble::output output;

		network_ensemble(const _N1& n1, const _N2& n2, const _Args&... args)
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
			m_gradient.fill(0.0);
			m_ensemble.compute_gradient(grad, m_local, m_gradient);
			return m_gradient;
		}

		void update_weights(
			const double rate)
		{
			m_ensemble.update_weights(rate);
		}

	private:
		_Ensemble m_ensemble;
		output m_output;
		input m_gradient;
		typename _Ensemble::common_output m_local;
	};

	template <class... _Networks>
	network_ensemble<typename std::_Unrefwrap<_Networks>::type...> make_ensemble(
		_Networks&&... args)
	{
		typedef network_ensemble<typename std::_Unrefwrap<_Networks>::type...> _Ntype;
		return (_Ntype(std::forward<_Networks>(args)...));
	}
}
