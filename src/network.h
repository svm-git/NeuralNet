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

#include "connected.h"
#include "activation.h"
#include "reshape.h"
#include "pooling.h"
#include "loss.h"
#include "ensemble.h"

namespace neural_network {

	template <class _Net, class _Loss>
	void _train_impl(
		_Net& net,
		const typename _Net::input& input,
		const typename _Net::output& truth,
		_Loss& loss,
		const double rate)
	{
		net.compute_gradient(
			loss.compute_gradient(
				net.process(input),
				truth));

		net.update_weights(-std::abs(rate));
	}

	template <class _Layer, class... _Args>
	class network : protected network <_Args...>
	{
	public:
		typedef typename network<_Layer, _Args...> _Self;
		typedef typename network<_Args...> _Base;

		typedef typename _Layer::input input;
		typedef typename _Base::output output;

		static_assert(std::is_same<typename _Layer::output, typename _Base::input>::value, "Output of the current layer does not match input of the next layer.");

		network(const _Layer& layer, const _Args&... args)
			: _Base(args...), m_layer(layer)
		{
		}

		const output& process(const input& input)
		{
			return _Base::process(
				m_layer.process(input));
		}

		const input& compute_gradient(const output& grad)
		{
			return m_layer.compute_gradient(
				_Base::compute_gradient(grad));
		}

		void update_weights(
			const double rate)
		{
			_Base::update_weights(rate);
			m_layer.update_weights(rate);
		}

		template <class _Loss>
		void train(
			const typename input& input,
			const typename output& truth,
			_Loss& loss,
			const double rate)
		{
			_train_impl(*this, input, truth, loss, rate);
		}

	private:
		_Layer m_layer;
	};

	template <class _Layer>
	class network<_Layer>
	{
	public:
		typedef typename network<_Layer> _Self;

		typedef typename _Layer::input input;
		typedef typename _Layer::output output;

		network(const _Layer& layer)
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
			const double rate)
		{
			m_layer.update_weights(rate);
		}

		template <class _Loss>
		void train(
			const typename input& input,
			const typename output& truth,
			_Loss& loss,
			const double rate)
		{
			_train_impl(*this, input, truth, loss, rate);
		}

	private:
		_Layer m_layer;
	};

	template <class... _Layers>
	network<typename std::_Unrefwrap<_Layers>::type...> make_network(
		_Layers&&... args)
	{
		typedef network<typename std::_Unrefwrap<_Layers>::type...> _Ntype;
		return (_Ntype(std::forward<_Layers>(args)...));
	}
}