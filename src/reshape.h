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

	template <typename _InputMetrics, typename _OutputMetrics>
	class reshape : public layer_base < _InputMetrics, _OutputMetrics >
	{
	public:
		typedef typename reshape<_InputMetrics, _OutputMetrics> _Self;
		typedef typename layer_base<_InputMetrics, _OutputMetrics> _Base;

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
			const double rate)
		{}
	};
}