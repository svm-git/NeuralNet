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

#include "tensor.h"

namespace neural_network {

	template <typename InputMetrics, typename OutputMetrics>
	class layer_base
	{
	public:
		typedef typename InputMetrics::tensor_type input;
		typedef typename OutputMetrics::tensor_type output;

		static_assert(
			std::is_same<typename input::number_type, typename output::number_type>::value,
			"Input and output tensor value types do not match.");

		typedef typename input::number_type number_type;

		layer_base()
			: m_output(), m_gradient()
		{}

		const output& get_output() const
		{
			return m_output;
		}

		const input& get_gradient() const
		{
			return m_gradient;
		}

	protected:
		output m_output;
		input m_gradient;
	};
}