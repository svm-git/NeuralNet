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

	template <typename ValueMetrics>
	class squared_error_loss
	{
	public:
		typedef typename squared_error_loss<ValueMetrics> this_type;
		typedef typename ValueMetrics::tensor_type tensor_type;
		typedef typename tensor_type::number_type number_type;

		squared_error_loss()
			: m_gradient()
		{}

		const number_type compute(
			const tensor_type& result,
			const tensor_type& truth)
		{
			number_type loss = 0.0f;

			tensor_type tmp;
			result.transform(
				truth,
				tmp,
				[&loss](const number_type& r, const number_type& t)
				{
					auto delta = (r - t);
					loss += (delta * delta);
					return r;
				});

			return loss;
		}

		const tensor_type& compute_gradient(
			const tensor_type& result,
			const tensor_type& truth)
		{
			result.transform(
				truth,
				m_gradient,
				[](const number_type& r, const number_type& t)
				{
					return (r - t);
				});

			return m_gradient;
		}
	
	private:
		tensor_type m_gradient;
	};
}
