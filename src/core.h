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
namespace algebra {
namespace detail {

	template <typename Metrics, typename Core, typename Stride, const size_t Rank>
	struct apply_core_with_stride
	{
		static_assert(Rank == Metrics::rank, "Rank mismatch.");
		static_assert(Core::rank == Metrics::rank, "Core rank must be the same as the input tensor rank.");
		static_assert(Core::rank == Stride::rank, "Stride rank must be the same as the core rank.");

		static_assert(Core::dimension_size <= Metrics::dimension_size, "Core dimension must be the same or smaller then the input tensor dimension.");
		static_assert(Stride::dimension_size <= Core::dimension_size, "Stride dimension must be the same or smaller then the core dimension.");

		static_assert(0 == (Metrics::dimension_size - Core::dimension_size) % Stride::dimension_size, "Current core and stride size cause some data in the input tensor to be ignored.");

		typedef typename apply_core_with_stride<
			typename Metrics::base_type, typename Core::base_type, typename Stride::base_type, (Rank - 1)>
				::metrics inner_metrics;

		typedef typename inner_metrics::
			template expand<(Metrics::dimension_size - Core::dimension_size + Stride::dimension_size) / Stride::dimension_size>
				::type metrics;
	};

	template <typename Metrics, typename Core, typename Stride>
	struct apply_core_with_stride<Metrics, Core, Stride, 1>
	{
		static_assert(1 == Metrics::rank, "Rank mismatch.");
		static_assert(Core::rank == Metrics::rank, "Core rank must be the same as the input tensor rank.");
		static_assert(Core::rank == Stride::rank, "Stride rank must be the same as the core rank.");

		static_assert(Core::dimension_size <= Metrics::dimension_size, "Core dimension must be the same or smaller then the input tensor dimension.");
		static_assert(Stride::dimension_size <= Core::dimension_size, "Stride dimension must be the same or smaller then the core dimension.");

		static_assert(0 == (Metrics::dimension_size - Core::dimension_size) % Stride::dimension_size, "Current core and stride size cause some data in the input tensor to be ignored.");

		typedef typename metrics<(Metrics::dimension_size - Core::dimension_size + Stride::dimension_size) / Stride::dimension_size> metrics;
	};
}
}
}