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

namespace neural_network { namespace algebra {

	template <typename _Metrics, const size_t _Rank>
	struct _shrink
	{
		static_assert(_Rank < _Metrics::rank, "Target rank is smaller than metrics rank.");

		typedef typename std::conditional <
			_Rank == 0,
			typename _Metrics,
			typename _shrink<typename _Metrics::_Base, (_Rank - 1)>::metrics >::type metrics;
	};

	template <typename _Metrics>
	struct _shrink<_Metrics, 0>
	{
		typedef typename _Metrics metrics;
	};

	template <typename _Metrics, typename _Core, const size_t _Rank>
	struct _is_valid_submetrics
	{
		typedef typename std::conditional<
			_Metrics::dimension_size >= _Core::dimension_size,
			typename _is_valid_submetrics<typename _Metrics::_Base, typename _Core::_Base, (_Rank - 1)>::type,
			std::false_type
		>::type type;
	};

	template <typename _Metrics, typename _Core>
	struct _is_valid_submetrics<_Metrics, _Core, 1>
	{
		typedef typename std::conditional<
			_Metrics::dimension_size >= _Core::dimension_size,
			std::true_type,
			std::false_type>::type type;
	};

	template <typename _Metrics, typename _Core>
	struct _shrink_to_core
	{
		static_assert(_Core::rank <= _Metrics::rank, "Core rank must be same or smaller than metrics rank to shrink.");

		typedef typename _shrink<_Metrics, _Metrics::rank - _Core::rank>::metrics metrics;

		static_assert(_is_valid_submetrics<metrics, _Core, metrics::rank>::type::value, "Dimensions of core are larger than dimensions of the metrics.");
	};

	template <typename _Metrics, typename _Core>
	struct _reshape_to_core
	{
		typedef typename std::conditional<
			_Metrics::rank == _Core::rank,
			typename _Metrics,
			typename _shrink_to_core<_Metrics, _Core>::metrics
		>::type _inner;

		static_assert(0 == (_Metrics::data_size % _inner::data_size), "Metrics computations are incorrect.");

		typedef typename _inner::template expand<_Metrics::data_size / _inner::data_size>::type metrics;
	};

	template <typename _Metrics, typename _Core, typename _Stride, const size_t _Rank>
	struct _apply_core_with_stride
	{
		static_assert(_Rank == _Metrics::rank, "Rank mismatch.");
		static_assert(_Core::rank == _Metrics::rank, "Core rank must be the same as the input tensor rank.");
		static_assert(_Core::rank == _Stride::rank, "Stride rank must be the same as the core rank.");

		static_assert(_Core::dimension_size <= _Metrics::dimension_size, "Core dimension must be the same or smaller then the input tensor dimension.");
		static_assert(_Stride::dimension_size <= _Core::dimension_size, "Stride dimension must be the same or smaller then the core dimension.");

		static_assert(0 == (_Metrics::dimension_size - _Core::dimension_size) % _Stride::dimension_size, "Current core and stride size cause some data in the input tensor to be ignored.");

		typedef typename _apply_core_with_stride<
			typename _Metrics::_Base, typename _Core::_Base, typename _Stride::_Base, (_Rank - 1)>
				::metrics _inner;

		typedef typename _inner::
			template expand<(_Metrics::dimension_size - _Core::dimension_size + _Stride::dimension_size) / _Stride::dimension_size>
				::type metrics;
	};

	template <typename _Metrics, typename _Core, typename _Stride>
	struct _apply_core_with_stride<_Metrics, _Core, _Stride, 1>
	{
		static_assert(1 == _Metrics::rank, "Rank mismatch.");
		static_assert(_Core::rank == _Metrics::rank, "Core rank must be the same as the input tensor rank.");
		static_assert(_Core::rank == _Stride::rank, "Stride rank must be the same as the core rank.");

		static_assert(_Core::dimension_size <= _Metrics::dimension_size, "Core dimension must be the same or smaller then the input tensor dimension.");
		static_assert(_Stride::dimension_size <= _Core::dimension_size, "Stride dimension must be the same or smaller then the core dimension.");

		static_assert(0 == (_Metrics::dimension_size - _Core::dimension_size) % _Stride::dimension_size, "Current core and stride size cause some data in the input tensor to be ignored.");

		typedef typename metrics<(_Metrics::dimension_size - _Core::dimension_size + _Stride::dimension_size) / _Stride::dimension_size> metrics;
	};
}
}