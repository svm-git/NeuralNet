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

#include <memory>
#include <array>

namespace neural_network { namespace algebra {

	template <const size_t _Size, const size_t... _Args>
	struct _metrics : public _metrics<_Args...>
	{
	public:
		typedef typename _metrics<_Size, _Args...> _Self;
		typedef typename _metrics<_Args...> _Base;

		enum { rank = _Base::rank + 1, size = _Size * _Base::size };

		template <typename ..._Other>
		static bool is_valid_index(const size_t index, _Other... args)
		{
			return (index < _Size) && _Base::is_valid_index(args...);
		}

		template <typename ..._Other>
		static size_t offset(const size_t index, _Other... args)
		{
			return index * _Base::size + _Base::offset(args...);
		}

		template <const size_t _Dim>
		static size_t dimension_size()
		{
			return _Dim == 0 ? _Size : _Base::dimension_size<(_Dim - 1)>();
		}
	};

	template <const size_t _Size>
	class _metrics<_Size>
	{
	public:
		typedef typename _metrics<_Size> _Self;

		enum { rank = 1, size = _Size };

		static bool is_valid_index(const size_t index)
		{
			return (index < _Size);
		}

		static size_t offset(const size_t index)
		{
			return index;
		}

		template <const size_t _Dim>
		static size_t dimension_size()
		{
			return _Dim == 0 ? _Size : 0;
		}
	};

	template <const size_t _Size, const size_t... _Args>
	class tensor
	{
	public:
		typedef typename _metrics<_Size, _Args...> _Metrics;

		enum { rank = _Metrics::rank, data_size = _Metrics::size };

		typedef typename std::array<double, data_size> _Data;

		tensor()
			: m_pData(std::make_shared<_Data>())
		{
			m_pData->fill(0.0);
		}

		template <typename ..._Idx>
		const double& operator()(_Idx... idx) const
		{
			if (false == _Metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->cbegin() + _Metrics::offset(idx...));
		}

		template <typename ..._Idx>
		double& operator()(_Idx... idx)
		{
			if (false == _Metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->begin() + _Metrics::offset(idx...));
		}

		template <const size_t _Dim>
		const size_t size() const
		{
			return _Metrics::dimension_size< _Dim >();
		}

	private:
		std::shared_ptr<_Data> m_pData;
	};
}
}