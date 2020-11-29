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
#include <functional>

namespace neural_network { namespace algebra {

	template <class _Metrics, const size_t _Dim>
	struct _dimension
	{
		typedef typename std::conditional <
			_Dim == 0,
			typename _Metrics,
			typename _dimension<typename _Metrics::_Base, (_Dim - 1)>::metrics >::type metrics;

		enum { size = metrics::dimension_size };
	};

	template <class _Metrics>
	struct _dimension<_Metrics, 0>
	{
		typedef typename _Metrics metrics;
		enum { size = metrics::dimension_size };
	};

	template <const size_t... _Metrics>
	class tensor;

	template <const size_t _Size, const size_t... _Args>
	struct metrics : public metrics<_Args...>
	{
	public:
		static_assert(0 < _Size, "0-size metrics are not supported.");

		typedef typename metrics<_Size, _Args...> _Self;
		typedef typename metrics<_Args...> _Base;

		typedef typename tensor< _Size, _Args...> tensor_type;

		enum { 
			rank = _Base::rank + 1,
			dimension_size = _Size,
			data_size = _Size * _Base::data_size };

		template <typename ..._Other>
		static bool is_valid_index(const size_t index, _Other... args)
		{
			return (index < _Size) && _Base::is_valid_index(args...);
		}

		template <typename ..._Other>
		static size_t offset(const size_t index, _Other... args)
		{
			return index * _Base::data_size + _Base::offset(args...);
		}
	};

	template <const size_t _Size>
	class metrics<_Size>
	{
	public:
		static_assert(0 < _Size, "0-size metrics are not supported.");

		typedef typename metrics<_Size> _Self;
		typedef typename metrics<_Size> _Base;

		typedef typename tensor< _Size> tensor_type;

		enum { 
			rank = 1,
			dimension_size = _Size,
			data_size = _Size };

		static bool is_valid_index(const size_t index)
		{
			return (index < _Size);
		}

		static size_t offset(const size_t index)
		{
			return index;
		}
	};

	template <const size_t... _Metrics>
	class tensor
	{
	public:
		typedef typename tensor<_Metrics...> _Self;
		typedef typename metrics<_Metrics...> metrics;

		enum { 
			rank = metrics::rank,
			dimension_size = metrics::dimension_size,
			data_size = metrics::data_size };

		typedef typename std::array<double, data_size> _Data;
		typedef typename std::shared_ptr<_Data> _DataPtr;

		tensor()
			: m_pData(std::make_shared<_Data>())
		{
			m_pData->fill(0.0);
		}

		tensor(std::function<double(const double&)> initializer)
			: m_pData(std::make_shared<_Data>())
		{
			std::transform(
				m_pData->cbegin(), m_pData->cend(),
				m_pData->begin(),
				initializer);
		}

		tensor(const _Self& other)
			: m_pData(other.m_pData)
		{}

		tensor(const _DataPtr& ptr)
			: m_pData(ptr)
		{}

		_Self& operator=(const _Self& other)
		{
			m_pData = other.m_pData;
			return (*this);
		}

		template <typename ..._Idx>
		const double& operator()(_Idx... idx) const
		{
			if (false == metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->cbegin() + metrics::offset(idx...));
		}

		template <typename ..._Idx>
		double& operator()(_Idx... idx)
		{
			if (false == metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->begin() + metrics::offset(idx...));
		}

		template <const size_t _Dim>
		const size_t size() const
		{
			static_assert(_Dim < _Self::rank, "Requested dimension is larger than tensor rank.");

			return _dimension<metrics, _Dim>::size;
		}

		template <class Operator>
		void transform(_Self& dst, Operator op) const
		{
			std::transform(
				m_pData->cbegin(), m_pData->cend(),
				dst.m_pData->begin(),
				op);
		}

		template <class Operator>
		void transform(const _Self& other, _Self& dst, Operator op) const
		{
			std::transform(
				m_pData->cbegin(), m_pData->cend(),
				other.m_pData->cbegin(),
				dst.m_pData->begin(),
				op);
		}

		template <class _Other>
		typename _Other::tensor_type reshape() const
		{
			static_assert(metrics::data_size == _Other::data_size, "Reshape data size must match this data size.");

			return _Other::tensor_type(m_pData);
		}

	private:
		std::shared_ptr<_Data> m_pData;
	};
}
}