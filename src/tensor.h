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

#ifdef NEURAL_NET_ENABLE_OPEN_CL

#pragma warning (push)
#pragma warning (disable: 4127 4512)

#include <boost/compute/core.hpp>
#include <boost/compute/container/mapped_view.hpp>

#pragma warning (pop)

#endif

namespace neural_network {
namespace algebra {

namespace detail {

	template <class Metrics, const size_t Dimension>
	struct dimension
	{
		typedef typename std::conditional <
			Dimension == 0,
			typename Metrics,
			typename dimension<typename Metrics::base_type, (Dimension - 1)>::metrics >::type metrics;

		enum { size = metrics::dimension_size };
	};

	template <class Metrics>
	struct dimension<Metrics, 0>
	{
		typedef typename Metrics metrics;
		enum { size = metrics::dimension_size };
	};

}

	template <const size_t... Metrics>
	class tensor;

	template <const size_t Size, const size_t... Args>
	struct metrics : public metrics<Args...>
	{
	public:
		static_assert(0 < Size, "0-size metrics are not supported.");

		typedef typename metrics<Size, Args...> this_type;
		typedef typename metrics<Args...> base_type;

		typedef typename tensor< Size, Args...> tensor_type;

		enum { 
			rank = base_type::rank + 1,
			dimension_size = Size,
			data_size = Size * base_type::data_size };

		template <const size_t Dimension>
		struct expand
		{
			typedef typename metrics<Dimension, Size, Args...> type;
		};

		struct shrink
		{
			typedef typename base_type type;
		};

		template <typename ...IndexArgs>
		static bool is_valid_index(const size_t index, IndexArgs... args)
		{
			return (index < Size) && base_type::is_valid_index(args...);
		}

		template <typename ...IndexArgs>
		static size_t offset(const size_t index, IndexArgs... args)
		{
			return index * base_type::data_size + base_type::offset(args...);
		}
	};

	template <const size_t Size>
	class metrics<Size>
	{
	public:
		static_assert(0 < Size, "0-size metrics are not supported.");

		typedef typename metrics<Size> this_type;
		typedef typename metrics<Size> base_type;

		typedef typename tensor< Size> tensor_type;

		enum { 
			rank = 1,
			dimension_size = Size,
			data_size = Size };

		template <const size_t Dimension>
		struct expand
		{
			typedef typename metrics<Dimension, Size> type;
		};

		static bool is_valid_index(const size_t index)
		{
			return (index < Size);
		}

		static size_t offset(const size_t index)
		{
			return index;
		}
	};

	template <const size_t... Metrics>
	class tensor
	{
	public:
		typedef typename tensor<Metrics...> this_type;
		typedef typename metrics<Metrics...> metrics;

		enum { 
			rank = metrics::rank,
			dimension_size = metrics::dimension_size,
			data_size = metrics::data_size };

		typedef float number_type;
		typedef typename std::array<number_type, data_size> buffer_type;
		typedef typename std::shared_ptr<buffer_type> buffer_ptr;

		tensor()
			: m_pData(std::make_shared<buffer_type>())
		{
			m_pData->fill(0.0f);
		}

		tensor(std::function<number_type()> initializer)
			: m_pData(std::make_shared<buffer_type>())
		{
			std::generate(
				m_pData->begin(), m_pData->end(),
				initializer);
		}

		tensor(const this_type& other)
			: m_pData(other.m_pData)
		{}

		tensor(const buffer_ptr& ptr)
			: m_pData(ptr)
		{}

		this_type& operator=(const this_type& other)
		{
			m_pData = other.m_pData;
			return (*this);
		}

		template <typename ...IndexArgs>
		const number_type& operator()(IndexArgs... idx) const
		{
			if (false == metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->cbegin() + metrics::offset(idx...));
		}

		template <typename ...IndexArgs>
		number_type& operator()(IndexArgs... idx)
		{
			if (false == metrics::is_valid_index(idx...))
				throw std::invalid_argument("Index out of range.");

			return *(m_pData->begin() + metrics::offset(idx...));
		}

		template <const size_t Dimension>
		const size_t size() const
		{
			static_assert(Dimension < this_type::rank, "Requested dimension is larger than tensor rank.");

			return detail::dimension<metrics, Dimension>::size;
		}

		template <class Operator>
		void transform(this_type& dst, Operator op) const
		{
			std::transform(
				m_pData->cbegin(), m_pData->cend(),
				dst.m_pData->begin(),
				op);
		}

		template <class Operator>
		void transform(const this_type& other, this_type& dst, Operator op) const
		{
			std::transform(
				m_pData->cbegin(), m_pData->cend(),
				other.m_pData->cbegin(),
				dst.m_pData->begin(),
				op);
		}

		template <class Other>
		typename Other::tensor_type reshape() const
		{
			static_assert(metrics::data_size == Other::data_size, "Reshape data size must match this data size.");

			return Other::tensor_type(m_pData);
		}

		void fill(const number_type val)
		{
			m_pData->fill(val);
		}

#ifdef NEURAL_NET_ENABLE_OPEN_CL

		::boost::compute::mapped_view<number_type> get_device_view(
			const ::boost::compute::context& context)
		{
			return ::boost::compute::mapped_view<number_type>(
				std::addressof((*m_pData)[0]),
				this_type::data_size,
				context);
		}

		::boost::compute::mapped_view<number_type> get_device_view(
			const ::boost::compute::context& context) const
		{
			return ::boost::compute::mapped_view<number_type>(
				std::addressof((*m_pData)[0]),
				this_type::data_size,
				context);
		}

#endif

	private:
		std::shared_ptr<buffer_type> m_pData;
	};

}
}