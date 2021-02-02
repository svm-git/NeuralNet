/*

Copyright (c) 2020-2021 svm-git

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

#include <iostream>

#include "tensor.h"

namespace neural_network {
namespace serialization {

	enum chunk_types : size_t
	{
		none = 0x0,

		tensor,

		reshape_layer,

		relu_activation_layer,

		logistic_activation_layer,

		convolution_layer,

		max_pooling_layer,

		max_pooling_with_core_layer,

		fully_connected_layer,

		ensemble_layer,

		tanh_activation_layer,
	};

namespace detail {

	struct serializer_base
	{
		static void throw_io_error(const char* message)
		{
			throw std::ios_base::failure(message);
		}
	};

	template <typename Metrics, const size_t Rank>
	struct metrics_serializer_impl : public detail::serializer_base
	{
		typedef typename metrics_serializer_impl<typename Metrics::base_type, (Rank - 1)> base_type;
		typedef typename base_type::value_type value_type;

		enum : size_t { serialized_data_size = base_type::serialized_data_size + sizeof(value_type) };

		static void read(
			std::istream& in)
		{
			value_type dim;
			if (!in.read(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value_type)))
				throw_io_error("Failed to read metrics dimension value.");

			base_type::read(in);
		}

		static void write(
			std::ostream& out)
		{
			value_type dim = Metrics::dimension_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value_type)))
				throw_io_error("Failed to write metrics dimension value.");

			base_type::write(out);
		}
	};

	template <typename Metrics>
	struct metrics_serializer_impl<Metrics, 1> : public detail::serializer_base
	{
		typedef unsigned int value_type;
		enum : size_t { serialized_data_size = sizeof(value_type) };

		static void read(
			std::istream& in)
		{
			value_type dim;
			if (!in.read(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value_type)))
				throw_io_error("Failed to read metrics dimension value.");
		}

		static void write(
			std::ostream& out)
		{
			value_type dim = Metrics::dimension_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value_type)))
				throw_io_error("Failed to write metrics dimension value.");
		}
	};

}

	template <typename Value>
	struct value_serializer : public detail::serializer_base
	{
		static_assert(std::is_arithmetic<Value>::value, "Only primitive arithmetic types are supported.");

		typedef Value value_type;
		enum : size_t { serialized_data_size = sizeof(value_type) };

		static void read(
			std::istream& in,
			value_type& result)
		{
			if (!in.read(reinterpret_cast<char*>(std::addressof(result)), sizeof(result)))
				throw_io_error("Failed to read value.");
		}

		static void write(
			std::ostream& out,
			const value_type& val)
		{
			if (!out.write(reinterpret_cast<const char*>(std::addressof(val)), sizeof(value_type)))
				throw_io_error("Failed to write value.");
		}
	};

	template <const chunk_types ChunkType, typename ValueSerializer>
	struct chunk_serializer : public detail::serializer_base
	{
		typedef chunk_serializer<ChunkType, ValueSerializer> this_type;
		typedef typename ValueSerializer::value_type value_type;

		enum : size_t { serialized_data_size = 2 * sizeof(unsigned int) + ValueSerializer::serialized_data_size };

		template <class... Values>
		static void read(
			std::istream& in,
			Values&... args)
		{
			unsigned int tmp = 0;
			if (!in.read(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				throw_io_error("Failure to read chunk size.");

			if (tmp != this_type::serialized_data_size)
				throw_io_error("Invalid chunk size.");

			if (!in.read(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				throw_io_error("Failure to read chunk type.");

			if (tmp != ChunkType)
				throw_io_error("Invalid chunk size.");

			ValueSerializer::read(in, args...);
		}

		template <class... Values>
		static void write(
			std::ostream& out,
			const Values&... args)
		{
			unsigned int tmp = this_type::serialized_data_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				throw_io_error("Failure to write chunk size.");

			tmp = ChunkType;
			if (!out.write(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				throw_io_error("Failure to write chunk type.");

			ValueSerializer::write(out, args...);
		}
	};

	template <typename Metrics>
	struct metrics_serializer : public detail::serializer_base
	{
		typedef typename detail::metrics_serializer_impl<Metrics, Metrics::rank> Impl;

		typedef unsigned int value_type;
		enum : size_t { serialized_data_size = Impl::serialized_data_size + sizeof(value_type) };

		static void read(
			std::istream& in)
		{
			value_type rank;
			if (!in.read(reinterpret_cast<char*>(std::addressof(rank)), sizeof(value_type)))
				throw_io_error("Failed to read metrics rank value.");

			if (Metrics::rank != rank)
				throw_io_error("Incompatible metrics rank value.");

			Impl::read(in);
		}

		static void write(
			std::ostream& out)
		{
			value_type rank = Metrics::rank;
			if (!out.write(reinterpret_cast<char*>(std::addressof(rank)), sizeof(value_type)))
				throw_io_error("Failed to write metrics rank value.");

			Impl::write(out);
		}
	};

	template <typename Tensor>
	struct tensor_serializer : public detail::serializer_base
	{
		typedef typename metrics_serializer<typename Tensor::metrics> MetricsSerializer;
		typedef typename Tensor::number_type number_type;

		enum : size_t { serialized_data_size = MetricsSerializer::serialized_data_size + sizeof(number_type) * Tensor::data_size };

		typedef typename Tensor value_type;

		static void read(
			std::istream& in,
			value_type& result)
		{
			MetricsSerializer::read(in);

			typedef typename neural_network::algebra::metrics<Tensor::data_size>::tensor_type flat;
			flat flatValue;

			for (size_t i = 0; i < flatValue.size<0>(); ++i)
			{
				if (!in.read(reinterpret_cast<char*>(std::addressof(flatValue(i))), sizeof(number_type)))
					throw_io_error("Failed to read tensor element value.");
			}

			result = flatValue.reshape<typename Tensor::metrics>();
		}

		static void write(
			std::ostream& out,
			const value_type& tensor)
		{
			MetricsSerializer::write(out);

			typedef typename neural_network::algebra::metrics<Tensor::data_size> flat;
			flat::tensor_type flatValue = tensor.reshape<flat>();

			for (size_t i = 0; i < flatValue.size<0>(); ++i)
			{
				if (!out.write(reinterpret_cast<char*>(std::addressof(flatValue(i))), sizeof(number_type)))
					throw_io_error("Failed to write tensor element value.");
			}
		}
	};

	template <class Serializer, class... Args>
	struct composite_serializer : public composite_serializer<Args...>
	{
		typedef typename composite_serializer<Serializer, Args...> this_type;
		typedef typename composite_serializer<Args...> base_type;

		typedef typename Serializer::value_type value_type;
		typedef typename Serializer serializer;

		enum : size_t { serialized_data_size = Serializer::serialized_data_size + base_type::serialized_data_size };

		template <class... Values>
		static void read(
			std::istream& in,
			value_type& val,
			Values&... args)
		{
			serializer::read(in, val);

			base_type::read(in, args...);
		}

		template <class... Values>
		static void write(
			std::ostream& out,
			const value_type& val,
			const Values&... args)
		{
			serializer::write(out, val);

			base_type::write(out, args...);
		}
	};

	template <class Serializer>
	struct composite_serializer<Serializer> : public detail::serializer_base
	{
		typedef typename composite_serializer<Serializer> this_type;

		enum : size_t { serialized_data_size = Serializer::serialized_data_size };

		typedef typename Serializer::value_type value_type;
		typedef typename Serializer serializer;

		static void read(
			std::istream& in,
			value_type& val)
		{
			serializer::read(in, val);
		}

		static void write(
			std::ostream& out,
			const value_type& val)
		{
			serializer::write(out, val);
		}
	};

	template <class Layer>
	void read(
		std::istream& input,
		Layer& layer)
	{
		Layer::serializer::read(input, layer);
	}

	template <class Layer>
	void write(
		std::ostream& output,
		const Layer& layer)
	{
		Layer::serializer::write(output, layer);
	}

	template <class Layer>
	size_t model_size(const Layer&)
	{
		return Layer::serializer::serialized_data_size;
	}
}
}