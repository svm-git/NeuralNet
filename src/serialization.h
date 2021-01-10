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

#include <iostream>

#include "tensor.h"

namespace neural_network { namespace serialization {

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
	};

	struct _serializer_base
	{
		static void _throw_io_error(const char* message)
		{
			throw std::ios_base::failure(message);
		}
	};

	template <typename _Value>
	struct value_serializer : public _serializer_base
	{
		static_assert(std::is_arithmetic<_Value>::value, "Only primitive arithmetic types are supported.");

		enum : size_t { serialized_data_size = sizeof(_Value)};
		typedef _Value value;

		static void read(
			std::istream& in,
			value& result)
		{
			if (!in.read(reinterpret_cast<char*>(std::addressof(result)), sizeof(result)))
				_throw_io_error("Failed to read value.");
		}

		static void write(
			std::ostream& out,
			const value& val)
		{
			if (!out.write(reinterpret_cast<const char*>(std::addressof(val)), sizeof(_Value)))
				_throw_io_error("Failed to write value.");
		}
	};

	template <const chunk_types _Type, typename _ValueSerializer>
	struct chunk_serializer : public _serializer_base
	{
		typedef chunk_serializer<_Type, _ValueSerializer> _Self;
		typedef typename _ValueSerializer::value value;

		enum : size_t { serialized_data_size = 2 * sizeof(unsigned int) + _ValueSerializer::serialized_data_size };

		template <class... _Values>
		static void read(
			std::istream& in,
			_Values&... args)
		{
			unsigned int tmp = 0;
			if (!in.read(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				_throw_io_error("Failure to read chunk size.");

			if (tmp != _Self::serialized_data_size)
				_throw_io_error("Invalid chunk size.");

			if (!in.read(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				_throw_io_error("Failure to read chunk type.");

			if (tmp != _Type)
				_throw_io_error("Invalid chunk size.");

			_ValueSerializer::read(in, args...);
		}

		template <class... _Values>
		static void write(
			std::ostream& out,
			const _Values&... args)
		{
			unsigned int tmp = _Self::serialized_data_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				_throw_io_error("Failure to write chunk size.");

			tmp = _Type;
			if (!out.write(reinterpret_cast<char*>(std::addressof(tmp)), sizeof(tmp)))
				_throw_io_error("Failure to write chunk type.");

			_ValueSerializer::write(out, args...);
		}
	};

	template <typename _Metrics, const size_t _Rank>
	struct _metrics_serializer_impl : public _serializer_base
	{
		typedef typename _metrics_serializer_impl<typename _Metrics::base_type, (_Rank - 1)> base_type;
		typedef typename base_type::value value;

		enum : size_t { serialized_data_size = base_type::serialized_data_size + sizeof(value) };

		static void read(
			std::istream& in)
		{
			value dim;
			if (!in.read(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value)))
				_throw_io_error("Failed to read metrics dimension value.");

			base_type::read(in);
		}

		static void write(
			std::ostream& out)
		{
			value dim = _Metrics::dimension_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value)))
				_throw_io_error("Failed to write metrics dimension value.");

			base_type::write(out);
		}
	};

	template <typename _Metrics>
	struct _metrics_serializer_impl<_Metrics, 1> : public _serializer_base
	{
		typedef unsigned int value;
		enum : size_t { serialized_data_size = sizeof(value) };

		static void read(
			std::istream& in)
		{
			value dim;
			if (!in.read(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value)))
				_throw_io_error("Failed to read metrics dimension value.");
		}

		static void write(
			std::ostream& out)
		{
			value dim = _Metrics::dimension_size;
			if (!out.write(reinterpret_cast<char*>(std::addressof(dim)), sizeof(value)))
				_throw_io_error("Failed to write metrics dimension value.");
		}
	};

	template <typename _Metrics>
	struct metrics_serializer : public _serializer_base
	{
		typedef typename _metrics_serializer_impl<_Metrics, _Metrics::rank> _Impl;

		typedef unsigned int value;
		enum : size_t { serialized_data_size = _Impl::serialized_data_size + sizeof(value) };

		static void read(
			std::istream& in)
		{
			value rank;
			if (!in.read(reinterpret_cast<char*>(std::addressof(rank)), sizeof(value)))
				_throw_io_error("Failed to read metrics rank value.");

			if (_Metrics::rank != rank)
				_throw_io_error("Incompatible metrics rank value.");

			_Impl::read(in);
		}

		static void write(
			std::ostream& out)
		{
			value rank = _Metrics::rank;
			if (!out.write(reinterpret_cast<char*>(std::addressof(rank)), sizeof(value)))
				_throw_io_error("Failed to write metrics rank value.");

			_Impl::write(out);
		}
	};

	template <typename _Tensor>
	struct tensor_serializer : public _serializer_base
	{
		typedef typename metrics_serializer<typename _Tensor::metrics> _MetricsSerializer;

		enum : size_t { serialized_data_size = _MetricsSerializer::serialized_data_size + sizeof(double) * _Tensor::data_size };

		typedef typename _Tensor value;

		static void read(
			std::istream& in,
			value& result)
		{
			_MetricsSerializer::read(in);

			typedef typename neural_network::algebra::metrics<_Tensor::data_size>::tensor_type flat;
			flat flatValue;

			for (size_t i = 0; i < flatValue.size<0>(); ++i)
			{
				if (!in.read(reinterpret_cast<char*>(std::addressof(flatValue(i))), sizeof(double)))
					_throw_io_error("Failed to read tensor element value.");
			}

			result = flatValue.reshape<typename _Tensor::metrics>();
		}

		static void write(
			std::ostream& out,
			const value& tensor)
		{
			_MetricsSerializer::write(out);

			typedef typename neural_network::algebra::metrics<_Tensor::data_size> flat;
			flat::tensor_type flatValue = tensor.reshape<flat>();

			for (size_t i = 0; i < flatValue.size<0>(); ++i)
			{
				if (!out.write(reinterpret_cast<char*>(std::addressof(flatValue(i))), sizeof(double)))
					_throw_io_error("Failed to write tensor element value.");
			}
		}
	};

	template <class _S, class... _Args>
	struct composite_serializer : public composite_serializer<_Args...>
	{
		typedef typename composite_serializer<_S, _Args...> _Self;
		typedef typename composite_serializer<_Args...> base_type;

		typedef typename _S::value value;
		typedef typename _S serializer;

		enum : size_t { serialized_data_size = _S::serialized_data_size + base_type::serialized_data_size };

		template <class... _Values>
		static void read(
			std::istream& in,
			value& val,
			_Values&... args)
		{
			serializer::read(in, val);

			base_type::read(in, args...);
		}

		template <class... _Values>
		static void write(
			std::ostream& out,
			const value& val,
			const _Values&... args)
		{
			serializer::write(out, val);

			base_type::write(out, args...);
		}
	};

	template <class _S>
	struct composite_serializer<_S> : public _serializer_base
	{
		typedef typename composite_serializer<_S> _Self;

		enum : size_t { serialized_data_size = _S::serialized_data_size };

		typedef typename _S::value value;
		typedef typename _S serializer;

		static void read(
			std::istream& in,
			value& val)
		{
			serializer::read(in, val);
		}

		static void write(
			std::ostream& out,
			const value& val)
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