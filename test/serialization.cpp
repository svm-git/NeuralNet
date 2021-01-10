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

#include "stdafx.h"

#include <iostream>
#include <random>

#include "unittest.h"
#include "serializationtest.h"

#include "..\src\serialization.h"

template <class _Val>
void test_value_serializer(
	const char* log,
	const _Val expected)
{
	char buffer[0x100] = { 0x0 };

	test::log(log);

	typedef neural_network::serialization::value_serializer<_Val> serializer;

	static_assert(serializer::serialized_data_size <= sizeof(buffer), "Stack buffer is not large enough.");

	membuf outbuf(buffer, sizeof(buffer));
	std::ostream out(&outbuf);

	serializer::write(out, expected);
	test::assert(serializer::serialized_data_size == out.tellp(), "Invalid stream position after writing value.");

	membuf inbuf(buffer, sizeof(buffer));
	std::istream in(&inbuf);

	_Val actual;
	serializer::read(in, actual);
	test::assert(serializer::serialized_data_size == in.tellg(), "Invalid stream position after reading value.");

	test::assert(expected == actual, "Value that was read does not match the value that was written.");
}

void test_serialization()
{
	scenario sc("Test for neural_network::serialization primitives");

	test_value_serializer<double>(
		"Test value_serializer<double>",
		123.456);

	test_value_serializer<size_t>(
		"Test value_serializer<size_t>",
		123456);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(-0.5, 0.5);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		test::log("Test tensor_serializer.");

		char buffer[0x100] = { 0x0 };

		typedef neural_network::algebra::tensor<4, 3, 2> m4x3x2;
		typedef neural_network::serialization::tensor_serializer<m4x3x2> serializer;

		static_assert(serializer::serialized_data_size <= sizeof(buffer), "Stack buffer is not large enough.");

		membuf outbuf(buffer, sizeof(buffer));
		std::ostream out(&outbuf);

		m4x3x2 expected(random_values);
		serializer::write(out, expected);
		test::assert(serializer::serialized_data_size == out.tellp(), "Invalid stream position after writing tensor.");

		membuf inbuf(buffer, sizeof(buffer));
		std::istream in(&inbuf);

		m4x3x2 actual;
		serializer::read(in, actual);
		test::assert(serializer::serialized_data_size == in.tellg(), "Invalid stream position after reading tensor.");

		for (size_t x = 0; x < expected.size<0>(); ++x)
		{
			for (size_t y = 0; y < expected.size<1>(); ++y)
			{
				for (size_t z = 0; z < expected.size<2>(); ++z)
				{
					test::assert(actual(x, y, z) == expected(x, y, z), "Tensor element that was read does not match the expected element that was written.");
				}
			}
		}
	}

	{
		test::log("Test composite_serializer.");

		char buffer[0x1000] = { 0x0 };

		typedef neural_network::algebra::tensor<4, 3, 2> m4x3x2;
		typedef neural_network::algebra::tensor<4, 3> m4x3;

		typedef neural_network::serialization::chunk_serializer<
			neural_network::serialization::chunk_types::fully_connected_layer,
			neural_network::serialization::composite_serializer<
				neural_network::serialization::value_serializer<double>,
				neural_network::serialization::tensor_serializer<m4x3x2>,
				neural_network::serialization::tensor_serializer<m4x3>>
		> serializer;

		static_assert(serializer::serialized_data_size <= sizeof(buffer), "Stack buffer is not large enough.");

		double expectedDouble = 654.321;
		m4x3x2 expectedT1(random_values);
		m4x3 expectedT2(random_values);

		membuf outbuf(buffer, sizeof(buffer));
		std::ostream out(&outbuf);

		serializer::write(out, expectedDouble, expectedT1, expectedT2);
		test::assert(serializer::serialized_data_size == out.tellp(), "Invalid stream position after writing composite chunk value.");

		membuf inbuf(buffer, sizeof(buffer));
		std::istream in(&inbuf);

		double actualDouble = 0.0;
		m4x3x2 actualT1;
		m4x3 actualT2;

		serializer::read(in, actualDouble, actualT1, actualT2);
		test::assert(serializer::serialized_data_size == in.tellg(), "Invalid stream position after reading composite chunk value.");

		test::assert(actualDouble == expectedDouble, "Double value that was read does not match the expected value that was written.");

		for (size_t x = 0; x < expectedT1.size<0>(); ++x)
		{
			for (size_t y = 0; y < expectedT1.size<1>(); ++y)
			{
				for (size_t z = 0; z < expectedT1.size<2>(); ++z)
				{
					test::assert(actualT1(x, y, z) == expectedT1(x, y, z), "Tensor element that was read does not match the expected element that was written for rank-3 tensor.");
				}
			}
		}

		for (size_t x = 0; x < actualT2.size<0>(); ++x)
		{
			for (size_t y = 0; y < actualT2.size<1>(); ++y)
			{
				test::assert(actualT2(x, y) == expectedT2(x, y), "Tensor element that was read does not match the expected element that was written for rank-2 tensor.");
			}
		}
	}

	sc.pass();
}