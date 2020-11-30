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

#include <random>

#include "unittest.h"
#include "..\src\tensor.h"

void test_tensor()
{
	scenario sc("Test for neural_network::algebra::tensor");

	const double minRnd = 0.5;
	const double maxRnd = 1.5;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minRnd, maxRnd);

	auto random_values = [&distr, &gen]() { return distr(gen); };

	{
		typedef neural_network::algebra::tensor<4> tensor;

		test::verbose("Rank 1 Tensor Tests");

		static_assert(tensor::rank == 1, "Invalid rank of 1-dimension tensor.");
		static_assert(tensor::data_size == 4, "Invalid size of 1-dimension tensor.");

		tensor t;

		test::assert(t.size<0>() == 4, "Invalid size<0> of 1-dimension tensor.");

		for (int i = 0; i < t.size<0>(); ++i)
			test::assert(0.0 == t(i), "Invalid initial value (i) of 1-dimension tensor.");

		t(2) = 2;
		test::assert(2 == t(2), "Invalid value at position (2)");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(4);  },
			"Invalid index of 1-dimension tensor.");

		tensor t2(random_values);

		for (int i = 0; i < t2.size<0>(); ++i)
		{
			test::assert(minRnd <= t2(i), "Invalid initial value (i) of randomly initialized 1-dimension tensor.");
			test::assert(t2(i) <= maxRnd, "Invalid initial value (i) of randomly initialized 1-dimension tensor.");
		}

		typedef tensor::metrics::expand<5>::type expanded;
		static_assert(expanded::rank == 2, "Invalid rank of tensor expanded from 1-dimension tensor.");

		expanded::tensor_type e;
		test::assert(e.size<0>() == 5, "Invalid size<0> of tensor expanded from 1-dimension tensor.");
		test::assert(e.size<1>() == 4, "Invalid size<1> of tensor expanded from 1-dimension tensor.");

		static_assert(std::is_same<expanded::shrink::type, tensor::metrics>::value, "Invalid type of expanded metric shrinked back to 1-dimenstion.");
		static_assert(std::is_same<expanded::shrink::type::tensor_type, tensor>::value, "Invalid type of expanded tensor shrinked back to 1-dimenstion.");
	}

	{
		test::verbose("Rank 2 Tensor Tests");

		typedef neural_network::algebra::tensor<4, 3> tensor;

		static_assert(tensor::rank == 2, "Invalid rank of 2-dimension tensor.");
		static_assert(tensor::data_size == 4 * 3, "Invalid size of 2-dimension tensor.");

		tensor t;

		test::assert(t.size<0>() == 4, "Invalid size<0> of 2-dimension tensor.");
		test::assert(t.size<1>() == 3, "Invalid size<1> of 2-dimension tensor.");

		for (int i = 0; i < t.size<0>(); ++i)
			for (int j = 0; j < t.size<1>(); ++j)
				test::assert(0.0 == t(i, j), "Invalid initial value (i, j) of 2-dimension tensor.");

		t(3, 2) = 3.2;
		test::assert(3.2 == t(3, 2), "Invalid value at position (3, 2)");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(4, 0);  },
			"Invalid first index of 2-dimension tensor.");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(1, 4);  },
			"Invalid first index of 2-dimension tensor.");

		tensor t2(random_values);

		for (int i = 0; i < t2.size<0>(); ++i)
			for (int j = 0; j < t2.size<1>(); ++j)
			{
				test::assert(minRnd <= t2(i, j), "Invalid initial value (i, j) of randomly initialized 2-dimension tensor.");
				test::assert(t2(i, j) <= maxRnd, "Invalid initial value (i, j) of randomly initialized 2-dimension tensor.");
			}

		typedef tensor::metrics::expand<5>::type expanded;
		static_assert(expanded::rank == 3, "Invalid rank of tensor expanded from 2-dimension tensor.");

		expanded::tensor_type e;
		test::assert(e.size<0>() == 5, "Invalid size<0> of tensor expanded from 2-dimension tensor.");
		test::assert(e.size<1>() == 4, "Invalid size<1> of tensor expanded from 2-dimension tensor.");
		test::assert(e.size<2>() == 3, "Invalid size<2> of tensor expanded from 2-dimension tensor.");

		static_assert(std::is_same<expanded::shrink::type, tensor::metrics>::value, "Invalid type of expanded metric shrinked back to 2-dimenstion.");
		static_assert(std::is_same<expanded::shrink::type::tensor_type, tensor>::value, "Invalid type of expanded tensor shrinked back to 2-dimenstion.");
	}

	{
		test::verbose("Rank 3 Tensor Tests");

		typedef neural_network::algebra::tensor<4, 3, 2> tensor;

		static_assert(tensor::rank == 3, "Invalid rank of 3-dimension tensor.");
		static_assert(tensor::data_size == 4 * 3 * 2, "Invalid size of 3-dimension tensor.");

		tensor t;

		test::assert(t.size<0>() == 4, "Invalid size<0> of 3-dimension tensor.");
		test::assert(t.size<1>() == 3, "Invalid size<1> of 3-dimension tensor.");
		test::assert(t.size<2>() == 2, "Invalid size<2> of 3-dimension tensor.");

		for (int i = 0; i < t.size<0>(); ++i)
			for (int j = 0; j < t.size<1>(); ++j)
				for (int k = 0; k < t.size<2>(); ++k)
					test::assert(0.0 == t(i, j, k), "Invalid initial value (i, j, k) of 3-dimension tensor.");

		t(3, 2, 1) = 3.21;
		test::assert(3.21 == t(3, 2, 1), "Invalid value at position (3, 2, 1)");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(4, 0, 0);  },
			"Invalid 1st index of 3-dimension tensor.");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(1, 3, 0);  },
			"Invalid 2nd index of 3-dimension tensor.");

		test::check_exception<std::invalid_argument>(
			[&t]() { t(1, 1, 2);  },
			"Invalid 3rd index of 3-dimension tensor.");

		typedef neural_network::algebra::metrics<4, 6> _Reshaped;

		_Reshaped::tensor_type r = t.reshape<_Reshaped>();

		static_assert(_Reshaped::tensor_type::rank == 2, "Invalid tensor rank after reshape.");
		test::assert(r.size<0>() == 4, "Invalid size<0> of reshaped 2-dimension tensor.");
		test::assert(r.size<1>() == 6, "Invalid size<1> of reshaped 2-dimension tensor.");

		tensor t2(random_values);

		for (int i = 0; i < t2.size<0>(); ++i)
			for (int j = 0; j < t2.size<1>(); ++j)
				for (int k = 0; k < t2.size<2>(); ++k)
				{
					test::assert(minRnd <= t2(i, j, k), "Invalid initial value (i, j, k) of randomly initialized 3-dimension tensor.");
					test::assert(t2(i, j, k) <= maxRnd, "Invalid initial value (i, j, k) of randomly initialized 3-dimension tensor.");
				}

		typedef tensor::metrics::expand<5>::type expanded;
		static_assert(expanded::rank == 4, "Invalid rank of tensor expanded from 3-dimension tensor.");

		expanded::tensor_type e;
		test::assert(e.size<0>() == 5, "Invalid size<0> of tensor expanded from 3-dimension tensor.");
		test::assert(e.size<1>() == 4, "Invalid size<1> of tensor expanded from 3-dimension tensor.");
		test::assert(e.size<2>() == 3, "Invalid size<2> of tensor expanded from 3-dimension tensor.");
		test::assert(e.size<3>() == 2, "Invalid size<3> of tensor expanded from 3-dimension tensor.");

		static_assert(std::is_same<expanded::shrink::type, tensor::metrics>::value, "Invalid type of expanded metric shrinked back to 3-dimenstion.");
		static_assert(std::is_same<expanded::shrink::type::tensor_type, tensor>::value, "Invalid type of expanded tensor shrinked back to 3-dimenstion.");
	}

	sc.pass();
}