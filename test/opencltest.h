/*

Copyright (c) 2020-21 svm-git

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

#include <boost/compute/core.hpp>

template <const int DeviceType = ::boost::compute::device::cpu>
::boost::compute::device find_test_device()
{
	for (auto device : ::boost::compute::system::devices())
	{
		if (device.type() & DeviceType)
			return device;
	}

	return ::boost::compute::system::default_device();
}

template <typename Tensor>
void check_tensors_2d(
	const Tensor& expected,
	const Tensor& actual)
{
	for (size_t x = 0; x < expected.size<0>(); ++x)
	{
		for (size_t y = 0; y < expected.size<1>(); ++y)
		{
			test::check_true(expected(x, y) == actual(x, y), "Unexpected mismatch between C++ and OpenCL results.");
		}
	}
}

template <typename Tensor>
void check_tensors_3d(
	const Tensor& expected,
	const Tensor& actual)
{
	for (size_t x = 0; x < expected.size<0>(); ++x)
	{
		for (size_t y = 0; y < expected.size<1>(); ++y)
		{
			for (size_t z = 0; z < expected.size<2>(); ++z)
			{
				test::check_true(expected(x, y, z) == actual(x, y, z), "Unexpected mismatch between C++ and OpenCL results.");
			}
		}
	}
}

