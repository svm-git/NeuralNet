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
#include "training.h"
#include "..\src\core.h"

void test_core()
{
	scenario sc("Test for convolution/pooling core utility classes");

	{
		typedef neural_network::algebra::metrics<2> _2;
		typedef neural_network::algebra::metrics<3> _3;
		typedef neural_network::algebra::metrics<7> _7;

		static_assert(std::is_same<neural_network::algebra::_apply_core_with_stride<_7, _3, _2, 1>::metrics, neural_network::algebra::metrics<3>>::value, "Invalid metrics after applying rank 1 core and stride.");
	}

	{
		typedef neural_network::algebra::metrics<3, 3> _3x3;
		typedef neural_network::algebra::metrics<4, 4> _4x4;
		typedef neural_network::algebra::metrics<19, 19> _19x19;

		static_assert(std::is_same<neural_network::algebra::_apply_core_with_stride<_19x19, _4x4, _3x3, 2>::metrics, neural_network::algebra::metrics<6, 6>>::value, "Invalid metrics after applying rank 2 core and stride.");
	}

	{
		typedef neural_network::algebra::metrics<1, 1, 1> _1x1x1;
		typedef neural_network::algebra::metrics<2, 2, 2> _2x2x2;
		typedef neural_network::algebra::metrics<17, 17, 3> _17x17x3;

		static_assert(std::is_same<neural_network::algebra::_apply_core_with_stride<_17x17x3, _2x2x2, _1x1x1, 3>::metrics, neural_network::algebra::metrics<16, 16, 2>>::value, "Invalid metrics after applying rank 3 core and stride.");
	}

	sc.pass();
}