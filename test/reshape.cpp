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

#include "unittest.h"
#include "serializationtest.h"

#include "..\src\reshape.h"

void test_reshape()
{
	scenario sc("Test for neural_network::reshape layer");

	typedef neural_network::algebra::metrics<3, 2, 1> m3x2x1;
	typedef neural_network::algebra::metrics<6> m6;
	
	m3x2x1::tensor_type i;

	auto layer = neural_network::make_reshape_layer<m3x2x1, m6>();

	auto r = layer.process(i);

	test::assert(r.size<0>() == 6, "Invalid size of reshaped tensor.");

	auto g = layer.compute_gradient(r);

	test::assert(g.size<0>() == 3, "Invalid size<0> of reshaped gradient tensor.");
	test::assert(g.size<1>() == 2, "Invalid size<1> of reshaped gradient tensor.");
	test::assert(g.size<2>() == 1, "Invalid size<2> of reshaped gradient tensor.");

	test_layer_serialization("Reshape Layer Serialization Tests", layer);

	sc.pass();
}