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

#include "..\src\connected.h"

void test_connected()
{
	scenario sc("Test for neural_network::fully_connected_layer");

	typedef neural_network::algebra::metrics<3, 2, 1> _3x2x1;
	typedef neural_network::algebra::metrics<5, 4> _5x4;

	_5x4::tensor_type tmp;
	auto layer = neural_network::make_fully_connected_layer<_5x4, _3x2x1>();
	
	_3x2x1::tensor_type ret = layer.process(tmp);
	layer.compute_gradient(ret);
	layer.update_weights(0.9);

	test_layer_serialization("Fully Connected Layer Serialization Tests", layer);

	sc.pass();
}
