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

// Utility memory buffer
struct membuf : std::streambuf
{
	membuf(char* base, std::ptrdiff_t n) {
		this->setg(base, base, base + n);
		this->setp(base, base, base + n);
	}

protected:
	virtual pos_type seekoff(off_type type,
		std::ios_base::seekdir dir,
		std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		if (type == 0 && dir == std::ios_base::cur)
		{
			if (mode == std::ios_base::in)
			{
				return std::streampos(this->gptr() - this->eback());
			}
			else if (mode == std::ios_base::out)
			{
				return std::streampos(this->pptr() - this->pbase());
			}
		}

		return (std::streampos(std::_BADOFF));
	}
};

template <typename Layer>
void test_layer_serialization(
	const char* testName,
	Layer& layer)
{
	test::verbose(testName);

	typedef Layer::serializer serializer;
	char buffer[serializer::serialized_data_size] = { 0x0 };

	membuf outbuf(buffer, sizeof(buffer));
	std::ostream out(&outbuf);

	neural_network::serialization::write(out, layer);
	test::check_true(neural_network::serialization::model_size(layer) == static_cast<size_t>(out.tellp()), "Invalid stream position after writing layer.");

	membuf inbuf(buffer, sizeof(buffer));
	std::istream in(&inbuf);

	Layer other;

	neural_network::serialization::read(in, other);
	test::check_true(neural_network::serialization::model_size(layer) == static_cast<size_t>(in.tellg()), "Invalid stream position after reading layer.");
}