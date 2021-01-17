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

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "mnist.h"

mnist_data load_mnist(
	std::wstring dataPath)
{
	mnist_data result;
	std::ifstream::pos_type tsize;
	size_t size;

	for (int filecount = 0; filecount < 10; filecount++)
	{
		std::wstringstream test;
		test << dataPath << L"\\data" << filecount << L".data";

		std::wcout << L"Reading data file '" << test.str() << L"'\r\n";

		std::ifstream infile = std::ifstream(test.str(), std::ios::in | std::ios::binary | std::ios::ate);
		test.clear();

		if (infile.is_open())
		{
			tsize = infile.tellg();
			size = static_cast<size_t>(tsize);
			infile.seekg(0, std::ios::beg);

			std::vector<char> buffer;
			buffer.resize(digit::data_size);

			while (infile.tellg() < tsize)
			{
				infile.read(std::addressof(buffer[0]), buffer.size());

				digit d;

				for (size_t y = 0; y < d.size<1>(); ++y)
				{
					for (size_t x = 0; x < d.size<0>(); ++x)
					{
						d(y, x) = (float)(unsigned char)buffer[y * d.size<0>() + x] / 255.0f;
					}
				}

				result.insert(
					result.end(),
					std::move(std::make_pair(filecount, d)));
			}

			infile.close();
		}
		else
		{
			std::cout<<"The file couldn't be read";
		}
	}

	return result;
}
