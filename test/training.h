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

#include <sstream>

template <class Network, class Input, class Truth, class Loss>
void train_test_network(
	Network& net,
	const Input& input,
	const Truth& truth,
	Loss& loss,
	typename Input::number_type& initialLoss,
	typename Input::number_type& finalLoss)
{
	initialLoss = loss.compute(net.process(input), truth);

	{
		std::stringstream ss;
		ss << "Initial network loss:" << initialLoss << ".";
		test::verbose(ss.str().c_str());
	}

	typename Input::number_type rate = 7.0f;
	int retry = 0;
	int epoch = 0;
	int iteration = 0;

	while (retry < 20 && iteration < 100000)
	{
		++iteration;
		float pretrained = loss.compute(
			net.process(input),
			truth);

		net.train(input, truth, loss, rate);

		float posttrained = loss.compute(
			net.process(input),
			truth);

		if (posttrained < pretrained)
		{
			retry = 0;
		}
		else
		{
			if (5 < retry)
			{
				rate = rate * 0.9f;
				++epoch;
			}

			++retry;
		}
	}

	finalLoss = loss.compute(
		net.process(input),
		truth);

	{
		std::stringstream ss;
		ss << "Training converged at epoch=" << epoch << "; iteration=" << iteration << "; rate=" << rate << "; final loss=" << finalLoss << ".";
		test::verbose(ss.str().c_str());
	}
}
