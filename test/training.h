/*

Copyright (c) 2020-2021 svm-git

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

template <const int MaxIterations, class Input, class Result, class Process, class Train, class Loss>
void train_test_network_impl(
	const Input& input,
	const Result& truth,
	Process& process,
	Train& train,
	Loss& loss,
	typename Input::number_type startingRate,
	typename Input::number_type& initialLoss,
	typename Input::number_type& finalLoss)
{
	initialLoss = loss(process(input), truth);

	{
		std::stringstream ss;
		ss << "Initial network loss=" << initialLoss << ", starting rate=" << startingRate << ".";
		test::verbose(ss.str().c_str());
	}

	typename Input::number_type rate = startingRate;
	int retry = 0;
	int epoch = 0;
	int iteration = 0;

	while (retry < 20 && iteration < MaxIterations)
	{
		++iteration;
		float pretrained = loss(
			process(input),
			truth);

		train(input, truth, rate);

		float posttrained = loss(
			process(input),
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

	finalLoss = loss(
		process(input),
		truth);

	{
		std::stringstream ss;
		ss << "Training converged at epoch=" << epoch << "; iteration=" << iteration << "; rate=" << rate << "; final loss=" << finalLoss << ".";
		test::verbose(ss.str().c_str());
	}
}

template <class Network, class Input, class Result, class Loss>
void train_test_network(
	Network& net,
	const Input& input,
	const Result& truth,
	Loss& loss,
	typename Input::number_type& initialLoss,
	typename Input::number_type& finalLoss)
{
	auto processFunc = [&net](const typename Input& in)
	{
		return net.process(in);
	};

	auto trainFunc = [&net, &loss](
		const typename Input& input,
		const typename Result& truth,
		const typename Input::number_type rate)
	{
		net.train(input, truth, loss, rate);
	};

	auto lossFunc = [&loss](
		const typename Result& result,
		const typename Result& truth)
	{
		return loss.compute(result, truth);
	};

	train_test_network_impl<100000>(
		input,
		truth,
		processFunc,
		trainFunc,
		lossFunc,
		1.6f,
		initialLoss,
		finalLoss);
}
