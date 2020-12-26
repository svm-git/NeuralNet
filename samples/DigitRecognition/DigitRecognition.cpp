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

#include <fstream>
#include <iostream>
#include <random>

#include "mnist.h"

struct arguments
{
	bool train;
	std::wstring mnist_path;
	std::wstring model_path;
	size_t epochs;
	double start_rate;
	double epoch_step;
	size_t training_percent;
};

typedef neural_network::algebra::metrics<10> output_metrics;

void print_usage()
{
	std::cout << "DigitRecognition - NeuralNet library example for training and recognizing MNIST digit images.\r\n";
	std::cout << "\r\n";
	std::cout << "USAGE:\r\n";
	std::cout << "\r\n";
	std::cout << "DigitRecognition.exe <options>\r\n";
	std::cout << "\r\n";
	std::cout << "    -mnist:path    Path to the MNIST training data set.\r\n";
	std::cout << "    -model:file    File name for the serialized model. If -test mode is used, a model is\r\n";
	std::cout << "                   loaded from the file. In training mode, the final model will be saved into\r\n";
	std::cout << "                   the file.\r\n";
	std::cout << "    -epochs:value  Number of epochs in the training mode. The paramenter is a integer value.\r\n";
	std::cout << "    -rate:value    Starting learning rate. The parameter is a positive floating point value.\r\n";
	std::cout << "    -step:value    Factor by which learning rate will change between epochs. The parameter is a\r\n";
	std::cout << "                   positive floating point value less than 1.0.\r\n";
	std::cout << "    -train:value   Indicates percentage of the data set to use for traning. The parameter is a.\r\n";
	std::cout << "                   positive integer value between 1 and 100.\r\n";
	std::cout << "    -test	         Indicates that a pre-trained model should be tested.\r\n";
}

bool parse_arguments(
	int argc,
	_TCHAR* argv[],
	arguments& args)
{
	args.train = true;
	args.epochs = 35;
	args.start_rate = 0.3;
	args.epoch_step = 0.9;
	args.model_path = L"";
	args.mnist_path = L"";
	args.training_percent = 30;

	for (int i = 0; i < argc; ++i)
	{
		std::wstring rawArg = argv[i];

		if (0 == rawArg.find(L"-mnist:"))
		{
			args.mnist_path = rawArg.substr(7);
		}
		else if (0 == rawArg.find(L"-model:"))
		{
			args.model_path = rawArg.substr(7);
		}
		else if (0 == rawArg.find(L"-epochs:"))
		{
			args.epochs = _wtoi(rawArg.substr(8).c_str());
			if (args.epochs == 0)
				return false;
		}
		else if (0 == rawArg.find(L"-rate:"))
		{
			args.start_rate = _wtof(rawArg.substr(6).c_str());
			if (false == (0.0 < args.start_rate))
				return false;
		}
		else if (0 == rawArg.find(L"-step:"))
		{
			args.epoch_step = _wtof(rawArg.substr(6).c_str());
			if (false == (0.0 < args.epoch_step) || false == (args.epoch_step < 1.0))
				return false;
		}
		else if (0 == rawArg.find(L"-train:"))
		{
			args.training_percent = _wtoi(rawArg.substr(7).c_str());
			if (args.training_percent == 0)
				return false;
		}
		else if (0 == rawArg.find(L"-test"))
		{
			args.train = false;
		}
	}

	return args.mnist_path.size() > 0 
		&& (args.train || args.model_path.size() > 0)
		&& (!args.train || 0 < args.training_percent && args.training_percent <= 100);
}

const output_metrics::tensor_type& get_target(
	int digit)
{
	static std::vector<output_metrics::tensor_type> targets;
	
	if (0 == targets.size())
	{
		for (size_t i = 0; i < 10; ++i)
		{
			output_metrics::tensor_type target;
			target.fill(0.0);
			target(i) = 1.0;

			targets.push_back(target);
		}
	}

	if (digit < 0 || (int)targets.size() < digit)
		throw std::invalid_argument("Input digit is out of range");

	return targets[digit];
}

const int get_result(
	const output_metrics::tensor_type& output,
	double& confidence)
{
	double max = output(0);
	size_t maxIndex = 0;
	for (size_t i = 1; i < output.size<0>(); ++i)
	{
		if (max < output(i))
		{
			maxIndex = i;
			max = output(i);
		}
	}

	confidence = max;
	return (int)maxIndex;
}

std::vector<double> get_learning_rates(
	double rate,
	const double factor,
	const size_t levels)
{
	std::vector<double> result;
	result.reserve(levels);

	for (size_t i = 0; i < levels; ++i)
	{
		result.push_back(rate);
		rate *= factor;
	}

	return result;
}

template <class _Network>
void test_success_rate(
	_Network& network,
	const mnist_data& training,
	std::string prefix)
{
	size_t errors = 0;
	for (auto digit : training)
	{
		auto result = network.process(digit.second);
		double confidence;
		int recognized = get_result(result, confidence);

		if (digit.first != recognized)
		{
			++errors;
		}
	}

	std::cout << prefix.c_str()
		<< " success rate: " << 100.0 * ((double)(training.size() - errors) / (double)training.size())
		<< "% error rate: " << 100.0 * ((double)(errors) / (double)training.size())
		<< "%\r\n";
}

template <class Network>
int train_network(
	Network& network,
	arguments& args,
	mnist_data& full,
	std::mt19937& gen)
{
	std::uniform_real_distribution<double> distr(0, 1);

	mnist_data training;
	mnist_data test;

	while (full.size() > 0)
	{
		int digitId = full.back().first;

		size_t segment = full.size();
		while (segment > 0 && digitId == full[segment - 1].first)
		{
			--segment;
		}

		while (full.size() > segment)
		{
			size_t segmentSize = full.size() - segment;
			size_t nextIndex = segment + (size_t)(((double)segmentSize) * distr(gen));

			if (segmentSize > 1)
			{
				std::swap(full[nextIndex], full[full.size() - 1]);
			}

			if ((full.size() % 100) < args.training_percent)
			{
				training.push_back(full.back());
			}
			else
			{
				test.push_back(full.back());
			}

			full.pop_back();
		}
	}

	neural_network::squared_error_loss<output_metrics> loss;

	//test_success_rate(network, training, "Untrained");
	std::cout
		<< "Training new model on MNIST data set.\r\n"
		<< "Epochs: " << args.epochs << "; training set: " << training.size() << " images; test set: " << test.size() << " images."
		<< "\r\n";

	std::vector<double> rates = get_learning_rates(args.start_rate, args.epoch_step, args.epochs);

	std::vector<const mnist_digit*> input;

	// Training
	for (auto rate : rates)
	{
		std::cout << "Epoch: " << (std::find(rates.cbegin(), rates.cend(), rate) - rates.cbegin()) << "; learning rate: " << rate << "\r\n";

		for (int i = 0; i < 1; ++i)
		{
			input.resize(training.size());
			std::transform(
				training.cbegin(), training.cend(),
				input.begin(),
				[](const mnist_digit& digit) { return std::addressof(digit); });

			while (input.size() > 0)
			{
				size_t nextIndex = (size_t)(((double)input.size()) * distr(gen));
				const mnist_digit* digit = input[nextIndex];

				if (input.size() > 1)
				{
					std::swap(input[nextIndex], input[input.size() - 1]);
				}
				input.pop_back();

				network.train(digit->second, get_target(digit->first), loss, rate);
			}
		}

		test_success_rate(network, training, "Training set");
		test_success_rate(network, test, "Test set");
	}

	if (args.model_path.size() > 0)
	{
		std::wcout 
			<< L"Saving model to file '" << args.model_path.c_str() << L"' (" 
			<< neural_network::serialization::model_size(network) << L" bytes)"
			<<"\r\n";

		try
		{
			std::ofstream outfile = std::ofstream(args.model_path, std::ios::out | std::ios::binary | std::ios::ate);

			neural_network::serialization::write(outfile, network);
			outfile.flush();
			outfile.close();
		}
		catch (const std::exception& ex)
		{
			std::wcout << L"Cannot write to file '" << args.model_path.c_str() << "'\r\n";
			std::cout << "Exception: " << ex.what() << "'\r\n";

			return 3;
		}
	}

	return 0;
}

template <class Network>
int test_network(
	Network& network,
	arguments& args,
	mnist_data& full)
{
	bool modelLoaded = false;

	std::ifstream infile = std::ifstream(args.model_path, std::ios::in | std::ios::binary);
	if (infile.is_open())
	{
		try
		{
			infile.seekg(0, std::ios::beg);

			neural_network::serialization::read(infile, network);
			modelLoaded = true;
		}
		catch (const std::exception& ex)
		{
			std::wcout << L"Failure to load pretrained model from file '" << args.model_path.c_str() << "'\r\n";
			std::cout << "Exception: " << ex.what() << "'\r\n";
		}
	}
	else
	{
		std::wcout << "The file '" << args.model_path.c_str() << "' couldn't be read";
	}

	if (false == modelLoaded)
	{
		return 2;
	}

	std::wcout << "Running model '" << args.model_path.c_str() << "' on data set '" << args.mnist_path.c_str() << "'\r\n";

	test_success_rate(network, full, "Model");

	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
	arguments args;
	if (false == parse_arguments(argc, argv, args))
	{
		print_usage();
		return 1;
	}

	mnist_data full = load_mnist(args.mnist_path);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> weight_distr(-0.5, 0.5);

	auto random_values = [&weight_distr, &gen]() { return weight_distr(gen); };

	
	typedef neural_network::algebra::metrics<2, 2> _2x2;
	typedef neural_network::algebra::metrics<3, 10> _3x10;
	typedef neural_network::algebra::metrics<14, 14> _14x14;
	typedef neural_network::algebra::metrics<49> _49;
	typedef neural_network::algebra::metrics<196> _196;
	typedef neural_network::algebra::metrics<784> flat_digit_metrics;

	const size_t nKernels = 48;
	const size_t nKernels_2 = 24;

	typedef neural_network::algebra::metrics<4, 4> _4x4;
	typedef neural_network::algebra::metrics<3, 3> _3x3;
	typedef neural_network::algebra::metrics<nKernels, 3, 3> _Kx3x3;
	typedef neural_network::algebra::metrics<nKernels, 4, 4> _Kx4x4;
	typedef neural_network::algebra::metrics<nKernels, 9, 9> _Kx9x9;
	typedef neural_network::algebra::metrics<1, 3, 3> _1x3x3;
	typedef neural_network::algebra::metrics<1, 2, 2> _1x2x2;
	typedef neural_network::algebra::metrics<nKernels, 2, 2> _Kx2x2;
	typedef neural_network::algebra::metrics<nKernels_2, 1, 2, 2> _K2x1x2x2;
	typedef neural_network::algebra::metrics<nKernels_2, 2, 2> _K2x2x2;
	typedef neural_network::algebra::metrics<2, 1, 1> _Pooling;
	typedef neural_network::algebra::metrics<nKernels_2 / 2, 2, 2> _PoolingOut;
	typedef neural_network::algebra::metrics<_PoolingOut::data_size> _Flat;

	auto network = neural_network::make_network(
		neural_network::make_ensemble(
			neural_network::make_network(
				neural_network::make_reshape_layer<digit::metrics, flat_digit_metrics>(),
				neural_network::make_fully_connected_layer<flat_digit_metrics, _49>(
					random_values, 0.0003),
				neural_network::make_relu_activation_layer<_49>(),
				neural_network::make_fully_connected_layer<_49, output_metrics>(
					random_values, 0.0003),
				neural_network::make_logistic_activation_layer<output_metrics>()
			),
			neural_network::make_network(
				neural_network::make_max_pooling_layer<digit::metrics, _2x2, _2x2>(),
				neural_network::make_reshape_layer<_14x14, _196>(),
				neural_network::make_fully_connected_layer<_196, _49>(
					random_values, 0.0003),
				neural_network::make_relu_activation_layer<_49>(),
				neural_network::make_fully_connected_layer<_49, output_metrics>(
					random_values, 0.0003),
				neural_network::make_logistic_activation_layer<output_metrics>()
			),
			neural_network::make_network(
				neural_network::make_convolution_layer<digit::metrics, _4x4, _3x3, nKernels>(
					random_values),
				neural_network::make_relu_activation_layer<_Kx9x9>(),
				neural_network::make_max_pooling_layer<_Kx9x9, _1x3x3, _1x2x2>(),
				neural_network::make_convolution_layer<_Kx4x4, _Kx2x2, _Kx2x2, nKernels_2>(
					random_values),
				neural_network::make_reshape_layer<_K2x1x2x2, _K2x2x2>(),
				neural_network::make_relu_activation_layer<_K2x2x2>(),
				neural_network::make_max_pooling_layer<_K2x2x2, _Pooling, _Pooling>(),
				neural_network::make_reshape_layer<_PoolingOut, _Flat>(),
				neural_network::make_fully_connected_layer<_Flat, output_metrics>(
					random_values, 0.0003),
				neural_network::make_relu_activation_layer<output_metrics>(),
				neural_network::make_fully_connected_layer<output_metrics, output_metrics>(
					random_values, 0.0003),
				neural_network::make_logistic_activation_layer<output_metrics>()
			)
		),
		neural_network::make_max_pooling_layer<_3x10>()
	);

	if (args.train)
	{
		return train_network(network, args, full, gen);
	}
	else
	{
		return test_network(network, args, full);
	}
	
	return 0;
}
