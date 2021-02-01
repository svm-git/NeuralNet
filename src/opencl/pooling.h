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

#include "layer_kernels.h"

namespace neural_network {
namespace opencl {
namespace detail {

	struct max_pooling
	{
		template < typename Input, typename Output>
		static void process_1d(
			const typename Input& input,
			typename Output& result,
			typename Input& mask,
			const size_t coreSizeX,
			const size_t strideSizeX,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto inputView = input.get_device_view(context);
			auto resultView = result.get_device_view(context);
			auto maskView = mask.get_device_view(context);

			layer_kernels::execute_1d_max_pooling_kernel(
				inputView,
				resultView,
				maskView,
				coreSizeX,
				result.size<0>(),
				strideSizeX,
				program,
				kernelName,
				queue);
		}

		template < typename Input, typename Output>
		static void process_2d(
			const typename Input& input,
			typename Output& result,
			typename Input& mask,
			const size_t coreSizeX,
			const size_t coreSizeY,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto inputView = input.get_device_view(context);
			auto resultView = result.get_device_view(context);
			auto maskView = mask.get_device_view(context);

			layer_kernels::execute_2d_max_pooling_kernel(
				inputView,
				resultView,
				maskView,
				input.size<1>(),
				coreSizeX,
				coreSizeY,
				result.size<0>(),
				result.size<1>(),
				strideSizeX,
				strideSizeY,
				program,
				kernelName,
				queue);
		}

		template < typename Input, typename Output>
		static void process_3d(
			const typename Input& input,
			typename Output& result,
			typename Input& mask,
			const size_t coreSizeX,
			const size_t coreSizeY,
			const size_t coreSizeZ,
			const size_t strideSizeX,
			const size_t strideSizeY,
			const size_t strideSizeZ,
			const ::boost::compute::program& program,
			const std::string& kernelName,
			const ::boost::compute::context& context,
			::boost::compute::command_queue& queue)
		{
			auto inputView = input.get_device_view(context);
			auto resultView = result.get_device_view(context);
			auto maskView = mask.get_device_view(context);

			layer_kernels::execute_3d_max_pooling_kernel(
				inputView,
				resultView,
				maskView,
				input.size<1>(),
				input.size<2>(),
				coreSizeX,
				coreSizeY,
				coreSizeZ,
				result.size<0>(),
				result.size<1>(),
				result.size<2>(),
				strideSizeX,
				strideSizeY,
				strideSizeZ,
				program,
				kernelName,
				queue);
		}
	};

}
}
}
