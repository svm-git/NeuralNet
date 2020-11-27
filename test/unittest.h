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

#include <exception>
#include <iostream>

class test_exception : public std::exception
{
public:
	test_exception(const char* const& message)
		: std::exception(message) {}
};

template <const bool Verbose = false>
class _Test
{
private:
	typedef _Test<Verbose> _Self;

public:
	static bool is_verbose()
	{
		return Verbose;
	}

	// Utility function to log a message from a test scenario.
	//
	// Sample usage:
	//		test::log("This text is always logged.");
	//
	static void log(const char* message)
	{
		std::cout << message << "\r" << std::endl;
		std::cout.flush();
	}

	// Utility function to log a verbose message from a test scenario.
	//
	// Sample usage:
	//		test::verbose("This text is logged only if verbose mode is enabled.");
	//
	static void verbose(const char* message)
	{
		if (is_verbose())
		{
			log(message);
		}
	}

	// Utility function to assert a test condition.
	//
	// Sample usage:
	//		test::assert(some_check, "My check failed");
	//
	static void assert(bool test, const char* message)
	{
		if (true == test)
			return;

		static std::string log = "ASSERT FAILED: ";
		
		verbose((log + message).c_str());
		std::cout.flush();

		throw test_exception(message);
	}

	// Utility function to execute a test function and check for success.
	//
	// Sample usage:
	//		test::check([]() { do_something(); return success; }, "My test scenario failed.");
	//
	template <class _Func>
	static bool check(_Func func, const char* message)
	{
		if (func())
			return true;

		static std::string log = "TEST FAILED: ";
		verbose((log + message).c_str());
		std::cout.flush();

		throw test_exception(message);
	}

	// Utility function to execute a test function and check that expected exception is thrown.
	//
	// Sample usage:
	//		test::check_exception<foo>([]() { throw foo() }, "Foo was not thrown.");
	//
	template <class _Ex, class _Func>
	static bool check_exception(_Func func, const char* message)
	{
		return _Self::check([&func]()
			{
				bool pass = false;
				try
				{
					func();
				}
				catch (const _Ex& err)
				{
					static std::string log = "Caught expected exception: ";
					verbose((log + typeid(_Ex).name() + ": " + err.what()).c_str());
					pass = true;
				}

				return pass;
			},
			message);
	}
};

// Default to no verbose logging. Change that to true for debugging.
typedef _Test<true> test;

// Utility struct to automatically check for a test scenario success.
//
// Sample usage:
//		void my_test_scenario()
//		{
//			scenario s("My scenario");
//			...
//			s.pass();
//		}
//
struct scenario
{
	const char* name;
	bool passed;

	scenario(const char* _Name = "Unknown")
		: name(_Name), passed(false)
	{
		static std::string log = "Scenario: ";

		test::log((log + name + ": BEGIN").c_str());
	}

	void pass()
	{
		passed = true;
	}

	~scenario()
	{
		static std::string log = "Scenario: ";

		if (passed)
		{
			test::log((log + name + ": PASS").c_str());
		}
		else
		{
			std::string failure = log + name + ": FAILED";
			test::log(failure.c_str());
			std::cout.flush();

			throw test_exception(failure.c_str());
		}
	}
};

void test_tensor();
void test_activation();
void test_connected();
void test_reshape();

void run_tests();
