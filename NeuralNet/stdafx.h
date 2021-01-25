// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#pragma warning (disable: 4503)

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

// TODO: reference additional headers your program requires here

#define NEURAL_NET_ENABLE_OPEN_CL
#define CL_TARGET_OPENCL_VERSION 120
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#define _SCL_SECURE_NO_WARNINGS
