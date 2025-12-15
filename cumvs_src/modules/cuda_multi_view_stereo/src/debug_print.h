/*
Copyright 2025 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __DEBUG_PRINT_H__
#define __DEBUG_PRINT_H__

#include <cstdio>

#define ENABLE_DEBUG_PRINT
#ifdef ENABLE_DEBUG_PRINT
#define DEBUG_PRINT(fmt, ...) std::printf(fmt, __VA_ARGS__);
#else
#define DEBUG_PRINT(fmt, ...)
#endif

#endif // !__DEBUG_PRINT_H__
