// Copyright (c) 2017 Leosocy. All rights reserved.
// Use of this source code is governed by a MIT-style license 
// that can be found in the LICENSE file.

#ifndef __PUB_H__
#define __PUB_H__

#include <string>

namespace edcc
{

typedef unsigned char u_char;
typedef unsigned short u_short;
typedef unsigned int u_int;

#define CHECK_POINTER_NULL_RETURN(pointer, ret) do { \
    if((pointer) == NULL) { \
        EDCC_Log("%s is NULL!", #pointer); \
        return (ret); \
    } \
} while(0)

#define CHECK_POINTER_NULL_RETURN_VOID(pointer) do { \
    if((pointer) == NULL) { \
        EDCC_Log("%s is NULL!", #pointer); \
        return ; \
    } \
} while(0)

#define CHECK_NE_RETURN_VOID(param1, param2) do { \
    if((param1) != (param2)) { \
        return ; \
    } \
} while(0)

#define CHECK_NE_RETURN(param1, param2, ret) do { \
    if((param1) != (param2)) { \
        return (ret); \
    } \
} while(0)

#define CHECK_EQ_RETURN_VOID(param1, param2) do { \
    if((param1) == (param2)) { \
        return ; \
    } \
} while(0)

#define CHECK_EQ_RETURN(param1, param2, ret) do { \
    if((param1) == (param2)) { \
        return (ret); \
    } \
} while(0)

#define CHECK_TRUE_RETURN(param, ret) do { \
    if((param) == true) { \
        return (ret); \
    } \
} while(0)

#define CHECK_FALSE_RETURN(param, ret) do { \
    if((param) == false) { \
        return (ret); \
    } \
} while(0)

#define CHECK_TRUE_RETURN_VOID(param) do { \
    if((param) == true) { \
        return; \
    } \
} while(0)

#define CHECK_FALSE_RETURN_VOID(param) do { \
    if((param) == false) { \
        return; \
    } \
} while(0)

#define CHECK_STR_EMPTY_RETURN(str, value) do { \
    if((str).empty()) { \
        return (value); \
    } \
} while(0)

#define ANSI_COLOR_RED                  "\x1b[31m"
#define ANSI_COLOR_GREEN                "\x1b[32m"
#define ANSI_COLOR_LIGHT_GREEN          "\x1b[1;32m"
#define ANSI_COLOR_YELLOW               "\x1b[33m"
#define ANSI_COLOR_BLUE                 "\x1b[34m"
#define ANSI_COLOR_LIGHT_BLUE           "\x1b[1;34m"
#define ANSI_COLOR_MAGENTA              "\x1b[35m"
#define ANSI_COLOR_CYAN                 "\x1b[36m"
#define ANSI_COLOR_LIGHT_GRAY           "\x1b[0;37m"
#define ANSI_COLOR_DARY_GRAY            "\x1b[1;30m"
#define ANSI_COLOR_RESET                "\x1b[0m"

#define EDCC_Log(format, ...) EDCC_Log_Func(ANSI_COLOR_LIGHT_GREEN "[EDCC_Debug]\t" ANSI_COLOR_RESET \
                                            ANSI_COLOR_DARY_GRAY   "[%s: %u]\t" ANSI_COLOR_RESET     \
                                            format"\n"                                              \
                                            , __FUNCTION__, __LINE__, ##__VA_ARGS__)
void EDCC_Log_Func(const char *format, ...);

std::string itos(int n);

} //namespace edcc

#endif // !__PUB_H__
