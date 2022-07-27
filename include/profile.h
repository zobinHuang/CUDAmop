/*!
 * \file    profile.h
 * \brief   Macro for profiling
 * \author  Zhuobin Huang
 * \date    July 25, 2022
 */

#ifndef _PROFILE_H_
#define _PROFILE_H_

#define PROFILING_ENABLED true

#if PROFILING_ENABLED
    #define PROFILE(profile_command) profile_command
#else
    #define PROFILE(profile_command)
#endif

#endif