if (CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(sherpa-mlx)

# Remember to update
# ./CHANGELOG.md
set(SHERPA_MLX_VERSION "0.0.2")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ version to be used.")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT WIN32)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(NOT APPLE)
  set(SHERPA_MLX_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_MLX_RPATH_ORIGIN "@loader_path")
endif()

if(NOT WIN32)
  set(CMAKE_INSTALL_RPATH ${SHERPA_MLX_RPATH_ORIGIN})
  set(CMAKE_BUILD_RPATH ${SHERPA_MLX_RPATH_ORIGIN})
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")

if(BUILD_SHARED_LIBS)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(show-info)

if(NOT DEFINED ENV{MLX_CMAKE_DIR})
    message(FATAL_ERROR "\
MLX_CMAKE_DIR not defined.
1. Install mlx with
  pip install mlx
2. Use
  python3 -m mlx --cmake-dir
to find the directory of mlx
3. Use
  export MLX_CMAKE_DIR=$(python3 -m mlx --cmake-dir)
4. Re-try the failed command
  ")
endif()

add_subdirectory(sherpa-mlx)
