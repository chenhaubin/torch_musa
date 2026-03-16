cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(w8a8 CXX C)
add_definitions(-DHAVE_AVX2_CPU_DEFINITION)
add_definitions(-DHAVE_AVX512_CPU_DEFINITION)
set(ENV{MUSA_HOME} "/usr/local/musa")

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c "
import os
os.environ[\"TORCH_DEVICE_BACKEND_AUTOLOAD\"] = \"0\"
import torch, torch_musa
print(torch_musa.core.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_MUSA_PYTHONPATH)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_MUSA_PYTHONPATH
                     ${TORCH_MUSA_PYTHONPATH})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_MUSA_PYTHONPATH})

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "
import torch
print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "^(.+)\n$" "\\1" PYTORCH_CMAKE_PREFIX_PATH
                     ${PYTORCH_CMAKE_PREFIX_PATH_RAW})
if(NOT PYTORCH_CMAKE_PREFIX_PATH)
  message(
    FATAL_ERROR
      "Could not find PyTorch's cmake_prefix_path. Ensure PyTorch (patched by torch_musa) is installed in your Python environment."
  )
endif()
message(STATUS "Found PyTorch CMake Prefix Path: ${PYTORCH_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH_RAW})

find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
  message(
    FATAL_ERROR
      "PyTorch/LibTorch not found. Ensure it's installed and CMAKE_PREFIX_PATH is set correctly."
  )
endif()
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

find_package(TorchMusa REQUIRED CONFIG)
list(POP_BACK CMAKE_PREFIX_PATH)
link_directories("/usr/local/musa/lib")
list(APPEND CMAKE_MODULE_PATH "${MUSA_HOME}/cmake")
find_package(MUSA REQUIRED)
string(APPEND MUSA_MCC_FLAGS " -std=c++17")
include_directories(${MUSA_INCLUDE_DIRS})
message("${MUSA_HOME} ${MUSA_INCLUDE_DIRS}")
# add_executable(example-app example-app.cpp) set(MU_SRCS MatmulQuantized.mu)
set(MU_SRCS MatmulQuantized.mu)
set(CPP_SRCS Qlinear.cpp)
# set_source_files_properties(MatmulQuantized.mu PROPERTIES LANGUAGE CXX)
set(CMAKE_BUILD_RPATH
    "/usr/local/lib/python3.10/dist-packages/torch_musa-2.7.1-py3.10-linux-x86_64.egg/torch_musa/lib"
)
musa_add_executable(qlinear ${MU_SRCS} ${CPP_SRCS})
# add_library(w8a8 Qlinear.cpp)
target_link_libraries(
  qlinear
  torch_musa
  musart
  ${TORCH_LIBRARIES}
  torch_cpu)
