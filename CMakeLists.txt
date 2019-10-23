cmake_minimum_required(VERSION 3.10)
project(matrix)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(matrix_build_example "Build example usage" OFF)

add_library(matrix INTERFACE)

target_include_directories(matrix INTERFACE ${PROJECT_SOURCE_DIR}/include)

if (matrix_build_example)
    add_executable(matrix_example example/example.cpp)
    target_link_libraries(matrix_example matrix)
endif()
