## The library
This is a simple C++ implementation of a matrix library in C++.

## Usage
In order to use the library in your own project using cmake, add these commands to your CMakeLists.txt:

```
add_subdirectory(vendor/matrix EXCLUDE_FROM_ALL)
target_link_libraries(target_executable matrix)
```

Otherwise simply add the header file to your project.

To see how to use the library check the provided example file in the example directory.

## Running the example
To run the provided example use these commands:
```
bash
git clone https://github.com/StanislavPavlic/matrix-lib.git matrix-lib
cd matrix-lib
mkdir build
cd build
cmake -Dmatrix_build_example=ON -DCMAKE_BUILD_TYPE=Release
make
bin/matrix_example
```
