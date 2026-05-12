cmake_minimum_required(VERSION 3.20)
project(QEGcodes CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(sim_FER
        simulateFER.cpp
        stabilizerCodes.cpp
        fileReader.cpp
        helpers.cpp
)

add_executable(test_prog
        main.cpp
        stabilizerCodes.cpp
        fileReader.cpp
        helpers.cpp
)


find_package(OpenMP)
if(OPENMP_FOUND)
    target_link_libraries(sim_FER PUBLIC OpenMP::OpenMP_CXX)
endif()
