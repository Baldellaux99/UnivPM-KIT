cmake_minimum_required(VERSION 3.20)
project(Decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ---------- OpenMP ----------
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sim_FER PRIVATE OpenMP::OpenMP_CXX)
else()
    # macOS Apple Clang fallback for Homebrew libomp
    if(APPLE)
        execute_process(
                COMMAND brew --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )

        if(NOT LIBOMP_PREFIX)
            if(EXISTS "/opt/homebrew/opt/libomp")
                set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
            elseif(EXISTS "/usr/local/opt/libomp")
                set(LIBOMP_PREFIX "/usr/local/opt/libomp")
            endif()
        endif()

        if(LIBOMP_PREFIX)
            message(STATUS "Using Homebrew libomp: ${LIBOMP_PREFIX}")

            target_compile_options(sim_FER PRIVATE
                    -Xpreprocessor
                    -fopenmp
            )

            target_include_directories(sim_FER PRIVATE
                    ${LIBOMP_PREFIX}/include
            )

            target_link_libraries(sim_FER PRIVATE
                    ${LIBOMP_PREFIX}/lib/libomp.dylib
            )
        else()
            message(FATAL_ERROR
                    "OpenMP not found. On macOS, install it with: brew install libomp"
            )
        endif()
    else()
        message(FATAL_ERROR "OpenMP not found")
    endif()
endif()