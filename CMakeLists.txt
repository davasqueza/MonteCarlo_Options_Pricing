cmake_minimum_required(VERSION 3.12)
project(MonteCarlo_Secuencial)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(MonteCarlo_Secuencial main.c asa241.c)
