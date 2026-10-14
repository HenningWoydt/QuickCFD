cmake_minimum_required(VERSION 3.20)
project(QuickCFD VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs / clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if no build type is set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type set — defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Per-config compiler flags
set(QUICKCFD_FLAGS_COMMON   -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
set(QUICKCFD_FLAGS_DEBUG    -O0 -g3 -fsanitize=address,undefined)
set(QUICKCFD_FLAGS_RELEASE  -O3 -march=native -DNDEBUG)

# Sources
file(GLOB_RECURSE QUICKCFD_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(QuickCFD ${QUICKCFD_SOURCES})

target_include_directories(QuickCFD PRIVATE include)

target_compile_options(QuickCFD PRIVATE
    ${QUICKCFD_FLAGS_COMMON}
    $<$<CONFIG:Debug>:${QUICKCFD_FLAGS_DEBUG}>
    $<$<CONFIG:Release>:${QUICKCFD_FLAGS_RELEASE}>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native>
)

target_link_options(QuickCFD PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(QuickCFD PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP     : enabled")
else()
    message(STATUS "OpenMP     : not found — single-threaded")
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
