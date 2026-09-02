cmake_minimum_required(VERSION 3.22.1)
project(android_trainer LANGUAGES C CXX)

if(NOT DEFINED MNN_ROOT OR NOT EXISTS "${MNN_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "MNN_ROOT missing; checkout MNN 3.6.1 into vendor/MNN")
endif()

set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD ON CACHE BOOL "" FORCE)
set(MNN_BUILD_TRAIN ON CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_VULKAN ON CACHE BOOL "" FORCE)
set(MNN_VULKAN_IMAGE OFF CACHE BOOL "" FORCE)
set(MNN_USE_SYSTEM_LIB OFF CACHE BOOL "" FORCE)
set(MNN_ARM82 ON CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_LLM OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_OPENCV OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CODEGEN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384")
add_subdirectory("${MNN_ROOT}" "${CMAKE_BINARY_DIR}/mnn" EXCLUDE_FROM_ALL)

add_library(android_trainer SHARED native_bridge.cpp bundle.cpp model0001_gate.cpp native_opencl_conformance.cpp)
target_include_directories(android_trainer PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${MNN_ROOT}/include"
    "${MNN_ROOT}/source"
    "${MNN_ROOT}/tools/train/source"
    "${MNN_ROOT}/tools/train/source/grad"
    "${MNN_ROOT}/tools/train/source/optimizer"
    "${MNN_ROOT}/3rd_party"
    "${MNN_ROOT}/3rd_party/flatbuffers/include"
    "${MNN_ROOT}/schema/current"
    "${MNN_ROOT}/3rd_party/OpenCLHeaders"
)
target_compile_definitions(android_trainer PRIVATE
    ANDROID_TRAINER_MNN_COMMIT="d407447ed56c4121a11ccbd266dc184ca1ead0c2")
target_compile_options(android_trainer PRIVATE -Wall -Wextra -Werror=return-type)

find_library(log-lib log)
find_library(android-lib android)
find_library(dl-lib dl)
target_link_libraries(android_trainer PRIVATE
    MNN MNN_Express MNNTrain MNN_CL MNN_Vulkan
    ${log-lib} ${android-lib} ${dl-lib})
