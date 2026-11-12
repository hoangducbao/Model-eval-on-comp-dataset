cmake_minimum_required(VERSION 3.22.1)
project(htpshared C)
# libonnxruntime.so is loaded by ORT Java first; htpshared.c finds it with dlopen.
add_library(htpshared SHARED htpshared.c)
target_include_directories(htpshared PRIVATE include)
target_link_libraries(htpshared log dl)
