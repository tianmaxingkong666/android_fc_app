# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

# 设置native目录
set(NATIVE_DIR ${CMAKE_SOURCE_DIR}/../jni)

# 设置opencv include文件夹
include_directories(${NATIVE_DIR}/opencv/jni/include)

# 设置opencv so库
add_library(
        libopencv_java3
        SHARED
        IMPORTED)

set_target_properties(
        libopencv_java3
        PROPERTIES
        IMPORTED_LOCATION
        ${NATIVE_DIR}/opencv/libs/${ANDROID_ABI}/libopencv_java3.so)

add_library( # Sets the name of the library.
        FacePreprocessSo

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${NATIVE_DIR}/face_align/FacePreprocess.h
        ${NATIVE_DIR}/face_align/JniFacePreprocess.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        FacePreprocessSo
        libopencv_java3
        jnigraphics
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

# 指定release编译选项
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s -O3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -O3 -Wall")