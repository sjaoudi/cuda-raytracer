cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

#for FindGLM.cmake, FindGLFW3.cmake, other helpers
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

FIND_PACKAGE(OpenGL REQUIRED) #for graphics
FIND_PACKAGE(GLEW REQUIRED)
#FIND_PACKAGE(GLFW3 REQUIRED)
#FIND_PACKAGE(GLM REQUIRED)

#required CMake options for QT Apps
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

#for window widgets
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5Gui REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
ADD_DEFINITIONS(${Qt5Widgets_DEFINITIONS})
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

include_directories(${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})

#default OpenGL libs
#SET(EXT_LIBS ${OPENGL_LIBRARY} ${GLEW_LIBRARY})

#get symlink_data macro
include(${CMAKE_SOURCE_DIR}/cmake_modules/helpers.cmake)

INCLUDE_DIRECTORIES(
  ${GLEW_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
  ./
  )

#things to actually build
find_package(GLUT REQUIRED)

# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.
#set(CUDA_SDK /usr/local/src/NVIDIA_CUDA_SDK)
#set(CUDA_INC ${CUDA_SDK}/common/inc/)

find_package(CUDA QUIET REQUIRED) #Any version OK

if (CUDA_FOUND)
   message("CUDA found")
else()
   message("CUDA not found")
endif()

cuda_include_directories(${CUDA_INC} ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDA_HOST_COMPILER /usr/bin/gcc-4.9)
link_directories(${CUDA_SDK}/lib/)

#set(HDRS book.h cpu_anim.h cpu_bitmap.h
#  gl_helper.h gpu_anim.h
#  timer.h)
set(HDRS book.h cpu_bitmap.h timer.h)

cuda_add_library(mycuda
  timer.h book.h cudahelpers.h
  timer.cu book.cu cudahelpers.cu)


#set(GPROG dot ripple shared_bitmap ray ray_noconst heat)
set(GPROG ray)

foreach(PROG ${GPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG}
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} m mycuda)
endforeach()

#QT headers requiring MOC (meta object compiler)
#set(QTHDRS
#  mainwindow.h
#  mypanelopengl.h
#  )

#window design files
#set(QTFORMS mainwindow.ui)

#Run qt tools to auto generate C++ compatible files
#QT5_WRAP_CPP(QTHDRS_MOC ${QTHDRS})
#QT5_WRAP_UI(QTFORMS_HDRS ${QTFORMS})
#SET(MYQTFILES ${QTHDRS_MOC} ${QTFORMS_HDRS})

#normal source files
#set(SRC
#  main.cpp
#  mainwindow.cpp
#  mypanelopengl.cpp
#)

#set(SHADERS
#  vshader.glsl
#  fshader.glsl
#)

#ADD_EXECUTABLE(interop ${SRC} ${SHDRS} ${MYQTFILES} ${SHADERS})

#TARGET_LINK_LIBRARIES(interop ${EXT_LIBS} libcommon mycuda
#  Qt5::Core Qt5::Gui Qt5::OpenGL)
TARGET_LINK_LIBRARIES(${EXT_LIBS} mycuda)

symlink_data(interop_shaders *.glsl)
