project(OT)

if(APPLE)
  add_definitions(-DMAC_OS)
endif()

# Include the 'include' of the project itself
include_directories("include")
# Include DartLib core and Eigen in 3rdparty
include_directories("${DartLib_DIR}/core"
                    "${CMAKE_SOURCE_DIR}/3rdparty/eigen")

# Include the directory of OpenGL and GLUT
if(MSVC)
  include_directories("${freeglut_DIR}/include")
else()
  include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
endif()

# Manually add the sources
set(SOURCES_ONE ${DartLib_DIR}/core/Utils/predicates.cxx)

# Also, the file(GLOB...) allows for wildcard additions
file(GLOB SOURCES_TWO
    "include/*.h"
    "src/*.cpp")

# Add an executable target called `OT` to be build from 
# the source files.
add_executable(OT ${SOURCES_ONE} ${SOURCES_TWO})

# Link the libraries of freeglut
if(MSVC)
  if(CMAKE_CL_64)
    target_link_libraries(OT "${freeglut_DIR}/lib/x64/freeglut.lib")
  else()
    target_link_libraries(OT "${freeglut_DIR}/lib/freeglut.lib")
  endif()
else()
  target_link_libraries(OT ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif()

# Install the executeable program in the bin folder of the
# current project directory.
install(TARGETS OT DESTINATION ${CMAKE_SOURCE_DIR}/bin)
