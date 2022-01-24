file(GLOB_RECURSE SRC_FILES ./*.cpp)
add_library(isqir SHARED ${SRC_FILES})
add_dependencies(isqir MLIRTablegen)