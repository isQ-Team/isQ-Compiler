file(GLOB_RECURSE SRC_FILES ./*.cpp)
add_library(isqir STATIC ${SRC_FILES})
add_dependencies(isqir MLIRTablegen)