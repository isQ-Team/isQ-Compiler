file(GLOB_RECURSE SRC_FILES ./*.cpp)
add_library(isqir STATIC ${SRC_FILES})
<<<<<<< HEAD
add_dependencies(isqir MLIRTablegen)
=======
add_dependencies(isqir MLIRTablegen)
target_link_libraries(isqir PRIVATE nlohmann_json::nlohmann_json)
>>>>>>> merge
