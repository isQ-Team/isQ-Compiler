function(qir_tool tool_name)
add_executable(qir-${tool_name} ${tool_name}.cpp)
target_link_libraries(qir-${tool_name} isqir LLVMCore ${mlir_libs})
add_custom_command(TARGET qir-${tool_name} POST_BUILD
    COMMAND ${CMAKE_STRIP} qir-${tool_name})
endfunction()

#qir_tool(opt)
#qir_tool(lsp-server)
#qir_tool(codegen)
qir_tool(lower)