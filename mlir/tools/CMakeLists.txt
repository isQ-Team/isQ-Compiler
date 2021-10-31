function(qir_tool tool_name)
add_executable(qir-${tool_name} ${tool_name}.cpp)
target_link_libraries(qir-${tool_name} isqir LLVMCore ${mlir_libs})
endfunction()

qir_tool(opt)
qir_tool(lsp-server)