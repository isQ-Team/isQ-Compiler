func @single(%a: !isq.qstate)->!isq.qstate{
    return %a : !isq.qstate
}
isq.defgate @foo {definition = [{type = "decomposition", value = @single}]}: !isq.gate<2, hermitian>