func @foo()->index{
    %a = arith.constant 114514 : index
    return %a : index
}

func @bar()->index{
    call @bar() : ()->index
    %b = call @foo() : ()->index
    return %b : index
}

func @baz()->index{
    %c = call @bar() : ()->index
    return %c : index

}