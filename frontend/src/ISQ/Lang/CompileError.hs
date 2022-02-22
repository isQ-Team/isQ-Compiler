module ISQ.Lang.CompileError where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.TypeCheck (TypeCheckError)
import ISQ.Lang.RAIICheck (RAIIError)


data CompileError = 
    GrammarError GrammarError
  | TypeCheckError TypeCheckError 
  | RAIIError RAIIError
  | InternalCompilerError InternalCompilerError 