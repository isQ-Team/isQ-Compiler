module ISQ.Lang.CompileError where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.TypeCheck (TypeCheckError)
import ISQ.Lang.RAIICheck (RAIIError)


data CompileError = 
    GrammarError GrammarError
  | TypeCheckError TypeCheckError 
  | RAIIError RAIIError
  | InternalCompilerError InternalCompilerError
  | SyntaxError String  deriving Show
class CompileErr e where
  fromError :: e->CompileError
instance CompileErr GrammarError where
  fromError = GrammarError 
instance CompileErr TypeCheckError where
  fromError = TypeCheckError 
instance CompileErr RAIIError where
  fromError = RAIIError 
instance CompileErr InternalCompilerError where
  fromError = ISQ.Lang.CompileError.InternalCompilerError 