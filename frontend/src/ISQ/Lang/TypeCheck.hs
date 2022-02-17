module ISQ.Lang.TypeCheck where
import ISQ.Lang.ISQv2Grammar


type EType = Type ()

data TypeCheckData = TypeCheckInfo{
    sourcePos :: Pos,
    termType :: EType
} deriving Show

type TypeCheckInfo = Either Pos TypeCheckData

data TypeCheckError = 
    TypeMismatch {badExpr :: LExpr, expectedType :: LType, foundType :: LType}


data TypeCheckEnv = TypeCheckEnv {

}

-- By now we only support bottom-up type checking.
-- All leaf nodes have their own type, and intermediate types are calculated.
--typeCheckExpr :: Expr Pos->Either TypeCheckError (Expr TypeCheckData)


--typeCheckAST :: AST Pos->Either TypeCheckError (AST TypeCheckData)