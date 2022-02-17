{-# LANGUAGE Rank2Types #-}
module ISQ.Lang.ISQv2Grammar (module ISQ.Lang.ISQv2Grammar, Pos) where
import ISQ.Lang.ISQv2Tokenizer
import Data.Complex
import Numeric.SpecFunctions (log2)
import qualified Data.Map.Lazy as Map
import Control.Monad.Fix
type Ident = String
type ASTBlock v ann = [AST v ann]
data BuiltinType = Ref | Unit | Qbit | Int | Bool | Double | Complex | FixedArray Int | UnknownArray | UserType String | IntRange String deriving Show
data Type ann = Type { annotationType :: ann, ty :: BuiltinType, subTypes :: [Type ann]} deriving Show

intType ann = Type ann Int []
qbitType ann = Type ann Qbit []
boolType ann = Type ann Bool []
doubleType ann = Type ann Double []
complexType ann = Type ann Complex []
unitType ann = Type ann Unit []


data CmpType = Equal | NEqual | Greater | Less | GreaterEq | LessEq deriving Show
data BinaryOperator = Add | Sub | Mul | Div | Cmp CmpType deriving Show
data UnaryOperator = Neg | Positive deriving Show
data Expr v ann = 
       EIdent { annotationExpr :: ann, identName :: String}
     | EBinary { annotationExpr :: ann, binaryOp :: BinaryOperator, binaryLhs :: Expr v ann, binaryRhs :: Expr v ann}
     | EUnary { annotationExpr :: ann, unaryOp :: UnaryOperator, unaryOperand :: Expr v ann}
     | ESubscript { annotationExpr :: ann, usedExpr :: Expr v ann, subscript :: Expr v ann}
     | ECall { annotationExpr :: ann, callee :: Expr v ann, callArgs :: [Expr v ann]}
     | EIntLit { annotationExpr :: ann, intLitVal :: Int}
     | EFloatingLit { annotationExpr :: ann, floatingLitVal :: Double}
     | EImagLit { annotationExpr :: ann, imagLitVal :: Double}
     | EBoolLit { annotationExpr :: ann, boolLitVal :: Bool }
     | ERange {annotationExpr :: ann, rangeLo :: Maybe (Expr v ann), rangeHi :: Maybe (Expr v ann), rangeStep :: Maybe (Expr v ann)}
     | ECoreMeasure { annotationExpr :: ann, measureOperand :: Expr v ann }
     | EList { annotationExpr :: ann, exprListElems :: [Expr v ann] }
     -- Analysis/transformation intermediate nodes.
     | EDeref {annotationExpr :: ann, derefVal :: Expr v ann}
     | EImplicitCast { annotationExpr :: ann, castedExpr :: Expr v ann}
     | EVar { annotationExpr :: ann,  varExpr :: v}
     deriving Show
instance Annotated (Expr v) where
  annotation = annotationExpr
instance Annotated Type where
  annotation = annotationType

newtype HOAS x = HOAS {unHOAS :: x}
instance Show (HOAS x) where
  show _ = "<<<hoas>>>"
data AST v ann = 
       NIf { annotationAST :: ann, condition :: Expr v ann, thenBlock :: [AST v ann], elseBlock :: [AST v ann]}
     | NFor { annotationAST :: ann, forVar :: Ident, forRange :: Expr v ann, body :: ASTBlock v ann}
     | NPass { annotationAST :: ann }
     | NWhile { annotationAST :: ann, condition :: Expr v ann,  body :: ASTBlock v ann}
     | NCall { annotationAST :: ann, callExpr :: Expr v ann}
     | NDefvar { annotationAST :: ann, definitions :: [(Type ann, Ident, Maybe (Expr v ann))]}
     | NAssign { annotationAST :: ann, assignLhs :: Expr v ann, assignRhs :: Expr v ann}
     | NGatedef { annotationAST :: ann, gateName :: String, gateRhs :: [[Expr v ann]]}
     | NReturn { annotationAST :: ann, returnedVal :: Expr v ann}
     | NCoreUnitary { annotationAST :: ann, unitaryGate :: Expr v ann, unitaryOperands :: [Expr v ann], gateModifiers :: [GateModifier]}
     | NCoreReset { annotationAST :: ann, resetOperands :: Expr v ann}
     | NCorePrint { annotationAST :: ann, printOperands :: Expr v ann}
     | NCoreMeasure {annotationAST :: ann, measExpr :: Expr v ann}
     | NProcedure { annotationAST :: ann, procReturnType :: Type ann, procName :: String, procArgs :: [(Type ann, Ident)], procBody :: [AST v ann]}
    -- Analysis/transformation intermediate nodes.
     | NAllocRef { annotationAST :: ann, next :: HOAS (v->[AST v ann])}
     | NFreeRef { annotationAST :: ann, freedVal :: AST v ann}
     | NResolvedGatedef { annotationAST :: ann, gateName :: String, resolvedGateRhs :: [[Complex Double]], gateSize :: Int}
     deriving Show

data GateModifier = Inv | Ctrl Bool Int deriving Show
-- Literal AST: no need to tie the knot.
type LAST = AST () Pos
type LExpr = Expr () Pos
type LType = Type Pos
instance Annotated (AST v) where
  annotation = annotationAST



data CompileError = 
    TypeMismatch {badExpr :: LExpr, expectedType :: LType, foundType :: LType}
  | BadMatrixElement {badExpr :: LExpr}
  | BadMatrixShape {badMatrix :: LAST}
  | IdentifierNotFound
foldConstantComplex :: LExpr->Either LExpr (Complex Double)
foldConstantComplex x@(EBinary _ op lhs rhs) = do
    l<-foldConstantComplex lhs
    r<-foldConstantComplex rhs
    case op of
      Add -> return $ l+r
      Sub -> return $ l-r
      Mul -> return $ l*r
      Div -> return $ l/r
      _ -> Left x
foldConstantComplex x@(EUnary _ op rhs) = do
    r<-foldConstantComplex rhs
    case op of
      Neg -> return (-r)
      Positive -> return r
foldConstantComplex x@(EIntLit _ val) = Right $ fromIntegral val
foldConstantComplex x@(EFloatingLit _ val) = Right $ val :+ 0.0
foldConstantComplex x@(EImagLit _ val) = Right $ 0.0 :+ val
foldConstantComplex x = Left x

foldConstantComplex' = go . foldConstantComplex where
  go (Left x) = Left $ BadMatrixElement x
  go (Right z) = Right z
checkGateSize :: [[a]]-> Maybe Int
checkGateSize arr = let logsize = log2 $ length arr in if all ((==length arr) . length) arr  && 2 ^ logsize == length arr then Just logsize else Nothing
passVerifyDefgate :: [LAST] -> Either CompileError [LAST]
passVerifyDefgate = mapM go where
    go g@(NGatedef ann name mat) = do
        new_mat<-mapM (mapM foldConstantComplex') mat
        case checkGateSize new_mat of
          Just sz -> return $ NResolvedGatedef ann name new_mat sz
          Nothing -> Left $ BadMatrixShape g

    go x = Right x

type SymbolTable v = Map.Map String v

