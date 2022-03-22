{-# LANGUAGE Rank2Types, DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
module ISQ.Lang.ISQv2Grammar (module ISQ.Lang.ISQv2Grammar, Pos) where
import ISQ.Lang.ISQv2Tokenizer
import Data.Complex
import Numeric.SpecFunctions (log2)
import qualified Data.Map.Lazy as Map
import Control.Monad.Fix
type Ident = String
type ASTBlock ann = [AST ann]
data BuiltinType = Ref | Unit | Qbit | Int | Bool | Double | Complex | FixedArray Int | UnknownArray | UserType String | IntRange | Gate Int | FuncTy deriving (Show, Eq)
data Type ann = Type { annotationType :: ann, ty :: BuiltinType, subTypes :: [Type ann]} deriving (Show,Functor, Eq)

intType ann = Type ann Int []
qbitType ann = Type ann Qbit []
boolType ann = Type ann Bool []
doubleType ann = Type ann Double []
complexType ann = Type ann Complex []
unitType ann = Type ann Unit []
refType ann s = Type ann Ref [s]


data CmpType = Equal | NEqual | Greater | Less | GreaterEq | LessEq deriving Show
data BinaryOperator = Add | Sub | Mul | Div | Cmp CmpType deriving Show
data UnaryOperator = Neg | Positive deriving Show
data Expr ann = 
       EIdent { annotationExpr :: ann, identName :: String}
     | EBinary { annotationExpr :: ann, binaryOp :: BinaryOperator, binaryLhs :: Expr ann, binaryRhs :: Expr ann}
     | EUnary { annotationExpr :: ann, unaryOp :: UnaryOperator, unaryOperand :: Expr ann}
     | ESubscript { annotationExpr :: ann, usedExpr :: Expr ann, subscript :: Expr ann}
     | ECall { annotationExpr :: ann, callee :: Expr ann, callArgs :: [Expr ann]}
     | EIntLit { annotationExpr :: ann, intLitVal :: Int}
     | EFloatingLit { annotationExpr :: ann, floatingLitVal :: Double}
     | EImagLit { annotationExpr :: ann, imagLitVal :: Double}
     | EBoolLit { annotationExpr :: ann, boolLitVal :: Bool }
     | ERange {annotationExpr :: ann, rangeLo :: Maybe (Expr ann), rangeHi :: Maybe (Expr ann), rangeStep :: Maybe (Expr ann)}
     | ECoreMeasure { annotationExpr :: ann, measureOperand :: Expr ann }
     | EList { annotationExpr :: ann, exprListElems :: [Expr ann] }
     -- Analysis/transformation intermediate nodes.
     | EDeref {annotationExpr :: ann, derefVal :: Expr ann}
     | EImplicitCast { annotationExpr :: ann, castedExpr :: Expr ann}
     | ETempVar {annotationExpr :: ann, tempVarId :: Int}
     | ETempArg {annotationExpr :: ann, tempArgId :: Int}
     | EUnitLit {annotationExpr :: ann}
     | EResolvedIdent {annotationExpr :: ann, resolvedId :: Int}
     | EGlobalName {annotationExpr :: ann, globalName :: String}
     | EEraselist {annotationExpr :: ann, subList :: Expr ann}
     deriving (Show,Functor)
instance Annotated Expr where
  annotation = annotationExpr
instance Annotated Type where
  annotation = annotationType



data AST ann = 
       NIf { annotationAST :: ann, condition :: Expr ann, thenBlock :: [AST ann], elseBlock :: [AST ann]}
     | NFor { annotationAST :: ann, forVar :: Ident, forRange :: Expr ann, body :: ASTBlock ann}
     | NPass { annotationAST :: ann }
     | NWhile { annotationAST :: ann, condition :: Expr ann,  body :: ASTBlock ann}
     | NCall { annotationAST :: ann, callExpr :: Expr ann}
     | NDefvar { annotationAST :: ann, definitions :: [(Type ann, Ident, Maybe (Expr ann))]}
     | NAssign { annotationAST :: ann, assignLhs :: Expr ann, assignRhs :: Expr ann}
     | NGatedef { annotationAST :: ann, gateName :: String, gateRhs :: [[Expr ann]]}
     | NReturn { annotationAST :: ann, returnedVal :: Expr ann}
     | NCoreUnitary { annotationAST :: ann, unitaryGate :: Expr ann, unitaryOperands :: [Expr ann], gateModifiers :: [GateModifier], rotation :: Maybe ([Expr ann])}
     | NCoreU3 { annotationAST :: ann, unitaryGate :: Expr ann, unitaryOperands :: [Expr ann], gateModifiers :: [GateModifier], angle :: [Expr ann]}
     | NCoreReset { annotationAST :: ann, resetOperands :: Expr ann}
     | NCorePrint { annotationAST :: ann, printOperands :: Expr ann}
     | NCoreMeasure {annotationAST :: ann, measExpr :: Expr ann}
     | NProcedure { annotationAST :: ann, procReturnType :: Type ann, procName :: String, procArgs :: [(Type ann, Ident)], procBody :: [AST ann]}
     | NContinue { annotationAST :: ann }
     | NBreak { annotationAST :: ann }
    -- Analysis/transformation intermediate nodes.
     | NResolvedFor { annotationAST :: ann, forVarId :: Int, forRange :: Expr ann, body :: ASTBlock ann}
     | NResolvedGatedef { annotationAST :: ann, gateName :: String, resolvedGateRhs :: [[Complex Double]], gateSize :: Int}
     | NWhileWithGuard { annotationAST :: ann, condition :: Expr ann,  body :: ASTBlock ann, breakFlag :: Expr ann}
     | NProcedureWithRet { annotationAST :: ann, procReturnType :: Type ann, procName :: String, procArgs :: [(Type ann, Ident)], procBody :: [AST ann], retVal :: Expr ann}
     | NResolvedProcedureWithRet { annotationAST :: ann, resolvedProcReturnType :: Type (), procName :: String, resolvedProcArgs :: [(Type (), Int)], procBody :: [AST ann], retValR :: Maybe (Expr ann), retVarSSA :: Maybe (Type (), Int)}
     -- Shortcut for jump to end of current region.
     -- In funciton body: jumps to end of function.
     -- In while guard: jumps out of loop (continue)
     -- In while body: jumps to end of loop (continue and break)
     -- In if body: jumps to end of branch.
     -- The jump check should be placed after every operation that has a block.
     | NJumpToEndOnFlag { annotationAST :: ann , flagId :: Expr ann } 
     | NJumpToEnd { annotationAST :: ann }
     | NTempvar { annotationAST :: ann, tempVar :: (Type (), Int, Maybe (Expr ann))}
     | NResolvedDefvar { annotationAST :: ann, resolvedDefinitions :: [(Type (), Int, Maybe (Expr ann))]}
     | NGlobalDefvar {annotationAST :: ann, globalDefinitions :: [(Type (), Int, String, Maybe (Expr ann))]}
     deriving (Show,Functor)

data GateModifier = Inv | Ctrl Bool Int deriving (Show, Eq)

addedQubits :: GateModifier->Int
addedQubits Inv = 0
addedQubits (Ctrl _ x) = x


type LAST = AST Pos
type LExpr = Expr Pos
type LType = Type Pos
instance Annotated AST where
  annotation = annotationAST

newtype InternalCompilerError = InternalCompilerError String deriving Show

data GrammarError = 
    BadMatrixElement {badExpr :: LExpr}
  | BadMatrixShape {badMatrix :: LAST}
  | BadMatrixName {badDefPos :: Pos, badDefName :: String}
  | MissingGlobalVarSize {badDefPos :: Pos, badDefName :: String} 
  | UnexpectedToken {token :: Token Pos} deriving Show

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

baseGate = ["H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "Rx", "Ry", "Rz", "u3"]

passVerifyDefgate :: [LAST] -> Either GrammarError [LAST]
passVerifyDefgate = mapM go where
    go g@(NGatedef ann name mat) = do
        new_mat<-mapM (mapM foldConstantComplex') mat
        case checkGateSize new_mat of
          Just sz -> if name `elem` baseGate then Left $ BadMatrixName ann name else return $ NResolvedGatedef ann name new_mat sz
          Nothing -> Left $ BadMatrixShape g

    go x = Right x


checkTopLevelVardef :: [LAST]->Either GrammarError [LAST]
checkTopLevelVardef = mapM go where
  go v@(NDefvar pos defs) = NDefvar pos <$> mapM go' defs where
    go' (Type _ UnknownArray _,b,c) = Left $ MissingGlobalVarSize pos b
    go' (a,b,c) = Right (a,b,c)
  go v = return v


