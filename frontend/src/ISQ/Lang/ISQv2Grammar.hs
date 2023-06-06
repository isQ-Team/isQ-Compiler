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
data BuiltinType = Ref | Unit | Qbit | Int | Bool | Double | Complex | Array Int | UserType String | IntRange | Gate Int | Logic Int | FuncTy deriving (Show, Eq)
data Type ann = Type { annotationType :: ann, ty :: BuiltinType, subTypes :: [Type ann]} deriving (Show,Functor, Eq)

intType ann = Type ann Int []
qbitType ann = Type ann Qbit []
boolType ann = Type ann Bool []
doubleType ann = Type ann Double []
complexType ann = Type ann Complex []
unitType ann = Type ann Unit []
refType ann s = Type ann Ref [s]
refIntType ann = Type ann Ref [intType ann]
refBoolType ann = Type ann Ref [boolType ann]

data CmpType = Equal | NEqual | Greater | Less | GreaterEq | LessEq deriving (Eq, Show)
data BinaryOperator = Add | Sub | Mul | Div | CeilDiv | Mod | And | Or | Andi | Ori | Xori | Cmp CmpType | Pow | Shl | Shr deriving (Eq, Show)
data UnaryOperator = Neg | Positive | Not | Noti deriving (Eq, Show)
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
     | EListCast {annotationExpr :: ann, subList :: Expr ann}
     | EArrayLen {annotationExpr :: ann, subList :: Expr ann}
     deriving (Eq, Show, Functor)
instance Annotated Expr where
  annotation = annotationExpr
instance Annotated Type where
  annotation = annotationType

-- A procedure marked with derive-gate will be seen as a parametric gate on its last n qubits.
-- A procedure marked with derive-oracle will be seen as an oracle on its last n boolean parameters.
data DerivingType = DeriveGate | DeriveOracle Int deriving (Eq, Show)

data AssignOperator = AssignEq | AddEq | SubEq deriving (Eq, Show)
data AST ann = 
       NBlock { annotationAST :: ann, statementList :: ASTBlock ann}
     | NIf { annotationAST :: ann, condition :: Expr ann, ifStat :: ASTBlock ann, elseStat :: ASTBlock ann}
     | NFor { annotationAST :: ann, forVar :: Ident, forRange :: Expr ann, body :: ASTBlock ann}
     | NEmpty { annotationAST :: ann }
     | NPass { annotationAST :: ann }
     | NAssert { annotationAST :: ann, condition :: Expr ann }
     | NBp { annotationAST :: ann }
     | NWhile { annotationAST :: ann, condition :: Expr ann,  body :: ASTBlock ann}
     | NCall { annotationAST :: ann, callExpr :: Expr ann}
     | NCallWithInv { annotationAST :: ann, callExpr :: Expr ann, gateModifiers :: [GateModifier]}
     -- The tuple elements are: type, identifier, initilizer, length (only valid for array)
     | NDefvar { annotationAST :: ann, definitions :: [(Type ann, Ident, Maybe (Expr ann), Maybe (Expr ann))]}
     | NAssign { annotationAST :: ann, assignLhs :: Expr ann, assignRhs :: Expr ann, operator :: AssignOperator}
     | NGatedef { annotationAST :: ann, gateName :: String, gateRhs :: [[Expr ann]], externQirName :: Maybe String}
     | NReturn { annotationAST :: ann, returnedVal :: Expr ann}
     | NCoreUnitary { annotationAST :: ann, unitaryGate :: Expr ann, unitaryOperands :: [Expr ann], gateModifiers :: [GateModifier], rotation :: Maybe ([Expr ann])}
     | NCoreU3 { annotationAST :: ann, unitaryGate :: Expr ann, unitaryOperands :: [Expr ann], gateModifiers :: [GateModifier], angle :: [Expr ann]}
     | NCoreReset { annotationAST :: ann, resetOperands :: Expr ann}
     | NCorePrint { annotationAST :: ann, printOperands :: Expr ann}
     | NCoreMeasure {annotationAST :: ann, measExpr :: Expr ann}
     -- procedure my_rx(double theta, qbit a, qbit b) deriving gate { }
     -- bool database(bool flags[4], bool a, bool b) deriving oracle(2) { } 
     | NTopLevel { package :: Maybe (AST ann), importList :: [AST ann], defMemberList :: [AST ann] }
     | NPackage { annotationAST :: ann, packageName :: String }
     | NImport { annotationAST :: ann, importName :: String }
     | NProcedureWithDerive { annotationAST :: ann, procReturnType :: Type ann, procName :: String, procArgs :: [(Type ann, Ident)], procBody :: [AST ann], deriveGate :: Maybe DerivingType}
     | NContinue { annotationAST :: ann }
     | NBreak { annotationAST :: ann }
    -- Analysis/transformation intermediate nodes.
    -- Derived gates can be called as if they were procedures.
    -- Since deriving gates do not change their signature.
     | NResolvedExternGate { annotationAST :: ann, gateName :: String, extraArgs' :: [Type ()], gateSize :: Int, qirName :: String}
     | NDerivedGatedef { annotationAST :: ann, gateName :: String, sourceProcName :: String, extraArgs' :: [Type ()], gateSize :: Int}
     | NDerivedOracle { annotationAST :: ann, gateName :: String, sourceProcName :: String, extraArgs' :: [Type ()], gateSize :: Int}
     | NProcedure { annotationAST :: ann, procReturnType :: Type ann, procName :: String, procArgs :: [(Type ann, Ident)], procBody :: [AST ann]}
     | NResolvedFor { annotationAST :: ann, forVarId :: Int, forRange :: Expr ann, body :: ASTBlock ann}
     | NResolvedGatedef { annotationAST :: ann, gateName :: String, resolvedGateRhs :: [[Complex Double]], gateSize :: Int, externQirName :: Maybe String}
     | NOracleTable {annotationAST :: ann, gateName :: String, sourceProcName :: String, oracleValue :: [[Int]], gateSize :: Int}
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
     | NOracle { annotationAST :: ann, oracleName :: String, oracleN :: Int, oracleM :: Int, oracleMap :: [Expr ann] }
     | NOracleFunc { annotationAST :: ann, gateName :: String, oracleN :: Int, oracleM :: Int, inVar :: String, procBody :: [AST ann] }
     | NOracleLogic { annotationAST :: ann, resolvedProcReturnType :: Type (), procName :: String, logicArgs :: [(Type (), Ident)], procBody :: [AST ann] }
     | NResolvedOracleLogic { annotationAST :: ann, resolvedProcReturnType :: Type (), procName :: String, resolvedProcArgs :: [(Type (), Int)], procBody :: [AST ann] }
     deriving (Eq, Show, Functor)

data GateModifier = Inv | Ctrl Bool Int deriving (Show, Eq)

addedQubits :: GateModifier->Int
addedQubits Inv = 0
addedQubits (Ctrl _ x) = x

logicSuffix :: String
logicSuffix = "__logic"

type LAST = AST Pos
type LExpr = Expr Pos
type LType = Type Pos
instance Annotated AST where
  annotation = annotationAST

newtype InternalCompilerError = InternalCompilerError String deriving (Eq, Show)

data GrammarError = 
    BadMatrixElement {badExpr :: LExpr}
  | BadMatrixShape {badMatrix :: LAST}
  | BadGlobalVarSize {badDefPos :: Pos, badDefName :: String}
  | UnexpectedToken {token :: Token Pos}
  | UnexpectedEOF
  | BadPackageName {badPackageName :: String}
  | InconsistentRoot {importedFile :: String, rootPath :: String}
  | ReadFileError {badFileName :: String}
  | ImportNotFound {missingImport :: String}
  | DuplicatedImport {duplicatedImport :: String}
  | CyclicImport {cyclicImport :: [String]}
  | AmbiguousImport {ambImport :: String, candidate1 :: String, candidate2 :: String} deriving (Eq, Show)

foldConstantComplex :: LExpr->Either LExpr (Complex Double)
foldConstantComplex x@(EBinary _ op lhs rhs) = do
    l<-foldConstantComplex lhs
    r<-foldConstantComplex rhs
    case op of
      Add -> return $ l+r
      Sub -> return $ l-r
      Mul -> return $ l*r
      Div -> return $ l/r
      Pow -> return $ l**r
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

passVerifyDefgate :: [LAST] -> Either GrammarError [LAST]
passVerifyDefgate = mapM go where
    go g@(NGatedef ann name mat qir) = do
        new_mat<-mapM (mapM foldConstantComplex') mat
        case checkGateSize new_mat of
          Just sz -> return $ NResolvedGatedef ann name new_mat sz qir
          Nothing -> Left $ BadMatrixShape g

    go x = Right x

checkTopLevelVardef :: [LAST]->Either GrammarError [LAST]
checkTopLevelVardef = mapM go where
  go v@(NDefvar pos defs) = NDefvar pos <$> mapM go' defs where
    go' (Type ann (Array 0) [sub], b, Nothing, Just len) = do
      case len of
        EIntLit _ val -> Right (Type ann (Array val) [sub], b, Nothing, Nothing)
        other -> Left $ BadGlobalVarSize pos b
    go' (a, b, c, d) = Right (a, b, c, d)
  go v = return v
