module ISQ.Lang.MLIRTree where
import ISQ.Lang.ISQv2Grammar hiding (Bool)
import Control.Monad.Fix
import Data.List (intercalate)
import Text.Printf (printf)
import Data.Complex (Complex ((:+)))

data MLIRType =
    Bool | Index | Qubit | Memref (Maybe Int) MLIRType
  | Gate Int deriving Show
mlirType :: MLIRType->String
mlirType Bool = "i1"
mlirType Index = "index"
mlirType Qubit = "!isq.qubit"
mlirType (Memref Nothing ty) = "memref<?x" ++ mlirType ty ++ ">"
mlirType (Memref x ty) = "memref<"++show x++"x" ++ mlirType ty ++ ">"
mlirType (Gate x) = "!isq.gate<"++show x++">"
newtype BlockName = BlockName {unBlockName :: String} deriving Show
newtype SSA = SSA {unSsa :: String} deriving Show
newtype FuncName = FuncName {unFuncName :: String} deriving Show
data MLIRBlock = MLIRBlock {
    blockId :: BlockName,
    blockArgs :: [(MLIRType, SSA)],
    blockBody :: [MLIROp]
} deriving Show

data MLIRPos = MLIRLoc {
  fileName :: String, line :: String, column :: String
} | MLIRPosUnknown deriving Show

mlirPos :: MLIRPos->String
mlirPos (MLIRLoc file line column) = printf "loc(%s:%d:%d)" file line column
mlirPos MLIRPosUnknown = ""

data MLIRBinaryOp = MLIRBinaryOp {binaryOpType :: String, lhsType :: MLIRType, rhsType :: MLIRType, resultType :: MLIRType} deriving Show

mlirBinaryOp (a, b, c, d) = MLIRBinaryOp a b c d
mlirAddi = mlirBinaryOp ("addi", Index, Index, Index)
mlirSubi = mlirBinaryOp ("subi", Index, Index, Index)
mlirMuli = mlirBinaryOp ("muli", Index, Index, Index)
mlirDivsi = mlirBinaryOp ("divsi", Index, Index, Index)
mlirFloorDiv = mlirBinaryOp ("floordivsi", Index, Index, Index)
mlirSltI = mlirBinaryOp ("slt", Index, Index, Bool)
mlirSgtI = mlirBinaryOp ("sgt", Index, Index, Bool)
mlirSleI = mlirBinaryOp ("sle", Index, Index, Bool)
mlirSgeI = mlirBinaryOp ("sge", Index, Index, Bool)
mlirEqI = mlirBinaryOp ("eq", Index, Index, Bool)
mlirNeI = mlirBinaryOp ("ne", Index, Index, Bool)
mlirEqB = mlirBinaryOp ("eq", Bool, Bool, Bool)
mlirNeB = mlirBinaryOp ("ne", Bool, Bool, Bool)

data MLIRUnaryOp = MLIRUnaryOp {unaryOpType :: String, argType :: MLIRType, unaryResultType :: MLIRType} deriving Show

mlirUnaryOp (a, b, c) = MLIRUnaryOp a b c

mlirNeg = mlirUnaryOp ("negi", Index, Index)

data MLIROp =
      MFunc {location :: MLIRPos, funcName :: FuncName, funcReturnType :: Maybe MLIRType, funcRegion :: [MLIRBlock]}
    | MQDefGate { location :: MLIRPos, gateName :: FuncName, matrixRep :: [[Complex Double]], gateSize :: Int}
    | MQUseGate { location :: MLIRPos, value :: SSA, usedGate :: String, usedGateType :: MLIRType}
    | MQDecorate { location :: MLIRPos, value :: SSA, trait :: ([Bool], Bool), gateSize :: Int }
    | MQApplyGate{ location :: MLIRPos, values :: [SSA], qubitOperands :: [SSA], gateOperand :: SSA}
    | MQMeasure { location :: MLIRPos, measResult :: SSA, measQOut :: SSA, measQIn :: SSA}
    | MQReset { location :: MLIRPos, resetQOut :: SSA, resetQIn :: SSA}
    | MQPrint { location :: MLIRPos, printIn :: SSA}
    -- | MQCallQop { location :: MLIRPos, values :: [(MLIRType, SSA)], funcName :: FuncName, operands :: [(MLIRType, SSA)]}
    | MBinary {location :: MLIRPos, value :: SSA, lhs :: SSA, rhs :: SSA, bopType :: MLIRBinaryOp}
    | MUnary {location :: MLIRPos, value :: SSA, unaryOperand :: SSA, uopType :: MLIRUnaryOp}
    | MLoad {location :: MLIRPos, value :: SSA, array :: SSA, arrayOffset :: SSA}
    | MStore {location :: MLIRPos, array ::SSA, arrayOffset :: SSA, storedVal :: SSA}
    | MTakeRef {location :: MLIRPos, value :: SSA, array :: SSA, arrayOffset :: SSA}
    | MEraseMemref {location :: MLIRPos, value :: SSA, rankedMemref :: SSA}
    | MLitInt {location :: MLIRPos, value :: SSA, litInt :: Int}
    | MAllocMemref {location :: MLIRPos, value :: SSA, allocType :: MLIRType}
    | MFreeMemref {location :: MLIRPos, value :: SSA, freeType :: MLIRType}
    | MJmp {location :: MLIRPos, value :: SSA, jmpBlock :: String}
    | MBranch {location :: MLIRPos, value :: SSA, branches :: (String, String)}
    | MModule {location :: MLIRPos, topOps :: [MLIROp]}
    | MCall {location :: MLIRPos, values :: [SSA], funcName :: FuncName, operands :: [SSA]}
    deriving Show

data MLIREmitEnv = MLIREmitEnv {
  indent :: Int
} deriving Show

incrIndent :: MLIREmitEnv->MLIREmitEnv
incrIndent x = x {indent = 1+indent x}
emitWithIndent :: MLIREmitEnv->[String]->String
emitWithIndent env s = intercalate "\n" $ fmap (indented env) s

indented :: MLIREmitEnv->String->String
indented env = (replicate (4*indent env) ' '++)
blockArg :: (MLIRType, SSA)->String
blockArg (ty, ssa) = unSsa ssa ++ ": "++mlirType ty
blockHeader :: MLIREmitEnv->MLIRBlock->String
blockHeader env blk@(MLIRBlock id [] _) = indented env $ (unBlockName id) ++ ":"
blockHeader env blk@(MLIRBlock id args _) = indented env $ (unBlockName id) ++ "(" ++ (intercalate ", " $ fmap blockArg args)++ ")"
emitBlock :: (MLIREmitEnv->MLIROp->String)->MLIREmitEnv->MLIRBlock->String
emitBlock f env blk@(MLIRBlock id args body) = 
  let s = fmap (f (incrIndent env)) body in emitWithIndent env 
  ([blockHeader env blk]++s)

funcHeader :: MLIRPos->FuncName->Maybe MLIRType->[MLIRType]->String
funcHeader loc name ret args = printf "func %s(%s)%s %s " (unFuncName name) (intercalate ", " $ fmap mlirType args) (go ret) (mlirPos loc) where
  go Nothing = ""
  go (Just ty) = "->"++mlirType ty
printComplex :: Complex Double -> String
printComplex (a :+ b) = printf "#isq.complex<%f, %f>" a b

printRow :: (a->String)->[a]->String
printRow f xs = "["++intercalate "," (fmap f xs)++"]"

decorToDict :: ([Bool], Bool)->(String, Int)
decorToDict (a, b) = (printf "{ctrl = [%s], adjoint = %s}" (intercalate ", " $  map (\x->if x then "true" else "false") a) (if b then "true" else "false"), length a)


emitOpStep :: (MLIREmitEnv->MLIROp->String)->(MLIREmitEnv->MLIROp->String)
emitOpStep f env (MModule _ ops) =
  let s = fmap (f (incrIndent env)) ops in emitWithIndent env
  (["module{"] ++ s ++ ["}"])
emitOpStep f env (MFunc loc name ret blocks) = let s = fmap (emitBlock f env) blocks in emitWithIndent env ([funcHeader loc name ret (fmap fst $ blockArgs $ head blocks), "{"] ++ s ++ ["}"])
emitOpStep f env (MQDefGate loc name mat size) = indented env $ printf "isq.defgate %s {definition = [{type=\"unitary\", value = %s}]}: !isq.gate<%d> %s" (unFuncName name) (printRow (printRow printComplex) mat) size (mlirPos loc)
emitOpStep f env (MQDecorate loc value trait size) = let (d, sz) = decorToDict trait in indented env $ printf "%s = isq.decorate(%s: !isq.gate<%d> %s : !isq.gate<%d> %s)" (unSsa value) size d (size+sz) (mlirPos loc)
emitOpStep f env (MQApplyGate loc values args gate) = indented env $ printf "%s = isq.apply %s(%s) : !isq.gate<%d> %s" (intercalate ", " $ (fmap unSsa values)) (unSsa gate) (intercalate ", " $ (fmap (unSsa) args)) (length args) (mlirPos loc)
emitOpStep f env (MQMeasure loc result out arg) = indented env $ printf "%s, %s = isq.call_qop @isq_builtin::@measure(%s): [1]()->i1 %s" (unSsa out) (unSsa result) (unSsa arg) (mlirPos loc)
emitOpStep f env (MQReset loc out arg) = indented env $ printf "%s = isq.call_qop @isq_builtin::@reset(%s): [1]()->() %s" (unSsa out)  (unSsa arg) (mlirPos loc)
emitOpStep f env (MQPrint loc arg) = indented env $ printf "isq.call_qop @isq_builtin::@print_int(%s): [1]()->() %s" (unSsa arg) (mlirPos loc)
emitOpStep f env (MBinary loc value lhs rhs (MLIRBinaryOp op lt rt rest)) = indented env $ printf "%s = arith.%s %s, %s : %s %s" (unSsa value) op (unSsa lhs) (unSsa rhs) (mlirType lt) (mlirPos loc)
emitOpStep f env (MUnary loc value arg (MLIRUnaryOp op at rest)) = indented env $ printf "%s = arith.%s %s : %s %s" (unSsa value) op (unSsa arg) (mlirType at) (mlirPos loc)

emitOp :: MLIREmitEnv -> MLIROp -> String
emitOp = fix emitOpStep

