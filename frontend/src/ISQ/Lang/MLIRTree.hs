module ISQ.Lang.MLIRTree where
import ISQ.Lang.ISQv2Grammar hiding (Bool, Gate, Double)
import Control.Monad.Fix
import Data.List (intercalate)
import Text.Printf (printf)
import Data.Complex (Complex ((:+)))
import GHC.Stack (HasCallStack)

data MLIRType =
    Bool | I2 | I64 | Index | QState | BorrowedRef MLIRType | Memref (Maybe Int) MLIRType
  | Gate Int | Double | QIRQubit deriving Show
mlirType :: MLIRType->String
mlirType Bool = "i1"
mlirType I2 = "i2"
mlirType I64 = "i64"
mlirType Index = "index"
mlirType Double = "f64"
mlirType QState = "!isq.qstate"
mlirType QIRQubit = "!isq.qir.qubit"
mlirType (Memref Nothing ty) = "memref<?x" ++ mlirType ty ++ ">"
mlirType (Memref (Just x) ty) = "memref<"++show x++"x" ++ mlirType ty ++ ">"
mlirType (BorrowedRef ty) = "memref<1x"++ mlirType ty ++", affine_map<(d0)[s0]->(d0+s0)>>"
mlirType (Gate x) = "!isq.gate<"++show x++">"
newtype BlockName = BlockName {unBlockName :: String} deriving Show
newtype SSA = SSA {unSsa :: String} deriving Show
newtype FuncName = FuncName {unFuncName :: String} deriving Show

fromBlockName :: Int->BlockName
fromBlockName = BlockName . printf "^block%d"
fromFuncName :: String->FuncName
fromFuncName = FuncName . printf "@%s" . show
fromSSA :: Int->SSA
fromSSA = SSA . printf "%%ssa_%d"
data MLIRBlock = MLIRBlock {
    blockId :: BlockName,
    blockArgs :: [(MLIRType, SSA)],
    blockBody :: [MLIROp]
} deriving Show

data MLIRPos = MLIRLoc {
  fileName :: String, line :: Int, column :: Int
} | MLIRPosUnknown deriving Show

mlirPos :: MLIRPos->String
mlirPos (MLIRLoc file line column) = printf "loc(%s:%d:%d)" (show file) line column
mlirPos MLIRPosUnknown = ""

data MLIRBinaryOp = MLIRBinaryOp {binaryOpType :: String, lhsType :: MLIRType, rhsType :: MLIRType, resultType :: MLIRType} deriving Show

mlirBinaryOp (a, b, c, d) = MLIRBinaryOp ("arith."++a) b c d
mlirMathBinaryOp (a,b,c,d) = MLIRBinaryOp ("math."++a) b c d
mlirAddi = mlirBinaryOp ("addi", Index, Index, Index)
mlirSubi = mlirBinaryOp ("subi", Index, Index, Index)
mlirMuli = mlirBinaryOp ("muli", Index, Index, Index)
--mlirDivsi = mlirBinaryOp ("divsi", Index, Index, Index)
mlirRemsi = mlirBinaryOp ("remsi", Index, Index, Index)
mlirFloorDivsi = mlirBinaryOp ("floordivsi", Index, Index, Index) -- Use this by default
mlirAnd = mlirBinaryOp ("andi", Bool, Bool, Bool)
mlirOr = mlirBinaryOp ("ori", Bool, Bool, Bool)
mlirAndi = mlirBinaryOp ("andi", Index, Index, Index)
mlirOri = mlirBinaryOp ("ori", Index, Index, Index)
mlirXori = mlirBinaryOp ("xori", Index, Index, Index)
mlirAddf = mlirBinaryOp ("addf", Double, Double, Double)
mlirSubf = mlirBinaryOp ("subf", Double, Double, Double)
mlirMulf = mlirBinaryOp ("mulf", Double, Double, Double)
mlirDivf = mlirBinaryOp ("divf", Double, Double, Double)
mlirPowf = mlirMathBinaryOp ("powf", Double, Double, Double)
mlirSltI = mlirBinaryOp ("cmpi \"slt\",", Index, Index, Bool)
mlirSgtI = mlirBinaryOp ("cmpi \"sgt\",", Index, Index, Bool)
mlirSleI = mlirBinaryOp ("cmpi \"sle\",", Index, Index, Bool)
mlirSgeI = mlirBinaryOp ("cmpi \"sge\",", Index, Index, Bool)
mlirEqI = mlirBinaryOp ("cmpi \"eq\",", Index, Index, Bool)
mlirNeI = mlirBinaryOp ("cmpi \"ne\",", Index, Index, Bool)
mlirSltF = mlirBinaryOp ("cmpf \"slt\",", Index, Index, Bool)
mlirSgtF = mlirBinaryOp ("cmpf \"sgt\",", Index, Index, Bool)
mlirSleF = mlirBinaryOp ("cmpf \"sle\",", Index, Index, Bool)
mlirSgeF = mlirBinaryOp ("cmpf \"sge\",", Index, Index, Bool)
mlirEqF = mlirBinaryOp ("cmpf \"eq\",", Index, Index, Bool)
mlirNeF = mlirBinaryOp ("cmpf \"ne\",", Index, Index, Bool)
mlirEqB = mlirBinaryOp ("cmpi \"eq\",", Bool, Bool, Bool)
mlirNeB = mlirBinaryOp ("cmpi \"ne\",", Bool, Bool, Bool)
mlirShl = mlirBinaryOp ("shli", Index, Index, Index)
mlirShr = mlirBinaryOp ("shrui", Index, Index, Index)

data MLIRUnaryOp = MLIRUnaryOp {unaryOpType :: String, argType :: MLIRType, unaryResultType :: MLIRType} deriving Show

mlirUnaryOp (a, b, c) = MLIRUnaryOp ("arith."++a) b c

mlirNegI = mlirUnaryOp ("negi", Index, Index)
mlirNegF = mlirUnaryOp ("negf", Double, Double)

mlirI1toI2 = mlirUnaryOp ("extui", Bool, I2)
mlirI2toIndex = mlirUnaryOp ("index_cast", I2, Index)
mlirIndextoI64 = mlirUnaryOp ("index_cast", Index, I64)
mlirI64toDouble = mlirUnaryOp ("sitofp", I64, Double)

type TypedSSA = (MLIRType, SSA)

data GateRep = MatrixRep [[Complex Double]] | QIRRep FuncName | DecompositionRep FuncName | OracleRep FuncName | OracleTableRep [[Int]] deriving Show

gateRep :: GateRep->String
gateRep (MatrixRep mat) = printf "{type=\"unitary\", value = %s}" (printRow (printRow printComplex) mat)
gateRep (QIRRep fn) = printf "{type = \"qir\", value = %s}" (unFuncName fn)
gateRep (DecompositionRep fn) = printf "{type = \"decomposition_raw\", value = %s}" (unFuncName fn)
gateRep (OracleRep fn) = printf "{type = \"oracle\", value = %s}" (unFuncName fn)
gateRep (OracleTableRep mat) = printf "{type = \"oracle_table\", value = %s}" (printRow (printRow printInt) mat)


data MLIROp =
      MFunc {location :: MLIRPos, funcName :: FuncName, funcReturnType :: Maybe MLIRType, funcRegion :: [MLIRBlock]}
    | MQDefGate { location :: MLIRPos, gateName :: FuncName, gateSize :: Int, extraArgTypes :: [MLIRType], representations :: [GateRep]}
    | MQOracleTable { location :: MLIRPos, gateName :: FuncName, gateSize :: Int, representations :: [GateRep] }
    | MQUseGate { location :: MLIRPos, value :: SSA, usedGate :: FuncName, usedGateType :: MLIRType, useGateParams :: [(MLIRType, SSA)]}
    | MExternFunc { location :: MLIRPos, funcName :: FuncName, funcReturnType :: Maybe MLIRType, funcArgTypes :: [MLIRType]}
    | MQDecorate { location :: MLIRPos, value :: SSA, decoratedGate :: SSA, trait :: ([Bool], Bool), gateSize :: Int }
    | MQApplyGate{ location :: MLIRPos, values :: [SSA], qubitOperands :: [SSA], gateOperand :: SSA}
    | MQMeasure { location :: MLIRPos, measResult :: SSA, measQOut :: SSA, measQIn :: SSA}
    | MQReset { location :: MLIRPos, resetQOut :: SSA, resetQIn :: SSA}
    | MQPrint { location :: MLIRPos, printIn :: (MLIRType, SSA)}
    -- | MQCallQop { location :: MLIRPos, values :: [(MLIRType, SSA)], funcName :: FuncName, operands :: [(MLIRType, SSA)]}
    | MBinary {location :: MLIRPos, value :: SSA, lhs :: SSA, rhs :: SSA, bopType :: MLIRBinaryOp}
    | MUnary {location :: MLIRPos, value :: SSA, unaryOperand :: SSA, uopType :: MLIRUnaryOp}
    | MCast {location::MLIRPos, value :: SSA, unaryOperand :: SSA, uopType :: MLIRUnaryOp}
    | MLoad {location :: MLIRPos, value :: SSA, array :: (MLIRType, SSA)}
    | MStore {location :: MLIRPos, array :: (MLIRType, SSA), storedVal :: SSA}
    | MTakeRef {location :: MLIRPos, value :: SSA, array :: (MLIRType, SSA), arrayOffset :: SSA}
    | MEraseMemref {location :: MLIRPos, value :: SSA, rankedMemref :: (MLIRType, SSA)}
    | MLitInt {location :: MLIRPos, value :: SSA, litInt :: Int}
    | MLitBool {location :: MLIRPos, value :: SSA, litBool :: Bool}
    | MLitDouble {location :: MLIRPos, value :: SSA, litDouble :: Double}
    | MAllocMemref {location :: MLIRPos, value :: SSA, allocType :: MLIRType}
    | MFreeMemref {location :: MLIRPos, value :: SSA, freeType :: MLIRType}
    | MJmp {location :: MLIRPos, jmpBlock :: BlockName}
    | MBranch {location :: MLIRPos, value :: SSA, branches :: (BlockName, BlockName)}
    | MModule {location :: MLIRPos, topOps :: [MLIROp]}
    | MCall {location :: MLIRPos, callRet :: Maybe (MLIRType, SSA), funcName :: FuncName, operands :: [(MLIRType, SSA)]}
    | MSCFIf {location :: MLIRPos, ifCondition :: SSA, thenRegion :: MLIROp, elseRegion :: MLIROp}
    | MSCFWhile {location :: MLIRPos, breakBlock :: [MLIROp], condBlock :: [MLIROp], condExpr :: SSA, breakCond :: SSA, whileBody :: [MLIROp]}
    | MAffineFor {location :: MLIRPos, forLo :: SSA, forHi :: SSA, forStep :: Int, forVar :: SSA, forRegion :: [MLIROp]}
    | MSCFFor {location :: MLIRPos, forLo :: SSA, forHi :: SSA, forStep :: Int, forVar :: SSA, forRegion :: [MLIROp]}
    | MSCFExecRegion {location :: MLIRPos, blocks :: [MLIRBlock]}
    | MSCFYield {location :: MLIRPos}
    | MReturn {location :: MLIRPos, returnVal :: TypedSSA}
    | MReturnUnit {location :: MLIRPos}
    | MBp {location :: MLIRPos, bpLine :: SSA}
    | MGlobalMemref {location :: MLIRPos, globalMemrefName :: FuncName, globalMemrefType :: MLIRType}
    | MUseGlobalMemref {location :: MLIRPos, usedVal :: SSA, usedName :: FuncName, globalMemrefType :: MLIRType}
    deriving Show

data MLIREmitEnv = MLIREmitEnv {
  indent :: Int, isTopLevel :: Bool
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
blockHeader env blk@(MLIRBlock id args _) = indented env $ (unBlockName id) ++ "(" ++ (intercalate ", " $ fmap blockArg args)++ "):"
emitBlock :: (MLIREmitEnv->MLIROp->String)->MLIREmitEnv->MLIRBlock->String
emitBlock f env blk@(MLIRBlock id args body) =
  let s = fmap (f (incrIndent env)) body in intercalate "\n"
  ([blockHeader env blk]++s)

funcHeader :: FuncName->Maybe MLIRType->[MLIRType]->String
funcHeader name ret args = printf "func %s(%s)%s " (unFuncName name) (intercalate ", " $ fmap mlirType args) (go ret)  where
  go Nothing = ""
  go (Just ty) = "->"++mlirType ty
printComplex :: Complex Double -> String
printComplex (a :+ b) = printf "#isq.complex<%f, %f>" a b

printInt :: Int -> String
printInt x = printf "%d" x

printRow :: (a->String)->[a]->String
printRow f xs = "["++intercalate "," (fmap f xs)++"]"

decorToDict :: ([Bool], Bool)->(String, Int)
decorToDict (a, b) = (printf "{ctrl = [%s], adjoint = %s}" (intercalate ", " $ map (\x->if x then "true" else "false") a) (if b then "true" else "false"), length a)


data AffineSet = AffineEq | AffineSlt | AffineSle | AffineSgt | AffineSge deriving Show

affineSet AffineEq = "()[s0, s1]: (s0-s1 == 0)"
-- s0-s1<0 s0-s1+1<=0
affineSet AffineSlt = "()[s0, s1]: (s1-s0-1 >= 0)"
-- s0-s1<=0
affineSet AffineSle = "()[s0, s1]: (s1-s0 >= 0)"
-- s0-s1>0 s0-s1-1>=0
affineSet AffineSgt = "()[s0, s1]: (s0-s1-1 >= 0)"
-- s0-s1>=0  ++ [indented env $ "affine.yield"]
affineSet AffineSge = "()[s0, s1]: (s0-s1 >= 0)"

emitOpStep :: (MLIREmitEnv->MLIROp->String)->(MLIREmitEnv->MLIROp->String)
emitOpStep f env (MModule _ ops) =
  let s = fmap (f (incrIndent env)) ops in intercalate "\n" $
  ([indented env "module{"] ++ [
      indented env $ "    isq.declare_qop @__isq__builtin__measure : [1]()->i1",
      indented env $ "    isq.declare_qop @__isq__builtin__reset : [1]()->()",
      indented env $ "    isq.declare_qop @__isq__builtin__bp : [0](index)->()",
      indented env $ "    isq.declare_qop @__isq__builtin__print_int : [0](index)->()",
      indented env $ "    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()"
  ]++s ++ [indented env "}"])
emitOpStep f env (MFunc loc name ret blocks) = let s = fmap (emitBlock f env) blocks in intercalate "\n" ([indented env $ funcHeader name ret (fmap fst $ blockArgs $ head blocks), indented env "{"] ++ s ++ [indented env $ printf "} %s" (mlirPos loc)])
emitOpStep f env (MQDefGate loc name size extra reps) = indented env $ printf "isq.defgate %s%s {definition = [%s]}: !isq.gate<%d> %s" (unFuncName name) (case extra of {[]->""; xs-> "("++intercalate ", " (map mlirType extra)++")"}) (intercalate ", " $ map gateRep reps) size (mlirPos loc)
emitOpStep f env (MQOracleTable loc name size reps) = indented env $ printf "isq.defgate %s {definition=[%s]}: !isq.gate<%d> %s" (unFuncName name) (intercalate ", " $ map gateRep reps) size (mlirPos loc)
emitOpStep f env (MExternFunc loc name Nothing args) = indented env $ printf "func private %s(%s) %s" (unFuncName name) (intercalate ", " $ map mlirType args) (mlirPos loc)
emitOpStep f env (MExternFunc loc name (Just returns) args) = indented env $ printf "func private %s(%s)->%s %s"(unFuncName name) (intercalate ", " $ map mlirType args) (mlirType returns) (mlirPos loc)
emitOpStep f env (MQUseGate loc val usedgate usedtype@(Gate sz) []) = indented env $ printf "%s = isq.use %s : !isq.gate<%d> %s " (unSsa val) (unFuncName usedgate) sz (mlirPos loc)
emitOpStep f env (MQUseGate loc val usedgate usedtype@(Gate sz) xs) = indented env $ printf "%s = isq.use %s(%s) : (%s) -> !isq.gate<%d> %s " (unSsa val) (unFuncName usedgate) (intercalate ", " $ fmap (unSsa.snd) xs) (intercalate ", " $ fmap (mlirType.fst) xs) sz (mlirPos loc)
emitOpStep f env (MQUseGate loc val usedgate usedtype _) = error "wtf?"
emitOpStep f env (MQDecorate loc value source trait size) = let (d, sz) = decorToDict trait in indented env $ printf "%s = isq.decorate(%s: !isq.gate<%d>) %s : !isq.gate<%d> %s" (unSsa value) (unSsa source) size d (size+sz) (mlirPos loc)
emitOpStep f env (MQApplyGate loc values [] gate) = indented env $ printf "isq.apply_gphase %s : !isq.gate<0> %s" (unSsa gate) (mlirPos loc)
emitOpStep f env (MQApplyGate loc values args gate) = indented env $ printf "%s = isq.apply %s(%s) : !isq.gate<%d> %s" (intercalate ", " $ (fmap unSsa values)) (unSsa gate) (intercalate ", " $ (fmap (unSsa) args)) (length args) (mlirPos loc)
emitOpStep f env (MQMeasure loc result out arg) = indented env $ printf "%s, %s = isq.call_qop @__isq__builtin__measure(%s): [1]()->i1 %s" (unSsa out) (unSsa result) (unSsa arg) (mlirPos loc)
emitOpStep f env (MQReset loc out arg) = indented env $ printf "%s = isq.call_qop @__isq__builtin__reset(%s): [1]()->() %s" (unSsa out)  (unSsa arg) (mlirPos loc)
emitOpStep f env (MBp loc arg) = indented env $ printf "isq.call_qop @__isq__builtin__bp(%s): [0](index)->() %s" (unSsa arg) (mlirPos loc)
emitOpStep f env (MQPrint loc (Index, arg)) = indented env $ printf "isq.call_qop @__isq__builtin__print_int(%s): [0](index)->() %s" (unSsa arg) (mlirPos loc)
emitOpStep f env (MQPrint loc (Double, arg)) = indented env $ printf "isq.call_qop @__isq__builtin__print_double(%s): [0](f64)->() %s" (unSsa arg) (mlirPos loc)
emitOpStep f env (MQPrint loc (t, arg)) = error $ "unsupported "++ show t
emitOpStep f env (MBinary loc value lhs rhs (MLIRBinaryOp "arith.floordivsi" lt rt rest)) =  intercalate "\n" $
  [
    indented env $ printf "%s_zero = arith.constant 0: index %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s_unequal = arith.cmpi \"ne\", %s, %s_zero : index %s" (unSsa value) (unSsa rhs) (unSsa value) (mlirPos loc),
    indented env $ printf "isq.assert %s_unequal : i1, 1 %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s = arith.floordivsi %s, %s : %s %s" (unSsa value) (unSsa lhs) (unSsa rhs) (mlirType lt) (mlirPos loc)
  ]
emitOpStep f env (MBinary loc value lhs rhs (MLIRBinaryOp "arith.divf" lt rt rest)) =  intercalate "\n" $
  [
    indented env $ printf "%s_zero = arith.constant 0.0: f64 %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s_unequal = arith.cmpf \"one\", %s, %s_zero : f64 %s" (unSsa value) (unSsa rhs) (unSsa value) (mlirPos loc),
    indented env $ printf "isq.assert %s_unequal : i1, 1 %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s = arith.divf %s, %s : %s %s" (unSsa value) (unSsa lhs) (unSsa rhs) (mlirType lt) (mlirPos loc)
  ]
emitOpStep f env (MBinary loc value lhs rhs (MLIRBinaryOp op lt rt rest)) = indented env $ printf "%s = %s %s, %s : %s %s" (unSsa value) op (unSsa lhs) (unSsa rhs) (mlirType lt) (mlirPos loc)
emitOpStep f env (MUnary loc value arg (MLIRUnaryOp op at rest)) = indented env $ printf "%s = %s %s : %s %s" (unSsa value) op (unSsa arg) (mlirType at) (mlirPos loc)
emitOpStep f env (MCast loc value arg (MLIRUnaryOp op at rest)) = indented env $ printf "%s = %s %s : %s to %s %s" (unSsa value) op (unSsa arg) (mlirType at) (mlirType rest) (mlirPos loc)
emitOpStep f env (MLoad loc value (arr_type, arr_val)) = intercalate "\n" $
  [indented env $ printf "%s_load_zero = arith.constant 0: index %s" (unSsa value) (mlirPos loc),
  indented env $ printf "%s = affine.load %s[%s_load_zero] : %s %s" (unSsa value) (unSsa arr_val) (unSsa value) (mlirType arr_type) (mlirPos loc)]

emitOpStep f env (MStore loc (arr_type, arr_val) value) = intercalate "\n" $
  [
    indented env $ printf "%s_store_zero = arith.constant 0: index %s" (unSsa value) (mlirPos loc),
    indented env $ printf "affine.store %s, %s[%s_store_zero] : %s %s" (unSsa value) (unSsa arr_val) (unSsa value) (mlirType arr_type) (mlirPos loc)
  ]
emitOpStep f env (MTakeRef loc value (arr_ty@(Memref _ elem_ty), arr_val) offset) = intercalate "\n" $
  [
    indented env $ printf "%s_zero = arith.constant 0 : index %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s_length = memref.dim %s, %s_zero : %s %s" (unSsa value) (unSsa arr_val) (unSsa value) (mlirType arr_ty) (mlirPos loc),
    indented env $ printf "%s_less = arith.cmpi \"slt\", %s, %s_length : index %s" (unSsa value) (unSsa offset) (unSsa value) (mlirPos loc),
    indented env $ printf "%s_non_neg = arith.cmpi \"sge\", %s, %s_zero : index %s" (unSsa value) (unSsa offset) (unSsa value) (mlirPos loc),
    indented env $ printf "%s_both = arith.andi %s_less, %s_non_neg : i1 %s" (unSsa value) (unSsa value) (unSsa value) (mlirPos loc),
    indented env $ printf "isq.assert %s_both : i1, 2 %s" (unSsa value) (mlirPos loc),
    indented env $ printf "%s = memref.subview %s[%s][1][1] : %s to %s %s" (unSsa value) (unSsa arr_val) (unSsa offset) (mlirType arr_ty) (mlirType $ BorrowedRef elem_ty) (mlirPos loc)
  ]
emitOpStep f env (MTakeRef loc value (arr_ty, arr_val) offset) = error "wtf?"
emitOpStep f env (MEraseMemref loc value (arr_ty@(Memref (Just x) elem_ty), arr_val)) = indented env $ printf "%s = memref.cast %s : %s to %s %s" (unSsa value) (unSsa arr_val) (mlirType arr_ty) (mlirType $ Memref Nothing elem_ty) (mlirPos loc)
emitOpStep f env (MEraseMemref loc value (arr_ty, arr_val)) = error "wtf?"
emitOpStep f env (MLitInt loc value val) = indented env $ printf "%s = arith.constant %d : index %s" (unSsa value) val (mlirPos loc)
emitOpStep f env (MLitBool loc value val) = indented env $ printf "%s = arith.constant %d : i1 %s" (unSsa value) (if val then 1::Int else 0) (mlirPos loc)
emitOpStep f env (MLitDouble loc value val) = indented env $ printf "%s = arith.constant %f : f64 %s" (unSsa value) val (mlirPos loc)

-- TODO: remove this dirty hack
emitOpStep f env (MAllocMemref loc val ty@(BorrowedRef subty)) = intercalate "\n" $ fmap (indented env) [
  printf "%s_real = memref.alloc() : memref<1x%s> %s" (unSsa val) (mlirType subty) (mlirPos loc),
  printf "%s_zero = arith.constant 0 : index" (unSsa val),
  printf "%s = memref.subview %s_real[%s_zero][1][1] : memref<1x%s> to %s %s" (unSsa val) (unSsa val) (unSsa val) (mlirType subty) (mlirType ty) (mlirPos loc)]
emitOpStep f env (MAllocMemref loc val ty) = indented env $ printf "%s = memref.alloc() : %s %s" (unSsa val) (mlirType ty) (mlirPos loc)
emitOpStep f env (MFreeMemref loc val ty@(BorrowedRef subty)) = intercalate "\n" $ [indented env $ printf "isq.accumulate_gphase %s_real : memref<1x%s> %s" (unSsa val) (mlirType subty) (mlirPos loc),
  indented env $ printf "memref.dealloc %s_real : memref<1x%s> %s" (unSsa val) (mlirType subty) (mlirPos loc)]
emitOpStep f env (MFreeMemref loc val ty) = intercalate "\n" $ [indented env $ printf "isq.accumulate_gphase %s : %s %s" (unSsa val) (mlirType ty) (mlirPos loc),
  indented env $ printf "memref.dealloc %s : %s %s" (unSsa val) (mlirType ty) (mlirPos loc)]
emitOpStep f env (MJmp loc blk) = indented env $ printf "br %s %s" (unBlockName blk) (mlirPos loc)
emitOpStep f env (MBranch loc val (trueDst, falseDst)) = indented env $ printf "cond_br %s, %s, %s %s" (unSsa val) (unBlockName trueDst) (unBlockName falseDst) (mlirPos loc)
emitOpStep f env (MCall loc Nothing fn args) = indented env $ printf "call %s(%s) : (%s)->() %s" (unFuncName fn) (intercalate ", " $ fmap (unSsa.snd) args) (intercalate ", " $ fmap (mlirType.fst) args) (mlirPos loc)
emitOpStep f env (MCall loc (Just (retty, retval)) fn args) = indented env $ printf "%s = call %s(%s) : (%s)->%s %s" (unSsa retval) (unFuncName fn) (intercalate ", " $ fmap (unSsa.snd) args) (intercalate ", " $ fmap (mlirType.fst) args) (mlirType retty) (mlirPos loc)
emitOpStep f env (MSCFIf loc cond then' else') = intercalate "\n" $ [
  indented env $ printf "scf.if %s {" $ unSsa cond]
  ++[f (incrIndent env{isTopLevel=False}) then']
  ++[indented env $ "} else {"]
  ++[f (incrIndent env{isTopLevel=False}) else']
  ++[indented env $ "}"]
emitOpStep f env (MSCFWhile loc breakb condb cond break body) = intercalate "\n" $
  [indented env $ "scf.while : ()->() {"]
  ++ [indented env $ "    %cond = scf.execute_region->i1 {"]
  ++ [indented env $ "        ^break_check:"]
  ++ fmap (f (incrIndent $ incrIndent $ incrIndent env)) breakb
  ++ [indented env $ printf "            cond_br %s, ^break, ^while_cond" (unSsa break)]
  ++ [indented env $ printf "        ^while_cond:"]
  ++ fmap (f (incrIndent $ incrIndent $ incrIndent env)) condb
  ++ [indented env $ printf "            scf.yield %s: i1" (unSsa cond)]
  ++ [indented env $ "        ^break:" ]
  ++ [indented env $ "            %zero=arith.constant 0: i1" ]
  ++ [indented env $ "            scf.yield %zero: i1" ]
  ++ [indented env $ "    }"]
  ++ [indented env $ "    scf.condition(%cond)"]
  ++ [indented env $ "} do {"]
  ++ fmap (f (incrIndent env{isTopLevel=False})) body
  ++ [indented env $ "scf.yield"]
  ++ [indented env $ printf "} %s" (mlirPos loc)]
emitOpStep f env (MSCFExecRegion loc blocks) = intercalate "\n"
  ([indented env $ "scf.execute_region {"]
  ++ fmap (emitBlock f env) blocks ++
  [indented env $ printf "} %s" (mlirPos loc)])
emitOpStep f env (MSCFYield loc) = indented env $ printf "scf.yield %s" (mlirPos loc)
emitOpStep f env@MLIREmitEnv{isTopLevel=True} (MAffineFor loc lo hi step var body) = intercalate "\n" $
  [indented env $ printf "affine.for %s = %s to %s step %d {" (unSsa var) (unSsa lo) (unSsa hi) step]
  ++ fmap (f (incrIndent env)) body
  ++ [indented env $ printf "} %s" (mlirPos loc)]
emitOpStep f env (MAffineFor loc lo hi step var body) = emitOpStep f env (MSCFFor loc lo hi step var body)
emitOpStep f env (MSCFFor loc lo hi step var body) = intercalate "\n" $
  [ indented env $ printf "%s_one = arith.constant %d : index %s" (unSsa var) step (mlirPos loc),
    indented env $ printf "scf.for %s = %s to %s step %s_one {" (unSsa var) (unSsa lo) (unSsa hi) (unSsa var)]
  ++ fmap (f (incrIndent (env{isTopLevel=False}))) body
  ++ [indented env $ printf "} %s" (mlirPos loc)]
emitOpStep f env (MReturn loc (ty, v)) = indented env $ printf "return %s : %s %s" (unSsa v) (mlirType ty) (mlirPos loc)
emitOpStep f env (MReturnUnit loc) = indented env $ printf "return %s" (mlirPos loc)
--emitOpStep f env (MBp loc) = indented env $ printf "isq.bp %s" (mlirPos loc)
emitOpStep f env (MGlobalMemref loc name ty@(BorrowedRef subty)) = indented env $ printf "memref.global %s : memref<1x%s> = uninitialized %s"  (unFuncName name) (mlirType subty) (mlirPos loc)
emitOpStep f env (MGlobalMemref loc name ty) = indented env $ printf "memref.global %s : %s = uninitialized %s"  (unFuncName name) (mlirType ty) (mlirPos loc)
emitOpStep f env (MUseGlobalMemref loc val name ty@(BorrowedRef subty)) = intercalate "\n" $ fmap (indented env) [
  printf "%s_uncast = memref.get_global %s : memref<1x%s> %s" (unSsa val) (unFuncName name) (mlirType subty) (mlirPos loc),
  printf "%s_zero = arith.constant 0 : index" (unSsa val),
  printf "%s = memref.subview %s_uncast[%s_zero][1][1] : memref<1x%s> to %s %s" (unSsa val) (unSsa val) (unSsa val) (mlirType subty) (mlirType ty) (mlirPos loc)
  ]
emitOpStep f env (MUseGlobalMemref loc val name ty) = indented env $ printf "%s = memref.get_global %s : %s %s" (unSsa val) (unFuncName name) (mlirType ty) (mlirPos loc)
emitOp' :: MLIREmitEnv -> MLIROp -> String
emitOp' = fix emitOpStep

emitOp :: MLIROp -> String
emitOp = emitOp' (MLIREmitEnv 0 True)