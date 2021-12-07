{-# LANGUAGE GeneralisedNewtypeDeriving, TemplateHaskell #-}
module ISQ.Lang.MLIRGen where
import ISQ.Lang.AST
import ISQ.Lang.Codegen hiding (ssa)
import Control.Lens
import qualified Data.Map as Map
import Control.Monad.State.Lazy
import Text.Printf
import Text.Parsec.Pos
import Data.List (intercalate)
import Data.Complex
import Data.Either

data MLIRGenState = MLIRGenState {
    _generatedCodes :: [[String]],
    _overloadedGlobalVars :: Map.Map SSA (VarDef Pos),
    _tempSSAs :: Int,
    _extraIndent :: Int
} deriving Show
makeLenses ''MLIRGenState

newtype MLIRGen a = MLIRGen{
    toState :: State MLIRGenState a
} deriving (MonadState MLIRGenState, Functor, Applicative, Monad)

locationInfo :: SourcePos -> String
locationInfo pos = printf "loc(%s:%d:%d)" (show $ sourceName pos) (sourceLine pos) (sourceColumn pos)

nextTempSSA :: MLIRGen String
nextTempSSA = do
    s <- use tempSSAs
    tempSSAs += 1
    return $ "t" ++ show s
indentStr :: String
indentStr = "    "
emit :: String->MLIRGen ()
emit s = do
    c<-use generatedCodes
    e<-use extraIndent
    let t = concat $ replicate e indentStr
    -- Append code in reversed direction!
    generatedCodes %= over _head ((t++s):)
emitBlock :: [String]->MLIRGen ()
emitBlock s = do
    c<-use generatedCodes
    -- Append code in reversed direction!
    mapM_ (\l->generatedCodes %= over _head (l:)) s
popBlock :: MLIRGen [String]
popBlock = do
    c<-use generatedCodes
    generatedCodes %= tail
    return $ reverse $ head c

ssa :: String->String
ssa = ("%"++)
--emit s = do

declareGlobalSSAOverride :: SSA->VarDef Pos->MLIRGen ()
declareGlobalSSAOverride ssa_val global_name = do
    overloadedGlobalVars.at ssa_val .= Just global_name

appendDim :: Bool->String
appendDim False = ""
appendDim True = ", affine_map<(d0)[s0]->(d0+s0)>"
mlirGlobalType :: Bool->VarType ann -> String
mlirGlobalType dimed (UnitType Int _) = "memref<1xindex"++(appendDim dimed)++">"
mlirGlobalType dimed (UnitType Qbit _) = "memref<1x!isq.qstate"++(appendDim dimed)++">"
mlirGlobalType dimed (Composite Int (Just n) _) = "memref<" ++ show n ++ "xindex"++(appendDim dimed)++">"
mlirGlobalType dimed (Composite Qbit (Just n) _) = "memref<" ++ show n ++ "x!isq.qstate"++(appendDim dimed)++">"
mlirGlobalType dimed (Composite Int Nothing _) = "memref<?xindex"++(appendDim dimed)++">"
mlirGlobalType dimed (Composite Qbit Nothing _) = "memref<?x!isq.qstate"++(appendDim dimed)++">"
mlirGlobalType _ _ = error "mlirGlobalType: unsupported type"

mlirArgType :: VarType ann -> String
mlirArgType (UnitType Int _) = "index"
mlirArgType x = mlirGlobalType True x

knownVarDefArrLen :: VarDef ann -> Int 
knownVarDefArrLen (VarDef (Composite _ (Just n) _) _ _) = n
knownVarDefArrLen (VarDef (Composite _ (Nothing) _) _ _) = error "array length is not known"
knownVarDefArrLen (VarDef (UnitType _ _) _ _) = 1
knownVarDefArrLen _ = error "knownVarDefArrLen: unsupported type"

overrideGlobalSSA :: Pos->SSA->MLIRGen SSA
overrideGlobalSSA p ssa_val = do
    l<-use $ overloadedGlobalVars.at ssa_val
    case l of
        Nothing -> return ssa_val -- Do nothing.
        Just v -> do
            -- Load from global.
            temp_ssa<-nextTempSSA
            temp_zero <- nextTempSSA
            casted_temp_ssa<-nextTempSSA
            emit $ printf "%s = memref.get_global @%s : %s %s" (ssa temp_ssa) (v^.varName^.identName) (mlirGlobalType False (v^.varType)) (locationInfo p)
            emit $ printf "%s = arith.constant 0 : index %s" (ssa temp_zero) (locationInfo p)
            emit $ printf "%s = memref.subview %s[%s][%d][1] : %s to %s %s" (ssa casted_temp_ssa) (ssa temp_ssa) (ssa temp_zero) (knownVarDefArrLen v) (mlirGlobalType False (v^.varType))  (mlirGlobalType True (v^.varType))  (locationInfo p)
            return temp_ssa

mapVarDefToArg :: Maybe SSA->VarType ann->String
mapVarDefToArg s v = case s of
    Nothing -> mlirArgType v
    Just x -> printf "%s: %s" (ssa x) (mlirArgType v)

pickoutIntSSAs :: [SSA]->ProcDef ann->MLIRGen ([SSA], [(SSA, SSA)])
pickoutIntSSAs ssas p = do
    tup<-zipWithM (\s pa->case pa of{
        (VarDef (UnitType Int _) _ _) -> do{
            new_ssa <- nextTempSSA;
            return $ Left (new_ssa, s);
        };
        _ -> return $ Right s
    }) ssas (p^.parameters)
    let new_ssas = map (\x-> case x of (Left (x,_))->x; Right y->y) tup
    let ssa_casts = lefts tup
    return (new_ssas, ssa_casts)

generateDefSignature::[SSA]->ProcDef ann->String
generateDefSignature ssas p =
    printf "func @%s(%s)->%s{" (p^.procName^.identName)
    (intercalate ", "(zipWith (\s pa->mapVarDefToArg (Just s) (pa^.varType)) ssas (p^.parameters))) (if (p^.returnType)==Int then "index" else "()")
generateCallingSignature :: ProcDef ann->String
generateCallingSignature p =
    printf "(%s)->%s" 
    (intercalate ", " (map (\pa->mapVarDefToArg Nothing (pa^.varType)) (p^.parameters))) (if (p^.returnType)==Int then "index" else "()")


loadQubit :: Pos->SSA->MLIRGen SSA
loadQubit pos s = do
    qstate<-nextTempSSA
    emit $ printf "%s = affine.load %s[0] : memref<1x!isq.qubit, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa qstate) (ssa s) (locationInfo pos)
    return qstate
storeQubit :: Pos->SSA->SSA->MLIRGen ()
storeQubit pos s qstate = do
    emit $ printf "affine.store %s, %s[0] : memref<1x!isq.qubit, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa qstate) (ssa s) (locationInfo pos)

decorToDict :: GateDecorator->String
decorToDict decor = printf "{ctrl = [%s], adjoint = %s}" (intercalate ", " $  map (\x->if x then "true" else "false") (decor^.controlStates)) (if decor^.adjoint then "true" else "false")

printComplex :: Complex Double->String
printComplex (a:+b) = printf "#isq.complex<%f, %f>" a b
printMatrix :: [[Complex Double]]->String
printMatrix xss = "[" ++ (intercalate ", " $ map (\xs->"["++(intercalate ", " $ map printComplex xs)++"]") xss) ++ "]"

-- In most cases: arrays represented in the form of memref<sizexindex, #onedim>
-- Upon allocation / get_global, we need to convert them to memref<sizexindex, #onedim>
instance CodeSunk MLIRGen where
    incrBlockIndent = extraIndent %= (+1)
    decrBlockIndent = extraIndent %= (\x->x-1)
    emitUnaryOp pos Neg ret val = do
        -- Problem: MLIR does not have arith.negi.
        temp_zero <- nextTempSSA
        emit $ printf "%s = arith.constant 0 : index %s" (ssa temp_zero) (locationInfo pos)
        emit $ printf "%s = arith.subi %s, %s : index %s" (ssa temp_zero) (ssa ret) (ssa val) (locationInfo pos)
    emitUnaryOp pos Positive ret val = error "Positive is not supported"
    emitBinaryOp pos bop ret lhs rhs = do
        let (op, is_cmp) = case bop of
                Add -> ("addi", False)
                Sub -> ("subi", False)
                Mul -> ("muli", False)
                Div -> ("divi", False)
                Less -> ("cmpi \"slt\",", True)
                Greater -> ("cmpi \"sgt\",", True)
                LessEqual -> ("cmpi \"sle\",", True)
                GreaterEqual -> ("cmpi \"sge\",", True)
                Equal -> ("cmpi \"eq\",", True)
                NEqual -> ("cmpi \"ne\",", True)
        if is_cmp then (do
            temp_value<-nextTempSSA
            -- Problem: comparison result is i1.
            emit $ printf "%s = arith.%s %s, %s : index %s" (ssa temp_value) op (ssa lhs) (ssa rhs) (locationInfo pos)
            emit $ printf "%s = arith.index_cast %s: i1 to index %s" (ssa ret) (ssa temp_value) (locationInfo pos)) 
            else (do
            emit $ printf "%s = arith.%s %s, %s : index %s" (ssa ret) op (ssa lhs) (ssa rhs) (locationInfo pos))
    emitSlice pos ret kind base offset arr_len = do
        base' <- overrideGlobalSSA pos base
        let prev_ty = Composite kind arr_len ()
        let curr_ty = Composite kind (Just 1) ()
        emit $ printf "%s = memref.subview %s[%s][1][1] : %s to %s %s" (ssa ret) (ssa base') (ssa offset) (mlirGlobalType True prev_ty) (mlirGlobalType True curr_ty) (locationInfo pos)
    emitReadIntOp pos ret val = do
        emit $ printf "%s = affine.load %s[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa ret) (ssa val) (locationInfo pos)
    emitWriteIntOp pos arr val = do
        emit $ printf "affine.store %s, %s[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa val) (ssa arr) (locationInfo pos)
    emitProcedure pos p args = do
        block<-popBlock
        (new_ssas, ssa_casts)<-pickoutIntSSAs args p
        emit $ generateDefSignature new_ssas p
        -- Store all values into temporary variables.
        mapM_ (\(new_ssa, old_ssa)->do
            temp_var <- nextTempSSA
            emit $ printf "    %s = memref.alloca() : memref<1xindex> %s" (ssa temp_var) (locationInfo pos)
            emit $ printf "    affine.store %s, %s[0] : memref<1xindex> %s" (ssa new_ssa) (ssa temp_var) (locationInfo pos)
            temp_zero <- nextTempSSA
            emit $ printf "    %s = arith.constant 0 : index %s" (ssa temp_zero) (locationInfo pos)
            emit $ printf "    %s = memref.subview %s[%s][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa old_ssa) (ssa temp_var) (ssa temp_zero) (locationInfo pos)
            ) ssa_casts
        emitBlock block
        -- Free return.
        when (p^.returnType == Void) $ emit "    return"
        emit "}"
        return ()
    emitCall pos p ret args = do
        let sig = generateCallingSignature p
        let ret_ty = if (p^.returnType)==Int then printf "%s = " (ssa ret) else ""
        emit $ printf "%scall @%s(%s) : %s %s" ret_ty (p^.procName^.identName) (intercalate ", " args) sig (locationInfo pos)
    emitPushBlock = do
        generatedCodes %= ([]:)
    emitIf pos cond = do
        b_else <- popBlock
        b_then <- popBlock
        emit $ printf "affine.if affine_set<(d0): (d0-1>=0)>(%s) {" (ssa cond)
        emitBlock b_then
        emit "} else {"
        emitBlock b_else
        emit $ printf "} %s" (locationInfo pos)
    emitWhile pos cond = do
        b_body <- popBlock
        b_cond <- popBlock
        emit "scf.while () : ()->(){"
        emitBlock b_cond
        temp_bool<-nextTempSSA
        emit $ printf "    %s = arith.index_cast %s : index to i1 %s" (ssa temp_bool) (ssa cond) (locationInfo pos)
        emit $ printf "    scf.condition(%s) %s" (ssa temp_bool) (locationInfo pos)
        emit "} do {"
        emitBlock b_body
        emit "    scf.yield"
        emit $ printf "} %s" (locationInfo pos)
    emitPrint pos val = do
        emit $ printf "call @isq_print(%s): (index)->() %s" (ssa val) (locationInfo pos)
    emitFor pos var lo hi = do
        b_body <- popBlock
        temp_val<-nextTempSSA
        emit $ printf "affine.for %s = %s to %s step 1 {" (ssa temp_val) (ssa lo) (ssa hi)
        --extraIndent%=(+1)
        emitLocalDef pos var (VarDef (UnitType Int ()) undefined ())
        emitWriteIntOp pos var temp_val
        --extraIndent%=(\x->x-1)
        emitBlock b_body
        
        emit $ printf "} %s" (locationInfo pos)
    emitReturn pos Nothing = do
        emit $ printf "return %s" (locationInfo pos)
    emitReturn pos (Just val) = do
        emit $ printf "return %s : index %s" (ssa val) (locationInfo pos)

    emitReset pos qubit = do
        qstate<-loadQubit pos qubit
        new_qstate<-nextTempSSA
        emit $ printf "%s = isq.call_qop @isq_builtin::@reset(%s): [1]()->() %s" (ssa new_qstate) (ssa qstate) (locationInfo pos)
        storeQubit pos qubit new_qstate
    emitGateApply pos decor gatedef args = do
        used_gate<-nextTempSSA
        emit $ printf "%s = isq.use @%s : !isq.gate<%d> %s" (ssa used_gate) (gatedef^.gateName^.identName) (length args) (locationInfo pos)
        lifted_gate<-nextTempSSA
        let new_length = ((length args)+(length $ decor^.controlStates))
        emit $ printf "%s = isq.decorate(%s: !isq.gate<%d>) %s :!isq.gate<%d> %s" (ssa lifted_gate) (ssa used_gate) (length args) (decorToDict decor) new_length (locationInfo pos)
        qstates <- mapM (loadQubit pos) args
        new_qstates <- sequence (replicate (length qstates) nextTempSSA)
        emit $ printf "%s = isq.apply %s(%s) : !isq.gate<%d> %s" (intercalate "," $ map ssa new_qstates) (ssa lifted_gate) (intercalate "," $ map ssa qstates) (length args) (locationInfo pos)
        zipWithM_ (storeQubit pos) args new_qstates
    emitGateDef gatedef matdef = do
        emit $ printf "isq.defgate @%s {definition = [{type=\"unitary\", value = %s }]}: !isq.gate<%d> %s" (gatedef^.gateName^.identName) (printMatrix matdef) (gateSize gatedef) (locationInfo $ gatedef^.annotation)
    emitMeasure pos qubit ret = do
        qstate<-loadQubit pos qubit
        new_qstate<-nextTempSSA
        temp_ret <-nextTempSSA
        emit $ printf "%s, %s = isq.call_qop @isq_builtin::@measure(%s): [1]()->i1 %s" (ssa new_qstate) (ssa temp_ret) (ssa qstate) (locationInfo pos)
        storeQubit pos qubit new_qstate
        emit $ printf "%s = arith.index_cast %s : i1 to index %s" (ssa ret) (ssa temp_ret) (locationInfo pos)
    emitConst pos val s = do
        emit $ printf "%s = arith.constant %d : index %s" (ssa s) val (locationInfo pos)
    emitPass pos = do
        emit $ printf "isq.pass %s" (locationInfo pos)
    emitGlobalDef pos s v = do
        overloadedGlobalVars %= Map.insert s v
        emit $ printf "memref.global @%s : %s = uninitialized %s" (v^.varName^.identName) (mlirGlobalType False (v^.varType)) (locationInfo pos)
    emitLocalDef pos s v = do
        temp_var <- nextTempSSA
        emit $ printf "%s = memref.alloca() : %s %s" (ssa temp_var) (mlirGlobalType False (v^.varType))  (locationInfo pos)
        temp_zero <- nextTempSSA
        emitConst pos 0 temp_zero
        emit $ printf "%s = memref.subview %s[%s][%d][1] : %s to %s %s" (ssa s) (ssa temp_var) (ssa temp_zero) (knownVarDefArrLen v) (mlirGlobalType False (v^.varType)) (mlirGlobalType True (v^.varType)) (locationInfo pos)

emptyMLIRGen :: MLIRGenState
emptyMLIRGen = MLIRGenState {
    _generatedCodes = [[]],
    _overloadedGlobalVars = Map.empty,
    _tempSSAs = 0,
    _extraIndent = 0
}

runMLIRGen :: MLIRGen (Either GrammarError ()) -> Either GrammarError MLIRGenState
runMLIRGen (MLIRGen m) = do
    let (a,res) = runState m emptyMLIRGen in 
        case a of
            Left err -> Left err
            Right _ -> Right res

extractCode = reverse . head . _generatedCodes

mlirGen :: Program Pos -> Either GrammarError [String]
mlirGen x  = fmap extractCode $ runMLIRGen $ runCodegen x