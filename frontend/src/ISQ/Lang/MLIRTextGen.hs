{-# LANGUAGE GeneralisedNewtypeDeriving, TemplateHaskell, FlexibleContexts #-}
{-# LANGUAGE ViewPatterns #-}
module ISQ.Lang.MLIRTextGen where
import ISQ.Lang.AST
import ISQ.Lang.Codegen hiding (ssa)
import Control.Lens
import qualified Data.Map as Map
import Control.Monad.State.Lazy
    ( when,
      StateT(StateT),
      MonadState,
      replicateM,
      zipWithM,
      zipWithM_,
      runState,
      evalState, get,
      State, execState )
import Text.Printf
import Text.Parsec.Pos
import Data.List (intercalate)
import Data.Complex
import Data.Either
import ISQ.Lang.Codegen (ScopeType)

locationInfo :: SourcePos -> String
locationInfo pos = printf "loc(%s:%d:%d)" (show $ sourceName pos) (sourceLine pos) (sourceColumn pos)

data MLIRLocalDef = LocalDef {
    _localDefName :: String,
    _localDefMLIRType :: String,
    _localDefPos :: Pos
} | Label Int deriving Show

data MLIRScope = MLIRScope{
    _scopeType :: ScopeType,
    _localDefs :: [MLIRLocalDef],
    _emittedCode :: [String],
    _usedLabels :: Int
} deriving Show
makeLenses ''MLIRScope

addDefinition :: (MonadState MLIRScope m)=>String -> String -> Pos -> m ()
addDefinition defType defName defPos = localDefs %= (LocalDef defName defType defPos:)

emitRAIIFence :: (MonadState MLIRScope m)=>m Int
emitRAIIFence = do
    curr<-use usedLabels
    localDefs%= (Label curr:)
    usedLabels%=(+1)
    return curr
raiiFence :: (MonadState MLIRScope m)=>m (Maybe Int)
raiiFence = do
    curr<-use localDefs
    case curr of
        [] -> return Nothing
        (Label x:_) -> return $ Just x
        _ -> Just <$> emitRAIIFence
emitLine :: (MonadState MLIRScope m)=>String -> m ()
emitLine line = emittedCode%= (line:)
cleanupCode :: Maybe String->MLIRScope->[String]
cleanupCode a b = cleanupCode' a (execState emitRAIIFence b)
cleanupCode' :: Maybe String->MLIRScope->[String]
cleanupCode' ty scope = concatMap (go ty) $ view localDefs scope where
    go (Just ty) (Label x) = [printf "    br ^raii_%d(%%ret_%d: %s)" x (x+1) ty | view usedLabels scope /= x + 1] ++ [printf "^raii_%d(%%ret_%d: %s):" x x ty]
    go Nothing (Label x) = [printf "    br ^raii_%d" x | view usedLabels scope /= x + 1] ++ [printf "^raii_%d:" x]
    go _ (LocalDef name t pos) = [printf "    memref.dealloc %s : %s %s" name t (locationInfo pos)]
jumpToReturn :: String->Maybe (String, String)->Maybe Int->String
jumpToReturn pos Nothing Nothing = printf "br ^clean_args %s" pos
jumpToReturn pos Nothing (Just x) = printf "br ^raii_%d %s" x pos
jumpToReturn pos (Just (ty, name)) Nothing = printf "br ^clean_args(%%%s: %s) %s"  name ty pos
jumpToReturn pos (Just (ty, name)) (Just x) = printf "br ^raii_%d(%%%s: %s) %s" x name ty pos

data MLIRGenState = MLIRGenState {
    _generatedCodes :: [MLIRScope],
    _overloadedGlobalVars :: Map.Map SSA (VarDef Pos),
    _tempSSAs :: Int,
    _extraIndent :: Int
} deriving Show
makeLenses ''MLIRGenState

newtype MLIRGen a = MLIRGen{
    toState :: State MLIRGenState a
} deriving (MonadState MLIRGenState, Functor, Applicative, Monad)



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
    generatedCodes %= over (_head.emittedCode) ((t++s):)
unemit :: MLIRGen ()
unemit = do
    generatedCodes %= over (_head.emittedCode) tail
emitBlock :: [String]->MLIRGen ()
emitBlock s = do
    c<-use generatedCodes
    -- Append code in reversed direction!
    mapM_ (\l->generatedCodes %= over (_head.emittedCode) (l:)) s
popBlock :: MLIRGen MLIRScope
popBlock = do
    c<-use generatedCodes
    generatedCodes %= tail
    return $ over emittedCode reverse $ head c

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
            return casted_temp_ssa

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

generateDefSignature::ProcDef ann->String
generateDefSignature p =
    printf "func @%s(%s)->%s{" (p^.procName^.identName)
    (intercalate ", " (map (mlirArgType . (^.varType)) $ p^.parameters)) (if (p^.returnType)==Int then "index" else "()")
generateInitialBlockSignature :: [SSA]->ProcDef ann->String
generateInitialBlockSignature [] p = "^start:"
generateInitialBlockSignature ssas p = 
    printf "^start(%s):" 
    (intercalate ", "(zipWith (\s pa->mapVarDefToArg (Just s) (pa^.varType)) ssas (p^.parameters))) 
generateCallingSignature :: ProcDef ann->String
generateCallingSignature p =
    printf "(%s)->%s"
    (intercalate ", " (map (\pa->mapVarDefToArg Nothing (pa^.varType)) (p^.parameters))) (if (p^.returnType)==Int then "index" else "()")


loadQubit :: Pos->SSA->MLIRGen SSA
loadQubit pos s = do
    s'<-overrideGlobalSSA pos s
    qstate<-nextTempSSA
    emit $ printf "%s = affine.load %s[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa qstate) (ssa s') (locationInfo pos)
    return qstate
storeQubit :: Pos->SSA->SSA->MLIRGen ()
storeQubit pos s qstate = do
    s'<-overrideGlobalSSA pos s
    emit $ printf "affine.store %s, %s[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa qstate) (ssa s') (locationInfo pos)

decorToDict :: GateDecorator->String
decorToDict decor = printf "{ctrl = [%s], adjoint = %s}" (intercalate ", " $  map (\x->if x then "true" else "false") (decor^.controlStates)) (if decor^.adjoint then "true" else "false")

printComplex :: Complex Double->String
printComplex (a:+b) = printf "#isq.complex<%f, %f>" a b
printMatrix :: [[Complex Double]]->String
printMatrix xss = "[" ++ (intercalate ", " $ map (\xs->"["++(intercalate ", " $ map printComplex xs)++"]") xss) ++ "]"


emitLocalDefWith :: SourcePos -> String -> VarDef ann -> p -> MLIRGen MLIRLocalDef
emitLocalDefWith pos s v scope = do
    temp_var <- nextTempSSA

    emit $ printf "%s = memref.alloc() : %s %s" (ssa temp_var) (mlirGlobalType False (v^.varType))  (locationInfo pos)
    temp_zero <- nextTempSSA
    emitConst pos 0 temp_zero
    emit $ printf "%s = memref.subview %s[%s][%d][1] : %s to %s %s" (ssa s) (ssa temp_var) (ssa temp_zero) (knownVarDefArrLen v) (mlirGlobalType False (v^.varType)) (mlirGlobalType True (v^.varType)) (locationInfo pos)
    return $ LocalDef (ssa temp_var) (mlirGlobalType False (v^.varType)) pos

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
                Div -> ("divsi", False)
                FloorDiv -> ("floordivsi", False)
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
        val' <- overrideGlobalSSA pos val
        emit $ printf "%s = affine.load %s[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa ret) (ssa val') (locationInfo pos)
    emitReadIntAtOp pos ret val offset = do
        val' <- overrideGlobalSSA pos val
        emit $ printf "%s = affine.load %s[%s] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa ret) (ssa val') (ssa offset) (locationInfo pos)
    emitWriteIntOp pos arr val = do
        arr' <- overrideGlobalSSA pos arr
        emit $ printf "affine.store %s, %s[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa val) (ssa arr') (locationInfo pos)
    emitWriteIntAtOp pos arr val offset = do
        arr' <- overrideGlobalSSA pos arr
        emit $ printf "affine.store %s, %s[%s]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa val) (ssa arr') (ssa offset) (locationInfo pos)
    emitProcedure pos p args = do
        (new_ssas, ssa_casts)<-pickoutIntSSAs args p
        stored_ssa_casts <- mapM (\(x,y)->do {s<-nextTempSSA; return (x,y,s)}) ssa_casts
        block<-popBlock
        let arg_stores = reverse (map (\(_,_,s)->LocalDef (ssa s) "memref<1xindex>" pos) stored_ssa_casts);
        emit $ generateDefSignature p
        emit $ generateInitialBlockSignature new_ssas p
        -- Store all values into temporary variables.
        mapM_ (\(new_ssa, old_ssa, store_ssa)->do
            let temp_var = store_ssa
            emit $ printf "    %s = memref.alloc() : memref<1xindex> %s" (ssa temp_var) (locationInfo pos)
            emit $ printf "    affine.store %s, %s[0] : memref<1xindex> %s" (ssa new_ssa) (ssa temp_var) (locationInfo pos)
            temp_zero <- nextTempSSA
            emit $ printf "    %s = arith.constant 0 : index %s" (ssa temp_zero) (locationInfo pos)
            emit $ printf "    %s = memref.subview %s[%s][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> %s" (ssa old_ssa) (ssa temp_var) (ssa temp_zero) (locationInfo pos)
            ) stored_ssa_casts
        emitBlock $ view emittedCode block
        if p^.returnType == Void then
            do
            ret<-use generatedCodes
            unemit
            --let current_scope = block
            --let (target, new_table) = runState raiiFence current_scope
            --emit $ "    " ++ (jumpToReturn (locationInfo pos) Nothing --target)
            emit "// Cleanup"
            emitBlock $ cleanupCode Nothing block
            emit $ printf "    br ^clean_args %s" (locationInfo pos)
            emit $ printf "^clean_args:"
            mapM_ (\localdef->emit $ printf "    memref.dealloc %s : memref<1xindex> %s" ((_localDefName localdef)) (locationInfo $ _localDefPos localdef)) arg_stores
            emit $ printf "    br ^return %s" (locationInfo pos)
            emit $ printf "^return:"
            emit $ printf "    return %s" (locationInfo pos)
        else
            do
            ret<-use generatedCodes
            unemit
            emitBlock $ cleanupCode (Just "index") block
            emit $ printf "    br ^clean_args(%%ret_0: index) %s" (locationInfo pos)
            emit $ printf "^clean_args(%%cleanup_ret: index):"
            mapM_ (\localdef->emit $ printf "    memref.dealloc %s : memref<1xindex> %s" ((_localDefName localdef)) (locationInfo $ _localDefPos localdef)) arg_stores
            emit $ printf "    br ^return(%%cleanup_ret: index) %s" (locationInfo pos)
            emit $ printf "^return(%%ret: index):"
            emit $ printf "    return %%ret: index %s" (locationInfo pos)
        
        emit "}"
        return ()
    emitCall pos p ret args = do
        let sig = generateCallingSignature p
        let ret_ty = if (p^.returnType)==Int then printf "%s = " (ssa ret) else ""
        args' <- mapM (overrideGlobalSSA pos) args
        emit $ printf "%scall @%s(%s) : %s %s" ret_ty (p^.procName^.identName) (intercalate ", " $ map ssa args') sig (locationInfo pos)
    emitPushBlock ty = do
        generatedCodes %= (MLIRScope ty [] [] 0 :)
    emitIf pos cond = do
        b_else <- popBlock
        b_then <- popBlock
        emit $ printf "affine.if affine_set<(d0): (d0-1>=0)>(%s) {" (ssa cond)
        emitBlock $ view emittedCode b_then
        emit "} else {"
        emitBlock $ view emittedCode b_else
        emit $ printf "} %s" (locationInfo pos)
    emitWhile pos cond = do
        b_body <- popBlock
        b_cond <- popBlock
        emit "scf.while () : ()->(){"
        emitBlock $ view emittedCode b_cond
        temp_bool<-nextTempSSA
        emit $ printf "    %s = arith.index_cast %s : index to i1 %s" (ssa temp_bool) (ssa cond) (locationInfo pos)
        emit $ printf "    scf.condition(%s) %s" (ssa temp_bool) (locationInfo pos)
        emit "} do {"
        emitBlock $ view emittedCode b_body
        emit "    scf.yield"
        emit $ printf "} %s" (locationInfo pos)
    emitPrint pos val = do
        emit $ printf "isq.print %s: index %s" (ssa val) (locationInfo pos)
    emitFor pos var lo hi step = do
        b_body <- popBlock
        temp_val<-nextTempSSA
        emit $ printf "affine.for %s = %s to %s step %d {" (ssa temp_val) (ssa lo) (ssa hi) step
        extraIndent%=(+1)
        iter_var_def <- emitLocalDefWith pos var (VarDef (UnitType Int ()) undefined ()) b_body
        emitWriteIntOp pos var temp_val
        extraIndent%=(\x->x-1)
        emitBlock $ view emittedCode (over localDefs (++[iter_var_def])b_body )
        emit $ printf "} %s" (locationInfo pos)
    emitReturn pos Nothing = do
        ret<-use generatedCodes
        let current_scope = head ret
        let (target, new_table) = runState raiiFence current_scope
        generatedCodes %= (\(x:xs)->new_table:xs)
        emit $ jumpToReturn (locationInfo pos) Nothing target
        next_label<-nextTempSSA
        extraIndent%=(\x->x-1)
        emit $ printf "^%s:" next_label
        extraIndent%=(+1)
    emitReturn pos (Just val) = do
        ret<-use generatedCodes
        let current_scope = head ret
        let (target, new_table) = runState raiiFence current_scope
        generatedCodes %= (\(x:xs)->new_table:xs)
        emit $ jumpToReturn (locationInfo pos) (Just ("index", val)) target
        next_label<-nextTempSSA
        extraIndent%=(\x->x-1)
        emit $ printf "^%s:" next_label
        extraIndent%=(+1)
    emitReset pos qubit = do
        qstate<-loadQubit pos qubit
        new_qstate<-nextTempSSA
        emit $ printf "%s = isq.call_qop @isq_builtin::@reset(%s): [1]()->() %s" (ssa new_qstate) (ssa qstate) (locationInfo pos)
        storeQubit pos qubit new_qstate
    emitGateApply pos decor gatedef args = do
        used_gate<-nextTempSSA
        let original_length = gateSize gatedef
        emit $ printf "%s = isq.use @%s : !isq.gate<%d> %s" (ssa used_gate) (gatedef^.gateName^.identName) (original_length) (locationInfo pos)
        lifted_gate<-nextTempSSA
        let new_length = length args
        emit $ printf "%s = isq.decorate(%s: !isq.gate<%d>) %s :!isq.gate<%d> %s" (ssa lifted_gate) (ssa used_gate) (original_length) (decorToDict decor) new_length (locationInfo pos)
        qstates <- mapM (loadQubit pos) args
        new_qstates <- replicateM (length qstates) nextTempSSA
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
        scope <- use generatedCodes
        new_def <- emitLocalDefWith pos s v (head scope)
        generatedCodes %= (\(x:xs)->over localDefs (new_def:) x :xs)

    emitProgramHeader = emitHeader
    emitEraseIntArray pos n src target = emit $ printf "%s = memref.cast %s : %s to %s" (ssa target) (ssa src) (mlirGlobalType True (Composite Int (Just n) ())) (mlirGlobalType True (Composite Int Nothing ())) 
    emitEraseQbitArray pos n src target = emit $ printf "%s = memref.cast %s : %s to %s" (ssa target) (ssa src) (mlirGlobalType True (Composite Qbit (Just n) ())) (mlirGlobalType True (Composite Qbit Nothing ())) 
    --emitMin :: Pos->[SSA]->m ()
    --emitIntArray :: Pos->[SSA]->SSA->m ()
emptyMLIRGen :: MLIRGenState
emptyMLIRGen = MLIRGenState {
    _generatedCodes = [MLIRScope GlobalScope [] [] 0],
    _overloadedGlobalVars = Map.empty,
    _tempSSAs = 0,
    _extraIndent = 0
}

emitHeader :: Pos->MLIRGen ()
emitHeader p = do
    emit "// This file is generated by the isQ Experimental compiler"
    emit $ "// Source file name: " ++ (sourceName p)
    emit $ "module @isq_builtin {"
    emit $ "    isq.declare_qop @measure : [1]()->i1"
    emit $ "    isq.declare_qop @reset : [1]()->()"
    emit $ "}"
runMLIRGen :: MLIRGen (Either GrammarError ()) -> Either GrammarError MLIRGenState
runMLIRGen (MLIRGen m) = do
    let (a,res) = runState m emptyMLIRGen in
        case a of
            Left err -> Left err
            Right _ -> Right res

extractCode = reverse . view emittedCode . head . view generatedCodes

mlirGen :: Program Pos -> Either GrammarError [String]
mlirGen x  = fmap extractCode $ runMLIRGen $ runCodegen x