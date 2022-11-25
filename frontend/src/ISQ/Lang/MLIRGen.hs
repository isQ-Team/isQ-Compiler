{-# LANGUAGE TemplateHaskell, FlexibleContexts, ViewPatterns #-}
module ISQ.Lang.MLIRGen where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.ISQv2Tokenizer(Pos(Pos), Annotated (annotation))
import ISQ.Lang.TypeCheck
import ISQ.Lang.MLIRTree hiding (Bool, Gate, Unit, Double)
import qualified ISQ.Lang.MLIRTree as M
import Control.Monad.State (fix, void, State, zipWithM_, evalState, execState, runState)
import Control.Lens
import Data.List (isSuffixOf, take)
import Data.List.Split (splitOn)
import Debug.Trace


data RegionBuilder = RegionBuilder{
    _nextBlockId :: Int,
    _nextTailBlockId :: Int,
    _currentBlock:: MLIRBlock,
    _headBlocks :: [MLIRBlock],
    _tailBlocks :: [MLIRBlock],
    _filename :: String,
    _ssaId :: Int,
    _inLogic :: Bool
} deriving Show

makeLenses ''RegionBuilder

nextSsaId :: State RegionBuilder Int
nextSsaId = do
    id <- use ssaId
    ssaId %= (+1)
    return id

mapType :: EType->MLIRType
mapType (Type () Unit []) = MUnit
mapType (Type () Bool []) = M.Bool
mapType (Type () Ref [x]) = BorrowedRef (mapType x)
mapType (Type () Int []) = Index
mapType (Type () Qbit []) = QState
mapType (Type () (Array 0) [x]) = Memref Nothing (mapType x)
mapType (Type () (Array n) [x]) = Memref (Just n) (mapType x)
mapType (Type () (Gate n) _) = M.Gate n
mapType (Type () (Logic n) _) = M.Gate n
mapType (Type () Double []) = M.Double
mapType (Type () FuncTy (ret:arg)) = Func (mapType ret) $ map mapType arg
mapType _ = error "unsupported type"


-- Create an empty region builder, with entry and exit.
generalRegion :: String -> [(MLIRType, SSA)] -> [MLIROp] -> Int -> Bool -> RegionBuilder
generalRegion name init_args end_body ssa in_logic = RegionBuilder 1 1 (MLIRBlock (BlockName "^entry") init_args [])
    [] [MLIRBlock (BlockName "^exit") [] end_body] name ssa in_logic

-- Called to emit a branch statement according to a flag.
pushBlock :: MLIRPos->SSA->State RegionBuilder ()
pushBlock pos v = do
    next_block_id<-use nextBlockId
    nextBlockId %= (+1)
    curr<-use currentBlock
    let new_block = MLIRBlock (fromBlockName next_block_id) [] []
    last<-use tailBlocks
    headBlocks %= (curr{blockBody=reverse $ MBranch pos v (blockId $ head last, fromBlockName next_block_id) : blockBody curr}:)
    currentBlock .= new_block
pushBlockUnconditioned :: MLIRPos->State RegionBuilder ()
pushBlockUnconditioned pos = do
    next_block_id<-use nextBlockId
    nextBlockId %= (+1)
    curr<-use currentBlock
    let new_block = MLIRBlock (fromBlockName next_block_id) [] []
    last<-use tailBlocks
    headBlocks %= (curr{blockBody=reverse $ MJmp pos (blockId $ head last) : blockBody curr}:)
    currentBlock .= new_block

finalizeBlock :: State RegionBuilder [MLIRBlock]
finalizeBlock = do
    last<-use tailBlocks
    curr<-use currentBlock
    let curr' = curr{blockBody=reverse $ MJmp MLIRPosUnknown (blockId $ head last) : blockBody curr}
    heads<-use headBlocks
    tails<-use tailBlocks
    return $ (reverse heads)++[curr']++tails


pushOp :: MLIROp->State RegionBuilder ()
pushOp op = currentBlock%=(\x->x{blockBody=op:blockBody x})

-- Pushes one RAII layer. Following jmp-to-end will jump here instead.
pushRAII :: [MLIROp]->State RegionBuilder ()
pushRAII ops = do
    next_tail_id <- use nextTailBlockId
    nextTailBlockId %= (+1);
    tails<-use tailBlocks
    let next_label = blockId $ head tails
    let new_block = MLIRBlock (BlockName ("^exit_"++show next_tail_id)) [] (ops++[MJmp MLIRPosUnknown next_label])
    tailBlocks%=(new_block:)

-- Pushes an alloc-free pair.
pushAllocFree:: MLIRPos->TypedSSA->State RegionBuilder ()
pushAllocFree pos (ty, val) = do
    pushOp (MAllocMemref pos val ty $ SSA "")
    pushRAII [MFreeMemref pos val ty]

mpos :: TypeCheckData->State RegionBuilder MLIRPos
mpos d = let Pos x y f = sourcePos d in fmap (\z->MLIRLoc f x y) (use filename)

ssa :: TypeCheckData->SSA
ssa x = fromSSA (termId x)

astMType :: (Annotated p)=>p TypeCheckData->MLIRType
astMType x = mapType $ termType $ annotation x

mType :: TypeCheckData->MLIRType
mType x = mapType $ termType $ x

binopTranslate :: BinaryOperator -> MLIRType -> MLIRBinaryOp
binopTranslate Add Index = mlirAddi
binopTranslate Sub Index = mlirSubi
binopTranslate Mul Index = mlirMuli
binopTranslate Div Index = mlirFloorDivsi
binopTranslate Mod Index = mlirRemsi
binopTranslate And M.Bool = mlirAnd
binopTranslate Or M.Bool = mlirOr
binopTranslate Andi Index = mlirAndi
binopTranslate Ori Index = mlirOri
binopTranslate Xori Index = mlirXori
binopTranslate Pow M.Double = mlirPowf 
binopTranslate Add M.Double = mlirAddf
binopTranslate Sub M.Double = mlirSubf
binopTranslate Mul M.Double = mlirMulf
binopTranslate Div M.Double = mlirDivf
binopTranslate (Cmp Less) Index = mlirSltI
binopTranslate (Cmp LessEq) Index = mlirSleI
binopTranslate (Cmp Greater) Index = mlirSgtI
binopTranslate (Cmp GreaterEq) Index = mlirSgeI
binopTranslate (Cmp Equal) Index = mlirEqI
binopTranslate (Cmp NEqual) Index = mlirNeI
binopTranslate (Cmp Less) M.Double = mlirSltF
binopTranslate (Cmp LessEq) M.Double = mlirSleF
binopTranslate (Cmp Greater) M.Double = mlirSgtF
binopTranslate (Cmp GreaterEq) M.Double = mlirSgeF
binopTranslate (Cmp Equal) M.Double = mlirEqF
binopTranslate (Cmp NEqual) M.Double = mlirNeF
binopTranslate (Cmp Equal) M.Bool = mlirEqB
binopTranslate (Cmp NEqual) M.Bool = mlirNeB
binopTranslate Shl Index = mlirShl
binopTranslate Shr Index = mlirShr
binopTranslate _ _ = undefined

unaryopTranslate Neg M.Double = mlirNegF
unaryopTranslate Neg Index = mlirNegI
unaryopTranslate _ _ = undefined

logicOpTranslate :: BinaryOperator -> String
logicOpTranslate Andi = "andv"
logicOpTranslate Ori = "orv"
logicOpTranslate Xori = "xorv"
logicOpTranslate And = "and"
logicOpTranslate Or = "or"
logicOpTranslate (Cmp NEqual) = "xor"
logicOpTranslate _ = undefined

emitExpr' :: (Expr TypeCheckData->State RegionBuilder SSA)->Expr TypeCheckData->State RegionBuilder SSA
emitExpr' f (EIdent ann name) = error "unreachable"
emitExpr' f x@(EBinary ann binop lhs rhs) = do
    lhs'<-f lhs
    rhs'<-f rhs
    pos<-mpos ann
    let lhsTy = astMType lhs
    let i = ssa ann
    logic <- use inLogic
    case logic of
        True -> pushOp $ MLBinary pos (lhsTy, i) lhs' rhs' $ logicOpTranslate binop
        False -> pushOp $ MBinary pos i lhs' rhs' $ binopTranslate binop lhsTy
    return i
emitExpr' f (EUnary ann uop lhs) = do
    lhs'<-f lhs
    pos<-mpos ann
    let i = ssa ann
    let lhsTy = astMType lhs
    case uop of
        Positive -> return lhs' -- Positive x is x
        Neg -> do
            case lhsTy of
                Index -> do
                    let zero = SSA (unSsa i ++ "_zero")
                    pushOp $ MLitInt pos zero 0
                    pushOp $ MBinary pos i zero lhs' (binopTranslate Sub lhsTy)
                    return i
                M.Double -> do
                    pushOp $ MUnary pos i lhs' (unaryopTranslate uop lhsTy)
                    return i
                _->error "bad neg type"
        Not -> do
            in_logic <- use inLogic
            case in_logic of
                True -> pushOp $ MLUnary pos (lhsTy, i) lhs' "not"
                False -> do
                    let zero = SSA (unSsa i ++ "_false")
                    pushOp $ MLitBool pos zero False
                    pushOp $ MBinary pos i zero lhs' (binopTranslate (Cmp Equal) lhsTy)
            return i
        Noti -> do
            pushOp $ MLUnary pos (lhsTy, i) lhs' "notv"
            return i

emitExpr' f (ESubscript ann base offset) = do
    base'<-f base
    offset'<-f offset
    pos<-mpos ann
    let i = ssa ann
    in_logic <- use inLogic
    pushOp $ MTakeRef pos i (astMType base, base') offset' in_logic
    return i
emitExpr' f (EArrayLen ann base) = do
    base' <- f base
    pos <- mpos ann
    let i = ssa ann
    pushOp $ MArrayLen pos i (astMType base, base')
    return i
emitExpr' f x@(ECall ann (EGlobalName ann2 mname) args) = do
    let name = if mname=="main" then "__isq__main" else mname
    let logic = isSuffixOf logicSuffix name
    -- remove logicSuffix
    let name' = if logic then take (length name - length logicSuffix) name else name
    args'<-mapM f args
    let args'' = zip (fmap astMType args) args'
    let ret = if (ty $ termType $ ann) == Unit then Nothing else Just (astMType x, ssa ann)
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MCall pos ret (fromFuncName name') args'' logic
    return i
emitExpr' f (ECall ann _ _) = error "indirect call not supported"
emitExpr' f (EIntLit ann val) = do
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MLitInt pos i val
    return i
emitExpr' f (EFloatingLit ann val) = do
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MLitDouble pos i val
    return i
emitExpr' f (EImagLit _ _) = error "complex number not supported"
emitExpr' f (EBoolLit ann val) = do
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MLitBool pos i val
    return i
emitExpr' f (ERange _ _ _ _) = error "first-class range not supported"
emitExpr' f (ECoreMeasure ann operand) = do
    operand'<-f operand
    pos<-mpos ann
    let i = ssa ann
    let i_in = SSA $ (unSsa (ssa ann)) ++"_in"
    let i_out = SSA $ (unSsa (ssa ann)) ++ "_out"
    pushOp $ MLoad pos i_in (BorrowedRef QState, operand')
    pushOp $ MQMeasure pos i i_out i_in
    pushOp $ MStore pos (BorrowedRef QState, operand') i_out
    return i
emitExpr' f (EList _ _) = error "first-class list not supported"
emitExpr' f x@(EDeref ann val) = do
    val'<-f val
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MLoad pos i (astMType val, val')
    return i
emitExpr' f x@(EImplicitCast ann@(mType->M.Index) val@(astMType->M.Bool)) = do
    val'<-f val
    pos<-mpos ann
    let i = ssa ann
    let i_i2 = SSA $ (unSsa i) ++"_i2"
    pushOp $ MCast pos i_i2 val' mlirI1toI2
    pushOp $ MCast pos i i_i2 mlirI2toIndex
    return i
emitExpr' f x@(EImplicitCast ann@(mType->M.Double) val@(astMType->M.Index)) = do
    val'<-f val
    pos<-mpos ann
    let i = ssa ann
    let i_i64 = SSA $ (unSsa i) ++"_i64"
    pushOp $ MCast pos i_i64 val' mlirIndextoI64
    pushOp $ MCast pos i i_i64 mlirI64toDouble
    return i
emitExpr' f x@(EImplicitCast {}) = error "not supported"
emitExpr' f (ETempVar {}) = error "unreachable"
emitExpr' f (ETempArg {}) = error "unreachable"
emitExpr' f (EUnitLit ann) = do
    let i = ssa ann
    return i
emitExpr' f (EResolvedIdent ann i) = do
    return $ fromSSA i
emitExpr' f (EGlobalName ann@(mType->BorrowedRef _) name) = do
    pos<-mpos ann
    pushOp $ MUseGlobalMemref pos (ssa ann) (fromFuncName name) (mType ann)
    return (ssa ann)
emitExpr' f (EGlobalName ann@(mType->(Memref (Just _) _)) name) = do
    pos<-mpos ann
    pushOp $ MUseGlobalMemref pos (ssa ann) (fromFuncName name) (mType ann)
    return (ssa ann)
emitExpr' f (EGlobalName ann _) = error "first-class global gate/function not supported"
emitExpr' f (EListCast ann sublist) = do
    let i = ssa ann
    pos<-mpos ann
    let Type () (Array llen) [sub_ty] = termType ann
    let Type () (Array rlen) [_] = termType $ annotation sublist
    let to_zero = llen == 0
    let len = if to_zero then rlen else llen
    right <- f sublist
    pushOp $ MListCast pos i right to_zero $ mapType $ Type () (Array len) [sub_ty]
    return i
emitExpr :: Expr TypeCheckData -> State RegionBuilder SSA
emitExpr = fix emitExpr'

emitStatement' :: (AST TypeCheckData-> State RegionBuilder ())->(AST TypeCheckData->State RegionBuilder ())
emitStatement' f (NBlock ann lis) = do
    pos<-mpos ann
    curSsa <- use ssaId
    lis' <- scopedStatement [] [MSCFYield pos] (mapM f lis) curSsa False
    pushOp $ MSCFExecRegion pos lis'
-- Generic if.
emitStatement' f (NIf ann cond bthen belse) = do
    pos<-mpos ann
    cond'<-emitExpr cond
    curSsa <- use ssaId
    then_block <- scopedStatement [] [MSCFYield pos] (mapM f bthen) curSsa False
    curSsa <- use ssaId
    else_block <- scopedStatement [] [MSCFYield pos] (mapM f belse) curSsa False
    pushOp $ MSCFIf pos cond' (MSCFExecRegion pos then_block) (MSCFExecRegion pos else_block)
emitStatement' f NFor{} = error "unreachable"
emitStatement' f NEmpty{} = return ()
emitStatement' f NPass{} = return ()
emitStatement' f (NBp ann) = do
    pos<-mpos ann
    let Pos x y f = sourcePos ann
    let i = ssa ann
    pushOp $ MLitInt pos i x
    pushOp $ MBp pos i
emitStatement' f NWhile{} = error "unreachable"
emitStatement' f (NCall ann expr) = void $ emitExpr expr
emitStatement' f (NDefvar ann defs) = error "unreachable"
emitStatement' f (NAssign ann lhs rhs op) = do
    rhs' <- emitExpr rhs
    in_logic <- use inLogic
    pos <- mpos ann
    case in_logic of
        True -> case lhs of
            ESubscript ann base offset -> do
                offset' <- emitExpr offset
                pushOp $ MStoreOffset pos (astMType base, ssa $ annotationExpr base) rhs' offset'
            _ -> return ()
        False -> do
            lhs' <- emitExpr lhs
            pushOp $ MStore pos (astMType lhs, lhs') rhs'
emitStatement' f NGatedef{} = error "unreachable"
emitStatement' f (NReturn ann expr) = do
    pos <- mpos ann
    let ty = astMType expr
    expr' <- emitExpr expr
    pushOp $ MReturn pos (ty, expr') True
emitStatement' f (NCoreUnitary ann (EGlobalName ann2 name) ops mods) = do
    let go Inv (l, f) = (l, not f)
        go (Ctrl x i) (l, f) = (replicate i x ++ l, f)
        folded_mods = foldr go ([], False) mods
    ops'<-mapM emitExpr ops
    pos<-mpos ann
    let i = ssa ann2
    
    let used_gate = i;
    let ty = termType ann2;
    let isq = case ty of
            Type _ (Gate _) _ -> True
            Type _ (Logic _) _ -> False
            other -> error "unexpected type"
    let len = length $ subTypes ty
    let gate_type@(M.Gate gate_size) = mapType ty;
    
    let (extra_args, qubit_args) = splitAt len ops
    let (extra_ssa, qubit_ssa) = splitAt len ops'
    let (ins, outs) = unzip $ map (\id->(SSA $ unSsa i ++ "_in_"++show id, SSA $ unSsa i ++ "_out_"++show id)) [1..length qubit_ssa]
    pushOp $ MQUseGate pos used_gate (fromFuncName name) gate_type (zipWith (\arg ssa->(astMType arg, ssa)) extra_args extra_ssa) isq
    decorated_gate<-case folded_mods of
            ([], False)->return i
            (ctrls, adj)->do
                let decorated_gate = SSA $ unSsa i ++ "_decorated"
                pushOp $ MQDecorate pos decorated_gate used_gate folded_mods gate_size
                return $ decorated_gate
    zipWithM_ (\in_state in_op->pushOp $ MLoad pos in_state (BorrowedRef QState, in_op)) ins qubit_ssa
    pushOp $ MQApplyGate pos outs ins decorated_gate isq
    zipWithM_ (\out_state in_op->pushOp $ MStore pos (BorrowedRef QState, in_op) out_state) outs qubit_ssa
emitStatement' f NCoreUnitary{} = error "first-class gate unsupported"
emitStatement' f (NCoreReset ann operand) = do
    operand'<-emitExpr operand
    pos<-mpos ann
    let i = ssa ann
    let i_in = SSA $ unSsa i ++"_in"
    let i_out = SSA $ unSsa i ++ "_out"
    pushOp $ MLoad pos i_in (BorrowedRef QState, operand')
    pushOp $ MQReset pos i_out i_in
    pushOp $ MStore pos (BorrowedRef QState, operand') i_out
emitStatement' f (NCorePrint ann expr) = do
    s<-emitExpr expr
    pos<-mpos ann
    pushOp $ MQPrint pos (astMType expr, s)
emitStatement' f (NCoreMeasure ann expr) = void $ emitExpr expr
emitStatement' f NProcedure{} = error "unreachable"
emitStatement' f NContinue{} = error "unreachable"
emitStatement' f NBreak{} = error "unreachable"
emitStatement' f (NResolvedFor ann fori (ERange _ (Just lo) (Just hi) (Just (EIntLit _ step))) body) = do
    lo'<-emitExpr lo
    hi'<-emitExpr hi
    pos<-mpos ann
    curSsa <- use ssaId
    r<-scopedStatement [] [MSCFYield pos] (mapM f body) curSsa False
    pushOp $ MSCFFor pos lo' hi' step (fromSSA fori) [MSCFExecRegion pos r]
emitStatement' f NResolvedFor{} = error "unreachable"
emitStatement' f (NResolvedGatedef ann name mat sz qir) = do
    pos<-mpos ann
    pushOp $ MQDefGate pos (fromFuncName name) sz [] (MatrixRep mat: case qir of {Just x->[QIRRep (fromFuncName x)]; Nothing->[]})
emitStatement' f (NOracleTable ann name source value size) = do
    pos<-mpos ann
    pushOp $ MQOracleTable pos (fromFuncName name) size [(DecompositionRep $ fromFuncName source), (OracleTableRep value)]
emitStatement' f NOracleLogic{} = error "unreachable"
emitStatement' f (NResolvedOracleLogic ann ty name args body) = do
    pos <- mpos ann
    let first_args = map (\(ty, s)->(mapType ty, fromSSA s)) args
    curSsa <- use ssaId
    body' <- scopedStatement first_args [] (mapM f body) curSsa True
    let entry = head body'
    let region = entry{blockBody = init $ blockBody entry}
    pushOp $ MQOracleLogic pos (fromFuncName name) (Just $ mapType ty) region
emitStatement' f (NWhileWithGuard ann cond body breakflag) = do
    pos<-mpos ann
    curSsa <- use ssaId
    break_block <- unscopedStatement (emitExpr breakflag) curSsa
    curSsa <- use ssaId
    cond_block <- unscopedStatement (emitExpr cond) curSsa
    curSsa <- use ssaId
    body_block <- scopedStatement [] [MSCFYield pos] (mapM f body) curSsa False
    let break_ssa = fromSSA $ termId $ annotation breakflag
    let cond_ssa = fromSSA $ termId $ annotation cond
    pushOp $ MSCFWhile pos break_block cond_block cond_ssa break_ssa [MSCFExecRegion pos body_block]
emitStatement' f NProcedureWithRet{} = error "unreachable"
emitStatement' f (NResolvedProcedureWithRet ann ret mname args body (Just retval) (Just retvar)) = do
    let name = if mname=="main" then "__isq__main" else mname
    pos<-mpos ann
    let first_args = map (\(ty, s)->(mapType ty, fromSSA s)) args
    curSsa <- use ssaId
    load_return_value <- unscopedStatement (emitExpr retval) curSsa
    let tail_ret = load_return_value ++ [MFreeMemref pos (fromSSA $ snd retvar) (mapType $ fst retvar), MReturn pos (astMType retval, ssa $ annotation retval) False]
    curSsa <- use ssaId
    body' <- scopedStatement first_args tail_ret (mapM f body) curSsa False
    let first_alloc = MAllocMemref pos (fromSSA $ snd retvar) (mapType $ fst retvar) $ SSA ""
    let body_head = head body'
    let body'' = (body_head{blockBody = first_alloc : blockBody body_head}):tail body'
    pushOp $ MFunc pos (fromFuncName name) (Just $ mapType ret) body''
emitStatement' f (NResolvedProcedureWithRet ann ret mname args body Nothing Nothing) = do
    let name = if mname=="main" then "__isq__main" else mname
    pos<-mpos ann
    let first_args = map (\(ty, s)->(mapType ty, fromSSA s)) args
    curSsa <- use ssaId
    body'<-scopedStatement first_args [MReturnUnit pos] (mapM f body) curSsa False
    pushOp $ MFunc pos (fromFuncName name) Nothing body'
emitStatement' f NResolvedProcedureWithRet{} = error "unreachable"
emitStatement' f (NJumpToEndOnFlag ann flag) = do
    pos<-mpos ann
    flag'<-emitExpr flag
    pushBlock pos flag'
emitStatement' f (NJumpToEnd ann) = do
    pos<-mpos ann
    pushBlockUnconditioned pos
emitStatement' f NTempvar{} = error "unreachable"
emitStatement' f (NResolvedDefvar ann defs) = do
    pos<-mpos ann
    in_logic <- use inLogic
    let one_def :: (Type (), Int, Maybe TCExpr) -> State RegionBuilder ()
        -- isQ source: sub_ty arr[] = lis;
        one_def ((Type () (Array _) [sub_ty]), ssa, Just (EList eann lis)) = do
            let rlen = length lis
            let mlir_ty = mapType $ Type () (Array rlen) [sub_ty]
            pushAllocFree pos (mlir_ty, fromSSA ssa)
            let one_assign base (index, right) = do
                    index_ssa <- nextSsaId
                    pushOp $ MLitInt pos (fromSSA index_ssa) index
                    ref_ssa <- nextSsaId
                    pushOp $ MTakeRef pos (fromSSA ref_ssa) (mlir_ty, fromSSA base) (fromSSA index_ssa) in_logic
                    initialized_val <- emitExpr right
                    pushOp $ MStore pos (mapType $ refType () sub_ty, fromSSA ref_ssa) initialized_val
            mapM_ (one_assign ssa) $ zip [0..rlen-1] lis

        -- isQ source: sub_ty arr[elen];
        one_def ((Type () (Array _) [sub_ty]), ssa, Just len) = do
            len' <- emitExpr len
            case in_logic of
                True -> return ()
                False -> do
                    let mlir_ty = mapType $ Type () (Array 0) [sub_ty]
                    pushOp (MAllocMemref pos (fromSSA ssa) mlir_ty len')
                    pushRAII [MFreeMemref pos (fromSSA ssa) mlir_ty]
        one_def (ty, ssa, Just initializer) = do
            initialized_val <- emitExpr initializer
            pushAllocFree pos (mapType ty, fromSSA ssa)
            pushOp $ MStore pos (mapType ty, fromSSA ssa) initialized_val
        one_def (ty, ssa, Nothing) = do
            pushAllocFree pos (mapType ty, fromSSA ssa)
    mapM_ one_def defs
emitStatement' f NExternGate{} = error "unreachable"
emitStatement' f NProcedureWithDerive{} = error "unreachable"
emitStatement' f (NResolvedExternGate ann name extraparams sz qirname) = do
    pos<-mpos ann
    let extern_name = fromFuncName qirname
    let extra_param_types = map mapType extraparams
    pushOp $ MExternFunc pos extern_name Nothing (extra_param_types++(replicate sz QIRQubit))
    pushOp $ MQDefGate pos (fromFuncName name) sz extra_param_types [QIRRep extern_name]
    -- Dirty hack to provide basic gates (WITHOUT PREFIX) for decomposition and syntax algorithms
    let bare_name = last $ splitOn "." name
    pushOp $ MQDefGate pos (fromFuncName bare_name) sz extra_param_types [QIRRep extern_name]
emitStatement' f (NDerivedGatedef ann name source extraparams sz ) = do
    pos<-mpos ann
    let extra_param_types = map mapType extraparams
    pushOp $ MQDefGate pos (fromFuncName name) sz extra_param_types [DecompositionRep $ fromFuncName source]
emitStatement' f (NDerivedOracle ann name source extraparams sz ) = do
    pos<-mpos ann
    let extra_param_types = map mapType extraparams
    pushOp $ MQDefGate pos (fromFuncName name) sz extra_param_types [OracleRep $ fromFuncName source]
emitStatement' f NGlobalDefvar{} = error "not top"
emitStatement' f NOracle {} = error "not top"
emitStatement' f other = error "unexpected statement to emit"
--emitStatement' f (NJumpToEndOnFlag)=
emitStatement :: AST TypeCheckData -> State RegionBuilder ()
emitStatement = fix emitStatement'

scopedStatement :: [(MLIRType, SSA)] -> [MLIROp] -> State RegionBuilder a -> Int -> Bool -> State RegionBuilder [MLIRBlock]
scopedStatement args tailops op curSsa in_logic = do
    file<-use filename
    let region = generalRegion file args tailops curSsa in_logic
    return $ evalState (op >> finalizeBlock) region

unscopedStatement :: State RegionBuilder a -> Int -> State RegionBuilder [MLIROp]
unscopedStatement op curSsa = do
    file<-use filename
    let region = generalRegion file [] [] curSsa False
    let finalized_block = evalState (op >> finalizeBlock) region
    let x = head finalized_block
    let y= init $ blockBody x
    return y

unscopedStatement' :: String -> State RegionBuilder a -> Int -> ([MLIROp], Int)
unscopedStatement' file op ssa =
    let region = generalRegion file [] [] ssa False
        (finalized_block, region') = runState (op >> finalizeBlock) region
        x = head finalized_block
        y = init $ blockBody x
    in (y, _ssaId region')


data TopBuilder = TopBuilder{
    _mainModule :: [MLIROp],
    _globalInitializers :: [MLIROp],
    _currentSsa :: Int
} deriving Show
makeLenses ''TopBuilder

nextCurrentSsa :: State TopBuilder Int
nextCurrentSsa = do
    id <- use currentSsa
    currentSsa %= (+1)
    return id

emitTop :: String->AST TypeCheckData->State TopBuilder ()
emitTop file x@NResolvedProcedureWithRet{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop file (NGlobalDefvar ann defs) = do
    let Pos l c f = sourcePos ann
    let pos = MLIRLoc f l c
    let def_one :: (Type (), Int, String, Maybe TCExpr) -> State TopBuilder ()
        def_one (ty, s, name, initializer) = do
            let stmt = MGlobalMemref pos (fromFuncName name) (mapType ty)
            mainModule %= (stmt:)
            case initializer of
                Nothing -> return ()
                Just initial -> do
                    let use_global_ref = MUseGlobalMemref pos (fromSSA s) (fromFuncName name) (mapType ty)
                    ops <- case ty of
                            Type () (Array _) [sub_ty] -> do
                                let mlir_ty = mapType ty
                                let lis = exprListElems initial -- initial must be an EList
                                let rlen = length lis
                                let one_assign :: Int -> (Int, TCExpr) -> State TopBuilder [MLIROp]
                                    one_assign base (index, right) = do
                                        index_ssa <- nextCurrentSsa
                                        let mint = MLitInt pos (fromSSA index_ssa) index
                                        ref_ssa <- nextCurrentSsa
                                        let mref = MTakeRef pos (fromSSA ref_ssa) (mlir_ty, fromSSA base) (fromSSA index_ssa) False
                                        curSsa <- use currentSsa
                                        let (right_ops, ssa') = unscopedStatement' file (emitExpr right) curSsa
                                        currentSsa .= ssa'
                                        let mstore = MStore pos (mapType $ refType () sub_ty, fromSSA ref_ssa) (ssa $ annotation right)
                                        return $ [mint, mref] ++ right_ops ++ [mstore]
                                ops <- mapM (one_assign s) $ zip [0..rlen-1] lis
                                return $ concat ops
                            other -> do
                                curSsa <- use currentSsa
                                let (compute_init_value, ssa') = unscopedStatement' file (emitExpr initial) curSsa
                                currentSsa .= ssa'
                                let store_back = MStore pos (mapType ty, fromSSA s) (ssa $ annotation initial)
                                return $ compute_init_value ++ [store_back]
                    let init = use_global_ref : ops
                    globalInitializers %= (reverse init++)
    mapM_ def_one defs
emitTop file x@NResolvedGatedef{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop file x@NOracleTable{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop file x@NResolvedOracleLogic{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop file x@NResolvedExternGate{} = do
    ssa <- use currentSsa
    let ([g, efn, efn2], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= ([efn,efn2,g]++)
emitTop file x@NDerivedGatedef{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop file x@NDerivedOracle{} = do
    ssa <- use currentSsa
    let ([fn], ssa') = unscopedStatement' file (emitStatement x) ssa
    currentSsa .= ssa'
    mainModule %= (fn:)
emitTop _ x = error $ "unreachable" ++ show x


generateMLIRModule :: String -> ([AST TypeCheckData], Int) -> MLIROp
generateMLIRModule file (xs, ssa) = 
    let builder = execState (mapM_ (emitTop file) xs) (TopBuilder [] [] ssa)
        main = _mainModule builder
        initialize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_initialize") Nothing [MLIRBlock (fromBlockName 1) [] (reverse (_globalInitializers builder) ++[MReturnUnit MLIRPosUnknown])]
        finalize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_finalize") Nothing [MLIRBlock (fromBlockName 1) [] [MReturnUnit MLIRPosUnknown]]
        args = [(Memref Nothing Index,SSA {unSsa = "%ssa_1"}),(Memref Nothing M.Double,SSA {unSsa = "%ssa_2"})]
        entry = MFunc MLIRPosUnknown (fromFuncName "__isq__entry") Nothing  [MLIRBlock (fromBlockName 1) args [
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_initialize") [] False,
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__main") args False,
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_finalize") [] False,
                MReturnUnit MLIRPosUnknown 
            ]]
    in MModule MLIRPosUnknown (reverse $ entry : finalize : initialize:view mainModule builder)
