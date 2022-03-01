{-# LANGUAGE TemplateHaskell, FlexibleContexts, ViewPatterns #-}
module ISQ.Lang.MLIRGen where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.TypeCheck
import ISQ.Lang.MLIRTree hiding (Bool, Gate, Unit, Double)
import qualified ISQ.Lang.MLIRTree as M
import Control.Monad.State
import Control.Lens
import ISQ.Lang.ISQv2Tokenizer(Pos(Pos), Annotated (annotation))



data RegionBuilder = RegionBuilder{
    _nextBlockId :: Int,
    _nextTailBlockId :: Int,
    _currentBlock:: MLIRBlock,
    _headBlocks :: [MLIRBlock],
    _tailBlocks :: [MLIRBlock],
    _filename :: String
} deriving Show

makeLenses ''RegionBuilder

type MLIRGen = State RegionBuilder

mapType :: EType->MLIRType
mapType (Type () Bool []) = M.Bool
mapType (Type () Ref [x]) = BorrowedRef (mapType x)
mapType (Type () Int []) = Index
mapType (Type () Qbit []) = QState
mapType (Type () UnknownArray [x]) = Memref Nothing (mapType x)
mapType (Type () (FixedArray n) [x]) = Memref (Just n) (mapType x)
mapType (Type () (Gate n) []) = M.Gate n
mapType (Type () Double []) = M.Double
mapType _ = error "unsupported type"



-- Create an empty region builder, with entry and exit.
generalRegion :: String->[(MLIRType, SSA)]->[MLIROp]->RegionBuilder
generalRegion name init_args end_body = RegionBuilder 1 1 (MLIRBlock (BlockName "^entry") init_args []) [] [MLIRBlock (BlockName "^exit") [] end_body] name

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
    pushOp (MAllocMemref pos val ty)
    pushRAII [MFreeMemref pos val ty]

mpos :: TypeCheckData->State RegionBuilder MLIRPos
mpos d = let Pos x y = sourcePos d in fmap (\z->MLIRLoc z x y) (use filename)

ssa :: TypeCheckData->SSA
ssa x = fromSSA (termId x)

astMType :: (Annotated p)=>p TypeCheckData->MLIRType
astMType x = mapType $ termType $ annotation x

mType :: TypeCheckData->MLIRType
mType x = mapType $ termType $ x

binopTranslate :: BinaryOperator->MLIRType->MLIRBinaryOp
binopTranslate Add Index = mlirAddi
binopTranslate Sub Index = mlirSubi
binopTranslate Mul Index = mlirMuli
binopTranslate Div Index = mlirFloorDivsi 
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
binopTranslate _ _ = undefined

unaryopTranslate Neg M.Double = mlirNegF
unaryopTranslate Neg Index = mlirNegI
unaryopTranslate _ _ = undefined

emitExpr' :: (Expr TypeCheckData->State RegionBuilder SSA)->Expr TypeCheckData->State RegionBuilder SSA
emitExpr' f (EIdent ann name) = error "unreachable"
emitExpr' f x@(EBinary ann binop lhs rhs) = do
    lhs'<-f lhs
    rhs'<-f rhs
    pos<-mpos ann
    let lhsTy = astMType lhs
    let rhsTy = astMType rhs
    let i = ssa ann
    pushOp $ MBinary pos i lhs' rhs' (binopTranslate binop lhsTy)
    return i
emitExpr' f (EUnary ann uop lhs) = do
    lhs'<-f lhs
    case uop of
        Positive -> return lhs' -- Positive x is x
        _ -> do
            pos<-mpos ann
            let lhsTy = astMType lhs
            let i = ssa ann
            pushOp $ MUnary pos i lhs' (unaryopTranslate uop lhsTy)
            return i
emitExpr' f (ESubscript ann base offset) = do
    base'<-f base
    offset'<-f offset
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MTakeRef pos i (astMType base, base') offset'
    return i
emitExpr' f x@(ECall ann (EGlobalName ann2 mname) args) = do
    let name = if mname=="main" then "__isq__main" else mname
    args'<-mapM f args
    let args'' = zip (fmap astMType args) args'
    let ret = if (ty $ termType $ ann) == Unit then Nothing else Just (astMType x, ssa ann)
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MCall pos ret (fromFuncName name) args''
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
emitExpr' f (EEraselist ann sublist) = do
    l<-f sublist
    let i = ssa ann
    pos<-mpos ann
    pushOp $ MEraseMemref pos i (astMType sublist, l)
    return i
emitExpr :: Expr TypeCheckData -> State RegionBuilder SSA
emitExpr = fix emitExpr'

emitStatement' :: (AST TypeCheckData-> State RegionBuilder ())->(AST TypeCheckData->State RegionBuilder ())
-- Generic if.
emitStatement' f (NIf ann cond bthen belse) = do
    pos<-mpos ann
    cond'<-emitExpr cond
    then_block <- scopedStatement [] [MSCFYield pos] (mapM f bthen)
    else_block <- scopedStatement [] [MSCFYield pos] (mapM f belse)
    pushOp $ MSCFIf pos cond' [MSCFExecRegion pos then_block] [MSCFExecRegion pos else_block]
emitStatement' f NFor{} = error "unreachable"
emitStatement' f NPass{} = return ()
emitStatement' f NWhile{} = error "unreachable"
emitStatement' f (NCall ann expr) = void $ emitExpr expr
emitStatement' f (NDefvar ann defs) = error "unreachable"
emitStatement' f (NAssign ann lhs rhs) = do
    lhs'<-emitExpr lhs
    rhs'<-emitExpr rhs
    pos<-mpos ann
    pushOp $ MStore pos (astMType lhs, lhs') rhs'
emitStatement' f NGatedef{} = error "unreachable"
emitStatement' f NReturn{} = error "unreachable"
emitStatement' f (NCoreUnitary ann (EGlobalName ann2 name) ops mods) = do
    let go Inv (l, f) = (l, not f)
        go (Ctrl x i) (l, f) = (replicate i x ++ l, f)
        folded_mods = foldr go ([], False) mods
    ops'<-mapM emitExpr ops
    pos<-mpos ann
    let i = ssa ann2
    let (ins, outs) = unzip $ map (\id->(SSA $ unSsa i ++ "_in_"++show id, SSA $ unSsa i ++ "_out_"++show id)) [1..(length ops)]
    let used_gate = i;
    let gate_type@(M.Gate gate_size) = mType ann2;
    pushOp $ MQUseGate pos used_gate (fromFuncName name) gate_type
    decorated_gate<-case folded_mods of
            ([], False)->return i
            (ctrls, adj)->do
                let decorated_gate = SSA $ unSsa i ++ "_decorated"
                pushOp $ MQDecorate pos decorated_gate used_gate folded_mods gate_size
                return $ decorated_gate
    zipWithM_ (\in_state in_op->pushOp $ MLoad pos in_state (BorrowedRef QState, in_op)) ins ops'
    pushOp $ MQApplyGate pos outs ins decorated_gate
    zipWithM_ (\out_state in_op->pushOp $ MStore pos (BorrowedRef QState, in_op) out_state) outs ops'
emitStatement' f NCoreUnitary{} = error "first-class gate unsupported"
emitStatement' f (NCoreReset ann operand) = do
    operand'<-emitExpr operand
    pos<-mpos ann
    let i_in = SSA $ unSsa operand' ++"_in"
    let i_out = SSA $ unSsa operand' ++ "_out"
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
    r<-scopedStatement [] [MSCFYield pos] (mapM f body)
    pushOp $ MAffineFor pos lo' hi' step (fromSSA fori) [MSCFExecRegion pos r]
emitStatement' f NResolvedFor{} = error "unreachable"
emitStatement' f (NResolvedGatedef ann name mat sz) = do
    pos<-mpos ann
    pushOp $ MQDefGate pos (fromFuncName name) mat sz
emitStatement' f (NWhileWithGuard ann cond body breakflag) = do
    pos<-mpos ann
    break_block<-unscopedStatement (emitExpr breakflag)
    cond_block<-unscopedStatement (emitExpr cond)
    body_block<-scopedStatement [] [MSCFYield pos] (mapM f body)
    let break_ssa = fromSSA $ termId $ annotation breakflag
    let cond_ssa = fromSSA $ termId $ annotation cond
    pushOp $ MSCFWhile pos break_block cond_block cond_ssa break_ssa [MSCFExecRegion pos body_block]
emitStatement' f NProcedureWithRet{} = error "unreachable"
emitStatement' f (NResolvedProcedureWithRet ann ret mname args body (Just retval) (Just retvar)) = do
    let name = if mname=="main" then "__isq__main" else mname
    pos<-mpos ann
    let first_args = map (\(ty, s)->(mapType ty, fromSSA s)) args
    load_return_value<-unscopedStatement (emitExpr retval)
    let tail_ret = load_return_value ++ [MFreeMemref pos (fromSSA $ snd retvar) (mapType $ fst retvar), MReturn pos (astMType retval, ssa $ annotation retval)]
    body'<-scopedStatement first_args tail_ret (mapM f body)
    let first_alloc = MAllocMemref pos (fromSSA $ snd retvar) (mapType $ fst retvar)
    let body_head = head body'
    let body'' = (body_head{blockBody = first_alloc : blockBody body_head}):tail body'
    pushOp $ MFunc pos (fromFuncName name) (Just $ mapType ret) body''
emitStatement' f (NResolvedProcedureWithRet ann ret mname args body Nothing Nothing) = do
    let name = if mname=="main" then "__isq__main" else mname
    pos<-mpos ann
    pos<-mpos ann
    let first_args = map (\(ty, s)->(mapType ty, fromSSA s)) args
    body'<-scopedStatement first_args [MReturnUnit pos] (mapM f body)
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
    let one_def (ty, ssa, Just initializer) = do
            initialized_val <-emitExpr initializer
            pushAllocFree pos (mapType ty, fromSSA ssa)
            pushOp $ MStore pos (mapType ty, fromSSA ssa) initialized_val
        one_def (ty, ssa, Nothing) = do
            pushAllocFree pos (mapType ty, fromSSA ssa)
    mapM_ one_def defs
emitStatement' f NGlobalDefvar{} = error "not top"
--emitStatement' f (NJumpToEndOnFlag)=
emitStatement :: AST TypeCheckData -> State RegionBuilder ()
emitStatement = fix emitStatement'

scopedStatement :: [(MLIRType, SSA)]->[MLIROp]->State RegionBuilder a->State RegionBuilder [MLIRBlock]
scopedStatement args tailops op = do
    file<-use filename
    let region = generalRegion file args tailops
    return $ evalState (op >> finalizeBlock) region

unscopedStatement :: State RegionBuilder a->State RegionBuilder [MLIROp]
unscopedStatement op = do
    file<-use filename
    let region = generalRegion file [] []
    let finalized_block = evalState (op >> finalizeBlock) region
    let x = head finalized_block
    let y= init $ blockBody x
    return y


unscopedStatement' :: String->State RegionBuilder a->[MLIROp]
unscopedStatement' file op =
    let region = generalRegion file [] []
        finalized_block = evalState (op >> finalizeBlock) region
        x = head finalized_block
        y= init $ blockBody x
    in y


data TopBuilder = TopBuilder{
    _mainModule :: [MLIROp],
    _globalInitializers :: [MLIROp]
} deriving Show
makeLenses ''TopBuilder
emitTop :: String->AST TypeCheckData->State TopBuilder ()
emitTop file x@NResolvedProcedureWithRet{} = do
    let [fn] = unscopedStatement' file (emitStatement x)
    mainModule %= (fn:)
emitTop file (NGlobalDefvar ann defs) = do
    let Pos l c = sourcePos ann
    let pos = MLIRLoc file l c
    let def_one (ty, s, name, initializer) = do
            let stmt = MGlobalMemref pos (fromFuncName name) (mapType ty)
            mainModule %= (stmt:)
            let maybe_initializer = do
                    let use_global_ref = MUseGlobalMemref pos (fromSSA s) (fromFuncName name) (mapType ty)
                    i<-initializer
                    let compute_init_value = unscopedStatement' file (emitExpr i)
                    let store_back = MStore pos (mapType ty, fromSSA s) (ssa $ annotation i)
                    return $ use_global_ref : (compute_init_value ++ [store_back])
            case maybe_initializer of
                Just init->globalInitializers%=(reverse init++)
                Nothing->return ()
    mapM_ def_one defs
emitTop file x@NResolvedGatedef{} = do
    let [fn] = unscopedStatement' file (emitStatement x)
    mainModule %= (fn:)
emitTop _ x = error $ "unreachable" ++ show x


generateMLIRModule :: String->[AST TypeCheckData]->MLIROp
generateMLIRModule file xs = 
    let builder = execState (mapM_ (emitTop file) xs) (TopBuilder [] [])
        main = _mainModule builder
        initialize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_initialize") Nothing [MLIRBlock (fromBlockName 1) [] (reverse (_globalInitializers builder) ++[MReturnUnit MLIRPosUnknown])]
        finalize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_finalize") Nothing [MLIRBlock (fromBlockName 1) [] [MReturnUnit MLIRPosUnknown]]
        entry = MFunc MLIRPosUnknown (fromFuncName "__isq__entry") Nothing  [MLIRBlock (fromBlockName 1) [] [
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_initialize") [],
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__main") [],
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_finalize") [],
                MReturnUnit MLIRPosUnknown 
            ]]
    in MModule MLIRPosUnknown (reverse $ entry : finalize : initialize:view mainModule builder)


--mlirGen :: [TCAST]->MLIRGen [MLIROp]
