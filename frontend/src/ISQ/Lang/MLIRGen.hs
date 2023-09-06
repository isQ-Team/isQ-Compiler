{-# LANGUAGE TemplateHaskell, FlexibleContexts, ViewPatterns #-}
module ISQ.Lang.MLIRGen where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.ISQv2Tokenizer(Pos(Pos), Annotated (annotation))
import ISQ.Lang.TypeCheck
import ISQ.Lang.MLIRTree hiding (Bool, Gate, Unit, Double, Complex, Ket)
import qualified ISQ.Lang.MLIRTree as M
import Control.Monad (when)
import Control.Monad.State (fix, void, State, zipWithM_, evalState, execState, runState)
import Control.Lens
import Data.List (isSuffixOf, take)
import Data.List.Split (splitOn)
import Data.Maybe (fromJust)
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
mapType (Type () Complex []) = M.Complex
mapType (Type () Ket []) = M.Ket
mapType (Type () FuncTy (ret:arg)) = Func (mapType ret) $ map mapType arg
mapType (Type () Param []) = I8
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

swapCurrentBlock :: MLIRBlock -> State RegionBuilder MLIRBlock
swapCurrentBlock new_curr = do
    curr <- use currentBlock
    currentBlock .= new_curr
    return curr

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
binopTranslate CeilDiv Index = mlirCeilDivsi
binopTranslate Mod Index = mlirRemsi
binopTranslate Pow Index = mlirPowi
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
binopTranslate Add M.Complex = mlirAddc
binopTranslate Sub M.Complex = mlirSubc
binopTranslate Mul M.Complex = mlirMulc
binopTranslate Div M.Complex = mlirDivc
binopTranslate Add M.Ket = mlirAddk
binopTranslate Sub M.Ket = mlirSubk
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
logicOpTranslate (Cmp Equal) = "xnor"
logicOpTranslate (Cmp NEqual) = "xor"
logicOpTranslate _ = undefined

getParamName :: String -> String
getParamName name = last $ splitOn "." name

paramGate :: [TCExpr] -> Bool
paramGate [] = False
paramGate (x:xs) = do
    let rx = case x of
            EGlobalName ann@(mType->I8) name -> True
            EDeref ann@(mType->I8) val -> True
            _ -> False
    rx || (paramGate xs)


paramExpr :: TCExpr -> State RegionBuilder SSA
paramExpr x = do
    case x of
        EGlobalName ann@(mType->I8) name -> do
            pos <-mpos ann
            let i = ssa ann
            --let index = SSA $ unSsa i ++ "_index"
            pushOp $ MParamref pos i (fromFuncName name) (length $ getParamName name)
            return i
        EDeref ann@(mType->I8) val -> do
            --let index = emitExpr (subscript val)
            pos<-mpos ann
            let i = ssa ann
            let name = globalName (usedExpr val)
            pushOp $ MParamref pos i (fromFuncName name) (length $ getParamName name)
            return i
        _ -> error "bad param expr"

paramIndex :: TCExpr -> State RegionBuilder SSA
paramIndex x = do
    case x of
        EGlobalName ann@(mType->I8) name -> do
            let i = ssa ann
            let index = SSA $ unSsa i ++ "_index"
            return index
        EDeref ann@(mType->I8) val -> do
            pos<-mpos ann
            index <- emitExpr (subscript val)
            pushOp $ MAssertParamIndex pos index
            return index
        _ -> error "bad param expr"

moveQubitBack :: (Expr TypeCheckData, SSA) -> State RegionBuilder ()
moveQubitBack ((EList ann lis), qubit') = do
    let lhs_ty = mType ann
    let Memref _ ty = lhs_ty
    case ty of
        QState -> do
            pos <- mpos ann
            let storeBack (EDeref ann subview, index) = do
                    -- The SSA is defined in emitExpression' f (Elist ...)
                    let deref_ssa = unSsa $ ssa ann
                    let index_ssa = SSA $ deref_ssa ++ "_index"

                    let res_ssa = SSA $ deref_ssa ++ "_res"
                    pushOp $ MTakeRef pos res_ssa (lhs_ty, qubit') index_ssa True
                    pushOp $ MStore pos (BorrowedRef QState, ssa $ annotationExpr subview) res_ssa
            mapM storeBack $ zip lis [0..]
            pushOp $ MFreeIsq pos qubit' lhs_ty
        _ -> return ()
moveQubitBack _ = return ()


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
    base' <- f base
    pos <- mpos ann
    let i = ssa ann
    in_logic <- use inLogic
    case offset of
        ERange _ start size step -> do
            start' <- f $ fromJust start
            step' <- f $ fromJust step
            case fromJust size of
                EIntLit _ v -> pushOp $ MSlice pos i (astMType base, base') start' (Left v) step'
                other -> do
                    size' <- f other
                    pushOp $ MSlice pos i (astMType base, base') start' (Right size') step'
        _ -> do
            offset'<-f offset
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
    mapM moveQubitBack $ zip args args'
    return i
emitExpr' f x@(ECall ann (EResolvedIdent ann2 sym_ssa) args) = do
    args'<-mapM f args
    let args'' = zip (fmap astMType args) args'
    let ret = if (ty $ termType $ ann) == Unit then Nothing else Just (astMType x, ssa ann)
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MCallIndirect pos ret (fromSSA sym_ssa) args''
    return i
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
emitExpr' f (EImagLit ann val) = do
    pos <- mpos ann
    let i = ssa ann
    pushOp $ MLitImag pos i val
    return i
emitExpr' f (EBoolLit ann val) = do
    pos<-mpos ann
    let i = ssa ann
    pushOp $ MLitBool pos i val
    return i
emitExpr' f (EKet ann coe base) = do
    pos <- mpos ann
    coe' <- f coe
    let i = ssa ann
    pushOp $ MKet pos i coe' base
    return i
emitExpr' f (EVector ann vals) = do
    pos <- mpos ann
    let i = ssa ann
    pushOp $ MVec pos i $ MatrixRep [vals]
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
emitExpr' f (EList ann lis) = do
    pos <- mpos ann
    let base = ssa ann
    let ty = mType ann
    pushOp $ MAllocIsq pos base ty
    let emitOne (exp, index) = do
            exp' <- f exp
            let index_ssa = SSA $ (unSsa $ ssa $ annotationExpr exp) ++"_index"
            pushOp $ MLitInt pos index_ssa index
            pushOp $ MStoreOffset pos (ty, base) exp' index_ssa
    mapM emitOne $ zip lis [0..]
    return base
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
emitExpr' f x@(EImplicitCast ann@(mType->M.Bool) val@(astMType->M.Index)) = do
    val' <- f val
    pos <- mpos ann
    let i = ssa ann
    pushOp $ MCast pos i val' mlirIndextoI1
    return i
emitExpr' f x@(EImplicitCast ann@(mType->M.Double) val@(astMType->M.Index)) = do
    val'<-f val
    pos<-mpos ann
    let i = ssa ann
    let i_i64 = SSA $ (unSsa i) ++"_i64"
    pushOp $ MCast pos i_i64 val' mlirIndextoI64
    pushOp $ MCast pos i i_i64 mlirI64toDouble
    return i
emitExpr' f x@(EImplicitCast ann@(mType->M.Index) val@(astMType->M.Double)) = do
    val' <- f val
    pos <- mpos ann
    let i = ssa ann
    let i_i64 = SSA $ (unSsa i) ++"_i64"
    pushOp $ MCast pos i_i64 val' mlirDoubletoI64
    pushOp $ MCast pos i i_i64 mlirI64toIndex
    return i
emitExpr' f (EImplicitCast ann@(mType->M.Complex) val@(astMType->M.Double)) = do
    val' <- f val
    pos <- mpos ann
    let i = ssa ann
    pushOp $ MCast pos i val' mlirDoubletoComplex
    return i
emitExpr' f x@(EImplicitCast {}) = error "not supported"
emitExpr' f (ETempVar {}) = error "unreachable"
emitExpr' f (ETempArg {}) = error "unreachable"
emitExpr' f (EUnitLit ann) = do
    let i = ssa ann
    return i
emitExpr' f (EResolvedIdent ann i) = do
    return $ ssa ann
emitExpr' f (EGlobalName ann@(mType->BorrowedRef _) name) = do
    pos<-mpos ann
    pushOp $ MUseGlobalMemref pos (ssa ann) (fromFuncName name) (mType ann)
    return (ssa ann)
emitExpr' f (EGlobalName ann@(mType->(Memref (Just _) _)) name) = do
    pos<-mpos ann
    pushOp $ MUseGlobalMemref pos (ssa ann) (fromFuncName name) (mType ann)
    return (ssa ann)
emitExpr' f (EGlobalName ann@(mType->Func ret args) name) = do
    pos<-mpos ann
    pushOp $ MCallConst pos (ssa ann) (fromFuncName name) ret args 
    return (ssa ann)
emitExpr' f (EGlobalName ann name) = error "first-class global gate/function not supported"
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
emitStatement' f (NAssert ann exp Nothing) = do
    pos <- mpos ann
    exp' <- emitExpr exp
    pushOp $ MAssert pos (astMType exp, exp') Nothing
emitStatement' f (NResolvedAssert ann q mat) = do
    pos <- mpos ann
    q' <- emitExpr q
    pushOp $ MAssert pos (astMType q, q') $ Just $ MatrixRep mat
emitStatement' f (NAssertSpan ann q vecs) = do
    pos <- mpos ann
    q' <- emitExpr q
    vecs' <- mapM emitExpr $ head vecs
    pushOp $ MAssertSpan pos (astMType q, q') vecs'
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

    let paramgate = paramGate ops
    case paramgate of
        True -> do
            pos<-mpos ann
            let args = head ops;
            args_ssa <- paramExpr args;
            args_index <- paramIndex args;
            let qubit = last ops;
            qubit_ssa <- emitExpr qubit;
            let i = ssa ann2;
            let used_gate = i;
            let ty = termType ann2;
            let len = length $ subTypes ty;
            let gate_type@(M.Gate gate_size) = mapType ty;

            let (ins, outs) = unzip $ map (\id->(SSA $ unSsa i ++ "_in_"++show id, SSA $ unSsa i ++ "_out_"++show id)) [1..length [qubit_ssa]]
            pushOp $ MQUseGate pos used_gate (fromFuncName $ getParamName $ name ++ "p") gate_type [(astMType args, args_ssa), (Index, SSA $ unSsa args_ssa ++ "_len"), (Index, args_index)] True
            zipWithM_ (\in_state in_op->pushOp $ MLoad pos in_state (BorrowedRef QState, in_op)) ins [qubit_ssa]
            pushOp $ MQApplyGate pos outs ins i True
            zipWithM_ (\out_state in_op->pushOp $ MStore pos (BorrowedRef QState, in_op) out_state) outs [qubit_ssa]
        False -> do
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
emitStatement' f (NResolvedInit ann qubit space) = do
    qubit' <- emitExpr qubit
    pos <- mpos ann
    let Type () (Array len) [q] = termType $ annotationExpr qubit
    pushOp $ MQInit pos qubit' len space
    moveQubitBack (qubit, qubit')
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
emitStatement' f (NSwitch ann cond cases defau) = do
    pos <- mpos ann
    cond' <- emitExpr cond
    let emit (NCase ann num stats _) = do
            curr <- swapCurrentBlock $ MLIRBlock (fromBlockName 0) [] []
            mapM f stats
            case_block <- swapCurrentBlock curr
            return (num, reverse $ blockBody case_block)
    cases' <- mapM emit cases
    curr <- swapCurrentBlock $ MLIRBlock (fromBlockName 0) [] []
    mapM f defau
    defau_block <- swapCurrentBlock curr
    pushOp $ MSwitch pos (astMType cond, cond') cases' $ reverse $ blockBody defau_block
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
                    index_id <- nextSsaId
                    let index_ssa = fromSSA index_id
                    pushOp $ MLitInt pos index_ssa index
                    ref_ssa <- nextSsaId
                    initialized_val <- emitExpr right
                    case in_logic of
                        True -> pushOp $ MStoreOffset pos (mlir_ty, fromSSA base) initialized_val index_ssa
                        False -> do
                            pushOp $ MTakeRef pos (fromSSA ref_ssa) (mlir_ty, fromSSA base) index_ssa in_logic
                            pushOp $ MStore pos (mapType $ refType () sub_ty, fromSSA ref_ssa) initialized_val
            mapM_ (one_assign ssa) $ zip [0..rlen-1] lis

        one_def (ty@(Type () (Array _) [sub_ty]), ssa, Just initOrLen) = do
            initOrLen' <- emitExpr initOrLen
            let iol_ty = termType $ annotationExpr initOrLen
            case iol_ty of
                -- isQ source: sub_ty arr[len];
                Type () Int [] -> do
                    let mlir_ty = case in_logic of
                            True -> mapType ty
                            False -> mapType $ Type () (Array 0) [sub_ty]
                    pushOp (MAllocMemref pos (fromSSA ssa) mlir_ty initOrLen')
                    pushRAII [MFreeMemref pos (fromSSA ssa) mlir_ty]
                -- isQ source: sub_ty arr[] = init
                _ -> return ()
        one_def (ty, ssa, Just initializer) = do
            initialized_val <- emitExpr initializer
            case in_logic of
                True -> return ()
                False -> do
                    pushAllocFree pos (mapType ty, fromSSA ssa)
                    pushOp $ MStore pos (mapType ty, fromSSA ssa) initialized_val
        one_def (ty, ssa, Nothing) = do
            case in_logic of
                True -> return ()
                False -> do
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
emitTop file (NResolvedDefParam ann params) = do
    let Pos l c f = sourcePos ann
    let pos = MLIRLoc f l c
    let def_one :: (String, String) -> State TopBuilder ()
        def_one (name, val) = do
            let stmt = MParamDef pos (fromFuncName name) val (length val)
            mainModule %= (stmt:)

    mapM_ def_one params

emitTop _ x = error $ "unreachable" ++ show x


generateMLIRModule :: String -> ([AST TypeCheckData], Int) -> MLIROp
generateMLIRModule file (xs, ssa) = 
    let builder = execState (mapM_ (emitTop file) xs) (TopBuilder [] [] ssa)
        main = _mainModule builder
        initialize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_initialize") Nothing [MLIRBlock (fromBlockName 1) [] (reverse (_globalInitializers builder) ++[MReturnUnit MLIRPosUnknown])]
        finalize = MFunc MLIRPosUnknown (fromFuncName "__isq__global_finalize") Nothing [MLIRBlock (fromBlockName 1) [] [MReturnUnit MLIRPosUnknown]]
        ssa_arg_rank = SSA {unSsa = "%rank"}
        args = [(Memref Nothing Index, SSA {unSsa = "%ssa_1"}), (Memref Nothing M.Double, SSA {unSsa = "%ssa_2"})]
        ssa_rank = SSA {unSsa = "%r"}
        entry = MFunc MLIRPosUnknown (fromFuncName "__isq__entry") Nothing  [MLIRBlock (fromBlockName 1) (args ++ [(Index, ssa_arg_rank)]) [
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_initialize") [] False,
                MUseGlobalMemref MLIRPosUnknown ssa_rank (fromFuncName "qmpi.__qmpi_rank") (BorrowedRef Index),
                MStore MLIRPosUnknown (BorrowedRef Index, ssa_rank) ssa_arg_rank,
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__main") args False,
                MCall MLIRPosUnknown Nothing (fromFuncName "__isq__global_finalize") [] False,
                MReturnUnit MLIRPosUnknown 
            ]]
    in MModule MLIRPosUnknown (reverse $ entry : finalize : initialize:view mainModule builder)
