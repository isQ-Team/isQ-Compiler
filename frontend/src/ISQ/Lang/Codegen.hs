{-# LANGUAGE TemplateHaskell, ViewPatterns, FlexibleInstances #-}
module ISQ.Lang.Codegen where
import Control.Lens
import ISQ.Lang.AST
import qualified Data.Map as Map
import Control.Monad.State.Lazy
import Control.Monad.Except
import Data.Complex
import Data.Foldable
import Data.List.Extra (allSame)
type SSA = String

data Symbol ann = Symbol {_symName::String, _symType::VarType ann, _ssa::SSA} deriving (Show)
-- Type checking: check if all variables are defined and have correct type.
newtype SymbolTable ann = SymbolTable {
    _varMap :: Map.Map String (Symbol ann)
} deriving Show

makeLenses ''Symbol
makeLenses ''SymbolTable

class (Monad m)=>CodeSunk m where
    -- b = op a
    emitUnaryOp :: Pos->UnaryOp->SSA->SSA->m ()
    -- c = a op b
    emitBinaryOp :: Pos->BinaryOp->SSA->SSA->SSA->m ()
    -- ret = 1-sized memory view of array.
    emitSlice :: Pos->SSA->UnitKind->SSA->SSA->Maybe Int->m ()
    -- ret = arr [offset]
    emitReadIntOp :: Pos->SSA->SSA->m ()
    -- arr = val
    emitWriteIntOp :: Pos->SSA->SSA->m ()
    emitProcedure :: Pos->ProcDef ann->[SSA]->m ()
    emitCall :: Pos->ProcDef ann->SSA->[SSA]->m ()
    -- Push a block into code emission.
    emitPushBlock :: m ()
    emitIf :: Pos->SSA->m ()
    emitWhile :: Pos->SSA->m ()
    emitPrint :: Pos->SSA->m ()
    emitFor :: Pos->SSA->SSA->SSA->m ()
    emitReturn :: Pos->Maybe SSA->m ()
    emitReset :: Pos->SSA->m ()
    emitGateApply :: Pos->GateDecorator->GateDef ann->[SSA]->m ()
    emitGateDef :: GateDef Pos->[[Complex Double]]->m ()
    emitMeasure :: Pos->SSA->SSA->m ()
    emitConst :: Pos->Int->SSA->m ()
    emitPass :: Pos->m ()
    emitGlobalDef :: Pos->SSA->VarDef Pos-> m ()
    emitLocalDef :: Pos->SSA->VarDef ann-> m ()
    incrBlockIndent :: m ()
    decrBlockIndent :: m ()
    emitProgramHeader :: Pos->m ()


data Codegen = Codegen {
    _symbolTables :: [SymbolTable Pos],
    _definedProcs :: Map.Map String (ProcDef Pos),
    _definedGates :: Map.Map String (GateDef Pos),
    _ssaCounter :: Int
} deriving Show

data GrammarError =
      TypeMismatch {_wanted:: VarType (), _got:: VarType Pos}
    | UndefinedSymbol {_missingSymbol:: Ident Pos}
    | RedefinedSymbol {_newSymbol:: Ident Pos, _oldSymbol:: Symbol Pos}
    | RedefinedProc {_newProc:: Ident Pos, _oldProc:: ProcDef Pos}
    | RedefinedGate {_newGate:: Ident Pos, _oldGate:: GateDef Pos}
    | NonConstantExpr {_expr:: Expr Pos}
    | FloatNotSupportedYet {_expr :: Expr Pos}
    | IndexingIntoNonArray {_indexedSymbol :: Ident Pos}
    | ArraySizeMismatch {_expectedSize:: Maybe Int, _actualSize:: Maybe Int}
    | ArgumentNumberMismatch {_callingSite:: Pos, _expectedArgs:: Int, _actualArgs:: Int}
    | InvalidGateMatrix {_badMatrix :: GateDef Pos}
    | InternalCompilerError
    deriving Show
makeLenses ''Codegen
makeLenses ''GrammarError
type CodegenM m = ExceptT GrammarError (StateT Codegen m)


instance (CodeSunk m)=>CodeSunk (CodegenM m) where
    emitUnaryOp a b c d = lift $ lift (emitUnaryOp a b c d)
    emitBinaryOp a b c d e = lift $ lift (emitBinaryOp a b c d e)
    emitSlice a b c d e f = lift $ lift (emitSlice a b c d e f)
    emitReadIntOp a b c = lift $ lift (emitReadIntOp a b c)
    emitWriteIntOp a b c = lift $ lift (emitWriteIntOp a b c)
    emitProcedure a b c = lift $ lift (emitProcedure a b c)
    emitCall a b c d = lift $ lift (emitCall a b c d)
    emitPushBlock = lift $ lift emitPushBlock
    emitIf a b = lift $ lift (emitIf a b)
    emitWhile a b = lift $ lift (emitWhile a b)
    emitPrint a b = lift $ lift (emitPrint a b)
    emitFor a b c d = lift $ lift (emitFor a b c d)
    emitReturn a b = lift $ lift (emitReturn a b)
    emitReset a b = lift $ lift (emitReset a b)
    emitGateApply a b c d = lift $ lift (emitGateApply a b c d)
    emitGateDef a b = lift $ lift (emitGateDef a b)
    emitMeasure a b c = lift $ lift (emitMeasure a b c)
    emitConst a b c = lift $ lift (emitConst a b c)
    emitPass a = lift $ lift (emitPass a)
    emitGlobalDef a b c = lift $ lift (emitGlobalDef a b c)
    emitLocalDef a b c = lift $ lift (emitLocalDef a b c)
    incrBlockIndent = lift $ lift incrBlockIndent
    decrBlockIndent = lift $ lift decrBlockIndent
    emitProgramHeader a = lift $ lift (emitProgramHeader a)

nextSSA :: (Monad m)=>CodegenM m SSA
nextSSA = do
    s<-use ssaCounter
    ssaCounter .= s+1
    return $ "x" ++ show s

scope :: (Monad m)=>CodegenM m ()
scope = do
    m<-use symbolTables
    symbolTables .= SymbolTable Map.empty:m

unscope :: (Monad m)=>CodegenM m ()
unscope = do
    xs<-use symbolTables
    symbolTables .= tail xs

scoped :: (Monad m)=>CodegenM m a->CodegenM m a
scoped a = do
    scope
    x<-a
    unscope
    return x

firstJustsM :: (Monad m, Foldable f)=>f (m (Maybe a))->m (Maybe a)
firstJustsM = foldlM go Nothing where
  go :: Monad m => Maybe a -> m (Maybe a) -> m (Maybe a)
  go Nothing         action  = action
  go result@(Just _) _action = return result

querySymbolInTable :: String->SymbolTable ann->Maybe (Symbol ann)
querySymbolInTable name sym = Map.lookup name (sym^.varMap)

querySymbol :: (Monad m)=>String->CodegenM m (Maybe (Symbol Pos))
querySymbol name = do
    m<-use symbolTables
    return $ msum $ map (querySymbolInTable name) m



addSymbol :: (Monad m)=>Ident Pos->VarType Pos->CodegenM m SSA
addSymbol name ty = do
    tables <- use symbolTables
    let sym = querySymbolInTable (name^.identName) (head tables)
    case sym of
        Just sym->do
            throwError $ RedefinedSymbol name sym
        Nothing->do
            ssa<-nextSSA
            symbolTables._head.varMap %= Map.insert (name^.identName) (Symbol (name^.identName) ty ssa)
            return ssa

evalComplexNumber :: (Monad m)=>Expr Pos->CodegenM m (Complex Double)
evalComplexNumber (ImmComplex x _) = return x
evalComplexNumber (ImmInt x _) = return $ fromIntegral x
evalComplexNumber e@(BinaryOp bin_op x y _) = do
    a<-evalComplexNumber x
    b<-evalComplexNumber y
    o<-op
    return $ a `o` b
    where
        op = case bin_op of
            Add->return (+)
            Sub->return (-)
            Mul->return (*)
            Div->return (/)
            _ ->throwError $ NonConstantExpr e
evalComplexNumber e@(UnaryOp unary_op x _) = do
    a<-evalComplexNumber x
    o<-op
    return $ o a
    where
        op = case unary_op of
            Neg->return negate
            Positive -> return id
evalComplexNumber x = throwError $ NonConstantExpr x



data TypedSSA = TypedSSA {
    _ssaVal::SSA,
    _ssaType::VarType Pos
} deriving Show
makeLenses ''TypedSSA

requireInt :: (Monad m)=>TypedSSA->CodegenM m SSA
requireInt tssa@(TypedSSA ssa (UnitType Int _)) = return ssa
requireInt tssa@(TypedSSA _ t@(view annotation->pos)) = throwError $ TypeMismatch (UnitType Int ()) t
requireQbit :: (Monad m)=>TypedSSA->CodegenM m SSA
requireQbit tssa@(TypedSSA ssa (UnitType Qbit _)) = return ssa
requireQbit tssa@(TypedSSA _ t@(view annotation->pos)) = throwError $ TypeMismatch (UnitType Qbit ()) t

requireUnit :: (Monad m)=>UnitKind->TypedSSA->CodegenM m SSA
requireUnit Int = requireInt
requireUnit Qbit = requireQbit
requireUnit _ = const $ throwError InternalCompilerError
requireIntArray :: (Monad m)=>TypedSSA->CodegenM m SSA
requireIntArray tssa@(TypedSSA ssa (Composite Int _ _)) = return ssa
requireIntArray tssa@(TypedSSA _ t@(view annotation->pos)) = throwError $ TypeMismatch (Composite Int Nothing ()) t
requireQbitArray :: (Monad m)=>TypedSSA->CodegenM m SSA
requireQbitArray tssa@(TypedSSA ssa (Composite Qbit _ _)) = return ssa
requireQbitArray tssa@(TypedSSA _ t@(view annotation->pos)) = throwError $ TypeMismatch (Composite Qbit Nothing ()) t
requireArray :: (Monad m)=>UnitKind->TypedSSA->CodegenM m SSA
requireArray Int  = requireIntArray
requireArray Qbit = requireQbitArray
requireArray _ = const $ throwError InternalCompilerError

-- Array size casting rule: only allowing discard of array size.
requireArrayWithCompatibleSize :: (Monad m)=>(UnitKind, Maybe Int)->TypedSSA->CodegenM m SSA
requireArrayWithCompatibleSize (k, sz) tssa= do
    ssa<-requireArray k tssa
    case sz of
        Just sz'-> let real_length = _arrLen (tssa^.ssaType) in when ((Just sz') /= real_length) $ throwError $ ArraySizeMismatch sz real_length
        Nothing -> return ()
    return ssa


evalLeftExpr :: (CodeSunk m)=>LeftValueExpr Pos->CodegenM m TypedSSA
evalLeftExpr (VarRef varname loc) = do
    let name = varname^.identName
    sym<-querySymbol name
    case sym of
        Just sym->do
            return $ TypedSSA (sym^.ssa) (sym^.symType)
        Nothing->throwError $ UndefinedSymbol varname
evalLeftExpr (ArrayRef arrname arr_offset loc) = do
    offset_expr <- evalExpr arr_offset
    offset_expr' <- requireInt offset_expr
    let name = arrname^.identName
    sym<-querySymbol name
    case sym of
        Just sym -> do
            case sym^.symType of
                Composite base_kind arr_len _ -> do
                    sliced_ssa<-nextSSA
                    emitSlice loc sliced_ssa base_kind (sym^.ssa) offset_expr' arr_len
                    return $ TypedSSA sliced_ssa (UnitType base_kind (sym^.symType^.annotation))
                _ -> throwError $ IndexingIntoNonArray arrname
        Nothing -> throwError $ UndefinedSymbol arrname

evalExpr :: (CodeSunk m)=>Expr Pos->CodegenM m TypedSSA
evalExpr e@(ImmComplex _ _) = throwError $ FloatNotSupportedYet e
evalExpr (ImmInt x pos) = do
    ssa<-nextSSA
    emitConst pos x ssa
    return $ TypedSSA ssa (UnitType Int pos)

evalExpr (BinaryOp bin_op x y pos) = do
    a<-evalExpr x
    b<-evalExpr y
    a'<-requireInt a
    b'<-requireInt b
    ssa<-nextSSA
    emitBinaryOp pos bin_op ssa a' b'
    return $ TypedSSA ssa (UnitType Int pos)

evalExpr (UnaryOp unary_op x pos) = do
    a<-evalExpr x
    a'<-requireInt a
    case unary_op of
        Positive ->return $ TypedSSA a' (UnitType Int pos)
        _ -> do
            ssa<-nextSSA
            emitUnaryOp pos unary_op a' ssa
            return $ TypedSSA ssa (UnitType Int pos)
evalExpr (LeftExpr e pos) = do
    a<-evalLeftExpr e
    -- If the integer is unit value, read its value.
    ssa<-case a^.ssaType of
        UnitType Int _ ->  do {ssa<-nextSSA; emitReadIntOp pos ssa (a^.ssaVal); return ssa}
        _ -> return $ a^.ssaVal
    return $ TypedSSA ssa (a^.ssaType)



evalExpr (MeasureExpr leftvalue e') = do
    v<-evalLeftExpr leftvalue
    ssa<-requireQbit v
    ret<-nextSSA
    emitMeasure e' ssa ret
    return $ TypedSSA ret (UnitType Int e')
evalExpr (CallExpr (ProcedureCall name args pos') pos) = do
    let i = name^.identName
    procs<-use definedProcs
    let query_proc = procs Map.!? i
    case query_proc of
        Just p->do
            args'<-mapM evalExpr args
            let requiredTypes = fmap _varType $ p^.parameters
            let check_type r@(UnitType Int _) = requireInt
                check_type r@(UnitType Qbit _) = requireQbit
                check_type r@(Composite Int _ _) = requireIntArray
                check_type r@(Composite Qbit _ _) = requireQbitArray
                check_type _ = \_->throwError InternalCompilerError
            when (length args' /= length requiredTypes) $ throwError $ ArgumentNumberMismatch pos (length requiredTypes) (length args')
            ssas<-zipWithM check_type requiredTypes args'
            return_value<-nextSSA
            emitCall pos p return_value ssas
            return $ TypedSSA return_value (UnitType (p^.returnType) pos)
        Nothing->throwError $ UndefinedSymbol name

declareSymbol :: (CodeSunk m)=>VarDef Pos->CodegenM m SSA
declareSymbol (VarDef t name pos) = addSymbol name t
defineSymbol :: (CodeSunk m)=>Bool->VarDef Pos->CodegenM m SSA
defineSymbol isGlobal v@(VarDef t name loc) = do
    ssa<-addSymbol name t
    let emit = if isGlobal then emitGlobalDef else emitLocalDef
    emit loc ssa v
    return ssa


evalStatement :: (CodeSunk m)=>Statement Pos-> CodegenM m ()
evalStatement st@(QbitInitStmt arg loc) = do
    arg'<-evalLeftExpr arg
    arg''<-requireQbit arg'
    emitReset loc arg''

evalStatement st@(QbitGateStmt decor operands gatename loc) = do
    gatedefs<-use definedGates
    let query_gate = gatedefs Map.!? (gatename^.identName)
    case query_gate of
        Nothing->throwError $ UndefinedSymbol gatename
        Just g -> do
            arg'<-mapM evalLeftExpr operands
            arg''<-mapM requireQbit arg'
            let required_arg_count = (gateSize g) + (length $ decor^.controlStates)
            when (length arg'' /= required_arg_count) $ throwError $ ArgumentNumberMismatch loc required_arg_count (length arg'')
            emitGateApply loc decor g arg''

evalStatement st@(CintAssignStmt lhs rhs loc) = do
    lhs'<-evalLeftExpr lhs
    rhs'<-evalExpr rhs
    lhs_ssa<-requireInt lhs'
    rhs_ssa<-requireInt rhs'
    emitWriteIntOp loc lhs_ssa rhs_ssa
evalStatement st@(IfStatement cond b_then b_else loc) = do
    cond'<-evalExpr cond
    cond''<-requireInt cond'
    incrBlockIndent
    scope
    emitPushBlock
    b_then'<-mapM evalStatement b_then
    unscope
    scope
    emitPushBlock
    b_else'<-mapM evalStatement b_else
    unscope
    decrBlockIndent
    emitIf loc cond''

evalStatement st@(WhileStatement cond b loc) = do
    incrBlockIndent
    emitPushBlock
    scope
    cond'<-evalExpr cond
    cond''<-requireInt cond'
    unscope
    emitPushBlock
    scope
    b'<-mapM evalStatement b
    unscope
    decrBlockIndent
    emitWhile loc cond''



evalStatement st@(ForStatement vname vlo vhi b loc) = do
    vlo'<-evalExpr vlo
    vlo''<-requireInt vlo'
    vhi'<-evalExpr vhi
    vhi''<-requireInt vhi'
    incrBlockIndent
    scope
    v_ssa<-declareSymbol (VarDef (UnitType Int loc) vname loc)
    emitPushBlock
    b'<-mapM evalStatement b
    unscope
    decrBlockIndent
    emitFor loc v_ssa vlo'' vhi''

evalStatement st@(PrintStatement expr loc) = do
    expr'<-evalExpr expr
    expr''<-requireInt expr'
    emitPrint loc expr''

evalStatement st@(ReturnStatement expr loc) = do
    e<-case expr of
        Nothing -> return Nothing
        Just e' -> do
            e''<-evalExpr e'
            e'''<-requireInt e''
            return $ Just e'''
    emitReturn loc e

evalStatement st@(PassStatement loc) = do
    emitPass loc

evalStatement st@(CallStatement c ann) = do
    evalExpr (CallExpr c ann)
    return ()
evalStatement st@(VarDefStatement defs loc) = do
    s<-use symbolTables
    mapM_ (defineSymbol (length s==1)) defs

evalProcDef :: (CodeSunk m)=>ProcDef Pos->CodegenM m ()
evalProcDef x = do
    let i = x^.procName^.identName
    procs<-use definedProcs
    case procs Map.!? i of
        Just y->throwError $ RedefinedProc (x^.procName) y
        Nothing->return ()
    scope
    let params = x^.parameters
    param_ssas<-mapM declareSymbol params
    -- define arguments
    incrBlockIndent
    emitPushBlock
    mapM_ evalStatement (x^.body)
    unscope
    decrBlockIndent
    emitProcedure (x^.annotation) x param_ssas
    definedProcs %= Map.insert i x

evalGateDef :: (CodeSunk m)=>GateDef Pos->CodegenM m ()
evalGateDef x = do
    let i = x^.gateName^.identName
    gatedefs<-use definedGates
    case gatedefs Map.!? i of
        Just y->throwError $ RedefinedGate (x^.gateName) y
        Nothing->return ()
    let sz = gateSize x
    let mat_size = 2^sz
    when (mat_size /= length (x^.matrix^.matrixData)) $ throwError $ InvalidGateMatrix x
    when (not $ allSame ((mat_size):(map length (x^.matrix^.matrixData)))) $ throwError $ InvalidGateMatrix x
    let dat = x^.matrix^.matrixData
    mat<-mapM (mapM evalComplexNumber ) dat
    emitGateDef x mat
    definedGates %= Map.insert i x

evalVarDef :: (CodeSunk m)=>VarDef Pos->CodegenM m ()
evalVarDef x = do
    ssa<-defineSymbol True x
    --emitGlobalDef (x^.annotation) ssa x
    return ()

evalProgram :: (CodeSunk m) => Program Pos -> ExceptT GrammarError (StateT Codegen m) ()
evalProgram prog = do
    emitProgramHeader (prog^.annotation)
    mapM_ evalGateDef (prog^.topGatedefs)
    mapM_ evalVarDef (prog^.topVardefs)
    mapM_ evalProcDef (prog^.procedures)


emptyCodegen :: Codegen
emptyCodegen = Codegen {
    _symbolTables = [SymbolTable Map.empty],
    _definedProcs = Map.empty,
    _definedGates = Map.empty,
    _ssaCounter = 0
}

runCodegen :: (CodeSunk m)=>Program Pos-> m (Either GrammarError ())
runCodegen program = do
    let e = runExceptT $ evalProgram program
    let s = runStateT e emptyCodegen
    fst <$> s


--evalExpr e@(_value->MeasureExpr name) = do
--evalExpr e@(_value->CallExpr name args)
