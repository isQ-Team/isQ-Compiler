module ISQ.Lang.TypeCheck where
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.ISQv2Tokenizer
import qualified Data.MultiMap as MultiMap
import Data.List.Extra (firstJust, zip4)
import Data.Maybe
import Control.Monad (join)
import Control.Monad.Except
import Control.Monad.State.Lazy
import Debug.Trace

type EType = Type ()
type TCAST = AST TypeCheckData

data TypeCheckData = TypeCheckData{
    sourcePos :: Pos,
    termType :: EType,
    termId :: Int
} deriving (Eq, Show)

type TypeCheckInfo = TypeCheckData

data Symbol = SymVar String | SymTempVar Int | SymTempArg Int deriving (Show, Eq, Ord)

getSymbolName :: Symbol -> String
getSymbolName sym = case sym of {SymVar str -> str; _ -> ""}

data TypeCheckError =
      RedefinedSymbol { pos :: Pos, symbolName :: Symbol, firstDefinedAt :: Pos}
    | UndefinedSymbol { pos :: Pos, symbolName :: Symbol}
    | AmbiguousSymbol { pos :: Pos, symbolName :: Symbol, firstDefinedAt :: Pos, secondDefinedAt :: Pos}
    | TypeMismatch {pos :: Pos, expectedType :: [MatchRule], actualType :: Type ()}
    | UnsupportedType { pos :: Pos, actualType :: Type () }
    | UnsupportedLeftSide { pos :: Pos }
    | ViolateNonCloningTheorem { pos :: Pos }
    | GateNameError { pos :: Pos }
    | ArgNumberMismatch { pos :: Pos, expectedArgs :: Int, actualArgs :: Int }
    | BadProcedureArgType { pos :: Pos, arg :: (Type (), Ident)}
    | BadProcedureReturnType { pos :: Pos, ret :: (Type (), Ident)}
    | ICETypeCheckError
    | MainUndefined
    | BadMainSignature { actualMainSignature :: Type () }
    deriving (Eq, Show)
type SymbolTableLayer = MultiMap.MultiMap Symbol DefinedSymbol
type SymbolTable = [SymbolTableLayer]

querySymbol :: Symbol -> SymbolTable -> [DefinedSymbol]
querySymbol sym [] = []
querySymbol sym (x:xs) = do
    let res = MultiMap.lookup sym x
    case res of
        [] -> querySymbol sym xs
        lis -> lis

insertSymbol :: Symbol->DefinedSymbol->SymbolTable->Either TypeCheckError SymbolTable
insertSymbol sym ast [] = insertSymbol sym ast [MultiMap.empty]
insertSymbol sym ast (x:xs) = case MultiMap.lookup sym x of
    [] -> Right $ MultiMap.insert sym ast x : xs
    (y:ys) -> Left $ RedefinedSymbol (definedPos ast) sym (definedPos y)


data TypeCheckEnv = TypeCheckEnv {
    symbolTable :: SymbolTable,
    ssaAllocator :: Int,
    mainDefined :: Bool,
    inOracle :: Bool
}

type TypeCheck = ExceptT TypeCheckError (State TypeCheckEnv)

data DefinedSymbol = DefinedSymbol{
    definedPos :: Pos,
    definedType :: EType,
    definedSSA :: Int,
    isGlobal :: Bool,
    isDerive :: Bool,
    qualifiedName :: String
} deriving (Show)

addSym :: Symbol->DefinedSymbol->TypeCheck ()
addSym k v = do
    symtable<-gets symbolTable
    new_table <-liftEither $ insertSymbol k v symtable
    modify' (\x->x{symbolTable=new_table})

getSym :: Pos->Symbol->TypeCheck DefinedSymbol
getSym pos k = do
    symtable<-gets symbolTable
    case querySymbol k symtable of
        [] -> throwError $ UndefinedSymbol pos k
        [x] -> return x
        (x:y:rest) -> throwError $ AmbiguousSymbol pos k (definedPos x) (definedPos y)

defineSym :: Symbol->Pos->EType->TypeCheck Int
defineSym a b c= do
    ssa<-nextId
    addSym a (DefinedSymbol b c ssa False False "")
    return ssa

defineGlobalSym :: String -> String -> Pos -> EType -> Bool -> Bool -> TypeCheck Int
defineGlobalSym prefix name b c logic d = do
    ssa<-nextId
    when (name == "main" && c /= Type () FuncTy [Type () Unit []] && c /= Type () FuncTy [Type () Unit [], Type () (Array 0) [intType ()], Type () (Array 0) [doubleType ()]]) $ do
        throwError $ BadMainSignature c
    when (name == "main") $ do
        modify' (\x->x{mainDefined = True})
    let qualifiedName = prefix ++ name
    let qualifiedName' = if logic then qualifiedName ++ logicSuffix else qualifiedName
    addSym (SymVar name) (DefinedSymbol b c ssa True d qualifiedName')
    addSym (SymVar qualifiedName) (DefinedSymbol b c ssa True d qualifiedName')
    return ssa

setSym :: Symbol -> Pos -> TypeCheckData -> TypeCheck Int
setSym sym pos (TypeCheckData _ ty rid) = do
    sym_tables <- gets symbolTable
    let cur_table = head sym_tables
    let deleted = MultiMap.delete sym cur_table
    let new_data = DefinedSymbol pos ty rid False False ""
    let new_curr = MultiMap.insert sym new_data deleted
    modify' (\x -> x{symbolTable=new_curr : tail sym_tables})
    return rid

scope :: TypeCheck ()
scope = modify (\x->x{symbolTable = MultiMap.empty:symbolTable x})

unscope :: TypeCheck SymbolTableLayer
unscope = do {x<-gets (head.symbolTable); modify' (\x->x{symbolTable = tail $ symbolTable x}); return x}

astType :: (Annotated p)=>p TypeCheckData->EType
astType = termType . annotation

nextId :: TypeCheck Int
nextId = do
    id<-gets ssaAllocator
    modify' (\x->x{ssaAllocator = id+1})
    return id

typeToInt :: Type () -> Int
typeToInt (Type () Bool []) = 3
typeToInt (Type () Int []) = 2
typeToInt (Type () Double []) = 1
typeToInt (Type () Complex []) = 0
typeToInt (Type () Ref [sub_type]) = typeToInt sub_type
typeToInt _ = -1

intToType :: Int -> Type ()
intToType 3 = boolType ()
intToType 2 = intType ()
intToType 1 = doubleType ()
intToType 0 = complexType ()
intToType _ = error "Unreachable."

type TCExpr = Expr TypeCheckData

data MatchRule = Exact EType | AnyUnknownList | AnyKnownList Int | AnyList | AnyFunc | AnyGate | AnyRef
    | ArrayType MatchRule
    deriving (Show, Eq)

checkRule :: MatchRule->EType->Bool
checkRule (Exact x) y = x==y
checkRule AnyUnknownList (Type () (Array 0) [_]) = True
checkRule (AnyKnownList x) (Type () (Array y) [_]) = x==y
checkRule AnyList (Type () (Array _) [_]) = True
checkRule AnyFunc (Type () FuncTy _) = True
checkRule AnyGate (Type () (Gate _) _) = True
checkRule AnyGate (Type () (Logic _) _) = True
checkRule AnyRef (Type () Ref [_]) = True
checkRule (ArrayType subRule) (Type () (Array _) [subType]) = checkRule subRule subType
checkRule _ _ = False

-- try to match two types, using auto dereference and int-to-bool implicit conversion.
matchType' :: [MatchRule]->TCExpr->TypeCheck (Maybe TCExpr)
matchType' wanted e = do
    let current_type = astType e
    let pos = sourcePos $ annotation e
    if any (`checkRule` current_type) wanted then return $ Just e
    else
        case current_type of
            -- Auto dereference rule
            Type () Ref [x] -> do
                id<-nextId
                matchType' wanted (EDeref (TypeCheckData pos x id) e)
            -- Bool-to-int implicit cast
            Type () Bool [] -> if Exact (Type () Int []) `notElem` wanted then return Nothing else do
                id<-nextId
                matchType' wanted (EImplicitCast (TypeCheckData pos (Type () Int [] ) id) e)
            -- int-to-double implicit cast
            Type () Int [] -> if Exact (Type () Double []) `notElem` wanted then return Nothing else do
                id<-nextId
                matchType' wanted (EImplicitCast (TypeCheckData pos (Type () Double [] ) id) e)
            -- float-to-complex implicit cast
            Type () Double [] -> if Exact (Type () Complex []) `notElem` wanted then return Nothing else do
                id<-nextId
                matchType' wanted (EImplicitCast (TypeCheckData pos (Type () Complex [] ) id) e)
            -- Auto cast. Only the first rule is considered
            Type () (Array 0) [y] -> case head wanted of
                    Exact (Type () (Array x) [y]) -> do
                        id <- nextId
                        matchType' wanted (EListCast (TypeCheckData pos (Type () (Array x) [y]) id) e)
                    _ -> return Nothing
            -- Auto list erasure
            Type () (Array x) [y] -> do
                id<-nextId
                matchType' wanted (EListCast (TypeCheckData pos (Type () (Array 0) [y]) id) e)
            _ -> return Nothing
matchType :: [MatchRule]->TCExpr->TypeCheck TCExpr
matchType wanted e = do
    new_e<-matchType' wanted e
    case new_e of
        Just x->return x
        Nothing -> throwError $ TypeMismatch (sourcePos $ annotation e) wanted (astType e)

exactBinaryCheck :: (Expr Pos->TypeCheck (Expr TypeCheckData)) -> EType -> Pos -> BinaryOperator -> LExpr -> LExpr -> TypeCheck (Expr TypeCheckData)
exactBinaryCheck f etype pos op lhs rhs = do
    ref_lhs <- f lhs
    ref_rhs <- f rhs
    lhs' <- matchType (map Exact [etype]) ref_lhs
    rhs' <- matchType (map Exact [etype]) ref_rhs
    ssa <- nextId
    return $ EBinary (TypeCheckData pos etype ssa) op lhs' rhs'

buildBinaryExpr :: Pos -> BinaryOperator -> Expr TypeCheckData -> Expr TypeCheckData -> TypeCheck (Expr TypeCheckData)
buildBinaryExpr pos op ref_lhs ref_rhs = do
    ssa <- nextId
    logic <- gets inOracle
    case logic of
        True -> do
            let ty = astType ref_lhs
            case ty of
                Type () Bool [] -> do
                    rhs <- matchType [Exact ty] ref_rhs
                    return $ EBinary (TypeCheckData pos ty ssa) op ref_lhs rhs
                Type () (Array _) [Type () Bool []] -> do
                    rhs <- matchType [Exact ty] ref_rhs
                    return $ EBinary (TypeCheckData pos ty ssa) op ref_lhs rhs
                other -> throwError $ UnsupportedType pos other
        False -> do
            lhs' <- matchType (map Exact [intType (), doubleType (), complexType ()]) ref_lhs
            rhs' <- matchType (map Exact [intType (), doubleType (), complexType ()]) ref_rhs
            let lty = astType lhs'
            let rty = astType rhs'
            --traceM $ show rty
            matched_lhs <- case ty rty of
                    Double -> matchType [Exact (doubleType ())] lhs'
                    _ -> matchType (map Exact [intType (), doubleType (), complexType ()]) lhs'
            matched_rhs <- case ty lty of
                    Double -> matchType [Exact (doubleType ())] rhs'
                    _ -> matchType (map Exact [intType (), doubleType (), complexType ()]) rhs'
            --traceM $ show matched_lhs
            let return_type = case op of
                    Cmp _ -> boolType ()
                    _ -> astType matched_lhs
            case op of
                Mod -> if (return_type /= intType ()) then throwError $ TypeMismatch pos [Exact (intType ())] return_type
                    else return $ EBinary (TypeCheckData pos return_type ssa) op matched_lhs matched_rhs
                _ -> return $ EBinary (TypeCheckData pos return_type ssa) op matched_lhs matched_rhs

-- By now we only support bottom-up type checking.
-- All leaf nodes have their own type, and intermediate types are calculated.
typeCheckExpr' :: (Expr Pos->TypeCheck (Expr TypeCheckData))->Expr Pos->TypeCheck (Expr TypeCheckData)
typeCheckExpr' f (EIdent pos ident) = do
    sym<-getSym pos (SymVar ident)
    ssa<-nextId
    in_oracle <- gets inOracle
    case isGlobal sym of
        True ->return $ EGlobalName (TypeCheckData pos (definedType sym) ssa) (qualifiedName sym)
        False -> do
            let sym_ssa = definedSSA sym
            case in_oracle of
                True -> return $ EResolvedIdent (TypeCheckData pos (definedType sym) sym_ssa) sym_ssa
                False -> return $ EResolvedIdent (TypeCheckData pos (definedType sym) ssa) sym_ssa

typeCheckExpr' f (EBinary pos And lhs rhs) = exactBinaryCheck f (boolType ()) pos And lhs rhs
typeCheckExpr' f (EBinary pos Or lhs rhs) = exactBinaryCheck f (boolType ()) pos Or lhs rhs
typeCheckExpr' f (EBinary pos Shl lhs rhs) = exactBinaryCheck f (intType ()) pos Shl lhs rhs
typeCheckExpr' f (EBinary pos Shr lhs rhs) = exactBinaryCheck f (intType ()) pos Shr lhs rhs
typeCheckExpr' f (EBinary pos Pow lhs rhs) = do
    ref_lhs<-f lhs
    ref_rhs<-f rhs
    lhs' <- matchType (map Exact [doubleType ()]) ref_lhs
    rhs' <- matchType (map Exact [doubleType ()]) ref_rhs
    ssa<-nextId
    let lty = astType lhs'
    let rty = astType rhs'
    let return_type = lty
    return $ EBinary (TypeCheckData pos return_type ssa) Pow lhs' rhs'
typeCheckExpr' f (EBinary pos op lhs rhs) = do
    ref_lhs<-f lhs
    ref_rhs<-f rhs
    buildBinaryExpr pos op ref_lhs ref_rhs
typeCheckExpr' f (EUnary pos op lhs) = do
    lhs'<-f lhs
    matched_lhs <- case op of
        Not -> matchType [Exact $ boolType ()] lhs'
        Noti -> matchType [ArrayType $ Exact $ boolType ()] lhs'
        _ -> matchType (map Exact [intType (), doubleType (), complexType ()]) lhs'
    ssa<-nextId
    let return_type = astType matched_lhs
    return $ EUnary (TypeCheckData pos return_type ssa) op matched_lhs
typeCheckExpr' f (ESubscript pos base (ERange epos lo hi step)) = do
    base' <- f base
    base'' <- matchType [AnyList] base'
    let lo' = case lo of
            Just exp -> exp
            Nothing -> EIntLit epos 0
    let step' = case step of
            Just exp -> exp
            Nothing -> EIntLit epos 1
    let hi' = case hi of
            Just exp -> exp
            Nothing -> EArrayLen epos base
    -- size = ceil[(hi - lo) / step]
    let size = EBinary epos CeilDiv (EBinary epos Sub hi' lo') step'
    range <- f $ ERange epos (Just lo') (Just size) (Just step')
    ssa <- nextId
    in_oracle <- gets inOracle
    let sub_ty = head $ subTypes $ astType base''
    let ty = Type () (Array 0) [sub_ty]
    return $ ESubscript (TypeCheckData pos ty ssa) base'' range
typeCheckExpr' f (ESubscript pos base offset) = do
    base' <- f base
    base'' <- matchType [AnyList] base'
    offset' <- f offset
    offset'' <- matchType [Exact $ intType ()] offset'
    ssa <- nextId
    in_oracle <- gets inOracle
    let sub_ty = head $ subTypes $ astType base''
    let ty = case in_oracle of
            True -> sub_ty
            False -> refType () sub_ty
    return $ ESubscript (TypeCheckData pos ty ssa) base'' offset''
typeCheckExpr' f (ECall pos callee callArgs) = do
    callee'<-f callee
    callee''<-matchType [AnyFunc] callee'
    let callee_ty = astType callee''
    let (ret:args) = subTypes callee_ty
    callArgs'<-mapM f callArgs
    when (length args /= length callArgs') $ throwError $ ArgNumberMismatch pos (length args) (length callArgs')
    callArgs''<-zipWithM (\a->matchType [Exact a]) args callArgs'
    ssa<-nextId
    return $ ECall (TypeCheckData pos ret ssa) callee'' callArgs''
typeCheckExpr' f (EIntLit pos x) = do
    ssa<-nextId
    return $ EIntLit (TypeCheckData pos (intType ()) ssa) x
typeCheckExpr' f (EFloatingLit pos x) = do
    ssa<-nextId
    return $ EFloatingLit (TypeCheckData pos (doubleType ()) ssa) x
typeCheckExpr' f (EImagLit pos x) = do
    ssa<-nextId
    return $ EImagLit (TypeCheckData pos (complexType ()) ssa) x
typeCheckExpr' f (EBoolLit pos x) = do
    ssa<-nextId
    return $ EBoolLit (TypeCheckData pos (boolType ()) ssa) x
typeCheckExpr' f (ERange pos lo hi Nothing) = do
    let step = Just (EIntLit pos 1)
    f (ERange pos lo hi step)
typeCheckExpr' f (ERange pos lo hi step) = do
    let resolve (Just x) = do {x'<-f x; x''<-matchType [Exact (Type () Int [])] x'; return $ Just x''}
        resolve Nothing = return Nothing
    lo'<-resolve lo
    hi'<-resolve hi
    step'<-resolve step
    ssa<-nextId
    return $ ERange (TypeCheckData pos (Type () IntRange []) ssa) lo' hi' step'
typeCheckExpr' f (ECoreMeasure pos qubit) = do
    qubit'<-f qubit
    ssa<-nextId
    is_qubit <- matchType' [Exact (refType () (qbitType ()))] qubit'
    case is_qubit of
        Just qubit'' -> return $ ECoreMeasure (TypeCheckData pos (boolType ()) ssa) qubit''
        Nothing -> do
            qubit'' <- matchType [Exact $ Type () (Array 0) [qbitType ()]] qubit'
            fun <- f (EIdent pos ".__measure_bundle")
            return $ ECall (TypeCheckData pos (intType ()) ssa) fun [qubit'']
typeCheckExpr' f (EList pos lis) = do
    lis' <- mapM f lis
    let levels = map (typeToInt . termType . annotationExpr) lis'
    let (min_level, min_idx) = minimum $ zip levels [0..]
    when (min_level < 0) $ throwError $ do
        let ann = annotationExpr $ lis' !! min_idx
        UnsupportedType (sourcePos ann) (termType ann)
    let ele_type = intToType min_level
    lis'' <- mapM (matchType [Exact ele_type]) lis'
    let ty = Type () (Array $ length lis) [ele_type]
    ssa <- nextId
    return $ EList (TypeCheckData pos ty ssa) lis''
typeCheckExpr' f x@EDeref{} = error "Unreachable."
typeCheckExpr' f x@EImplicitCast{} = error "Unreachable."
typeCheckExpr' f (ETempVar pos ident) = do
    sym<-getSym pos (SymTempVar ident)
    ssa<-nextId
    return $ EResolvedIdent (TypeCheckData pos (definedType sym) ssa) (definedSSA sym)
typeCheckExpr' f (ETempArg pos ident) = do
    sym<-getSym pos (SymTempArg ident)
    ssa<-nextId
    return $ EResolvedIdent (TypeCheckData pos (definedType sym) ssa) (definedSSA sym)
typeCheckExpr' f (EUnitLit pos) = EUnitLit . TypeCheckData pos (unitType ()) <$> nextId
typeCheckExpr' f x@EResolvedIdent{} = error "Unreachable."
typeCheckExpr' f x@EGlobalName{} = error "Unreachable."
typeCheckExpr' f x@EListCast{} = error "Unreachable."
typeCheckExpr' f (EArrayLen pos array) = do
    array' <- f array
    ssa <- nextId
    let ty = termType $ annotationExpr array'
    case ty of
        Type () (Array 0) [_] -> return $ EArrayLen (TypeCheckData pos (intType()) ssa) array'
        Type () (Array x) [_] -> return $ EIntLit (TypeCheckData pos (intType()) ssa) x
        _ -> throwError $ TypeMismatch pos [AnyList] ty
typeCheckExpr :: Expr Pos -> TypeCheck (Expr TypeCheckData)
typeCheckExpr = fix typeCheckExpr'

okStmt :: Pos->TypeCheckData
okStmt pos = (TypeCheckData pos (unitType ()) (-1))


-- Transforms a defvar-type into a ref type.
-- Lists are passed by value and thus are right-values.
definedRefType :: EType->EType
definedRefType x@(Type () (Array _) _) = x
definedRefType x = Type () Ref [x]

getBaseFromArray :: TCExpr -> TCExpr
getBaseFromArray x@(EResolvedIdent _ _) = x
getBaseFromArray x@(EGlobalName _ _) = x
getBaseFromArray (ESubscript ann base _) = base
getBaseFromArray _ = error "Unreachable."

generateRange :: TCExpr -> TypeCheckData -> TypeCheck TCExpr
generateRange symbol ann = do
    let pos = sourcePos ann
    init_id <- nextId
    let init = EIntLit (TypeCheckData pos (intType ()) init_id) 0
    hi_id <- nextId
    let hi = EArrayLen (TypeCheckData pos (intType ()) hi_id) symbol
    step_id <- nextId
    let step = EIntLit (TypeCheckData pos (intType ()) step_id) 1
    return $ ERange ann (Just init) (Just hi) (Just step)

getRangeFromArray :: TCExpr -> TypeCheck TCExpr
getRangeFromArray array@(EResolvedIdent ann _) = generateRange array ann
getRangeFromArray array@(EGlobalName ann _) = generateRange array ann
getRangeFromArray (ESubscript ann _ subscript) = return $ subscript
getRangeFromArray x = error "Unreachable."

generateIteratorDef :: TCExpr -> TypeCheck [Maybe TCAST]
generateIteratorDef (ERange ann lo hi step) = do
    int0_id <- nextId
    let c0 = EIntLit (TypeCheckData (sourcePos ann) (intType ()) int0_id) 0
    it_id <- nextId
    let it_def = Just $ NResolvedDefvar ann [(refIntType (), it_id, Just c0)]
    lo_id <- nextId
    let lo_def = Just $ NResolvedDefvar ann [(refIntType (), lo_id, lo)]
    hi_def <- case hi of
            Nothing -> return $ Nothing
            Just _ -> do
                hi_id <- nextId
                return $ Just $ NResolvedDefvar ann [(refIntType (), hi_id, hi)]
    estep <- case step of
            Nothing -> error "Unreachable."
            Just x -> return $ Just $ x
    step_id <- nextId
    let step_def = Just $ NResolvedDefvar ann [(refIntType (), step_id, estep)]
    return [it_def, lo_def, hi_def, step_def]

getVariableRef :: Pos -> Type () -> Int -> TypeCheck TCExpr
getVariableRef pos var_type var_id = do
    var_ref_id <- nextId
    let var_ref = TypeCheckData pos (refType () var_type) var_ref_id
    return $ EResolvedIdent var_ref var_id

getVariableDeref :: Pos -> Type () -> Int -> TypeCheck TCExpr
getVariableDeref pos var_type var_id = do
    var_ref <- getVariableRef pos var_type var_id
    var_deref_id <- nextId
    let var_deref = TypeCheckData pos var_type var_deref_id
    return $ EDeref var_deref var_ref

increaseIterator :: Pos -> Int -> TypeCheck TCAST
increaseIterator pos it_id = do
    it <- getVariableDeref pos (intType ()) it_id
    add_id <- nextId
    let add_ann = TypeCheckData pos (intType ()) add_id
    int1_id <- nextId
    let step = EIntLit (TypeCheckData pos (intType ()) int1_id) 1
    let add = EBinary add_ann Add it step
    left <- getVariableRef pos (intType ()) it_id
    return $ NAssign (okStmt pos) left add AssignEq

andRangeCondition :: Pos -> Int -> (Int, Int) -> TypeCheck TCAST
andRangeCondition pos in_id (it_id, hi_id) = do
    cond_id <- nextId
    let cond_ann = TypeCheckData pos (boolType ()) cond_id
    cond <- case hi_id of
            -1 -> return $ EBoolLit cond_ann True
            x -> do
                ident <- getVariableDeref pos (intType ()) it_id
                hi <- getVariableDeref pos (intType ()) hi_id
                return $ EBinary cond_ann (Cmp Less) ident hi
    cond_in <- getVariableDeref pos (boolType ()) in_id
    and_id <- nextId
    let and_ann = TypeCheckData pos (boolType ()) and_id
    let and = EBinary and_ann And cond_in cond
    left <- getVariableRef pos (boolType ()) in_id
    return $ NAssign (okStmt pos) left and AssignEq

getItemFromArray :: Pos -> (TCExpr, Int, Int, Int) -> TypeCheck TCExpr
getItemFromArray pos (base, it_id, lo_id, step_id) = do
    it <- getVariableDeref pos (intType ()) it_id
    step <- getVariableDeref pos (intType ()) step_id
    mul_ssa <- nextId
    let mul = EBinary (TypeCheckData pos (intType ()) mul_ssa) Mul it step
    lo <- getVariableDeref pos (intType ()) lo_id
    add_ssa <- nextId
    let add = EBinary (TypeCheckData pos (intType ()) add_ssa) Add lo mul
    let subtype = head $ subTypes $ astType base
    ssa <- nextId
    base' <- case base of
            EGlobalName (TypeCheckData pos ty _) name -> do
                new_id <- nextId
                return $ EGlobalName (TypeCheckData pos ty new_id) name
            other -> return other
    return $ ESubscript (TypeCheckData pos subtype ssa) base' add

typeCheckAST' :: (AST Pos->TypeCheck (AST TypeCheckData))->AST Pos->TypeCheck (AST TypeCheckData)
typeCheckAST' f (NBlock pos lis) = do
    scope
    lis' <- mapM f lis
    unscope
    return $ NBlock (okStmt pos) lis'
typeCheckAST' f (NIf pos cond bthen belse) = do
    cond'<-typeCheckExpr cond
    cond''<-matchType [Exact (boolType ())] cond'
    scope
    bthen'<-mapM f bthen
    unscope
    scope
    belse'<-mapM f belse
    unscope
    return $ NIf (okStmt pos) cond'' bthen' belse'
typeCheckAST' f (NFor pos v r b) = do
    scope
    r'<-typeCheckExpr r
    v'<-defineSym (SymVar v) pos (intType ())
    r''<-matchType [Exact (Type () IntRange [])] r'
    b'<-mapM f b
    unscope
    return $  NResolvedFor (okStmt pos) v' r'' b'
typeCheckAST' f (NEmpty pos) = return $ NEmpty (okStmt pos)
typeCheckAST' f (NPass pos) = return $ NPass (okStmt pos)
typeCheckAST' f (NAssert pos exp Nothing) = do
    exp' <- typeCheckExpr exp
    exp'' <- matchType [Exact (boolType ())] exp'
    return $ NAssert (okStmt pos) exp'' Nothing
typeCheckAST' f (NAssert pos exp _) = error "unreachable"
typeCheckAST' f (NResolvedAssert pos q space) = do
    q' <- typeCheckExpr q
    q'' <- matchType [ArrayType $ Exact (qbitType ())] q'
    return $ NResolvedAssert (okStmt pos) q'' space
typeCheckAST' f (NBp pos) = do
    temp_ssa<-nextId
    let annotation = TypeCheckData pos (unitType ()) temp_ssa
    return $ NBp annotation
typeCheckAST' f (NWhile pos cond body) = error "unreachable"
typeCheckAST' f (NCall pos c@(ECall _ callee args)) = do
    callee'<-typeCheckExpr callee
    callee''<-matchType [AnyFunc, AnyGate] callee'
    let callee_ty = astType callee''
    case ty $ callee_ty of
        FuncTy -> do
            c'<-typeCheckExpr c
            return $ NCall (okStmt pos) c'
        Gate _ -> f (NCoreUnitary pos callee args [])
        Logic _ -> f (NCoreUnitary pos callee args [])
        _ -> undefined
typeCheckAST' f (NCallWithInv pos c@(ECall _ callee args) mods) = do
    callee'<-typeCheckExpr callee
    callee''<-matchType [AnyFunc, AnyGate] callee'
    let callee_ty = astType callee''
    case ty $ callee_ty of
        FuncTy -> throwError $ UnsupportedType pos callee_ty
        Gate _ -> f (NCoreUnitary pos callee args mods)
        Logic _ -> throwError $ UnsupportedType pos callee_ty
        _ -> undefined
typeCheckAST' f (NCall pos c) = error "unreachable"
typeCheckAST' f (NDefvar pos defs) = do
    in_oracle <- gets inOracle
    let def_one (ty, name, initializer, length) = do
            let left_type = void ty
            case in_oracle of
                True -> do
                    let sym = SymVar name
                    case initializer of
                        Just r -> do
                            r' <- typeCheckExpr r
                            case left_type of
                                Type () (Array 0) [Type () Bool []] -> do
                                    r'' <- matchType [ArrayType $ Exact $ boolType ()] r'
                                    rid <- setSym sym pos $ annotationExpr r''
                                    return (left_type, rid, Just r'')
                                Type () Bool [] -> do
                                    r''<-matchType [Exact left_type] r'
                                    rid <- setSym sym pos $ annotationExpr r''
                                    return (left_type, rid, Just r'')
                                other -> throwError $ UnsupportedType pos other
                        Nothing -> case length of
                            Nothing -> do
                                case left_type of
                                    Type () Bool [] -> do
                                        rid <- defineSym sym pos left_type
                                        return (left_type, rid, Nothing)
                                    _ -> throwError $ UnsupportedType pos left_type
                            Just elen@(EIntLit _ len) -> do
                                case left_type of
                                    Type () (Array 0) [Type () Bool []] -> do
                                        let ty = Type () (Array len) [Type () Bool []]
                                        rid <- defineSym sym pos ty
                                        elen' <- typeCheckExpr elen
                                        return (ty, rid, Just elen')
                                    other -> throwError $ UnsupportedType pos other
                            _ -> throwError $ UnsupportedLeftSide pos
                False -> do
                    (i', ty') <- case initializer of
                        Just r -> do
                            r' <- typeCheckExpr r
                            case left_type of
                                Type () (Array llen) [lsub] -> do
                                    let right_type = termType $ annotationExpr r'
                                    case right_type of
                                        Type () (Array rlen) [rsub] -> do
                                            let li = typeToInt lsub
                                            let ri = typeToInt rsub
                                            let min = minimum [li, ri]
                                            when (min < 0 || li > ri) $ throwError $ TypeMismatch pos [Exact left_type] right_type
                                            let llen' = case llen of
                                                    0 -> rlen
                                                    _ -> llen
                                            return (Just r', Type () (Array llen') [lsub])
                                        _ -> throwError $ TypeMismatch pos [Exact left_type] right_type
                                _ -> do
                                    r''<-matchType [Exact left_type] r'
                                    return (Just r'', definedRefType left_type)
                        Nothing -> case length of
                            Nothing -> return (Nothing, definedRefType left_type)
                            Just len -> do
                                len' <- typeCheckExpr len
                                len'' <- matchType [Exact $ intType ()] len'
                                return (Just len'', left_type)
                    s <- defineSym (SymVar name) pos ty'
                    return (ty', s, i')
    defs'<-mapM def_one defs
    return $ NResolvedDefvar (okStmt pos) defs'
typeCheckAST' f (NAssign pos lhs rhs op) = do
    rhs'<-typeCheckExpr rhs
    in_oracle <- gets inOracle
    lhs' <- typeCheckExpr lhs
    case in_oracle of
        True -> do
            lhs'' <- matchType [Exact $ boolType()] lhs'
            case lhs of
                EIdent lpos ident -> do
                    let sym = SymVar ident
                    sym_data <- getSym lpos sym
                    let lhs_ty = definedType sym_data
                    case lhs_ty of
                        Type () Bool [] -> do
                            rhs'' <- matchType [Exact lhs_ty] rhs'
                            setSym sym lpos $ annotationExpr rhs''
                            return $ NAssign (okStmt pos) lhs'' rhs'' AssignEq
                        other -> throwError $ UnsupportedType pos other
                ESubscript _ _ _ -> do
                    rhs'' <- matchType [Exact $ boolType()] rhs'
                    return $ NAssign (okStmt pos) lhs'' rhs'' AssignEq
                _ -> throwError $ UnsupportedLeftSide $ annotationExpr lhs
        False -> do
            let doAssign lhs' rhs' = do
                    lhs'' <- matchType [AnyRef] lhs'
                    let Type () Ref [lhs_ty] = astType lhs''
                    when (ty lhs_ty==Qbit) $ throwError $ ViolateNonCloningTheorem pos
                    rhs'' <- matchType [Exact lhs_ty] rhs'
                    return $ NAssign (okStmt pos) lhs'' rhs'' AssignEq
            case op of
                AssignEq -> doAssign lhs' rhs'
                AddEq -> do
                    let lhs_ty = termType $ annotationExpr lhs'
                    case lhs_ty of
                        Type () (Array _) [Type () Qbit []] -> do
                            lhs'' <- matchType [Exact $ Type () (Array 0) [qbitType ()]] lhs'
                            rhs'' <- matchType [Exact $ Type () (Array 0) [qbitType ()]] rhs'
                            call_id <- nextId
                            callee <- typeCheckExpr $ EIdent pos "__add"
                            let ecall = ECall (TypeCheckData pos (unitType ()) call_id) callee [rhs'', lhs'']
                            return $ NCall (okStmt pos) ecall
                        _ -> do
                            eadd <- buildBinaryExpr pos Add lhs' rhs'
                            doAssign lhs' eadd
                SubEq -> do
                    let lhs_ty = termType $ annotationExpr lhs'
                    case lhs_ty of
                        Type () (Array _) [Type () Qbit []] -> do
                            lhs'' <- matchType [Exact $ Type () (Array 0) [qbitType ()]] lhs'
                            rhs'' <- matchType [Exact $ Type () (Array 0) [qbitType ()]] rhs'
                            call_id <- nextId
                            callee <- typeCheckExpr $ EIdent pos "__sub"
                            let ecall = ECall (TypeCheckData pos (unitType ()) call_id) callee [rhs'', lhs'']
                            return $ NCall (okStmt pos) ecall
                        _ -> do
                            esub <- buildBinaryExpr pos Sub lhs' rhs'
                            doAssign lhs' esub
typeCheckAST' f (NGatedef pos lhs rhs _) = error "unreachable"
typeCheckAST' f (NReturn pos expr) = do
    expr' <- typeCheckExpr expr
    return $ NReturn (okStmt pos) expr'
typeCheckAST' f (NCoreUnitary pos gate operands modifiers) = do
    let go Inv (l, fv) = (l, not fv)
        go y@(Ctrl x i) (l, fv) =  ([y]++l, fv)
        (cm, fv) = foldr go ([], False) modifiers

    sym <- getSym pos (SymVar (identName gate))
    let (modifiers', new_gate) = case ((isDerive sym), fv) of
            (True, True)->(cm, EIdent (annotationExpr gate) ((identName gate)++"_inv"))
            (_, _)->(modifiers, gate)

    --traceM $ show new_gate
    gate'<-typeCheckExpr new_gate
    gate''<-matchType [AnyGate] gate'
    --traceM $ show gate''
    let (x, extra) = case astType gate'' of
            Type _ (Gate x) extra -> (x, extra)
            Type _ (Logic x) extra -> (x, extra)
    let total_qubits = sum (map addedQubits modifiers') + x
    let total_operands = length extra + total_qubits
    when (total_operands /= length operands) $ throwError $ ArgNumberMismatch pos total_operands (length operands)
    operands'<-mapM typeCheckExpr operands
    --traceM $ show operands'
    --traceM $ show extra
    let canparam = case ((identName gate), modifiers') of
            ("Rx", []) -> True
            ("Ry", []) -> True
            ("Rz", []) -> True
            (_, _) -> False
    let extra' = map (\(x, y)->case astType y of
            Type _ Param _ -> if canparam then Type () Param [] else x
            Type _ Ref [s] -> case s of
                Type _ Param _ -> if canparam then Type () Param [] else x
                _ -> x
            _ -> x) (zip extra operands')
    
    let (op_extra, op_qubits) = splitAt (total_operands - total_qubits) operands'
    op_extra'<-zipWithM (\x y->matchType [Exact x] y) extra' op_extra
    case null op_qubits of
        -- GPhase has no qubit operand
        True -> return $ NCoreUnitary (okStmt pos) gate'' op_extra' modifiers'
        False -> do
            is_qubit <- matchType' [Exact (refType () $ qbitType ())] $ head op_qubits
            case is_qubit of
                Just _ -> do
                    op_qubits' <- mapM (matchType [Exact (refType () $ qbitType ())]) op_qubits
                    let operands'' = op_extra' ++ op_qubits'
                    return $ NCoreUnitary (okStmt pos) gate'' operands'' modifiers'
                Nothing -> do
                    -- Bundle operation
                    op_qubits' <- mapM (matchType [ArrayType $ Exact $ qbitType ()]) op_qubits
                    ranges <- mapM getRangeFromArray op_qubits'
                    var_defs <- mapM generateIteratorDef ranges
                    let get_second (a, b, c) = b
                    let getIdFromDefvar Nothing = -1
                        getIdFromDefvar (Just x) = get_second $ head $ resolvedDefinitions x
                    let it_ids = map (getIdFromDefvar . head) var_defs
                    let lo_ids = map (getIdFromDefvar . head . tail) var_defs
                    let hi_ids = map (getIdFromDefvar . head . tail . tail) var_defs
                    let step_ids = map (getIdFromDefvar . last) var_defs

                    -- bool cond = true;
                    true_id <- nextId
                    let true_ann = TypeCheckData pos (boolType ()) true_id
                    let true_lit = Just $ EBoolLit true_ann True
                    cond_id <- nextId
                    let cond_def = NResolvedDefvar (okStmt pos) [(refBoolType (), cond_id, true_lit)]

                    apply_cond <- mapM (andRangeCondition pos cond_id) $ zip it_ids hi_ids
                    let bases = map getBaseFromArray op_qubits'
                    qubits <- mapM (getItemFromArray pos) $ zip4 bases it_ids lo_ids step_ids
                    let apply_unitary = NCoreUnitary (okStmt pos) gate'' qubits modifiers'
                    inc_its <- mapM (increaseIterator pos) $ it_ids
                    apply_cond2 <- mapM (andRangeCondition pos cond_id) $ zip it_ids hi_ids
                    let while_body = apply_unitary : inc_its ++ apply_cond2
                    ntemp_id <- nextId
                    flag <- getVariableDeref pos (boolType ()) ntemp_id
                    econd <- getVariableDeref pos (boolType ()) cond_id
                    let nwhile = NWhileWithGuard (okStmt pos) econd while_body flag

                    false_id <- nextId
                    let false_ann = TypeCheckData pos (boolType ()) false_id
                    let false_lit = Just $ EBoolLit false_ann False
                    let temp_def = NResolvedDefvar (okStmt pos) [(refBoolType (), ntemp_id, false_lit)]
                    let block_body = (catMaybes $ concat var_defs) ++ cond_def:apply_cond ++ [temp_def, nwhile]
                    let block = NBlock (okStmt pos) block_body 
                    return block
typeCheckAST' f (NCoreReset pos qubit) = do
    qubit'<-typeCheckExpr qubit
    qubit''<-matchType [Exact (refType () $ qbitType ())] qubit'
    temp_ssa<-nextId
    let annotation = TypeCheckData pos (unitType ()) temp_ssa
    return $ NCoreReset annotation qubit''
typeCheckAST' f (NCorePrint pos val) = do
    val'<-typeCheckExpr val
    val''<-matchType [
        Exact (intType ()),
        Exact (doubleType ()),
        Exact (complexType ())
        ] val'
    return $ NCorePrint (okStmt pos) val''
typeCheckAST' f (NCoreMeasure pos qubit) = do
    qubit'<-typeCheckExpr qubit
    return $ NCoreMeasure (okStmt pos) qubit'
typeCheckAST' f (NProcedure _ _ _ _ _) = error "unreachable"
typeCheckAST' f (NContinue _) = error "unreachable"
typeCheckAST' f (NBreak _) = error "unreachable"
typeCheckAST' f (NResolvedFor _ _ _ _) = error "unreachable"
typeCheckAST' f (NResolvedGatedef pos name matrix size _) = error "unreachable"
typeCheckAST' f (NWhileWithGuard pos cond body break) = do
    cond'<-typeCheckExpr cond
    cond''<-matchType [Exact (boolType ())] cond'
    break'<-typeCheckExpr break
    break''<-matchType [Exact (boolType ())] break'
    body'<-mapM f body
    return $ NWhileWithGuard (okStmt pos) cond'' body' break''
typeCheckAST' f (NProcedureWithRet _ _ _ _ _ _) = error "not top"
typeCheckAST' f (NResolvedProcedureWithRet _ _ _ _ _ _ _) = error "unreachable"
typeCheckAST' f (NJumpToEndOnFlag pos flag) = do
    flag'<-typeCheckExpr flag
    flag''<-matchType [Exact (boolType ())] flag'
    return $ NJumpToEndOnFlag (okStmt pos) flag''
typeCheckAST' f (NJumpToEnd pos) = return $ NJumpToEnd (okStmt pos)
typeCheckAST' f (NTempvar pos def) = do
    let def_one (ty, id, initializer) = do
            i'<-case initializer of
                Just r->do
                        r'<-typeCheckExpr r
                        r''<-matchType [Exact (void ty)] r'
                        return $ Just r''
                Nothing -> return Nothing
            s<-defineSym (SymTempVar id) pos $ definedRefType $ void ty
            return (definedRefType $ void ty, s, i')
    def'<-def_one def
    return $ NResolvedDefvar (okStmt pos) [def']
typeCheckAST' f x@NResolvedExternGate{} = return $ fmap okStmt x
typeCheckAST' f x@NDerivedGatedef{} = return $ fmap okStmt x
typeCheckAST' f x@NDerivedOracle{} = return $ fmap okStmt x
typeCheckAST' f NExternGate{} = error "unreachable"
typeCheckAST' f NProcedureWithDerive{} = error "unreachable"
typeCheckAST' f NResolvedDefvar{} = error "unreachable"
typeCheckAST' f NGlobalDefvar {} = error "unreachable"
typeCheckAST' f NOracle{} = error "unreachable"
typeCheckAST' f NOracleTable{} = error "unreachable"

typeCheckAST :: AST Pos -> TypeCheck (AST TypeCheckData)
typeCheckAST = fix typeCheckAST'


argType :: Type Pos->Ident->TypeCheck EType
argType ty = argType' (annotation ty) ty 
argType' :: Pos->Type ann->Ident->TypeCheck EType
argType' pos ty i = case ty of
    Type _ Int [] -> return $ void ty
    Type _ Double [] -> return $ void ty
    Type _ Bool [] -> return $ void ty
    Type _ Qbit [] -> return $ Type () Ref [void ty]
    Type _ (Array _) [a] -> return $ void ty
    _ -> throwError $ BadProcedureArgType pos (void ty, i)

typeCheckToplevel :: Bool -> String -> [AST Pos]-> Bool -> TypeCheck ([TCAST], SymbolTableLayer, Int)
typeCheckToplevel isMain prefix ast qcis = do
    
    (resolved_defvar, varlist)<-flip runStateT [] $ do
        mapM (\node->case node of
                NDefvar pos def -> do
                    -- Create separate scope to prevent cross-reference.
                    lift scope
                    p<-lift $ typeCheckAST node
                    let (NResolvedDefvar a defs') = p
                    s<-lift unscope
                    modify' (MultiMap.map (\x->x{isGlobal=True}) s:)
                    let node' = NGlobalDefvar a (zipWith (\(a1, a2, a3) (_, a4, _, _) ->(a1, a2, prefix ++ a4, a3)) defs' def)
                    return $ Right node'
                x -> return $ Left x
            ) ast
    -- Add all vars into table.
    let vars=concatMap MultiMap.toList varlist
    let qualifiedVars = concat $ map (\tup -> do
            let sym = fst tup
            let symName = getSymbolName sym
            let qualified = prefix ++ symName
            let qualifiedData = (snd tup){qualifiedName = qualified}
            [(sym, qualifiedData), (SymVar qualified, qualifiedData)]) vars
    mapM_ (uncurry addSym) $ reverse qualifiedVars
    
    -- Resolve all gates and procedures.
    resolved_headers<-mapM (\node->case node of
            Right x->return (Right x)
            Left (NResolvedGatedef pos name matrix size qir) -> do
                defineGlobalSym prefix name pos (Type () (Gate size) []) False False
                return $ Right (NResolvedGatedef (okStmt pos) (prefix ++ name) matrix size qir)
            Left (NDefParam pos params) -> do
                if qcis then do
                    mapM (\(param, b) -> case b of
                        Just _ -> defineGlobalSym prefix param pos (Type () (Array 0) [Type () Param []]) False False
                        Nothing -> defineGlobalSym prefix param pos (Type () Param []) False False) params
                    let params' = map (\(param, _) -> (prefix++param, param)) params
                    return $ Right $ NResolvedDefParam (okStmt pos) params'
                else throwError $ UnsupportedType pos (Type () Param [])
            Left (NExternGate pos name extra size qirname) -> do
                extra'<-mapM (\x->argType' pos x "<anonymous>") extra
                defineGlobalSym prefix name pos (Type () (Gate size) extra') False False
                return $ Right $ NResolvedExternGate (okStmt pos) (prefix ++ name) (fmap void extra) size qirname
            Left (NOracleTable pos name source value size) -> do
                defineGlobalSym prefix name pos (Type () (Gate size) []) False False
                return $ Right (NOracleTable (okStmt pos) (prefix ++ name) (prefix ++ source) value size)
            Left (NOracleLogic pos ty name args body) -> do
                let bool2qbit (Type () (Array x) [Type () Bool []]) = (Type () (Array x) [Type () Qbit []])
                let arg_types = map fst args
                let arg_types' = map bool2qbit $ arg_types ++ [ty]
                defineGlobalSym prefix name pos (Type () FuncTy $ unitType() : arg_types') True False
                scope
                ids <- mapM (\(ty, i) -> defineSym (SymVar i) pos ty) args
                modify' (\x->x{inOracle = True})
                body' <- mapM typeCheckAST body
                modify' (\x->x{inOracle = False})
                unscope
                return $ Right (NResolvedOracleLogic (okStmt pos) (Type () FuncTy $ ty:arg_types) (prefix ++ name) (zip arg_types ids) body')
            Left x@(NDerivedGatedef pos name source extra size) -> do
                extra'<-mapM (\x->argType' pos x "<anonymous>") extra
                defineGlobalSym prefix name pos (Type () (Gate size) extra') False True
                return $ Right (NDerivedGatedef (okStmt pos) (prefix ++ name) (prefix ++ source) extra' size)
            Left x@(NDerivedOracle pos name source extra size)->do
                extra'<-mapM (\x->argType' pos x "<anonymous>") extra
                defineGlobalSym prefix name pos (Type () (Gate size) extra') False False
                return $ Right (NDerivedOracle (okStmt pos) (prefix ++ name) (prefix ++ source) extra size)
            Left (NProcedureWithRet pos ty name args body ret) -> do
                -- check arg types and return types
                ty'<-case ty of
                    Type _ Int [] -> return $ void ty
                    Type _ Unit [] -> return $ void ty
                    Type _ Double [] -> return $ void ty
                    Type _ Bool [] -> return $ void ty
                    _ -> throwError $ BadProcedureReturnType pos (void ty, name)
                let new_args = if name == "main" && (length args) == 0 then [(Type pos (Array 0) [intType pos], "main$par1"), (Type pos (Array 0) [doubleType pos], "main$par2")] else args
                args'<-mapM (uncurry argType) new_args
                defineGlobalSym prefix name (annotation ty) (Type () FuncTy (ty':args')) False False
                -- NTempvar a (void b, procRet, Nothing)
                let procName = case name of {"main" -> "main"; x -> prefix ++ name}
                return $ Left (pos, ty', procName, zip args' (fmap snd new_args), body, ret)
            Left x -> error $ "unreachable" ++ show x
        ) resolved_defvar
    -- Finally, resolve procedure bodies.
    -- Note that we need to store byval-passed values (e.g. int) into new variables.
    
    body<-mapM (\node->case node of
        Right x->return x
        Left (pos, ty, func_name, args, body, ret@(ETempVar pret ret_id))-> do
            scope
            -- resolve return value
            ret_var<-case ty of
                Type _ Unit [] -> do
                    s<-defineSym (SymTempVar ret_id) pret (Type () Ref [ty])
                    return $ Nothing
                _ -> do
                    s<-defineSym (SymTempVar ret_id) pret (Type () Ref [ty])
                    return $ Just (Type () Ref [ty], s)
            -- resolve args
            (args', new_tempvars)<-flip runStateT [] $ mapM (\(ty, i)->case ty of
                Type _ Int [] -> do
                    temp_arg<-lift $ nextId -- Temporary argument
                    s<-lift $ defineSym (SymTempArg temp_arg) pos (intType ())
                    -- Leave the good name for defvar.
                    --real_arg<-lift $ defineSym (SymVar i) pos (refType () (intType ()))
                    modify' (++[NDefvar pos [(intType pos, i, Just $ ETempArg pos temp_arg, Nothing)]])
                    return (ty, s)
                x -> do
                    s<-lift $ defineSym (SymVar i) pos x
                    return (ty, s) 
                    ) args
            -- resolve body
            body'<-mapM typeCheckAST (new_tempvars++body)
            ret''<-case ty of
                Type _ Unit [] -> return Nothing
                _ -> do
                    ret'<-typeCheckExpr ret
                    ret''<-matchType [Exact ty] ret'
                    return $ Just ret''
            unscope
            return $ NResolvedProcedureWithRet (okStmt pos) ty func_name args' body' ret'' ret_var
        Left _ -> error "unreachable"
        ) resolved_headers

    m <- gets mainDefined
    when (isMain && not m) $ do
        throwError $ MainUndefined

    -- Extract global symbols
    symtable <- gets symbolTable
    let topLayer = getSecondLast symtable
    let lis = MultiMap.toList topLayer
    let globalLis = filter (isGlobal . snd) lis
    let globalLayer = MultiMap.fromList globalLis
    ssaId <- gets ssaAllocator
    return (body, globalLayer, ssaId)

getSecondLast :: [a] -> a
getSecondLast [] = error "Empty list"
getSecondLast [x] = error "Single-element list"
getSecondLast (x:_:[]) = x
getSecondLast (x:xs) = getSecondLast xs

typeCheckTop :: Bool -> String -> [LAST] -> SymbolTableLayer -> Int -> Bool -> Either TypeCheckError ([TCAST], SymbolTableLayer, Int)
typeCheckTop isMain prefix ast stl ssaId qcis= do
    let env = TypeCheckEnv [MultiMap.empty, stl] ssaId False False
    evalState (runExceptT $ typeCheckToplevel isMain prefix ast qcis) env

-- TODO: unification-based type check and type inference.

data TyAtom= TInt | TQbit | TBool | TDouble | TComplex | TList | TKnownList Int | TUser String | TRange | TGate Int | TRef | TVal | TFunc deriving (Show, Eq)
data Ty = TMultiple { tyArgs :: [Ty] } | TAtom TyAtom | TVar Int deriving (Show, Eq)

