{-# LANGUAGE TupleSections #-}
module ISQ.Lang.RAIICheck where
import Control.Monad.State
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.ISQv2Tokenizer (annotation, Annotated)
import Control.Monad.Except
import Data.Bifunctor
import Debug.Trace (trace)

{-
Module for transforming AST tree into RAII form, eliminating intermediate return/continue/break.
Instead, intermediate control flow statements will be replaced by flag registers and conditionally-jump-to-end. Return values will also be replaced by return value register.
-}

data Region = Func {value :: Int, flag :: Int} | While {headFlag :: Int, flag :: Int} | For {flag :: Int} | If {flag :: Int}  deriving (Show, Eq)

data RegionType = RFunc | RLoop | RIf deriving (Show, Eq)

regionType :: Region->RegionType
regionType Func{} = RFunc
regionType While{} = RLoop
regionType For{} = RLoop
regionType If{} = RIf

data RAIICheckEnv = RAIICheckEnv{
    flagCounter :: Int,
    regionFlags :: [Region]
} deriving Show

newRAIIEnv :: RAIICheckEnv
newRAIIEnv = RAIICheckEnv 0 []

-- Before region: push a flag onto stack. The flag means ``whether the parent region should stop right after this op.''
-- On multi-region jump: scan through the regions and set all the flags to 1/set return value to 1.

{-
While-statement can be transformed from
while {cond} do {body}
to 
loop { if(cond) {body} }
thus, body can be seen as ``subregion'' of head.
-}


-- Break writes both headFlag and bodyFlag
-- Continue writes bodyFlag only
-- Thus, (Continue, bodyFlag):(Break, headFlag):(Return, funcFlag)
pushRegion :: Region->RAIICheck ()
pushRegion r = modify' (\s->s{regionFlags = r:regionFlags s})
popRegion :: RAIICheck ()
popRegion = modify' (\s->s{regionFlags = tail $ regionFlags s})

skippedRegions :: Pos->RegionType->RAIICheck [Region]
skippedRegions pos ty = do
    regions <- gets regionFlags
    let go [] = Nothing
        go (x:xs) = if regionType x == ty  then Just [x] else fmap (x:) (go xs)
    case go regions of
        Nothing -> throwError $ UnmatchedScopeError pos ty
        Just x->return x


data RAIIError =
      UnmatchedScopeError {unmatchedPos :: Pos, wantedRegionType :: RegionType}
    deriving Show


type RAIICheck = ExceptT RAIIError (State RAIICheckEnv)

nextId :: RAIICheck Int
nextId = do
    st<-get
    id<-gets flagCounter
    put st{flagCounter=id+1}
    return id


-- A for-loop is affine safe if it keeps affine property (i.e. will finish its all iterations instead of break/return).
data AffineSafe = Safe | ContainsBreak | ContainsReturn deriving Show

instance Semigroup AffineSafe where
    (<>) ContainsReturn _ = ContainsReturn
    (<>) _ ContainsReturn = ContainsReturn
    (<>) ContainsBreak Safe = ContainsBreak
    (<>) Safe ContainsBreak = ContainsBreak
    (<>) ContainsBreak ContainsBreak = ContainsBreak
    (<>) Safe Safe = Safe
instance Monoid AffineSafe where
    mempty = Safe


isExprSafe :: Expr ann->Expr (AffineSafe, ann)
isExprSafe x@(ERange _ _ _ Nothing) = fmap (Safe,) x
isExprSafe x@(ERange _ _ _ (Just (EIntLit _ _))) = fmap (Safe,) x
isExprSafe x@(ERange ann lo hi (Just (EUnary ann2 Neg (EIntLit ann3 v)))) = isExprSafe (ERange ann lo hi (Just (EIntLit ann2 (-v))))
isExprSafe x@(ERange ann lo hi (Just (EUnary ann2 Positive (EIntLit ann3 v)))) = isExprSafe (ERange ann lo hi (Just (EIntLit ann2 v)))
isExprSafe x@ERange{} = fmap (ContainsBreak,) x
isExprSafe x = fmap (Safe,) x
checkSafe :: (Annotated p)=>p (AffineSafe, ann)->AffineSafe
checkSafe = fst . annotation

bodyEffect :: AST (AffineSafe, b) -> AffineSafe
bodyEffect NFor{annotationAST=(ContainsBreak,_)}=Safe
bodyEffect NWhile{annotationAST=(ContainsBreak,_)}=Safe
bodyEffect x = checkSafe x

isStatementAffineSafe' :: (AST ann->AST (AffineSafe, ann))->(AST ann->AST (AffineSafe, ann))
isStatementAffineSafe' f (NIf ann cond b1 b2) = let b1'=fmap f b1; b2' = fmap f b2 in NIf (mconcat $ fmap bodyEffect $ b1'++b2', ann) (isExprSafe cond) b1' b2'
isStatementAffineSafe' f (NFor ann v range body) = let b' = fmap f body; r' = isExprSafe range in NFor (mconcat $ checkSafe r':fmap bodyEffect b', ann) v r' b'
isStatementAffineSafe' _ x@(NBreak _) = fmap (ContainsBreak,) x
isStatementAffineSafe' _ x@(NReturn _ _) = fmap (ContainsReturn,) x
isStatementAffineSafe' f x@(NProcedure a b c d e)= let e' = fmap f e in NProcedure (Safe, a) (fmap (Safe,) b) c (fmap (first (fmap (Safe,))) d) e'
isStatementAffineSafe' _ x = fmap (Safe,) x

isStatementAffineSafe :: AST ann -> AST (AffineSafe, ann)
isStatementAffineSafe = fix isStatementAffineSafe'

eraseSafe :: (Functor p)=>p (a, b) -> p b
eraseSafe = fmap snd
eliminateNonAffineForStmts' :: (AST (AffineSafe, ann)->RAIICheck [AST ann])->AST (AffineSafe, ann)->RAIICheck [AST ann]
eliminateNonAffineForStmts' f (NIf a b c d) = do {c'<-mapM f c; d'<-mapM f d; return [NIf (snd a) (eraseSafe b) (concat c') (concat d')]}
eliminateNonAffineForStmts' f (NWhile a b c) = do {c'<-mapM f c; return [NWhile (snd a) (eraseSafe b) (concat c')]}
eliminateNonAffineForStmts' f (NFor (Safe, ann) v expr body) = do {b'<-mapM f body; return [NFor ann v (eraseSafe expr) (concat b')]}
eliminateNonAffineForStmts' f (NFor (s1, ann) v (ERange (s2, ann2) (Just a) (Just b) Nothing) body) = eliminateNonAffineForStmts' f (NFor (s1, ann) v (ERange (s2, ann2) (Just a) (Just b) (Just (EIntLit (Safe, ann) 1))) body)
eliminateNonAffineForStmts' f (NFor (_, ann) vn (ERange (_, ann2) (Just a) (Just b) (Just c)) body) = do {
    idlo<-nextId; idhi<-nextId;idstep<-nextId;
    b'<-mapM f body;
    let v = EIdent ann vn
        lo = ETempVar ann2 idlo
        hi = ETempVar ann2 idhi
        step = ETempVar ann2 idstep
    in return [
        NTempvar ann (intType (), idlo, Just $ eraseSafe a),
        NTempvar ann (intType (), idhi, Just $ eraseSafe b),
        NTempvar ann (intType (), idstep, Just $ eraseSafe c),
        NDefvar ann [(intType ann, vn, Just lo)],
        NWhile ann (EBinary ann2 (Cmp Less) v hi)
        (concat b' ++ [NAssign ann v (EBinary ann2 Add v step)])]
    }
eliminateNonAffineForStmts' f NFor{} = error "For-statement with non-standard range indices not supported."
eliminateNonAffineForStmts' f (NProcedure a b c d e) = do
    e'<-mapM f e
    return [NProcedure (snd a) (eraseSafe b) c (fmap (first eraseSafe) d) (concat e')]

eliminateNonAffineForStmts' _ x = return [eraseSafe x]

eliminateNonAffineForStmts :: AST (AffineSafe, ann) -> RAIICheck [AST ann]
eliminateNonAffineForStmts = fix eliminateNonAffineForStmts'


checkCurrentScope :: Pos->RAIICheck LAST
checkCurrentScope pos = do
    scope<-gets (flag.head.regionFlags)
    return $ NJumpToEndOnFlag pos (ETempVar pos scope)

tempBool :: ann->Int->AST ann
tempBool ann i = NTempvar ann (boolType (), i, Just (EBoolLit ann False))

raiiTransform' :: (LAST->RAIICheck [LAST])->(LAST->RAIICheck [LAST])
raiiTransform' f (NIf ann cond t e) = do
    i<-nextId
    pushRegion (If i)
    t'<-mapM f t
    e'<-mapM f e
    popRegion
    finalize<-checkCurrentScope ann
    return [tempBool ann i, NIf ann cond (concat t') (concat e'), finalize]
raiiTransform' f (NFor ann var range b) = do
    i<-nextId
    pushRegion (For i)
    b'<-mapM f b
    popRegion
    finalize<-checkCurrentScope ann
    return [tempBool ann i, NFor ann var range (concat b'), finalize]
raiiTransform' f (NWhile ann cond body) = do
    ihead<-nextId
    ibody<-nextId
    pushRegion (While ihead ibody)
    b'<-mapM f body
    popRegion
    finalize<-checkCurrentScope ann
    return [tempBool ann ihead, tempBool ann ibody, NWhileWithGuard ann cond (concat b') (ETempVar ann ihead)]
raiiTransform' f (NProcedure a b c d e) = do
    procRet<-nextId
    procFlag<-nextId
    pushRegion (Func procRet procFlag)
    e'<-mapM f (tempBool a procFlag : e)
    popRegion
    -- no finalizer
    return [NProcedureWithRet a b c d (concat e') (ETempVar a procRet)]

-- The transformations below should also work with labeled loops.
raiiTransform' f (NBreak ann) = do
    regions<-skippedRegions ann RLoop
    let break_all_loops  = concatMap (\x->case x of
            While h f-> [setFlag ann h]
            _ -> []) regions
    let break_passing_bodies = fmap (\x->setFlag ann (flag x)) $ tail regions
    return $ break_all_loops ++ break_passing_bodies ++ [NJumpToEnd ann]

raiiTransform' f (NContinue ann) = do
    regions<-skippedRegions ann RLoop
    let break_all_loops  = concatMap (\x->case x of
            While h f-> [setFlag ann h]
            _ -> []) $ init regions
    let break_passing_bodies = fmap (\x->setFlag ann (flag x)) $ tail regions
    return $ break_all_loops ++ break_passing_bodies ++ [NJumpToEnd ann]

raiiTransform' f (NReturn ann val) = do
    regions<-skippedRegions ann RFunc
    let break_all_loops  = concatMap (\x->case x of
            While h f-> [setFlag ann h]
            _ -> []) regions
    let break_passing_bodies = fmap (\x->setFlag ann (flag x)) $ tail regions
    let Func v f = last regions
    case val of
        EUnitLit _ -> return $ break_all_loops ++ break_passing_bodies ++ [NJumpToEnd ann]
        _ -> return $ break_all_loops ++ break_passing_bodies ++ [setReturnVal ann v val, NJumpToEnd ann]

raiiTransform' _ ast = return [ast]



setFlag :: ann->Int->AST ann
setFlag ann x= NAssign ann (ETempVar ann x) (EBoolLit ann True)
setReturnVal :: ann->Int->Expr ann->AST ann
setReturnVal ann x y = NAssign ann (ETempVar ann x) y
raiiTransform :: LAST -> RAIICheck [LAST]
raiiTransform = fix raiiTransform'


raiiCheck' :: [LAST]->RAIICheck [LAST]
raiiCheck' ast = do
    let safeCheck = fmap isStatementAffineSafe ast
    eliminateFor <- concat <$> mapM eliminateNonAffineForStmts safeCheck
    concat <$> mapM raiiTransform eliminateFor

raiiCheck ast = evalState (runExceptT (raiiCheck' ast)) newRAIIEnv

-- 

--isStatementAffineSafe b2
--isStatementAffineSafe 
--raiiTransform :: [LAST]->[LAST]
