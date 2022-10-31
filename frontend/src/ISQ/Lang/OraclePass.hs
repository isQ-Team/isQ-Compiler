{-# LANGUAGE ViewPatterns #-}
module ISQ.Lang.OraclePass where
import ISQ.Lang.ISQv2Grammar
import Control.Monad.Except
import Control.Monad.Extra (concatMapM)
import Control.Monad (void)
import Control.Monad.State.Lazy (evalState, get, put, State)
import Data.Bits
import Data.Complex
import Data.List (null)
--import Data.Either.Combinators (mapRight)
import qualified Data.Map.Lazy as Map
  
data OracleError =
      BadOracleShape Pos
    | BadOracleValue Pos
    | IllegalExpression Pos
    | MultipleDefined {sourcePos :: Pos, varName :: String}
    | NoReturnValue {sourcePos :: Pos, inputValue :: Int}
    | UndefinedSymbol {sourcePos :: Pos, varName :: String}
    | UnmatchedType Pos
    | UnsupportedType Pos
    deriving (Eq, Show)

mangleGate x = "$_ISQ_GATEDEF_"++x

pow2 :: Int->Int
pow2 0 = 1
pow2 x = 2 * pow2 (x-1)

toBit :: Int->[Int]
toBit 0 = [0]
toBit 1 = [1]
toBit x = (toBit $ div x 2) ++ (toBit $ mod x 2)

toBit' :: Int->Int->[Int]
toBit' x l = (replicate (l - (length $ toBit x)) 0) ++ (toBit x)

foldConstantValue :: LExpr->Either OracleError Int
foldConstantValue x@(EIntLit _ val) = Right val
foldConstantValue x = Left $ BadOracleValue (annotationExpr x)

getDerivingArgs :: Int -> Pos -> [(LType, Ident)]
getDerivingArgs a ann = map (\x -> (Type ann Qbit [], [x])) (take a ['a'..'z']) 

getMValue :: [Int]->Int->Int->[Int]
getMValue fx m y = [u | u <- (map (\(v, x) -> if (toBit' v m) !! y == 1 then x else -1) (zip fx [0..])), u > -1]

getOracleVale :: [Int]->Int->[[Int]]
getOracleVale fx m = map (\y -> getMValue fx m y) (take m [0..])

-- The execution result of expressions and statements
data Obj = OInt Int
    | OBool Bool
    | OUnit
    | OBreak
    | OContinue

isInt :: Obj -> Bool
isInt (OInt _) = True
isInt _ = False

isBool :: Obj -> Bool
isBool (OBool _) = True
isBool _ = False

type OracleEvaluate = ExceptT OracleError (State [Map.Map String Obj])

getInt :: Pos -> Obj -> OracleEvaluate Int
getInt _ (OInt val) = return $ val
getInt _ (OBool True) = return $ 1
getInt _ (OBool False) = return $ 0
getInt ann _ = throwError $ UnmatchedType ann

getBool :: Pos -> Obj -> OracleEvaluate Bool
getBool _ (OBool val) = return $ val
getBool ann _ = throwError $ UnmatchedType ann

binaryIntOperation :: Pos -> (Int -> Int -> Int) -> Obj -> Obj -> OracleEvaluate Obj
binaryIntOperation ann op lobj robj = do
    lint <- getInt ann lobj
    rint <- getInt ann robj
    return $ OInt $ op lint rint

binaryBoolOperation :: Pos -> (Bool -> Bool -> Bool) -> Obj -> Obj -> OracleEvaluate Obj
binaryBoolOperation ann op lobj robj = do
    lint <- getBool ann lobj
    rint <- getBool ann robj
    return $ OBool $ op lint rint

binaryComparison :: Pos -> (Int -> Int -> Bool) -> Obj -> Obj -> OracleEvaluate Obj
binaryComparison ann op lobj robj = do
    lint <- getInt ann lobj
    rint <- getInt ann robj
    return $ OBool $ op lint rint


evaluateExpression :: LExpr -> OracleEvaluate Obj

evaluateExpression (EIntLit ann val) = return $ OInt val

evaluateExpression (EBoolLit ann val) = return $ OBool val

evaluateExpression (EIdent ann ident) = do
    let find :: [Map.Map String Obj] -> OracleEvaluate Obj
        find [] = throwError $ UndefinedSymbol ann ident
        find (x:xs) = do
            let maybe = Map.lookup ident x
            case maybe of
                Just x -> return x
                Nothing -> find xs
    symbolTables <- get
    find symbolTables

evaluateExpression (EBinary ann op lexpr rexpr) = do
    lobj <- evaluateExpression(lexpr)
    robj <- evaluateExpression(rexpr)
    case op of
        Add -> binaryIntOperation ann (+) lobj robj
        Sub -> binaryIntOperation ann (-) lobj robj
        Mul -> binaryIntOperation ann (*) lobj robj
        Div -> binaryIntOperation ann div lobj robj
        Mod -> binaryIntOperation ann mod lobj robj
        Pow -> binaryIntOperation ann (^) lobj robj
        And -> binaryBoolOperation ann (&&) lobj robj
        Or -> binaryBoolOperation ann (||) lobj robj
        Andi -> binaryIntOperation ann (.&.) lobj robj
        Ori -> binaryIntOperation ann (.|.) lobj robj
        Xori -> binaryIntOperation ann xor lobj robj
        Cmp Equal -> do
            unless (isInt lobj && isInt robj || isBool lobj && isBool robj)
                (throwError $ UnmatchedType ann)
            if (isInt lobj) then binaryComparison ann (==) lobj robj
                else do
                    lbool <- getBool ann lobj
                    rbool <- getBool ann robj
                    return $ OBool $ lbool == rbool
        Cmp NEqual -> do
            unless (isInt lobj && isInt robj || isBool lobj && isBool robj)
                (throwError $ UnmatchedType ann)
            if (isInt lobj) then binaryComparison ann (/=) lobj robj
                else do
                    lbool <- getBool ann lobj
                    rbool <- getBool ann robj
                    return $ OBool $ lbool /= rbool
        Cmp Greater -> binaryComparison ann (>) lobj robj
        Cmp Less -> binaryComparison ann (<) lobj robj
        Cmp GreaterEq -> binaryComparison ann (>=) lobj robj
        Cmp LessEq -> binaryComparison ann (<=) lobj robj
        Shl -> binaryIntOperation ann shiftL lobj robj
        Shr -> binaryIntOperation ann shiftR lobj robj

evaluateExpression (EUnary ann op expr) = do
    obj <- evaluateExpression(expr)
    case op of
        Positive -> return obj
        Neg -> getInt ann obj >>= (\x -> return $ OInt $ 0 - x)
        Not -> getBool ann obj >>= (\x -> return $ OBool $ x == False)

evaluateExpression exp = throwError $ IllegalExpression $ annotationExpr exp


scope :: OracleEvaluate ()
scope = do
    symbolTables <- get
    put (Map.empty : symbolTables)

unscope :: OracleEvaluate ()
unscope = do
    symbolTables <- get
    put $ tail symbolTables

defineVariable :: Pos -> (LType, String, Maybe LExpr) -> OracleEvaluate ()
defineVariable pos (ltype, identifier, maybeExpr) = do
    symbolTables <- get
    let currentTable = head symbolTables
    let maybe = Map.lookup identifier currentTable
    case maybe of
        Just _ -> throwError $ MultipleDefined pos identifier
        Nothing -> do
            obj <- case maybeExpr of
                Nothing -> return OUnit
                Just expr -> evaluateExpression expr
            case ltype of
                Type ann Int _ -> put $ Map.insert identifier obj currentTable : tail symbolTables
                Type ann Bool _ -> put $ Map.insert identifier obj currentTable : tail symbolTables
                _ -> throwError $ UnsupportedType pos


evaluateStatement :: LAST -> OracleEvaluate Obj

evaluateStatement (NDefvar ann defs) = do
    mapM (defineVariable ann) defs
    return OUnit

evaluateStatement (NReturn ann exp) = evaluateExpression exp

evaluateStatement (NAssign ann lexpr rexpr op) = do
    case lexpr of
        EIdent eann ident -> do
            let findIdent :: [Map.Map String Obj] -> [Map.Map String Obj] -> OracleEvaluate [Map.Map String Obj]
                findIdent pre [] = throwError $ UndefinedSymbol eann ident
                findIdent pre (x:xs) = do
                    let maybe = Map.lookup ident x
                    case maybe of
                        Nothing -> findIdent (pre ++ [x]) xs
                        Just _ -> do
                            obj <- evaluateExpression rexpr
                            let newx = Map.insert ident obj x
                            return $ pre ++ [newx] ++ xs
            symbolTables <- get
            newTables <- findIdent [] symbolTables
            put newTables
            return OUnit
        _ -> throwError $ IllegalExpression ann

evaluateStatement (NBlock ann body) = do
    scope
    res <- evaluateStatements body
    unscope
    return res

evaluateStatement (NIf ann cond ifStat elseStat) = do
    ocond <- evaluateExpression cond
    bcond <- getBool ann ocond
    if bcond then evaluateStatement $ head ifStat
    else if null elseStat then return OUnit
        else evaluateStatement $ head elseStat

evaluateStatement (NWhile ann cond body) = do
    let executeWhile :: OracleEvaluate Obj
        executeWhile = do
            ocond <- evaluateExpression cond
            bcond <- getBool ann ocond
            case bcond of
                True -> do
                    res <- evaluateStatement $ head body
                    case res of
                        OBreak -> return OUnit
                        _ -> executeWhile
                False -> return OUnit
    executeWhile

evaluateStatement (NFor ann forVar range@(ERange eann lo hi step) body) = do
    let evar = EIdent ann forVar
    let econd = case hi of
            Nothing -> EBoolLit eann True
            Just x -> EBinary eann (Cmp Less) evar x
    let estep = case step of
            Nothing -> EIntLit eann 1
            Just x -> x
    let eadd = EBinary eann Add evar estep
    let executeFor :: OracleEvaluate Obj
        executeFor = do
            ocond <- evaluateExpression econd
            bcond <- getBool ann ocond
            case bcond of
                True -> do
                    res <- evaluateStatement $ head body
                    evaluateStatement $ NAssign eann evar eadd AssignEq
                    case res of
                        OBreak -> return OUnit
                        _ -> executeFor
                False -> return OUnit
    scope
    evaluateStatement $ NDefvar ann [(intType ann, forVar, lo)]
    res <- executeFor
    unscope
    return res

evaluateStatement (NBreak ann) = return OBreak

evaluateStatement (NContinue ann) = return OContinue

evaluateStatement (NCall ann _) = throwError $ IllegalExpression ann

evaluateStatement _ = return OUnit


evaluateStatements :: [LAST] -> OracleEvaluate Obj
evaluateStatements [] = return OUnit
evaluateStatements (x:xs) = do
    obj <- evaluateStatement x
    case obj of
        OUnit -> evaluateStatements xs
        other -> return other

evaluateFunc :: [LAST] -> Pos -> String -> Int -> Either OracleError Int
evaluateFunc body pos var val = do
    let defVar = NDefvar pos [(intType pos, var, Just $ EIntLit pos val)]
    obj <- evalState (runExceptT $ evaluateStatements $ [defVar] ++ body) [Map.empty]
    case obj of
        OInt val -> Right val
        OBool True -> Right 1
        OBool False -> Right 0
        OUnit -> Left $ NoReturnValue pos val
        _ -> Left $ UnmatchedType pos

--replace orcale with deriving gate (use multi-contrl-x)
passOracle' :: LAST -> Either OracleError [LAST]
passOracle' (NOracle ann name n m fx) = if (pow2 n) /= (length fx) then Left $ BadOracleShape ann else do 
    fv <- mapM foldConstantValue fx
    if all (\x -> x < (pow2 m)) fv then
        let proc_name = mangleGate name
        in Right [
            NProcedure ann (unitType ann) proc_name (getDerivingArgs (n+m) ann) [],
            NOracleTable ann name proc_name (getOracleVale fv m) (n+m)
            ]
    else Left $ BadOracleValue ann

passOracle' (NOracleFunc ann name n m inVar stats) = do
    let ins = take (pow2 n) [0..]
    outs <- mapM (evaluateFunc stats ann inVar) ins
    let eints = map (EIntLit ann) outs
    passOracle' (NOracle ann name n m eints)

passOracle' x = Right [x]

passOracle :: [LAST] -> Either OracleError [LAST]
passOracle = concatMapM passOracle'
