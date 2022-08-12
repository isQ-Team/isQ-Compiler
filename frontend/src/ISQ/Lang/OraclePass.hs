{-# LANGUAGE ViewPatterns #-}
module ISQ.Lang.OraclePass where
import ISQ.Lang.ISQv2Grammar
import Control.Monad.Extra (concatMapM)
import Control.Monad (void)
import Data.Complex
  
data OracleError =
      BadOracleShape Pos
    | BadOracleValue Pos deriving (Eq, Show)

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

bitToInt :: [Int]->Int
bitToInt [] = error "empty list"
bitToInt [x] = x
bitToInt (x:y) = x + 2*(bitToInt y) 

bitAdd :: Int -> Int -> Int
bitAdd a b = bitToInt $ go bit_a bit_b where 
    bit_a = toBit a
    bit_b = toBit b
    go [] [] = []
    go x [] = x
    go [] y = y
    go (x1:x2) (y1:y2) = (mod (x1+y1) 2):(go x2 y2)

foldConstantValue :: LExpr->Either OracleError Int
foldConstantValue x@(EIntLit _ val) = Right val
foldConstantValue x = Left $ BadOracleValue (annotationExpr x)

getDerivingArgs :: Int -> Pos -> [(LType, Ident)]
getDerivingArgs a ann = map (\x -> (Type ann Qbit [], [x])) (take a ['a'..'z']) 

getGateModifier :: Int -> Int -> Pos -> [GateModifier]
getGateModifier t 0 ann = [Ctrl True t]
getGateModifier 0 f ann = [Ctrl False f]
getGateModifier t f ann = getGateModifier t 0 ann ++ (getGateModifier 0 f ann)

--getQbitList :: [Int]->Int->Pos-> [LExpr]
--getQbitList fx y ann = let s_i = zip fx ['a'..'z'] in (map (\x -> (EIdent ann [x])) ([b | (a,b) <- s_i , a == 1] ++ [b | (a,b) <- s_i , a == 0]) ++ [EIdent ann [['a'..'z'] !! y]])

--getDerivingState :: Int->Int->Int->Pos-> LAST
--getDerivingState x y l ann = let b = toBit' x l in NCoreUnitary ann (EIdent ann "X") (getQbitList b y ann) (getGateModifier (sum b) (length b - (sum b)) ann) 

getCtrlList :: [(Int, Int)]->Int->Int->[(Int, Int)]
getCtrlList [(0, _)] n m = []
getCtrlList [(x, y)] n m = let b = toBit' x m in [(y, n+v) | (u, v) <- (zip b [0..]), u == 1]
getCtrlList (x:y) n m = getCtrlList [x] n m ++ (getCtrlList y n m)

--getDerivingBody :: [Int]->Int->Int->Pos->[LAST]
--getDerivingBody fx n m ann = map (\(x, y) -> getDerivingState x y n ann) (getCtrlList (zip fx [0..]) n m)

getMValue :: [Int]->Int->Int->[Int]
getMValue fx m y = [u | u <- (map (\(v, x) -> if (toBit' v m) !! y == 1 then x else -1) (zip fx [0..])), u > -1]

getOracleVale :: [Int]->Int->[[Int]]
getOracleVale fx m = map (\y -> getMValue fx m y) (take m [0..])

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

passOracle' x = Right [x]

passOracle :: [LAST] -> Either OracleError [LAST]
passOracle = concatMapM passOracle'