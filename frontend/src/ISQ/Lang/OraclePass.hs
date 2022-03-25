{-# LANGUAGE ViewPatterns #-}
module ISQ.Lang.OraclePass where
import ISQ.Lang.ISQv2Grammar
import Control.Monad.Extra (concatMapM)
import Control.Monad (void)
import Data.Complex
  
data OracleError =
      BadOracleShape Pos
    | BadOracleValue Pos deriving Show

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

{--
getMatRow :: [Int]->Int->Int->[[Complex Double]]
getMatRow [i] n m = let size = pow2 (n+m) in [(replicate i 0) ++ [1] ++ (replicate (size - 1 - i) 0)]
getMatRow (i:y) n m = let size = pow2 (n+m) in [(replicate i 0) ++ [1] ++ (replicate (size - 1 - i) 0)] ++ (getMatRow y n m)

getMat :: [[Int]] -> Int -> Int -> [[Complex Double]]
getMat [x] n m = getMatRow x n m
getMat (x:y) n m = getMatRow x n m ++ (getMat y n m)

getNewGateDef :: [Int] -> Int -> Int -> Maybe [[Complex Double]]
getNewGateDef a n m = if all (\x -> x < (pow2 m)) a then do
    Just $ getMat (map (\(x,i) -> map (\y -> (pow2 m * i) + bitAdd x y) [0..(pow2 m - 1)]) (zip a [0..(pow2 n-1)])) n m
    else Nothing
--}

getDerivingArgs :: Int -> Pos -> [(LType, Ident)]
getDerivingArgs a ann = map (\x -> (Type ann Qbit [], [x])) (take a ['a'..'z']) 

getGateModifier :: Int -> Int -> Pos -> [GateModifier]
getGateModifier t 0 ann = [Ctrl True t]
getGateModifier 0 f ann = [Ctrl False f]
getGateModifier t f ann = getGateModifier t 0 ann ++ (getGateModifier 0 f ann)

getQbitList :: [Int]->Int->Pos-> [LExpr]
getQbitList fx y ann = let s_i = zip fx ['a'..'z'] in (map (\x -> (EIdent ann [x])) ([b | (a,b) <- s_i , a == 1] ++ [b | (a,b) <- s_i , a == 0]) ++ [EIdent ann [['a'..'z'] !! y]])

getDerivingState :: Int->Int->Int->Pos-> LAST
getDerivingState x y l ann = let b = toBit' x l in NCoreUnitary ann (EIdent ann "X") (getQbitList b y ann) (getGateModifier (sum b) (length b - (sum b)) ann) 

getCtrlList :: [(Int, Int)]->Int->Int->[(Int, Int)]
getCtrlList [(0, _)] n m = []
getCtrlList [(x, y)] n m = let b = toBit' x m in [(y, n+v) | (u, v) <- (zip b [0..]), u == 1]
getCtrlList (x:y) n m = getCtrlList [x] n m ++ (getCtrlList y n m)

getDerivingBody :: [Int]->Int->Int->Pos->[LAST]
getDerivingBody fx n m ann = map (\(x, y) -> getDerivingState x y n ann) (getCtrlList (zip fx [0..]) n m)

--replace orcale with deriving gate (use multi-contrl-x)
passOracle :: [LAST] -> Either OracleError [LAST]
passOracle = mapM go where 
  go o@(NOracle ann name n m fx) = if (pow2 n) /= (length fx) then Left $ BadOracleShape ann else do
        fv <- mapM foldConstantValue fx
        if all (\x -> x < (pow2 m)) fv then return $ NProcedureWithDerive ann (unitType ann) name (getDerivingArgs (n+m) ann) (getDerivingBody fv n m ann) (Just DeriveGate) else Left $ BadOracleValue ann
  go x = return x