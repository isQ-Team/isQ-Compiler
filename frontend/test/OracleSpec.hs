module OracleSpec where
import ISQ.Driver.Passes
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.OraclePass

import Control.Monad.Except
import Control.Monad.State
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import Test.Hspec

oracleTestTemplate :: String -> Int -> IO (Either OracleError Int)
oracleTestTemplate input val = do
    errorOrAst <- evalStateT (runExceptT $ parseToAST "" input) $ ImportEnv MultiMap.empty Map.empty 0
    case errorOrAst of
        Left x -> error "input file error"
        Right ast -> do
            let oracleFunc = head $ defMemberList ast
            let body = procBody oracleFunc
            let ann = annotationAST oracleFunc
            return $ evaluateFunc body ann "x" val

evaluateExpect :: String -> Int -> Int -> IO ()
evaluateExpect input val expect = do
    res <- oracleTestTemplate input val
    res `shouldBe` (Right expect)

getOracleErrorType :: OracleError -> String
getOracleErrorType = head . words . show

illegalEvaluate :: String -> Int -> String -> IO ()
illegalEvaluate input val expect = do
    errorOrInt <- oracleTestTemplate input val
    case errorOrInt of
        Left err -> (getOracleErrorType err) `shouldBe` expect
        Right _ -> error "evaluate error"

oracleSpec :: SpecWith ()
oracleSpec = do
  describe "ISQ.Lang.OraclePass.evaluateFunc" $ do
    it "evaluates correctly with returning an integer literal" $ do
        let str = "oracle o(1, 1) : x { return 1;}"
        evaluateExpect str 1 1

    it "evaluates correctly with bool-to-int conversion" $ do
        let str = "oracle o(1, 1) : x { return true;}"
        evaluateExpect str 1 1

    it "evaluates correctly with returning a variable" $ do
        let str = "oracle o(1, 1) : x { return x;}"
        evaluateExpect str 1 1

    it "evaluates correctly with addition" $ do
        let str = "oracle o(1, 1) : x { return x + 1;}"
        evaluateExpect str 0 1

    it "evaluates correctly with subtraction" $ do
        let str = "oracle o(1, 1) : x { return x - 1;}"
        evaluateExpect str 1 0

    it "evaluates correctly with multiplication" $ do
        let str = "oracle o(2, 2) : x { return x * 2;}"
        evaluateExpect str 1 2

    it "evaluates correctly with division" $ do
        let str = "oracle o(2, 2) : x { return x / 2;}"
        evaluateExpect str 2 1

    it "evaluates correctly with modulo" $ do
        let str = "oracle o(2, 1) : x { return x % 2;}"
        evaluateExpect str 2 0

    it "evaluates correctly with pow" $ do
        let str = "oracle o(2, 3) : x { return x ** 2;}"
        evaluateExpect str 2 4

    it "evaluates correctly with &&" $ do
        let str = "oracle o(1, 1) : x { return true && false;}"
        evaluateExpect str 1 0

    it "evaluates correctly with and" $ do
        let str = "oracle o(1, 1) : x { return true and true;}"
        evaluateExpect str 1 1

    it "evaluates correctly with ||" $ do
        let str = "oracle o(1, 1) : x { return true || false;}"
        evaluateExpect str 1 1

    it "evaluates correctly with or" $ do
        let str = "oracle o(1, 1) : x { return false or false;}"
        evaluateExpect str 1 0

    it "evaluates correctly with andi" $ do
        let str = "oracle o(4, 4) : x { return x & 12;}"
        evaluateExpect str 9 8

    it "evaluates correctly with ori" $ do
        let str = "oracle o(4, 4) : x { return x | 12;}"
        evaluateExpect str 9 13

    it "evaluates correctly with xori" $ do
        let str = "oracle o(4, 4) : x { return x ^ 12;}"
        evaluateExpect str 9 5

    it "evaluates correctly with <<" $ do
        let str = "oracle o(4, 4) : x { return (x << 2) & 15;}"
        evaluateExpect str 9 4

    it "evaluates correctly with >>" $ do
        let str = "oracle o(4, 4) : x { return x >> 2;}"
        evaluateExpect str 9 2

    it "evaluates correctly with neg" $ do
        let str = "oracle o(2, 1) : x { return x + -1;}"
        evaluateExpect str 2 1

    it "evaluates correctly with !" $ do
        let str = "oracle o(1, 1) : x { return !false;}"
        evaluateExpect str 1 1

    it "evaluates correctly with not" $ do
        let str = "oracle o(1, 1) : x { return not true;}"
        evaluateExpect str 1 0

    it "evaluates correctly with assigned variables" $ do
        let str = "oracle o(2, 2) : x { x = 3; return x;}"
        evaluateExpect str 2 3

    it "evaluates correctly with defining and assignment" $ do
        let str = "oracle o(2, 2) : x { int y; y = 3; return y;}"
        evaluateExpect str 2 3

    it "evaluates correctly with plus and assign" $ do
        let str = "oracle o(2, 2) : x { int y = 2; y += x; return y;}"
        evaluateExpect str 1 3

    it "evaluates correctly with minus and assign" $ do
        let str = "oracle o(2, 2) : x { int y = 2; y -= x; return y;}"
        evaluateExpect str 1 1

    it "evaluates correctly with scoped assigned variables" $ do
        let str = "oracle o(2, 1) : x { { x = 3; } return x;}"
        evaluateExpect str 2 3

    it "evaluates correctly with shadowed variables" $ do
        let str = "oracle o(2, 1) : x { { int x = 3; } return x;}"
        evaluateExpect str 2 2

    it "evaluates correctly with if (match)" $ do
        let str = "oracle o(2, 1) : x { if (x == 3) return 1; return 0;}"
        evaluateExpect str 3 1

    it "evaluates correctly with if (not match)" $ do
        let str = "oracle o(2, 1) : x { if (x < 1) return 1; return 0;}"
        evaluateExpect str 2 0

    it "evaluates correctly with if-else" $ do
        let str = "oracle o(2, 1) : x { if (x > 2) return 1; else return 0;}"
        evaluateExpect str 3 1

    it "evaluates correctly with greater equal" $ do
        let str = "oracle o(2, 1) : x { if (x >= 2) return 1; else return 0;}"
        evaluateExpect str 2 1

    it "evaluates correctly with less equal" $ do
        let str = "oracle o(2, 1) : x { if (x <= 2) return 1; else return 0;}"
        evaluateExpect str 2 1

    it "evaluates correctly with not equal" $ do
        let str = "oracle o(2, 1) : x { if (x != 2) return 1; else return 0;}"
        evaluateExpect str 1 1

    it "evaluates correctly with newly defined variable" $ do
        let str = "oracle o(1, 1) : x { int y = 1; return y;}"
        evaluateExpect str 0 1

    it "evaluates correctly with while statement" $ do
        let str = "oracle o(2, 2) : x {\
            \int sum = 0; {\
            \int i = 1;\
            \while (i < 4) {\
                \sum = sum + i;\
                \i = i + 1;\
            \} }\
            \return sum;\
        \}"
        evaluateExpect str 3 6

    it "evaluates correctly with for statement" $ do
        let str = "oracle o(2, 3) : x {\
            \int sum = 0;\
            \for i in 1:4\
                \sum = sum + i;\
            \return sum;\
        \}"
        evaluateExpect str 3 6

    it "evaluates correctly with break statement" $ do
        let str = "oracle o(2, 3) : x {\
            \int sum = 0;\
            \int i = 1;\
            \while (i < 4) {\
                \if (i == 2) break;\
                \sum = sum + i;\
                \i = i + 1;\
            \}\
            \return sum;\
        \}"
        evaluateExpect str 3 1

    it "evaluates correctly with continue statement" $ do
        let str = "oracle o(2, 3) : x {\
            \int sum = 0;\
            \int i = 0;\
            \while (i < 3) {\
                \i = i + 1;\
                \if (i == 2) continue;\
                \sum = sum + i;\
            \}\
            \return sum;\
        \}"
        evaluateExpect str 3 4

    it "evaluates correctly with for and break statements" $ do
        let str = "oracle o(2, 3) : x {\
            \int sum = 0;\
            \for i in 1:4 {\
                \if (i == 2) break;\
                \sum = sum + i;\
            \}\
            \return sum;\
        \}"
        evaluateExpect str 3 1

    it "evaluates correctly with for and continue statement" $ do
        let str = "oracle o(2, 3) : x {\
            \int sum = 0;\
            \for i in 1:4 {\
                \if (i == 2) continue;\
                \sum = sum + i;\
            \}\
            \return sum;\
        \}"
        evaluateExpect str 3 4

    it "returns an error when assigned to an int" $ do
        let str = "oracle o(1, 1) : x { (2) = 3; }"
        illegalEvaluate str 0 "IllegalExpression"

    it "returns an error for function calling" $ do
        let str = "oracle o(1, 1) : x { fun(); }"
        illegalEvaluate str 0 "IllegalExpression"

    it "returns an error for function calling expression" $ do
        let str = "oracle o(1, 1) : x { return fun(); }"
        illegalEvaluate str 0 "IllegalExpression"

    it "returns an error when defining x" $ do
        let str = "oracle o(1, 1) : x { int x; }"
        illegalEvaluate str 0 "MultipleDefined"

    it "returns an error when defining a variable twice" $ do
        let str = "oracle o(1, 1) : x { int y; int y; }"
        illegalEvaluate str 0 "MultipleDefined"

    it "returns an error when no value is returned" $ do
        let str = "oracle o(1, 1) : x {}"
        illegalEvaluate str 0 "NoReturnValue"

    it "returns an error with undefined variable" $ do
        let str = "oracle o(1, 1) : x { return y; }"
        illegalEvaluate str 0 "UndefinedSymbol"

    it "returns an error when assigning to undefined variable" $ do
        let str = "oracle o(1, 1) : x { y = 3; }"
        illegalEvaluate str 0 "UndefinedSymbol"

    it "returns an error when anding an integer" $ do
        let str = "oracle o(1, 1) : x { return 1 && true; }"
        illegalEvaluate str 0 "UnmatchedType"

    it "returns an error when using int as condition" $ do
        let str = "oracle o(1, 1) : x { if (2) ; }"
        illegalEvaluate str 0 "UnmatchedType"

    it "returns an error when defining a qbit" $ do
        let str = "oracle o(1, 1) : x { qbit y; }"
        illegalEvaluate str 0 "UnsupportedType"
