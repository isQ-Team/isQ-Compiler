module PrecedenceSpec where
import ISQ.Driver.Passes
import ISQ.Lang.ISQv2Grammar

import Control.Monad.Except
import Control.Monad.State
import Control.Monad (void)
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import Test.Hspec

strToAst :: String -> IO LAST
strToAst input = do
    errorOrAst <- evalStateT (runExceptT $ parseToAST "" input) $ ImportEnv MultiMap.empty Map.empty 0 False []
    case errorOrAst of
        Left x -> error "input file error"
        Right ast -> return $ head $ defMemberList ast

precedenceSpec :: SpecWith ()
precedenceSpec = do
    describe "ISQ.parseToAST" $ do
        it "evaluate as before **" $ do
            let str1 = "int fun(){ return 2 ** 2.2 as int; }"
            let str2 = "int fun(){ return 2 **(2.2 as int); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate ** before -" $ do
            let str1 = "int fun(){ return - 2 ** 2; }"
            let str2 = "int fun(){ return -(2 ** 2); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate minus before *" $ do
            let str1 = "int fun(){ return -2 * 2; }"
            let str2 = "int fun(){ return(-2)* 2; }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate ** right to left" $ do
            let str1 = "int fun(){ return 2 ** 3 ** 4; }"
            let str2 = "int fun(){ return 2 **(3 ** 4); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate * before +" $ do
            let str1 = "int fun(){ return 2 * 3 + 4; }"
            let str2 = "int fun(){ return(2 * 3)+ 4; }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate + before >>" $ do
            let str1 = "int fun(){ return 2 + 3 >> 4; }"
            let str2 = "int fun(){ return(2 + 3)>> 4; }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate >> before >" $ do
            let str1 = "int fun(){ return 2 > 3 >> 4 ; }"
            let str2 = "int fun(){ return 2 >(3 >> 4); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate > before ==" $ do
            let str1 = "int fun(){ return false == 3 > 4 ; }"
            let str2 = "int fun(){ return false ==(3 > 4); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate == before &" $ do
            let str1 = "int fun(){ return true & 3 == 4 ; }"
            let str2 = "int fun(){ return true &(3 == 4); }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            ast1 `shouldBe` ast2

        it "evaluate *= as *" $ do
            let str1 = "procedure fun(int a){ a *= 3; }"
            let str2 = "procedure fun(int a){ a = a * 3; }"
            ast1 <- strToAst str1
            ast2 <- strToAst str2
            (void ast1) `shouldBe` (void ast2)

