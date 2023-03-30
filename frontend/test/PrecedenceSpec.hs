module PrecedenceSpec where
import ISQ.Driver.Passes
import ISQ.Lang.ISQv2Grammar

import Control.Monad.Except
import Control.Monad.State
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import Test.Hspec

strToAst :: String -> IO LAST
strToAst input = do
    errorOrAst <- evalStateT (runExceptT $ parseToAST "" input) $ ImportEnv MultiMap.empty Map.empty 0
    case errorOrAst of
        Left x -> error "input file error"
        Right ast -> return $ head $ defMemberList ast

precedenceSpec :: SpecWith ()
precedenceSpec = do
    describe "ISQ.paseToAST" $ do
        it "evaluate ** before *" $ do
            let str1 = "int fun(){ return 2 ** 3 * 4; }"
            let str2 = "int fun(){ return(2 ** 3)* 4; }"
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

