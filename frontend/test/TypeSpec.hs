module TypeSpec where
import ISQ.Driver.Passes
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.TypeCheck

import Control.Monad.Except
import Control.Monad.State
import Data.Aeson
import qualified Data.ByteString.Lazy as BS
import qualified Data.Map.Lazy as Map
import qualified Data.MultiMap as MultiMap
import System.IO (stdout)
import Test.Hspec

getTypeCheckError :: TypeCheckError -> String
getTypeCheckError = head . words . show

typeTestTemplate :: String -> String -> IO ()
typeTestTemplate input expect = do
    errorOrAst <- evalStateT (runExceptT $ parseToAST "" input) $ ImportEnv MultiMap.empty Map.empty 0 False []
    case errorOrAst of
        Left _ -> error "input file error"
        Right ast -> do
            let errOrRaii = compileRAII $ defMemberList ast
            case errOrRaii of
                Left _ -> error "raii error"
                Right raii -> do
                    --liftIO $ BS.hPut stdout (encode raii) -- for debug
                    let errOrTuple = typeCheckTop False "." raii MultiMap.empty 0 False
                    case errOrTuple of
                        Left err -> (getTypeCheckError err) `shouldBe` expect
                        Right _ -> error "evaluate error"

typeSpec :: SpecWith ()
typeSpec = do
    describe "ISQ.Lang.TypeCheck.typeCheckTop" $ do
        it "returns an error when using an integer as if condition" $ do
            let str = "procedure fun(){ if(1); }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when anding an integer" $ do
            let str = "procedure fun(){ print true && 1; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when oring an integer" $ do
            let str = "procedure fun(){ print true || 1; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when noting an integer" $ do
            let str = "bool fun(){ return !1; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when shifting a double" $ do
            let str = "int fun(){ return 1.2 << 3; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when adding an int and a qbit" $ do
            let str = "int fun(){ qbit q; return 1 + q; }"
            typeTestTemplate str "UnsupportedType"

        it "returns an error when increasing a qbit" $ do
            let str = "procedure fun(){ qbit q; q += 1; }"
            typeTestTemplate str "UnsupportedType"

        it "returns an error when measuring an integer" $ do
            let str = "bool fun(){ int a; return M<a>; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when a qubit appears in an array" $ do
            let str = "bool fun(){ qbit q; int a[] = {2, q}; }"
            typeTestTemplate str "UnsupportedType"

        it "returns an error when assigning a scalar to an array" $ do
            let str = "bool fun(){ int a[] = 2; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when assigning a double to an int array" $ do
            let str = "bool fun(){ int a[] = {2, 3.4}; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when using for over an int" $ do
            let str = "bool fun(){ for i in 4 {}; }"
            typeTestTemplate str "TypeMismatch"

        it "returns an error when getting the length of an int" $ do
            let str = "bool fun(){ int a; print a.length; }"
            typeTestTemplate str "TypeMismatch"
