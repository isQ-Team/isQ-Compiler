module ImportSpec where
import ISQ.Driver.Passes
import ISQ.Lang.CompileError
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.TypeCheck

import Control.Exception (evaluate)
import Data.Either (isRight)
import System.Directory (canonicalizePath)
import System.FilePath
import Test.Hspec

importSpec :: SpecWith ()
importSpec = do
  describe "ISQ.Driver.Passes.generateTcast" $ do
    it "returns an error for illegal package name" $ do
        let input = joinPath ["test", "input", "bad_package.isq"]
        err <- generateTcast "" input False
        err `shouldBe` (Left $ GrammarError $ BadPackageName "no_match")

    it "returns an error when compiling an unexisting file" $ do
        let input = joinPath ["unexisting", "file.isq"]
        err <- generateTcast "" input False
        fullPath <- canonicalizePath input
        err `shouldBe` (Left $ GrammarError $ ReadFileError fullPath)

    it "returns an error when importing an unexisting file" $ do
        let input = joinPath ["test", "input", "import_unexisting.isq"]
        err <- generateTcast "" input False
        err `shouldBe` (Left $ GrammarError $ ImportNotFound "unexisting.file")

    it "returns an error when importing a file more than once" $ do
        let input = joinPath ["test", "input", "duplicated_import.isq"]
        err <- generateTcast "" input False
        err `shouldBe` (Left $ GrammarError $ DuplicatedImport "b.cycle")

    it "returns an error when importing itself" $ do
        input <- canonicalizePath $ joinPath ["test", "input", "self_import.isq"]
        err <- generateTcast "" input False
        err `shouldBe` (Left $ GrammarError $ CyclicImport [input])

    it "returns an error for cyclic importing" $ do
        input <- canonicalizePath $ joinPath ["test", "input", "cyclic_import.isq"]
        importFile <- canonicalizePath $ joinPath ["test", "input", "b", "cycle.isq"]
        err <- generateTcast "" input False
        err `shouldBe` (Left $ GrammarError $ CyclicImport [importFile, input])

    it "returns an error when it is unsure about importing which file" $ do
        let input = joinPath ["test", "input", "ambiguous_import.isq"]
        let library = joinPath ["test", "input", "lib"]
        err <- generateTcast library input False
        cand1 <- canonicalizePath $ joinPath ["test", "input", "dummy.isq"]
        cand2 <- canonicalizePath $ joinPath ["test", "input", "lib", "dummy.isq"]
        err `shouldBe` (Left $ GrammarError $ AmbiguousImport "dummy" cand1 cand2)

    it "returns an error when there are multiple definations regarding a symbol" $ do
        let input = joinPath ["test", "input", "ambiguous_symbol.isq"]
        err <- generateTcast "" input False
        let isAmbiguousImport err = case err of
                Right _ -> False
                Left excpt -> case excpt of
                    TypeCheckError tc -> case tc of
                        AmbiguousSymbol _ _ _ _ -> True
                        _ -> False
                    _ -> False
        err `shouldSatisfy` isAmbiguousImport

    it "parses correctly with a naive import" $ do
        let input = joinPath ["test", "input", "naive_import.isq"]
        res <- generateTcast "" input False
        res `shouldSatisfy` isRight

    it "parses correctly when calling an imported procedure" $ do
        let input = joinPath ["test", "input", "use_imported_proc.isq"]
        res <- generateTcast "" input False
        res `shouldSatisfy` isRight

    it "parses correctly when using an imported variable" $ do
        let input = joinPath ["test", "input", "use_imported_var.isq"]
        res <- generateTcast "" input False
        res `shouldSatisfy` isRight

    it "parses correctly when shadow a local variable over an imported one" $ do
        let input = joinPath ["test", "input", "shadow_imported_var.isq"]
        res <- generateTcast "" input False
        res `shouldSatisfy` isRight

    it "parses correctly when referring a variable using its qualified name" $ do
        let input = joinPath ["test", "input", "qualified_name.isq"]
        res <- generateTcast "" input False
        res `shouldSatisfy` isRight

