import Test.Hspec
--import Test.QuickCheck
import Control.Exception (evaluate)
import ImportSpec
import OracleSpec
import PrecedenceSpec
import TypeSpec

main :: IO ()
main = hspec $ do
  importSpec
  oracleSpec
  precedenceSpec
  typeSpec