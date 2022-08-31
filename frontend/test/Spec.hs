import Test.Hspec
--import Test.QuickCheck
import Control.Exception (evaluate)
import ImportSpec
import OracleSpec

main :: IO ()
main = hspec $ do
  importSpec
  oracleSpec