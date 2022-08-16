import Test.Hspec
--import Test.QuickCheck
import Control.Exception (evaluate)
import ImportSpec

main :: IO ()
main = hspec $ do
  importSpec