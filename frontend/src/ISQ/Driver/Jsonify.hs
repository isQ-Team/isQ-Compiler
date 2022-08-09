{-# LANGUAGE DeriveGeneric, StandaloneDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
module ISQ.Driver.Jsonify where
import ISQ.Lang.CompileError 
import GHC.Generics (Generic)
import Data.Aeson.Types (ToJSON)
import ISQ.Lang.ISQv2Grammar
import ISQ.Lang.RAIICheck
import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.TypeCheck
import Data.Complex
import ISQ.Lang.TypeCheck (TypeCheckData(TypeCheckData))
import Control.DeepSeq
import ISQ.Lang.DeriveGate 
import ISQ.Lang.OraclePass
--import ISQ.Lang.FlatInc

deriving instance Generic CompileError
instance ToJSON CompileError
deriving instance Generic InternalCompilerError
instance ToJSON InternalCompilerError
deriving instance Generic RAIIError
instance ToJSON RAIIError
deriving instance Generic DeriveError
instance ToJSON DeriveError
deriving instance Generic OracleError
instance ToJSON OracleError
--deriving instance Generic IncFileError
--instance ToJSON IncFileError
deriving instance Generic RegionType
instance ToJSON RegionType
deriving instance Generic Pos
instance ToJSON Pos
deriving instance Generic TypeCheckError
instance ToJSON TypeCheckError
deriving instance (Generic e)=>Generic (Type e)
instance (ToJSON e, Generic e)=>ToJSON (Type e)
deriving instance (Generic e)=>Generic (Expr e)
instance (ToJSON e, Generic e)=>ToJSON (Expr e)
deriving instance (Generic e)=>Generic (AST e)
instance (ToJSON e, Generic e)=>ToJSON (AST e)
deriving instance (Generic e)=>Generic (Token e)
instance (ToJSON e, Generic e)=>ToJSON (Token e)
deriving instance Generic MatchRule 
instance ToJSON MatchRule
deriving instance Generic Symbol
instance ToJSON Symbol
deriving instance Generic BuiltinType
instance ToJSON BuiltinType
deriving instance Generic GrammarError
instance ToJSON GrammarError

deriving instance Generic UnaryOperator 
instance ToJSON UnaryOperator 
deriving instance Generic BinaryOperator 
instance ToJSON BinaryOperator 
deriving instance Generic CmpType
instance ToJSON CmpType
instance ToJSON (Complex Double)
deriving instance Generic GateModifier 
instance ToJSON GateModifier 
deriving instance Generic TypeCheckData
instance ToJSON TypeCheckData

instance NFData (AST Pos)
instance NFData (Expr Pos)
instance NFData Pos
instance NFData BuiltinType 
instance NFData UnaryOperator 
instance NFData BinaryOperator 
instance NFData CmpType 
instance NFData (Type Pos)
instance NFData (Type ())
instance NFData GateModifier 
instance NFData DerivingType 
deriving instance Generic DerivingType 
instance ToJSON DerivingType 