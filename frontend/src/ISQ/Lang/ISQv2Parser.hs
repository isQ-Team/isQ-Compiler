{-# OPTIONS_GHC -w #-}
module ISQ.Lang.ISQv2Parser where

import ISQ.Lang.ISQv2Tokenizer
import ISQ.Lang.ISQv2Grammar
import Data.Maybe (catMaybes)
import qualified Data.Array as Happy_Data_Array
import qualified Data.Bits as Bits
import Control.Applicative(Applicative(..))
import Control.Monad (ap)

-- parser produced by Happy Version 1.20.0

data HappyAbsSyn 
	= HappyTerminal (ISQv2Token)
	| HappyErrorToken Prelude.Int
	| HappyAbsSyn4 ([LAST])
	| HappyAbsSyn5 ([Maybe LAST])
	| HappyAbsSyn7 (LExpr)
	| HappyAbsSyn12 (Maybe LExpr)
	| HappyAbsSyn15 ([LExpr])
	| HappyAbsSyn18 ([ISQv2Token])
	| HappyAbsSyn19 (LAST)
	| HappyAbsSyn28 ([[LExpr]])
	| HappyAbsSyn32 (GateModifier)
	| HappyAbsSyn33 ([GateModifier])
	| HappyAbsSyn40 (Maybe LAST)
	| HappyAbsSyn41 (BuiltinType)
	| HappyAbsSyn42 (LType)
	| HappyAbsSyn45 (LType->(LType, ISQv2Token, Maybe LExpr))
	| HappyAbsSyn46 ([LType->(LType, ISQv2Token, Maybe LExpr)])
	| HappyAbsSyn48 ([(LType, Ident)])
	| HappyAbsSyn50 ((LType, Ident))

{- to allow type-synonyms as our monads (likely
 - with explicitly-specified bind and return)
 - in Haskell98, it seems that with
 - /type M a = .../, then /(HappyReduction M)/
 - is not allowed.  But Happy is a
 - code-generator that can just substitute it.
type HappyReduction m = 
	   Prelude.Int 
	-> (ISQv2Token)
	-> HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> m HappyAbsSyn)
	-> [HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> m HappyAbsSyn)] 
	-> HappyStk HappyAbsSyn 
	-> [(ISQv2Token)] -> m HappyAbsSyn
-}

action_0,
 action_1,
 action_2,
 action_3,
 action_4,
 action_5,
 action_6,
 action_7,
 action_8,
 action_9,
 action_10,
 action_11,
 action_12,
 action_13,
 action_14,
 action_15,
 action_16,
 action_17,
 action_18,
 action_19,
 action_20,
 action_21,
 action_22,
 action_23,
 action_24,
 action_25,
 action_26,
 action_27,
 action_28,
 action_29,
 action_30,
 action_31,
 action_32,
 action_33,
 action_34,
 action_35,
 action_36,
 action_37,
 action_38,
 action_39,
 action_40,
 action_41,
 action_42,
 action_43,
 action_44,
 action_45,
 action_46,
 action_47,
 action_48,
 action_49,
 action_50,
 action_51,
 action_52,
 action_53,
 action_54,
 action_55,
 action_56,
 action_57,
 action_58,
 action_59,
 action_60,
 action_61,
 action_62,
 action_63,
 action_64,
 action_65,
 action_66,
 action_67,
 action_68,
 action_69,
 action_70,
 action_71,
 action_72,
 action_73,
 action_74,
 action_75,
 action_76,
 action_77,
 action_78,
 action_79,
 action_80,
 action_81,
 action_82,
 action_83,
 action_84,
 action_85,
 action_86,
 action_87,
 action_88,
 action_89,
 action_90,
 action_91,
 action_92,
 action_93,
 action_94,
 action_95,
 action_96,
 action_97,
 action_98,
 action_99,
 action_100,
 action_101,
 action_102,
 action_103,
 action_104,
 action_105,
 action_106,
 action_107,
 action_108,
 action_109,
 action_110,
 action_111,
 action_112,
 action_113,
 action_114,
 action_115,
 action_116,
 action_117,
 action_118,
 action_119,
 action_120,
 action_121,
 action_122,
 action_123,
 action_124,
 action_125,
 action_126,
 action_127,
 action_128,
 action_129,
 action_130,
 action_131,
 action_132,
 action_133,
 action_134,
 action_135,
 action_136,
 action_137,
 action_138,
 action_139,
 action_140,
 action_141,
 action_142,
 action_143,
 action_144,
 action_145,
 action_146,
 action_147,
 action_148,
 action_149,
 action_150,
 action_151,
 action_152,
 action_153,
 action_154,
 action_155,
 action_156,
 action_157,
 action_158,
 action_159,
 action_160,
 action_161,
 action_162,
 action_163,
 action_164,
 action_165,
 action_166,
 action_167,
 action_168,
 action_169,
 action_170,
 action_171,
 action_172,
 action_173,
 action_174,
 action_175,
 action_176,
 action_177,
 action_178,
 action_179,
 action_180,
 action_181,
 action_182,
 action_183,
 action_184,
 action_185,
 action_186,
 action_187,
 action_188,
 action_189,
 action_190,
 action_191,
 action_192,
 action_193,
 action_194,
 action_195,
 action_196,
 action_197,
 action_198,
 action_199,
 action_200,
 action_201,
 action_202,
 action_203,
 action_204,
 action_205,
 action_206,
 action_207,
 action_208,
 action_209,
 action_210,
 action_211,
 action_212,
 action_213,
 action_214,
 action_215,
 action_216,
 action_217,
 action_218,
 action_219,
 action_220,
 action_221,
 action_222,
 action_223,
 action_224,
 action_225,
 action_226,
 action_227,
 action_228,
 action_229,
 action_230,
 action_231 :: () => Prelude.Int -> ({-HappyReduction (HappyIdentity) = -}
	   Prelude.Int 
	-> (ISQv2Token)
	-> HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)
	-> [HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)] 
	-> HappyStk HappyAbsSyn 
	-> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)

happyReduce_1,
 happyReduce_2,
 happyReduce_3,
 happyReduce_4,
 happyReduce_5,
 happyReduce_6,
 happyReduce_7,
 happyReduce_8,
 happyReduce_9,
 happyReduce_10,
 happyReduce_11,
 happyReduce_12,
 happyReduce_13,
 happyReduce_14,
 happyReduce_15,
 happyReduce_16,
 happyReduce_17,
 happyReduce_18,
 happyReduce_19,
 happyReduce_20,
 happyReduce_21,
 happyReduce_22,
 happyReduce_23,
 happyReduce_24,
 happyReduce_25,
 happyReduce_26,
 happyReduce_27,
 happyReduce_28,
 happyReduce_29,
 happyReduce_30,
 happyReduce_31,
 happyReduce_32,
 happyReduce_33,
 happyReduce_34,
 happyReduce_35,
 happyReduce_36,
 happyReduce_37,
 happyReduce_38,
 happyReduce_39,
 happyReduce_40,
 happyReduce_41,
 happyReduce_42,
 happyReduce_43,
 happyReduce_44,
 happyReduce_45,
 happyReduce_46,
 happyReduce_47,
 happyReduce_48,
 happyReduce_49,
 happyReduce_50,
 happyReduce_51,
 happyReduce_52,
 happyReduce_53,
 happyReduce_54,
 happyReduce_55,
 happyReduce_56,
 happyReduce_57,
 happyReduce_58,
 happyReduce_59,
 happyReduce_60,
 happyReduce_61,
 happyReduce_62,
 happyReduce_63,
 happyReduce_64,
 happyReduce_65,
 happyReduce_66,
 happyReduce_67,
 happyReduce_68,
 happyReduce_69,
 happyReduce_70,
 happyReduce_71,
 happyReduce_72,
 happyReduce_73,
 happyReduce_74,
 happyReduce_75,
 happyReduce_76,
 happyReduce_77,
 happyReduce_78,
 happyReduce_79,
 happyReduce_80,
 happyReduce_81,
 happyReduce_82,
 happyReduce_83,
 happyReduce_84,
 happyReduce_85,
 happyReduce_86,
 happyReduce_87,
 happyReduce_88,
 happyReduce_89,
 happyReduce_90,
 happyReduce_91,
 happyReduce_92,
 happyReduce_93,
 happyReduce_94,
 happyReduce_95,
 happyReduce_96,
 happyReduce_97,
 happyReduce_98,
 happyReduce_99,
 happyReduce_100,
 happyReduce_101,
 happyReduce_102,
 happyReduce_103,
 happyReduce_104,
 happyReduce_105,
 happyReduce_106,
 happyReduce_107,
 happyReduce_108,
 happyReduce_109,
 happyReduce_110,
 happyReduce_111,
 happyReduce_112,
 happyReduce_113,
 happyReduce_114,
 happyReduce_115,
 happyReduce_116,
 happyReduce_117,
 happyReduce_118,
 happyReduce_119,
 happyReduce_120,
 happyReduce_121,
 happyReduce_122,
 happyReduce_123 :: () => ({-HappyReduction (HappyIdentity) = -}
	   Prelude.Int 
	-> (ISQv2Token)
	-> HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)
	-> [HappyState (ISQv2Token) (HappyStk HappyAbsSyn -> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)] 
	-> HappyStk HappyAbsSyn 
	-> [(ISQv2Token)] -> (HappyIdentity) HappyAbsSyn)

happyExpList :: Happy_Data_Array.Array Prelude.Int Prelude.Int
happyExpList = Happy_Data_Array.listArray (0,541) ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32768,42003,0,0,0,0,0,0,0,512,0,0,0,0,0,0,0,0,0,0,0,0,8192,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,256,0,0,0,0,0,0,0,0,0,0,0,16384,64,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,16384,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,8192,20480,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,32832,385,57424,1,0,0,49152,8448,0,1024,0,0,0,0,0,2048,2,0,0,0,0,0,16384,0,0,0,12288,2112,0,256,0,0,0,0,0,256,0,0,0,0,0,0,4096,0,0,0,3072,528,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,0,0,4096,0,0,0,0,0,256,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,256,0,0,0,0,0,0,0,0,0,0,0,128,256,0,0,0,0,0,0,0,0,0,0,0,0,8192,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,256,0,0,0,0,0,0,16,0,0,0,0,49152,255,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32768,256,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,6148,24,7685,0,0,0,4096,24672,5120,120,0,0,0,32832,385,57424,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,0,0,0,0,0,0,0,61440,63,0,0,0,0,0,0,132,0,0,0,0,0,32768,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,256,0,0,0,0,0,64,4,0,0,0,6148,24,7685,0,0,0,4096,24672,5120,120,0,0,0,32832,385,57424,1,0,0,0,1537,16390,1921,0,0,0,1024,6168,1280,30,0,0,0,24592,96,30740,0,0,0,16384,33152,20481,480,0,0,0,256,1542,33088,7,0,0,0,6148,24,7685,0,0,0,4096,24672,5120,120,0,0,0,32832,385,57424,1,0,0,0,1537,16390,1921,0,0,0,1024,6168,1280,30,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,192,33,0,0,0,0,0,0,0,144,0,0,0,3072,528,0,64,0,0,0,0,0,64,0,0,0,0,0,1023,0,0,0,0,0,0,2176,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,6148,24,7685,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4108,2,0,0,0,0,0,0,16384,0,0,0,0,0,0,0,0,0,0,0,0,16384,2,0,0,0,0,0,8192,0,0,0,0,0,32640,0,0,0,0,0,0,30,0,0,0,0,0,30720,0,0,0,0,0,0,480,0,0,0,0,0,32768,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32768,1,0,0,0,0,0,1536,0,0,0,0,0,0,510,0,0,0,0,0,64512,15,0,0,0,0,0,0,256,0,0,0,0,0,0,0,0,0,0,0,0,384,0,0,0,0,0,256,4,0,0,0,0,0,0,0,0,0,0,32832,385,57424,1,0,0,0,0,0,0,0,0,0,0,64512,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16384,32501,10,24,1,0,0,0,0,0,8,0,0,0,0,0,5120,0,0,0,0,0,0,128,0,0,0,0,0,1024,0,0,0,0,0,65280,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16416,0,0,0,0,0,512,1024,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,512,0,0,0,0,0,0,8,0,0,0,0,0,8192,0,0,0,0,0,0,128,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,57344,0,64,4,0,0,0,0,32768,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,8192,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,6148,24,7685,0,0,0,0,0,0,64,0,0,0,32832,385,57424,1,0,0,0,1537,16390,1921,0,0,0,0,0,0,0,0,0,0,24592,96,30740,0,0,0,0,0,8,0,0,0,0,0,8192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1024,0,0,0,8192,0,0,0,0,0,0,0,0,16384,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24592,100,30740,0,0,0,0,0,4096,256,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,4096,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16388,0,0,0,0,0,0,0,0,0,0,256,1542,33088,7,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,1,0,0,0,0,0,2048,0,0,0,0,0,16640,0,0,0,0,0,0,0,0,0,0,0,0,0,16,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,8192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,16384,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8192,0,0,0,0,0,0,0,0
	])

{-# NOINLINE happyExpListPerState #-}
happyExpListPerState st =
    token_strs_expected
  where token_strs = ["error","%dummy","%start_isqv2","TopLevel","StatementListMaybe","StatementList","Expr","ExprCallable","Expr1Left","Expr1","CallExpr","MaybeExpr1","RangeExpr","Expr2","Expr1List","Expr1ListNonEmpty","Expr1LeftListNonEmpty","IdentListNonEmpty","ForStatement","WhileStatement","IfStatement","PassStatement","DefvarStatement","LetStyleDef","CallStatement","AssignStatement","ReturnStatement","ISQCore_GatedefMatrix","ISQCore_GatedefMatrixContent","ISQCore_GatedefMatrixRow","ISQCore_GatedefStatement","GateModifier","GateModifierListNonEmpty","ISQCore_UnitaryStatement","ISQCore_MeasureExpr","ISQCore_MeasureStatement","ISQCore_ResetStatement","ISQCore_PrintStatement","StatementNonEmpty","Statement","ArrayTypeDecorator","Type","SimpleType","CompositeType","ISQCore_CStyleVarDefTerm","ISQCore_CStyleVarDefList","ISQCore_CStyleVarDef","ProcedureArgListNonEmpty","ProcedureArgList","ProcedureArg","ISQCore_CStyleArrayArg","Procedure","TopLevelVar","TopLevelMember","if","else","for","in","while","procedure","int","qbit","M","print","defgate","pass","return","ctrl","nctrl","inv","bool","true","false","let","const","unit","'|0>'","'='","'=='","'+'","'-'","'*'","'/'","'<'","'>'","'<='","'>='","'!='","'&&'","'||'","','","';'","'('","')'","'['","']'","'{'","'}'","':'","'->'","'.'","NATURAL","FLOAT","IMAGPART","IDENTIFIER","%eof"]
        bit_start = st Prelude.* 106
        bit_end = (st Prelude.+ 1) Prelude.* 106
        read_bit = readArrayBit happyExpList
        bits = Prelude.map read_bit [bit_start..bit_end Prelude.- 1]
        bits_indexed = Prelude.zip bits [0..105]
        token_strs_expected = Prelude.concatMap f bits_indexed
        f (Prelude.False, _) = []
        f (Prelude.True, nr) = [token_strs Prelude.!! nr]

action_0 (4) = happyGoto action_2
action_0 _ = happyReduce_1

action_1 _ = happyFail (happyExpListPerState 1)

action_2 (60) = happyShift action_11
action_2 (61) = happyShift action_12
action_2 (62) = happyShift action_13
action_2 (65) = happyShift action_14
action_2 (71) = happyShift action_15
action_2 (74) = happyShift action_16
action_2 (76) = happyShift action_17
action_2 (106) = happyAccept
action_2 (23) = happyGoto action_3
action_2 (24) = happyGoto action_4
action_2 (31) = happyGoto action_5
action_2 (43) = happyGoto action_6
action_2 (47) = happyGoto action_7
action_2 (52) = happyGoto action_8
action_2 (53) = happyGoto action_9
action_2 (54) = happyGoto action_10
action_2 _ = happyFail (happyExpListPerState 2)

action_3 (92) = happyShift action_26
action_3 _ = happyFail (happyExpListPerState 3)

action_4 _ = happyReduce_53

action_5 (92) = happyShift action_25
action_5 _ = happyFail (happyExpListPerState 5)

action_6 (105) = happyShift action_24
action_6 (45) = happyGoto action_22
action_6 (46) = happyGoto action_23
action_6 _ = happyFail (happyExpListPerState 6)

action_7 _ = happyReduce_54

action_8 _ = happyReduce_121

action_9 _ = happyReduce_123

action_10 _ = happyReduce_2

action_11 (105) = happyShift action_21
action_11 _ = happyFail (happyExpListPerState 11)

action_12 _ = happyReduce_97

action_13 _ = happyReduce_98

action_14 (105) = happyShift action_20
action_14 _ = happyFail (happyExpListPerState 14)

action_15 _ = happyReduce_99

action_16 (105) = happyShift action_19
action_16 (18) = happyGoto action_18
action_16 _ = happyFail (happyExpListPerState 16)

action_17 _ = happyReduce_100

action_18 (91) = happyShift action_34
action_18 (99) = happyShift action_35
action_18 _ = happyFail (happyExpListPerState 18)

action_19 _ = happyReduce_46

action_20 (78) = happyShift action_33
action_20 _ = happyFail (happyExpListPerState 20)

action_21 (93) = happyShift action_32
action_21 _ = happyFail (happyExpListPerState 21)

action_22 _ = happyReduce_106

action_23 (91) = happyShift action_31
action_23 _ = happyReduce_108

action_24 (78) = happyShift action_28
action_24 (93) = happyShift action_29
action_24 (95) = happyShift action_30
action_24 (41) = happyGoto action_27
action_24 _ = happyReduce_102

action_25 _ = happyReduce_122

action_26 _ = happyReduce_120

action_27 (78) = happyShift action_73
action_27 _ = happyReduce_103

action_28 (63) = happyShift action_62
action_28 (72) = happyShift action_63
action_28 (73) = happyShift action_64
action_28 (80) = happyShift action_65
action_28 (81) = happyShift action_66
action_28 (93) = happyShift action_67
action_28 (95) = happyShift action_68
action_28 (102) = happyShift action_69
action_28 (103) = happyShift action_70
action_28 (104) = happyShift action_71
action_28 (105) = happyShift action_72
action_28 (7) = happyGoto action_53
action_28 (8) = happyGoto action_54
action_28 (9) = happyGoto action_55
action_28 (10) = happyGoto action_56
action_28 (11) = happyGoto action_57
action_28 (12) = happyGoto action_58
action_28 (13) = happyGoto action_59
action_28 (14) = happyGoto action_60
action_28 (35) = happyGoto action_61
action_28 _ = happyReduce_35

action_29 (61) = happyShift action_12
action_29 (62) = happyShift action_13
action_29 (71) = happyShift action_15
action_29 (76) = happyShift action_17
action_29 (105) = happyShift action_47
action_29 (43) = happyGoto action_42
action_29 (48) = happyGoto action_43
action_29 (49) = happyGoto action_52
action_29 (50) = happyGoto action_45
action_29 (51) = happyGoto action_46
action_29 _ = happyReduce_112

action_30 (96) = happyShift action_50
action_30 (102) = happyShift action_51
action_30 _ = happyFail (happyExpListPerState 30)

action_31 (105) = happyShift action_49
action_31 (45) = happyGoto action_48
action_31 _ = happyFail (happyExpListPerState 31)

action_32 (61) = happyShift action_12
action_32 (62) = happyShift action_13
action_32 (71) = happyShift action_15
action_32 (76) = happyShift action_17
action_32 (105) = happyShift action_47
action_32 (43) = happyGoto action_42
action_32 (48) = happyGoto action_43
action_32 (49) = happyGoto action_44
action_32 (50) = happyGoto action_45
action_32 (51) = happyGoto action_46
action_32 _ = happyReduce_112

action_33 (95) = happyShift action_41
action_33 (28) = happyGoto action_40
action_33 _ = happyFail (happyExpListPerState 33)

action_34 (105) = happyShift action_39
action_34 _ = happyFail (happyExpListPerState 34)

action_35 (61) = happyShift action_12
action_35 (62) = happyShift action_13
action_35 (71) = happyShift action_15
action_35 (76) = happyShift action_17
action_35 (42) = happyGoto action_36
action_35 (43) = happyGoto action_37
action_35 (44) = happyGoto action_38
action_35 _ = happyFail (happyExpListPerState 35)

action_36 (95) = happyShift action_30
action_36 (41) = happyGoto action_104
action_36 _ = happyReduce_55

action_37 _ = happyReduce_95

action_38 _ = happyReduce_96

action_39 _ = happyReduce_47

action_40 _ = happyReduce_64

action_41 (63) = happyShift action_62
action_41 (72) = happyShift action_63
action_41 (73) = happyShift action_64
action_41 (80) = happyShift action_65
action_41 (81) = happyShift action_66
action_41 (93) = happyShift action_67
action_41 (95) = happyShift action_68
action_41 (102) = happyShift action_69
action_41 (103) = happyShift action_70
action_41 (104) = happyShift action_71
action_41 (105) = happyShift action_72
action_41 (8) = happyGoto action_54
action_41 (9) = happyGoto action_55
action_41 (10) = happyGoto action_101
action_41 (11) = happyGoto action_57
action_41 (29) = happyGoto action_102
action_41 (30) = happyGoto action_103
action_41 (35) = happyGoto action_61
action_41 _ = happyFail (happyExpListPerState 41)

action_42 (105) = happyShift action_100
action_42 _ = happyFail (happyExpListPerState 42)

action_43 (91) = happyShift action_99
action_43 _ = happyReduce_111

action_44 (94) = happyShift action_98
action_44 _ = happyFail (happyExpListPerState 44)

action_45 _ = happyReduce_109

action_46 _ = happyReduce_114

action_47 (99) = happyShift action_97
action_47 _ = happyFail (happyExpListPerState 47)

action_48 _ = happyReduce_107

action_49 (78) = happyShift action_28
action_49 (95) = happyShift action_30
action_49 (41) = happyGoto action_27
action_49 _ = happyReduce_102

action_50 _ = happyReduce_93

action_51 (96) = happyShift action_96
action_51 _ = happyFail (happyExpListPerState 51)

action_52 (94) = happyShift action_95
action_52 _ = happyFail (happyExpListPerState 52)

action_53 _ = happyReduce_104

action_54 (93) = happyShift action_94
action_54 _ = happyReduce_10

action_55 (95) = happyShift action_93
action_55 _ = happyReduce_12

action_56 (79) = happyShift action_83
action_56 (80) = happyShift action_84
action_56 (81) = happyShift action_85
action_56 (82) = happyShift action_86
action_56 (83) = happyShift action_87
action_56 (84) = happyShift action_88
action_56 (85) = happyShift action_89
action_56 (86) = happyShift action_90
action_56 (87) = happyShift action_91
action_56 (88) = happyShift action_92
action_56 (99) = happyReduce_34
action_56 _ = happyReduce_6

action_57 _ = happyReduce_28

action_58 (99) = happyShift action_82
action_58 _ = happyFail (happyExpListPerState 58)

action_59 _ = happyReduce_38

action_60 _ = happyReduce_7

action_61 _ = happyReduce_32

action_62 (84) = happyShift action_80
action_62 (93) = happyShift action_81
action_62 _ = happyFail (happyExpListPerState 62)

action_63 _ = happyReduce_30

action_64 _ = happyReduce_31

action_65 (63) = happyShift action_62
action_65 (72) = happyShift action_63
action_65 (73) = happyShift action_64
action_65 (80) = happyShift action_65
action_65 (81) = happyShift action_66
action_65 (93) = happyShift action_67
action_65 (95) = happyShift action_68
action_65 (102) = happyShift action_69
action_65 (103) = happyShift action_70
action_65 (104) = happyShift action_71
action_65 (105) = happyShift action_72
action_65 (8) = happyGoto action_54
action_65 (9) = happyGoto action_55
action_65 (10) = happyGoto action_79
action_65 (11) = happyGoto action_57
action_65 (35) = happyGoto action_61
action_65 _ = happyFail (happyExpListPerState 65)

action_66 (63) = happyShift action_62
action_66 (72) = happyShift action_63
action_66 (73) = happyShift action_64
action_66 (80) = happyShift action_65
action_66 (81) = happyShift action_66
action_66 (93) = happyShift action_67
action_66 (95) = happyShift action_68
action_66 (102) = happyShift action_69
action_66 (103) = happyShift action_70
action_66 (104) = happyShift action_71
action_66 (105) = happyShift action_72
action_66 (8) = happyGoto action_54
action_66 (9) = happyGoto action_55
action_66 (10) = happyGoto action_78
action_66 (11) = happyGoto action_57
action_66 (35) = happyGoto action_61
action_66 _ = happyFail (happyExpListPerState 66)

action_67 (63) = happyShift action_62
action_67 (72) = happyShift action_63
action_67 (73) = happyShift action_64
action_67 (80) = happyShift action_65
action_67 (81) = happyShift action_66
action_67 (93) = happyShift action_67
action_67 (95) = happyShift action_68
action_67 (102) = happyShift action_69
action_67 (103) = happyShift action_70
action_67 (104) = happyShift action_71
action_67 (105) = happyShift action_72
action_67 (7) = happyGoto action_77
action_67 (8) = happyGoto action_54
action_67 (9) = happyGoto action_55
action_67 (10) = happyGoto action_56
action_67 (11) = happyGoto action_57
action_67 (12) = happyGoto action_58
action_67 (13) = happyGoto action_59
action_67 (14) = happyGoto action_60
action_67 (35) = happyGoto action_61
action_67 _ = happyReduce_35

action_68 (63) = happyShift action_62
action_68 (72) = happyShift action_63
action_68 (73) = happyShift action_64
action_68 (80) = happyShift action_65
action_68 (81) = happyShift action_66
action_68 (93) = happyShift action_67
action_68 (95) = happyShift action_68
action_68 (102) = happyShift action_69
action_68 (103) = happyShift action_70
action_68 (104) = happyShift action_71
action_68 (105) = happyShift action_72
action_68 (8) = happyGoto action_54
action_68 (9) = happyGoto action_55
action_68 (10) = happyGoto action_75
action_68 (11) = happyGoto action_57
action_68 (15) = happyGoto action_76
action_68 (35) = happyGoto action_61
action_68 _ = happyReduce_41

action_69 _ = happyReduce_25

action_70 _ = happyReduce_26

action_71 _ = happyReduce_27

action_72 _ = happyReduce_9

action_73 (63) = happyShift action_62
action_73 (72) = happyShift action_63
action_73 (73) = happyShift action_64
action_73 (80) = happyShift action_65
action_73 (81) = happyShift action_66
action_73 (93) = happyShift action_67
action_73 (95) = happyShift action_68
action_73 (102) = happyShift action_69
action_73 (103) = happyShift action_70
action_73 (104) = happyShift action_71
action_73 (105) = happyShift action_72
action_73 (7) = happyGoto action_74
action_73 (8) = happyGoto action_54
action_73 (9) = happyGoto action_55
action_73 (10) = happyGoto action_56
action_73 (11) = happyGoto action_57
action_73 (12) = happyGoto action_58
action_73 (13) = happyGoto action_59
action_73 (14) = happyGoto action_60
action_73 (35) = happyGoto action_61
action_73 _ = happyReduce_35

action_74 _ = happyReduce_105

action_75 (79) = happyShift action_83
action_75 (80) = happyShift action_84
action_75 (81) = happyShift action_85
action_75 (82) = happyShift action_86
action_75 (83) = happyShift action_87
action_75 (84) = happyShift action_88
action_75 (85) = happyShift action_89
action_75 (86) = happyShift action_90
action_75 (87) = happyShift action_91
action_75 (88) = happyShift action_92
action_75 _ = happyReduce_39

action_76 (91) = happyShift action_132
action_76 (96) = happyShift action_133
action_76 _ = happyFail (happyExpListPerState 76)

action_77 (94) = happyShift action_131
action_77 _ = happyFail (happyExpListPerState 77)

action_78 _ = happyReduce_23

action_79 _ = happyReduce_24

action_80 (93) = happyShift action_67
action_80 (105) = happyShift action_72
action_80 (8) = happyGoto action_128
action_80 (9) = happyGoto action_130
action_80 _ = happyFail (happyExpListPerState 80)

action_81 (93) = happyShift action_67
action_81 (105) = happyShift action_72
action_81 (8) = happyGoto action_128
action_81 (9) = happyGoto action_129
action_81 _ = happyFail (happyExpListPerState 81)

action_82 (63) = happyShift action_62
action_82 (72) = happyShift action_63
action_82 (73) = happyShift action_64
action_82 (80) = happyShift action_65
action_82 (81) = happyShift action_66
action_82 (93) = happyShift action_67
action_82 (95) = happyShift action_68
action_82 (102) = happyShift action_69
action_82 (103) = happyShift action_70
action_82 (104) = happyShift action_71
action_82 (105) = happyShift action_72
action_82 (8) = happyGoto action_54
action_82 (9) = happyGoto action_55
action_82 (10) = happyGoto action_126
action_82 (11) = happyGoto action_57
action_82 (12) = happyGoto action_127
action_82 (35) = happyGoto action_61
action_82 _ = happyReduce_35

action_83 (63) = happyShift action_62
action_83 (72) = happyShift action_63
action_83 (73) = happyShift action_64
action_83 (80) = happyShift action_65
action_83 (81) = happyShift action_66
action_83 (93) = happyShift action_67
action_83 (95) = happyShift action_68
action_83 (102) = happyShift action_69
action_83 (103) = happyShift action_70
action_83 (104) = happyShift action_71
action_83 (105) = happyShift action_72
action_83 (8) = happyGoto action_54
action_83 (9) = happyGoto action_55
action_83 (10) = happyGoto action_125
action_83 (11) = happyGoto action_57
action_83 (35) = happyGoto action_61
action_83 _ = happyFail (happyExpListPerState 83)

action_84 (63) = happyShift action_62
action_84 (72) = happyShift action_63
action_84 (73) = happyShift action_64
action_84 (80) = happyShift action_65
action_84 (81) = happyShift action_66
action_84 (93) = happyShift action_67
action_84 (95) = happyShift action_68
action_84 (102) = happyShift action_69
action_84 (103) = happyShift action_70
action_84 (104) = happyShift action_71
action_84 (105) = happyShift action_72
action_84 (8) = happyGoto action_54
action_84 (9) = happyGoto action_55
action_84 (10) = happyGoto action_124
action_84 (11) = happyGoto action_57
action_84 (35) = happyGoto action_61
action_84 _ = happyFail (happyExpListPerState 84)

action_85 (63) = happyShift action_62
action_85 (72) = happyShift action_63
action_85 (73) = happyShift action_64
action_85 (80) = happyShift action_65
action_85 (81) = happyShift action_66
action_85 (93) = happyShift action_67
action_85 (95) = happyShift action_68
action_85 (102) = happyShift action_69
action_85 (103) = happyShift action_70
action_85 (104) = happyShift action_71
action_85 (105) = happyShift action_72
action_85 (8) = happyGoto action_54
action_85 (9) = happyGoto action_55
action_85 (10) = happyGoto action_123
action_85 (11) = happyGoto action_57
action_85 (35) = happyGoto action_61
action_85 _ = happyFail (happyExpListPerState 85)

action_86 (63) = happyShift action_62
action_86 (72) = happyShift action_63
action_86 (73) = happyShift action_64
action_86 (80) = happyShift action_65
action_86 (81) = happyShift action_66
action_86 (93) = happyShift action_67
action_86 (95) = happyShift action_68
action_86 (102) = happyShift action_69
action_86 (103) = happyShift action_70
action_86 (104) = happyShift action_71
action_86 (105) = happyShift action_72
action_86 (8) = happyGoto action_54
action_86 (9) = happyGoto action_55
action_86 (10) = happyGoto action_122
action_86 (11) = happyGoto action_57
action_86 (35) = happyGoto action_61
action_86 _ = happyFail (happyExpListPerState 86)

action_87 (63) = happyShift action_62
action_87 (72) = happyShift action_63
action_87 (73) = happyShift action_64
action_87 (80) = happyShift action_65
action_87 (81) = happyShift action_66
action_87 (93) = happyShift action_67
action_87 (95) = happyShift action_68
action_87 (102) = happyShift action_69
action_87 (103) = happyShift action_70
action_87 (104) = happyShift action_71
action_87 (105) = happyShift action_72
action_87 (8) = happyGoto action_54
action_87 (9) = happyGoto action_55
action_87 (10) = happyGoto action_121
action_87 (11) = happyGoto action_57
action_87 (35) = happyGoto action_61
action_87 _ = happyFail (happyExpListPerState 87)

action_88 (63) = happyShift action_62
action_88 (72) = happyShift action_63
action_88 (73) = happyShift action_64
action_88 (80) = happyShift action_65
action_88 (81) = happyShift action_66
action_88 (93) = happyShift action_67
action_88 (95) = happyShift action_68
action_88 (102) = happyShift action_69
action_88 (103) = happyShift action_70
action_88 (104) = happyShift action_71
action_88 (105) = happyShift action_72
action_88 (8) = happyGoto action_54
action_88 (9) = happyGoto action_55
action_88 (10) = happyGoto action_120
action_88 (11) = happyGoto action_57
action_88 (35) = happyGoto action_61
action_88 _ = happyFail (happyExpListPerState 88)

action_89 (63) = happyShift action_62
action_89 (72) = happyShift action_63
action_89 (73) = happyShift action_64
action_89 (80) = happyShift action_65
action_89 (81) = happyShift action_66
action_89 (93) = happyShift action_67
action_89 (95) = happyShift action_68
action_89 (102) = happyShift action_69
action_89 (103) = happyShift action_70
action_89 (104) = happyShift action_71
action_89 (105) = happyShift action_72
action_89 (8) = happyGoto action_54
action_89 (9) = happyGoto action_55
action_89 (10) = happyGoto action_119
action_89 (11) = happyGoto action_57
action_89 (35) = happyGoto action_61
action_89 _ = happyFail (happyExpListPerState 89)

action_90 (63) = happyShift action_62
action_90 (72) = happyShift action_63
action_90 (73) = happyShift action_64
action_90 (80) = happyShift action_65
action_90 (81) = happyShift action_66
action_90 (93) = happyShift action_67
action_90 (95) = happyShift action_68
action_90 (102) = happyShift action_69
action_90 (103) = happyShift action_70
action_90 (104) = happyShift action_71
action_90 (105) = happyShift action_72
action_90 (8) = happyGoto action_54
action_90 (9) = happyGoto action_55
action_90 (10) = happyGoto action_118
action_90 (11) = happyGoto action_57
action_90 (35) = happyGoto action_61
action_90 _ = happyFail (happyExpListPerState 90)

action_91 (63) = happyShift action_62
action_91 (72) = happyShift action_63
action_91 (73) = happyShift action_64
action_91 (80) = happyShift action_65
action_91 (81) = happyShift action_66
action_91 (93) = happyShift action_67
action_91 (95) = happyShift action_68
action_91 (102) = happyShift action_69
action_91 (103) = happyShift action_70
action_91 (104) = happyShift action_71
action_91 (105) = happyShift action_72
action_91 (8) = happyGoto action_54
action_91 (9) = happyGoto action_55
action_91 (10) = happyGoto action_117
action_91 (11) = happyGoto action_57
action_91 (35) = happyGoto action_61
action_91 _ = happyFail (happyExpListPerState 91)

action_92 (63) = happyShift action_62
action_92 (72) = happyShift action_63
action_92 (73) = happyShift action_64
action_92 (80) = happyShift action_65
action_92 (81) = happyShift action_66
action_92 (93) = happyShift action_67
action_92 (95) = happyShift action_68
action_92 (102) = happyShift action_69
action_92 (103) = happyShift action_70
action_92 (104) = happyShift action_71
action_92 (105) = happyShift action_72
action_92 (8) = happyGoto action_54
action_92 (9) = happyGoto action_55
action_92 (10) = happyGoto action_116
action_92 (11) = happyGoto action_57
action_92 (35) = happyGoto action_61
action_92 _ = happyFail (happyExpListPerState 92)

action_93 (63) = happyShift action_62
action_93 (72) = happyShift action_63
action_93 (73) = happyShift action_64
action_93 (80) = happyShift action_65
action_93 (81) = happyShift action_66
action_93 (93) = happyShift action_67
action_93 (95) = happyShift action_68
action_93 (102) = happyShift action_69
action_93 (103) = happyShift action_70
action_93 (104) = happyShift action_71
action_93 (105) = happyShift action_72
action_93 (7) = happyGoto action_115
action_93 (8) = happyGoto action_54
action_93 (9) = happyGoto action_55
action_93 (10) = happyGoto action_56
action_93 (11) = happyGoto action_57
action_93 (12) = happyGoto action_58
action_93 (13) = happyGoto action_59
action_93 (14) = happyGoto action_60
action_93 (35) = happyGoto action_61
action_93 _ = happyReduce_35

action_94 (63) = happyShift action_62
action_94 (72) = happyShift action_63
action_94 (73) = happyShift action_64
action_94 (80) = happyShift action_65
action_94 (81) = happyShift action_66
action_94 (93) = happyShift action_67
action_94 (95) = happyShift action_68
action_94 (102) = happyShift action_69
action_94 (103) = happyShift action_70
action_94 (104) = happyShift action_71
action_94 (105) = happyShift action_72
action_94 (8) = happyGoto action_54
action_94 (9) = happyGoto action_55
action_94 (10) = happyGoto action_75
action_94 (11) = happyGoto action_57
action_94 (15) = happyGoto action_114
action_94 (35) = happyGoto action_61
action_94 _ = happyReduce_41

action_95 (97) = happyShift action_113
action_95 _ = happyFail (happyExpListPerState 95)

action_96 _ = happyReduce_94

action_97 (61) = happyShift action_12
action_97 (62) = happyShift action_13
action_97 (71) = happyShift action_15
action_97 (76) = happyShift action_17
action_97 (42) = happyGoto action_112
action_97 (43) = happyGoto action_37
action_97 (44) = happyGoto action_38
action_97 _ = happyFail (happyExpListPerState 97)

action_98 (97) = happyShift action_110
action_98 (100) = happyShift action_111
action_98 _ = happyFail (happyExpListPerState 98)

action_99 (61) = happyShift action_12
action_99 (62) = happyShift action_13
action_99 (71) = happyShift action_15
action_99 (76) = happyShift action_17
action_99 (105) = happyShift action_47
action_99 (43) = happyGoto action_42
action_99 (50) = happyGoto action_109
action_99 (51) = happyGoto action_46
action_99 _ = happyFail (happyExpListPerState 99)

action_100 (95) = happyShift action_30
action_100 (41) = happyGoto action_108
action_100 _ = happyReduce_113

action_101 (79) = happyShift action_83
action_101 (80) = happyShift action_84
action_101 (81) = happyShift action_85
action_101 (82) = happyShift action_86
action_101 (83) = happyShift action_87
action_101 (84) = happyShift action_88
action_101 (85) = happyShift action_89
action_101 (86) = happyShift action_90
action_101 (87) = happyShift action_91
action_101 (88) = happyShift action_92
action_101 _ = happyReduce_62

action_102 (92) = happyShift action_106
action_102 (96) = happyShift action_107
action_102 _ = happyFail (happyExpListPerState 102)

action_103 (91) = happyShift action_105
action_103 _ = happyReduce_60

action_104 _ = happyReduce_101

action_105 (63) = happyShift action_62
action_105 (72) = happyShift action_63
action_105 (73) = happyShift action_64
action_105 (80) = happyShift action_65
action_105 (81) = happyShift action_66
action_105 (93) = happyShift action_67
action_105 (95) = happyShift action_68
action_105 (102) = happyShift action_69
action_105 (103) = happyShift action_70
action_105 (104) = happyShift action_71
action_105 (105) = happyShift action_72
action_105 (8) = happyGoto action_54
action_105 (9) = happyGoto action_55
action_105 (10) = happyGoto action_145
action_105 (11) = happyGoto action_57
action_105 (35) = happyGoto action_61
action_105 _ = happyFail (happyExpListPerState 105)

action_106 (63) = happyShift action_62
action_106 (72) = happyShift action_63
action_106 (73) = happyShift action_64
action_106 (80) = happyShift action_65
action_106 (81) = happyShift action_66
action_106 (93) = happyShift action_67
action_106 (95) = happyShift action_68
action_106 (102) = happyShift action_69
action_106 (103) = happyShift action_70
action_106 (104) = happyShift action_71
action_106 (105) = happyShift action_72
action_106 (8) = happyGoto action_54
action_106 (9) = happyGoto action_55
action_106 (10) = happyGoto action_101
action_106 (11) = happyGoto action_57
action_106 (30) = happyGoto action_144
action_106 (35) = happyGoto action_61
action_106 _ = happyFail (happyExpListPerState 106)

action_107 _ = happyReduce_59

action_108 _ = happyReduce_116

action_109 _ = happyReduce_110

action_110 (5) = happyGoto action_140
action_110 (6) = happyGoto action_143
action_110 _ = happyReduce_4

action_111 (61) = happyShift action_12
action_111 (62) = happyShift action_13
action_111 (71) = happyShift action_15
action_111 (76) = happyShift action_17
action_111 (42) = happyGoto action_142
action_111 (43) = happyGoto action_37
action_111 (44) = happyGoto action_38
action_111 _ = happyFail (happyExpListPerState 111)

action_112 (95) = happyShift action_30
action_112 (41) = happyGoto action_104
action_112 _ = happyReduce_115

action_113 (5) = happyGoto action_140
action_113 (6) = happyGoto action_141
action_113 _ = happyReduce_4

action_114 (91) = happyShift action_132
action_114 (94) = happyShift action_139
action_114 _ = happyFail (happyExpListPerState 114)

action_115 (96) = happyShift action_138
action_115 _ = happyFail (happyExpListPerState 115)

action_116 (80) = happyShift action_84
action_116 (81) = happyShift action_85
action_116 (82) = happyShift action_86
action_116 (83) = happyShift action_87
action_116 (84) = happyShift action_88
action_116 (85) = happyShift action_89
action_116 (86) = happyShift action_90
action_116 (87) = happyShift action_91
action_116 _ = happyReduce_18

action_117 (80) = happyShift action_84
action_117 (81) = happyShift action_85
action_117 (82) = happyShift action_86
action_117 (83) = happyShift action_87
action_117 (84) = happyFail []
action_117 (85) = happyFail []
action_117 (86) = happyFail []
action_117 (87) = happyFail []
action_117 _ = happyReduce_21

action_118 (80) = happyShift action_84
action_118 (81) = happyShift action_85
action_118 (82) = happyShift action_86
action_118 (83) = happyShift action_87
action_118 (84) = happyFail []
action_118 (85) = happyFail []
action_118 (86) = happyFail []
action_118 (87) = happyFail []
action_118 _ = happyReduce_22

action_119 (80) = happyShift action_84
action_119 (81) = happyShift action_85
action_119 (82) = happyShift action_86
action_119 (83) = happyShift action_87
action_119 (84) = happyFail []
action_119 (85) = happyFail []
action_119 (86) = happyFail []
action_119 (87) = happyFail []
action_119 _ = happyReduce_19

action_120 (80) = happyShift action_84
action_120 (81) = happyShift action_85
action_120 (82) = happyShift action_86
action_120 (83) = happyShift action_87
action_120 (84) = happyFail []
action_120 (85) = happyFail []
action_120 (86) = happyFail []
action_120 (87) = happyFail []
action_120 _ = happyReduce_20

action_121 _ = happyReduce_16

action_122 _ = happyReduce_15

action_123 (82) = happyShift action_86
action_123 (83) = happyShift action_87
action_123 _ = happyReduce_14

action_124 (82) = happyShift action_86
action_124 (83) = happyShift action_87
action_124 _ = happyReduce_13

action_125 (80) = happyShift action_84
action_125 (81) = happyShift action_85
action_125 (82) = happyShift action_86
action_125 (83) = happyShift action_87
action_125 (84) = happyShift action_88
action_125 (85) = happyShift action_89
action_125 (86) = happyShift action_90
action_125 (87) = happyShift action_91
action_125 _ = happyReduce_17

action_126 (79) = happyShift action_83
action_126 (80) = happyShift action_84
action_126 (81) = happyShift action_85
action_126 (82) = happyShift action_86
action_126 (83) = happyShift action_87
action_126 (84) = happyShift action_88
action_126 (85) = happyShift action_89
action_126 (86) = happyShift action_90
action_126 (87) = happyShift action_91
action_126 (88) = happyShift action_92
action_126 _ = happyReduce_34

action_127 (99) = happyShift action_137
action_127 _ = happyReduce_37

action_128 _ = happyReduce_10

action_129 (94) = happyShift action_136
action_129 (95) = happyShift action_93
action_129 _ = happyFail (happyExpListPerState 129)

action_130 (85) = happyShift action_135
action_130 (95) = happyShift action_93
action_130 _ = happyFail (happyExpListPerState 130)

action_131 _ = happyReduce_8

action_132 (63) = happyShift action_62
action_132 (72) = happyShift action_63
action_132 (73) = happyShift action_64
action_132 (80) = happyShift action_65
action_132 (81) = happyShift action_66
action_132 (93) = happyShift action_67
action_132 (95) = happyShift action_68
action_132 (102) = happyShift action_69
action_132 (103) = happyShift action_70
action_132 (104) = happyShift action_71
action_132 (105) = happyShift action_72
action_132 (8) = happyGoto action_54
action_132 (9) = happyGoto action_55
action_132 (10) = happyGoto action_134
action_132 (11) = happyGoto action_57
action_132 (35) = happyGoto action_61
action_132 _ = happyFail (happyExpListPerState 132)

action_133 _ = happyReduce_29

action_134 (79) = happyShift action_83
action_134 (80) = happyShift action_84
action_134 (81) = happyShift action_85
action_134 (82) = happyShift action_86
action_134 (83) = happyShift action_87
action_134 (84) = happyShift action_88
action_134 (85) = happyShift action_89
action_134 (86) = happyShift action_90
action_134 (87) = happyShift action_91
action_134 (88) = happyShift action_92
action_134 _ = happyReduce_40

action_135 _ = happyReduce_74

action_136 _ = happyReduce_75

action_137 (63) = happyShift action_62
action_137 (72) = happyShift action_63
action_137 (73) = happyShift action_64
action_137 (80) = happyShift action_65
action_137 (81) = happyShift action_66
action_137 (93) = happyShift action_67
action_137 (95) = happyShift action_68
action_137 (102) = happyShift action_69
action_137 (103) = happyShift action_70
action_137 (104) = happyShift action_71
action_137 (105) = happyShift action_72
action_137 (8) = happyGoto action_54
action_137 (9) = happyGoto action_55
action_137 (10) = happyGoto action_126
action_137 (11) = happyGoto action_57
action_137 (12) = happyGoto action_180
action_137 (35) = happyGoto action_61
action_137 _ = happyReduce_35

action_138 _ = happyReduce_11

action_139 _ = happyReduce_33

action_140 (55) = happyShift action_170
action_140 (57) = happyShift action_171
action_140 (59) = happyShift action_172
action_140 (61) = happyShift action_12
action_140 (62) = happyShift action_13
action_140 (63) = happyShift action_62
action_140 (64) = happyShift action_173
action_140 (66) = happyShift action_174
action_140 (67) = happyShift action_175
action_140 (68) = happyShift action_176
action_140 (69) = happyShift action_177
action_140 (70) = happyShift action_178
action_140 (71) = happyShift action_15
action_140 (74) = happyShift action_16
action_140 (76) = happyShift action_17
action_140 (92) = happyShift action_179
action_140 (93) = happyShift action_67
action_140 (105) = happyShift action_72
action_140 (8) = happyGoto action_149
action_140 (9) = happyGoto action_150
action_140 (11) = happyGoto action_151
action_140 (19) = happyGoto action_152
action_140 (20) = happyGoto action_153
action_140 (21) = happyGoto action_154
action_140 (22) = happyGoto action_155
action_140 (23) = happyGoto action_156
action_140 (24) = happyGoto action_4
action_140 (25) = happyGoto action_157
action_140 (26) = happyGoto action_158
action_140 (27) = happyGoto action_159
action_140 (32) = happyGoto action_160
action_140 (33) = happyGoto action_161
action_140 (34) = happyGoto action_162
action_140 (35) = happyGoto action_163
action_140 (36) = happyGoto action_164
action_140 (37) = happyGoto action_165
action_140 (38) = happyGoto action_166
action_140 (39) = happyGoto action_167
action_140 (40) = happyGoto action_168
action_140 (43) = happyGoto action_169
action_140 (47) = happyGoto action_7
action_140 _ = happyReduce_5

action_141 (98) = happyShift action_148
action_141 _ = happyFail (happyExpListPerState 141)

action_142 (95) = happyShift action_30
action_142 (97) = happyShift action_147
action_142 (41) = happyGoto action_104
action_142 _ = happyFail (happyExpListPerState 142)

action_143 (98) = happyShift action_146
action_143 _ = happyFail (happyExpListPerState 143)

action_144 (91) = happyShift action_105
action_144 _ = happyReduce_61

action_145 (79) = happyShift action_83
action_145 (80) = happyShift action_84
action_145 (81) = happyShift action_85
action_145 (82) = happyShift action_86
action_145 (83) = happyShift action_87
action_145 (84) = happyShift action_88
action_145 (85) = happyShift action_89
action_145 (86) = happyShift action_90
action_145 (87) = happyShift action_91
action_145 (88) = happyShift action_92
action_145 _ = happyReduce_63

action_146 _ = happyReduce_117

action_147 (5) = happyGoto action_140
action_147 (6) = happyGoto action_201
action_147 _ = happyReduce_4

action_148 _ = happyReduce_119

action_149 (84) = happyShift action_200
action_149 (93) = happyShift action_94
action_149 _ = happyReduce_10

action_150 (78) = happyShift action_199
action_150 (95) = happyShift action_93
action_150 _ = happyFail (happyExpListPerState 150)

action_151 _ = happyReduce_56

action_152 _ = happyReduce_81

action_153 _ = happyReduce_82

action_154 _ = happyReduce_80

action_155 (92) = happyShift action_198
action_155 _ = happyFail (happyExpListPerState 155)

action_156 (92) = happyShift action_197
action_156 _ = happyFail (happyExpListPerState 156)

action_157 (92) = happyShift action_196
action_157 _ = happyFail (happyExpListPerState 157)

action_158 (92) = happyShift action_195
action_158 _ = happyFail (happyExpListPerState 158)

action_159 (92) = happyShift action_194
action_159 _ = happyFail (happyExpListPerState 159)

action_160 _ = happyReduce_71

action_161 (68) = happyShift action_176
action_161 (69) = happyShift action_177
action_161 (70) = happyShift action_178
action_161 (93) = happyShift action_67
action_161 (105) = happyShift action_72
action_161 (8) = happyGoto action_192
action_161 (32) = happyGoto action_193
action_161 _ = happyFail (happyExpListPerState 161)

action_162 (92) = happyShift action_191
action_162 _ = happyFail (happyExpListPerState 162)

action_163 _ = happyReduce_76

action_164 (92) = happyShift action_190
action_164 _ = happyFail (happyExpListPerState 164)

action_165 (92) = happyShift action_189
action_165 _ = happyFail (happyExpListPerState 165)

action_166 (92) = happyShift action_188
action_166 _ = happyFail (happyExpListPerState 166)

action_167 _ = happyReduce_91

action_168 _ = happyReduce_3

action_169 (105) = happyShift action_49
action_169 (45) = happyGoto action_22
action_169 (46) = happyGoto action_23
action_169 _ = happyFail (happyExpListPerState 169)

action_170 (63) = happyShift action_62
action_170 (72) = happyShift action_63
action_170 (73) = happyShift action_64
action_170 (80) = happyShift action_65
action_170 (81) = happyShift action_66
action_170 (93) = happyShift action_67
action_170 (95) = happyShift action_68
action_170 (102) = happyShift action_69
action_170 (103) = happyShift action_70
action_170 (104) = happyShift action_71
action_170 (105) = happyShift action_72
action_170 (7) = happyGoto action_187
action_170 (8) = happyGoto action_54
action_170 (9) = happyGoto action_55
action_170 (10) = happyGoto action_56
action_170 (11) = happyGoto action_57
action_170 (12) = happyGoto action_58
action_170 (13) = happyGoto action_59
action_170 (14) = happyGoto action_60
action_170 (35) = happyGoto action_61
action_170 _ = happyReduce_35

action_171 (105) = happyShift action_186
action_171 _ = happyFail (happyExpListPerState 171)

action_172 (63) = happyShift action_62
action_172 (72) = happyShift action_63
action_172 (73) = happyShift action_64
action_172 (80) = happyShift action_65
action_172 (81) = happyShift action_66
action_172 (93) = happyShift action_67
action_172 (95) = happyShift action_68
action_172 (102) = happyShift action_69
action_172 (103) = happyShift action_70
action_172 (104) = happyShift action_71
action_172 (105) = happyShift action_72
action_172 (7) = happyGoto action_185
action_172 (8) = happyGoto action_54
action_172 (9) = happyGoto action_55
action_172 (10) = happyGoto action_56
action_172 (11) = happyGoto action_57
action_172 (12) = happyGoto action_58
action_172 (13) = happyGoto action_59
action_172 (14) = happyGoto action_60
action_172 (35) = happyGoto action_61
action_172 _ = happyReduce_35

action_173 (63) = happyShift action_62
action_173 (72) = happyShift action_63
action_173 (73) = happyShift action_64
action_173 (80) = happyShift action_65
action_173 (81) = happyShift action_66
action_173 (93) = happyShift action_67
action_173 (95) = happyShift action_68
action_173 (102) = happyShift action_69
action_173 (103) = happyShift action_70
action_173 (104) = happyShift action_71
action_173 (105) = happyShift action_72
action_173 (7) = happyGoto action_184
action_173 (8) = happyGoto action_54
action_173 (9) = happyGoto action_55
action_173 (10) = happyGoto action_56
action_173 (11) = happyGoto action_57
action_173 (12) = happyGoto action_58
action_173 (13) = happyGoto action_59
action_173 (14) = happyGoto action_60
action_173 (35) = happyGoto action_61
action_173 _ = happyReduce_35

action_174 _ = happyReduce_52

action_175 (63) = happyShift action_62
action_175 (72) = happyShift action_63
action_175 (73) = happyShift action_64
action_175 (80) = happyShift action_65
action_175 (81) = happyShift action_66
action_175 (93) = happyShift action_67
action_175 (95) = happyShift action_68
action_175 (102) = happyShift action_69
action_175 (103) = happyShift action_70
action_175 (104) = happyShift action_71
action_175 (105) = happyShift action_72
action_175 (7) = happyGoto action_183
action_175 (8) = happyGoto action_54
action_175 (9) = happyGoto action_55
action_175 (10) = happyGoto action_56
action_175 (11) = happyGoto action_57
action_175 (12) = happyGoto action_58
action_175 (13) = happyGoto action_59
action_175 (14) = happyGoto action_60
action_175 (35) = happyGoto action_61
action_175 _ = happyReduce_35

action_176 (84) = happyShift action_182
action_176 _ = happyReduce_68

action_177 (84) = happyShift action_181
action_177 _ = happyReduce_69

action_178 _ = happyReduce_65

action_179 _ = happyReduce_92

action_180 _ = happyReduce_36

action_181 (102) = happyShift action_212
action_181 _ = happyFail (happyExpListPerState 181)

action_182 (102) = happyShift action_211
action_182 _ = happyFail (happyExpListPerState 182)

action_183 _ = happyReduce_58

action_184 _ = happyReduce_78

action_185 (97) = happyShift action_210
action_185 _ = happyFail (happyExpListPerState 185)

action_186 (58) = happyShift action_209
action_186 _ = happyFail (happyExpListPerState 186)

action_187 (97) = happyShift action_208
action_187 _ = happyFail (happyExpListPerState 187)

action_188 _ = happyReduce_90

action_189 _ = happyReduce_89

action_190 _ = happyReduce_88

action_191 _ = happyReduce_87

action_192 (84) = happyShift action_207
action_192 _ = happyFail (happyExpListPerState 192)

action_193 _ = happyReduce_70

action_194 _ = happyReduce_86

action_195 _ = happyReduce_85

action_196 _ = happyReduce_84

action_197 _ = happyReduce_83

action_198 _ = happyReduce_79

action_199 (63) = happyShift action_62
action_199 (72) = happyShift action_63
action_199 (73) = happyShift action_64
action_199 (77) = happyShift action_206
action_199 (80) = happyShift action_65
action_199 (81) = happyShift action_66
action_199 (93) = happyShift action_67
action_199 (95) = happyShift action_68
action_199 (102) = happyShift action_69
action_199 (103) = happyShift action_70
action_199 (104) = happyShift action_71
action_199 (105) = happyShift action_72
action_199 (7) = happyGoto action_205
action_199 (8) = happyGoto action_54
action_199 (9) = happyGoto action_55
action_199 (10) = happyGoto action_56
action_199 (11) = happyGoto action_57
action_199 (12) = happyGoto action_58
action_199 (13) = happyGoto action_59
action_199 (14) = happyGoto action_60
action_199 (35) = happyGoto action_61
action_199 _ = happyReduce_35

action_200 (93) = happyShift action_67
action_200 (105) = happyShift action_72
action_200 (8) = happyGoto action_128
action_200 (9) = happyGoto action_203
action_200 (17) = happyGoto action_204
action_200 _ = happyFail (happyExpListPerState 200)

action_201 (98) = happyShift action_202
action_201 _ = happyFail (happyExpListPerState 201)

action_202 _ = happyReduce_118

action_203 (95) = happyShift action_93
action_203 _ = happyReduce_44

action_204 (85) = happyShift action_219
action_204 (91) = happyShift action_220
action_204 _ = happyFail (happyExpListPerState 204)

action_205 _ = happyReduce_57

action_206 _ = happyReduce_77

action_207 (93) = happyShift action_67
action_207 (105) = happyShift action_72
action_207 (8) = happyGoto action_128
action_207 (9) = happyGoto action_203
action_207 (17) = happyGoto action_218
action_207 _ = happyFail (happyExpListPerState 207)

action_208 (5) = happyGoto action_140
action_208 (6) = happyGoto action_217
action_208 _ = happyReduce_4

action_209 (63) = happyShift action_62
action_209 (72) = happyShift action_63
action_209 (73) = happyShift action_64
action_209 (80) = happyShift action_65
action_209 (81) = happyShift action_66
action_209 (93) = happyShift action_67
action_209 (95) = happyShift action_68
action_209 (102) = happyShift action_69
action_209 (103) = happyShift action_70
action_209 (104) = happyShift action_71
action_209 (105) = happyShift action_72
action_209 (8) = happyGoto action_54
action_209 (9) = happyGoto action_55
action_209 (10) = happyGoto action_126
action_209 (11) = happyGoto action_57
action_209 (12) = happyGoto action_58
action_209 (13) = happyGoto action_216
action_209 (35) = happyGoto action_61
action_209 _ = happyReduce_35

action_210 (5) = happyGoto action_140
action_210 (6) = happyGoto action_215
action_210 _ = happyReduce_4

action_211 (85) = happyShift action_214
action_211 _ = happyFail (happyExpListPerState 211)

action_212 (85) = happyShift action_213
action_212 _ = happyFail (happyExpListPerState 212)

action_213 _ = happyReduce_67

action_214 _ = happyReduce_66

action_215 (98) = happyShift action_225
action_215 _ = happyFail (happyExpListPerState 215)

action_216 (97) = happyShift action_224
action_216 _ = happyFail (happyExpListPerState 216)

action_217 (98) = happyShift action_223
action_217 _ = happyFail (happyExpListPerState 217)

action_218 (85) = happyShift action_222
action_218 (91) = happyShift action_220
action_218 _ = happyFail (happyExpListPerState 218)

action_219 _ = happyReduce_72

action_220 (93) = happyShift action_67
action_220 (105) = happyShift action_72
action_220 (8) = happyGoto action_128
action_220 (9) = happyGoto action_221
action_220 _ = happyFail (happyExpListPerState 220)

action_221 (95) = happyShift action_93
action_221 _ = happyReduce_45

action_222 _ = happyReduce_73

action_223 (56) = happyShift action_227
action_223 _ = happyReduce_50

action_224 (5) = happyGoto action_140
action_224 (6) = happyGoto action_226
action_224 _ = happyReduce_4

action_225 _ = happyReduce_49

action_226 (98) = happyShift action_229
action_226 _ = happyFail (happyExpListPerState 226)

action_227 (97) = happyShift action_228
action_227 _ = happyFail (happyExpListPerState 227)

action_228 (5) = happyGoto action_140
action_228 (6) = happyGoto action_230
action_228 _ = happyReduce_4

action_229 _ = happyReduce_48

action_230 (98) = happyShift action_231
action_230 _ = happyFail (happyExpListPerState 230)

action_231 _ = happyReduce_51

happyReduce_1 = happySpecReduce_0  4 happyReduction_1
happyReduction_1  =  HappyAbsSyn4
		 ([]
	)

happyReduce_2 = happySpecReduce_2  4 happyReduction_2
happyReduction_2 (HappyAbsSyn19  happy_var_2)
	(HappyAbsSyn4  happy_var_1)
	 =  HappyAbsSyn4
		 (happy_var_1 ++ [happy_var_2]
	)
happyReduction_2 _ _  = notHappyAtAll 

happyReduce_3 = happySpecReduce_2  5 happyReduction_3
happyReduction_3 (HappyAbsSyn40  happy_var_2)
	(HappyAbsSyn5  happy_var_1)
	 =  HappyAbsSyn5
		 (happy_var_1 ++ [happy_var_2]
	)
happyReduction_3 _ _  = notHappyAtAll 

happyReduce_4 = happySpecReduce_0  5 happyReduction_4
happyReduction_4  =  HappyAbsSyn5
		 ([]
	)

happyReduce_5 = happySpecReduce_1  6 happyReduction_5
happyReduction_5 (HappyAbsSyn5  happy_var_1)
	 =  HappyAbsSyn4
		 (catMaybes happy_var_1
	)
happyReduction_5 _  = notHappyAtAll 

happyReduce_6 = happySpecReduce_1  7 happyReduction_6
happyReduction_6 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_6 _  = notHappyAtAll 

happyReduce_7 = happySpecReduce_1  7 happyReduction_7
happyReduction_7 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_7 _  = notHappyAtAll 

happyReduce_8 = happySpecReduce_3  8 happyReduction_8
happyReduction_8 _
	(HappyAbsSyn7  happy_var_2)
	_
	 =  HappyAbsSyn7
		 (happy_var_2
	)
happyReduction_8 _ _ _  = notHappyAtAll 

happyReduce_9 = happySpecReduce_1  8 happyReduction_9
happyReduction_9 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn7
		 (EIdent (annotation happy_var_1) (tokenIdentV happy_var_1)
	)
happyReduction_9 _  = notHappyAtAll 

happyReduce_10 = happySpecReduce_1  9 happyReduction_10
happyReduction_10 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_10 _  = notHappyAtAll 

happyReduce_11 = happyReduce 4 9 happyReduction_11
happyReduction_11 (_ `HappyStk`
	(HappyAbsSyn7  happy_var_3) `HappyStk`
	(HappyTerminal (TokenReservedOp happy_var_2 "[")) `HappyStk`
	(HappyAbsSyn7  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn7
		 (ESubscript happy_var_2 happy_var_1 happy_var_3
	) `HappyStk` happyRest

happyReduce_12 = happySpecReduce_1  10 happyReduction_12
happyReduction_12 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_12 _  = notHappyAtAll 

happyReduce_13 = happySpecReduce_3  10 happyReduction_13
happyReduction_13 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "+"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 Add happy_var_1 happy_var_3
	)
happyReduction_13 _ _ _  = notHappyAtAll 

happyReduce_14 = happySpecReduce_3  10 happyReduction_14
happyReduction_14 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "-"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 Sub happy_var_1 happy_var_3
	)
happyReduction_14 _ _ _  = notHappyAtAll 

happyReduce_15 = happySpecReduce_3  10 happyReduction_15
happyReduction_15 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "*"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 Mul happy_var_1 happy_var_3
	)
happyReduction_15 _ _ _  = notHappyAtAll 

happyReduce_16 = happySpecReduce_3  10 happyReduction_16
happyReduction_16 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "/"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 Div happy_var_1 happy_var_3
	)
happyReduction_16 _ _ _  = notHappyAtAll 

happyReduce_17 = happySpecReduce_3  10 happyReduction_17
happyReduction_17 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "=="))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp Equal) happy_var_1 happy_var_3
	)
happyReduction_17 _ _ _  = notHappyAtAll 

happyReduce_18 = happySpecReduce_3  10 happyReduction_18
happyReduction_18 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "!="))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp NEqual) happy_var_1 happy_var_3
	)
happyReduction_18 _ _ _  = notHappyAtAll 

happyReduce_19 = happySpecReduce_3  10 happyReduction_19
happyReduction_19 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 ">"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp Greater) happy_var_1 happy_var_3
	)
happyReduction_19 _ _ _  = notHappyAtAll 

happyReduce_20 = happySpecReduce_3  10 happyReduction_20
happyReduction_20 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "<"))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp Less) happy_var_1 happy_var_3
	)
happyReduction_20 _ _ _  = notHappyAtAll 

happyReduce_21 = happySpecReduce_3  10 happyReduction_21
happyReduction_21 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 ">="))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp GreaterEq) happy_var_1 happy_var_3
	)
happyReduction_21 _ _ _  = notHappyAtAll 

happyReduce_22 = happySpecReduce_3  10 happyReduction_22
happyReduction_22 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "<="))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (EBinary happy_var_2 (Cmp LessEq) happy_var_1 happy_var_3
	)
happyReduction_22 _ _ _  = notHappyAtAll 

happyReduce_23 = happySpecReduce_2  10 happyReduction_23
happyReduction_23 (HappyAbsSyn7  happy_var_2)
	(HappyTerminal (TokenReservedOp happy_var_1 "-"))
	 =  HappyAbsSyn7
		 (EUnary happy_var_1 Neg happy_var_2
	)
happyReduction_23 _ _  = notHappyAtAll 

happyReduce_24 = happySpecReduce_2  10 happyReduction_24
happyReduction_24 (HappyAbsSyn7  happy_var_2)
	(HappyTerminal (TokenReservedOp happy_var_1 "+"))
	 =  HappyAbsSyn7
		 (EUnary happy_var_1 Positive happy_var_2
	)
happyReduction_24 _ _  = notHappyAtAll 

happyReduce_25 = happySpecReduce_1  10 happyReduction_25
happyReduction_25 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn7
		 (EIntLit (annotation happy_var_1) (tokenNaturalV happy_var_1)
	)
happyReduction_25 _  = notHappyAtAll 

happyReduce_26 = happySpecReduce_1  10 happyReduction_26
happyReduction_26 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn7
		 (EFloatingLit (annotation happy_var_1) (tokenFloatV happy_var_1)
	)
happyReduction_26 _  = notHappyAtAll 

happyReduce_27 = happySpecReduce_1  10 happyReduction_27
happyReduction_27 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn7
		 (EImagLit (annotation happy_var_1) (tokenImagPartV happy_var_1)
	)
happyReduction_27 _  = notHappyAtAll 

happyReduce_28 = happySpecReduce_1  10 happyReduction_28
happyReduction_28 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_28 _  = notHappyAtAll 

happyReduce_29 = happySpecReduce_3  10 happyReduction_29
happyReduction_29 _
	(HappyAbsSyn15  happy_var_2)
	(HappyTerminal (TokenReservedOp happy_var_1 "["))
	 =  HappyAbsSyn7
		 (EList happy_var_1 happy_var_2
	)
happyReduction_29 _ _ _  = notHappyAtAll 

happyReduce_30 = happySpecReduce_1  10 happyReduction_30
happyReduction_30 (HappyTerminal (TokenReservedId happy_var_1 "true"))
	 =  HappyAbsSyn7
		 (EBoolLit happy_var_1 True
	)
happyReduction_30 _  = notHappyAtAll 

happyReduce_31 = happySpecReduce_1  10 happyReduction_31
happyReduction_31 (HappyTerminal (TokenReservedId happy_var_1 "false"))
	 =  HappyAbsSyn7
		 (EBoolLit happy_var_1 False
	)
happyReduction_31 _  = notHappyAtAll 

happyReduce_32 = happySpecReduce_1  10 happyReduction_32
happyReduction_32 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_32 _  = notHappyAtAll 

happyReduce_33 = happyReduce 4 11 happyReduction_33
happyReduction_33 (_ `HappyStk`
	(HappyAbsSyn15  happy_var_3) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn7
		 (ECall (annotation happy_var_1) happy_var_1 happy_var_3
	) `HappyStk` happyRest

happyReduce_34 = happySpecReduce_1  12 happyReduction_34
happyReduction_34 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn12
		 (Just happy_var_1
	)
happyReduction_34 _  = notHappyAtAll 

happyReduce_35 = happySpecReduce_0  12 happyReduction_35
happyReduction_35  =  HappyAbsSyn12
		 (Nothing
	)

happyReduce_36 = happyReduce 5 13 happyReduction_36
happyReduction_36 ((HappyAbsSyn12  happy_var_5) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn12  happy_var_3) `HappyStk`
	(HappyTerminal (TokenReservedOp happy_var_2 ":")) `HappyStk`
	(HappyAbsSyn12  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn7
		 (ERange happy_var_2 happy_var_1 happy_var_3 happy_var_5
	) `HappyStk` happyRest

happyReduce_37 = happySpecReduce_3  13 happyReduction_37
happyReduction_37 (HappyAbsSyn12  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 ":"))
	(HappyAbsSyn12  happy_var_1)
	 =  HappyAbsSyn7
		 (ERange happy_var_2 happy_var_1 happy_var_3 Nothing
	)
happyReduction_37 _ _ _  = notHappyAtAll 

happyReduce_38 = happySpecReduce_1  14 happyReduction_38
happyReduction_38 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn7
		 (happy_var_1
	)
happyReduction_38 _  = notHappyAtAll 

happyReduce_39 = happySpecReduce_1  15 happyReduction_39
happyReduction_39 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn15
		 ([happy_var_1]
	)
happyReduction_39 _  = notHappyAtAll 

happyReduce_40 = happySpecReduce_3  15 happyReduction_40
happyReduction_40 (HappyAbsSyn7  happy_var_3)
	_
	(HappyAbsSyn15  happy_var_1)
	 =  HappyAbsSyn15
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_40 _ _ _  = notHappyAtAll 

happyReduce_41 = happySpecReduce_0  15 happyReduction_41
happyReduction_41  =  HappyAbsSyn15
		 ([]
	)

happyReduce_42 = happySpecReduce_1  16 happyReduction_42
happyReduction_42 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn15
		 ([happy_var_1]
	)
happyReduction_42 _  = notHappyAtAll 

happyReduce_43 = happySpecReduce_3  16 happyReduction_43
happyReduction_43 (HappyAbsSyn7  happy_var_3)
	_
	(HappyAbsSyn15  happy_var_1)
	 =  HappyAbsSyn15
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_43 _ _ _  = notHappyAtAll 

happyReduce_44 = happySpecReduce_1  17 happyReduction_44
happyReduction_44 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn15
		 ([happy_var_1]
	)
happyReduction_44 _  = notHappyAtAll 

happyReduce_45 = happySpecReduce_3  17 happyReduction_45
happyReduction_45 (HappyAbsSyn7  happy_var_3)
	_
	(HappyAbsSyn15  happy_var_1)
	 =  HappyAbsSyn15
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_45 _ _ _  = notHappyAtAll 

happyReduce_46 = happySpecReduce_1  18 happyReduction_46
happyReduction_46 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn18
		 ([happy_var_1]
	)
happyReduction_46 _  = notHappyAtAll 

happyReduce_47 = happySpecReduce_3  18 happyReduction_47
happyReduction_47 (HappyTerminal happy_var_3)
	_
	(HappyAbsSyn18  happy_var_1)
	 =  HappyAbsSyn18
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_47 _ _ _  = notHappyAtAll 

happyReduce_48 = happyReduce 7 19 happyReduction_48
happyReduction_48 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_6) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyTerminal happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "for")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NFor happy_var_1 (tokenIdentV happy_var_2) happy_var_4 happy_var_6
	) `HappyStk` happyRest

happyReduce_49 = happyReduce 5 20 happyReduction_49
happyReduction_49 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "while")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NWhile happy_var_1 happy_var_2 happy_var_4
	) `HappyStk` happyRest

happyReduce_50 = happyReduce 5 21 happyReduction_50
happyReduction_50 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "if")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NIf happy_var_1 happy_var_2 happy_var_4 []
	) `HappyStk` happyRest

happyReduce_51 = happyReduce 9 21 happyReduction_51
happyReduction_51 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_8) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn4  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "if")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NIf happy_var_1 happy_var_2 happy_var_4 happy_var_8
	) `HappyStk` happyRest

happyReduce_52 = happySpecReduce_1  22 happyReduction_52
happyReduction_52 (HappyTerminal (TokenReservedId happy_var_1 "pass"))
	 =  HappyAbsSyn19
		 (NPass happy_var_1
	)
happyReduction_52 _  = notHappyAtAll 

happyReduce_53 = happySpecReduce_1  23 happyReduction_53
happyReduction_53 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_53 _  = notHappyAtAll 

happyReduce_54 = happySpecReduce_1  23 happyReduction_54
happyReduction_54 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_54 _  = notHappyAtAll 

happyReduce_55 = happyReduce 4 24 happyReduction_55
happyReduction_55 ((HappyAbsSyn42  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn18  happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "let")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NDefvar happy_var_1 (fmap (\x->(happy_var_4, tokenIdentV x, Nothing)) happy_var_2)
	) `HappyStk` happyRest

happyReduce_56 = happySpecReduce_1  25 happyReduction_56
happyReduction_56 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn19
		 (NCall (annotation happy_var_1) happy_var_1
	)
happyReduction_56 _  = notHappyAtAll 

happyReduce_57 = happySpecReduce_3  26 happyReduction_57
happyReduction_57 (HappyAbsSyn7  happy_var_3)
	(HappyTerminal (TokenReservedOp happy_var_2 "="))
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn19
		 (NAssign happy_var_2 happy_var_1 happy_var_3
	)
happyReduction_57 _ _ _  = notHappyAtAll 

happyReduce_58 = happySpecReduce_2  27 happyReduction_58
happyReduction_58 (HappyAbsSyn7  happy_var_2)
	(HappyTerminal (TokenReservedId happy_var_1 "return"))
	 =  HappyAbsSyn19
		 (NReturn happy_var_1 happy_var_2
	)
happyReduction_58 _ _  = notHappyAtAll 

happyReduce_59 = happySpecReduce_3  28 happyReduction_59
happyReduction_59 _
	(HappyAbsSyn28  happy_var_2)
	_
	 =  HappyAbsSyn28
		 (happy_var_2
	)
happyReduction_59 _ _ _  = notHappyAtAll 

happyReduce_60 = happySpecReduce_1  29 happyReduction_60
happyReduction_60 (HappyAbsSyn15  happy_var_1)
	 =  HappyAbsSyn28
		 ([happy_var_1]
	)
happyReduction_60 _  = notHappyAtAll 

happyReduce_61 = happySpecReduce_3  29 happyReduction_61
happyReduction_61 (HappyAbsSyn15  happy_var_3)
	_
	(HappyAbsSyn28  happy_var_1)
	 =  HappyAbsSyn28
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_61 _ _ _  = notHappyAtAll 

happyReduce_62 = happySpecReduce_1  30 happyReduction_62
happyReduction_62 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn15
		 ([happy_var_1]
	)
happyReduction_62 _  = notHappyAtAll 

happyReduce_63 = happySpecReduce_3  30 happyReduction_63
happyReduction_63 (HappyAbsSyn7  happy_var_3)
	_
	(HappyAbsSyn15  happy_var_1)
	 =  HappyAbsSyn15
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_63 _ _ _  = notHappyAtAll 

happyReduce_64 = happyReduce 4 31 happyReduction_64
happyReduction_64 ((HappyAbsSyn28  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyTerminal happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "defgate")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NGatedef happy_var_1 (tokenIdentV happy_var_2) happy_var_4
	) `HappyStk` happyRest

happyReduce_65 = happySpecReduce_1  32 happyReduction_65
happyReduction_65 _
	 =  HappyAbsSyn32
		 (Inv
	)

happyReduce_66 = happyReduce 4 32 happyReduction_66
happyReduction_66 (_ `HappyStk`
	(HappyTerminal happy_var_3) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	happyRest)
	 = HappyAbsSyn32
		 (Ctrl True (tokenNaturalV happy_var_3)
	) `HappyStk` happyRest

happyReduce_67 = happyReduce 4 32 happyReduction_67
happyReduction_67 (_ `HappyStk`
	(HappyTerminal happy_var_3) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	happyRest)
	 = HappyAbsSyn32
		 (Ctrl False (tokenNaturalV happy_var_3)
	) `HappyStk` happyRest

happyReduce_68 = happySpecReduce_1  32 happyReduction_68
happyReduction_68 _
	 =  HappyAbsSyn32
		 (Ctrl True 1
	)

happyReduce_69 = happySpecReduce_1  32 happyReduction_69
happyReduction_69 _
	 =  HappyAbsSyn32
		 (Ctrl False 1
	)

happyReduce_70 = happySpecReduce_2  33 happyReduction_70
happyReduction_70 (HappyAbsSyn32  happy_var_2)
	(HappyAbsSyn33  happy_var_1)
	 =  HappyAbsSyn33
		 (happy_var_1 ++ [happy_var_2]
	)
happyReduction_70 _ _  = notHappyAtAll 

happyReduce_71 = happySpecReduce_1  33 happyReduction_71
happyReduction_71 (HappyAbsSyn32  happy_var_1)
	 =  HappyAbsSyn33
		 ([happy_var_1]
	)
happyReduction_71 _  = notHappyAtAll 

happyReduce_72 = happyReduce 4 34 happyReduction_72
happyReduction_72 (_ `HappyStk`
	(HappyAbsSyn15  happy_var_3) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NCoreUnitary (annotation happy_var_1) happy_var_1 happy_var_3 []
	) `HappyStk` happyRest

happyReduce_73 = happyReduce 5 34 happyReduction_73
happyReduction_73 (_ `HappyStk`
	(HappyAbsSyn15  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn7  happy_var_2) `HappyStk`
	(HappyAbsSyn33  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NCoreUnitary (annotation happy_var_2) happy_var_2 happy_var_4 happy_var_1
	) `HappyStk` happyRest

happyReduce_74 = happyReduce 4 35 happyReduction_74
happyReduction_74 (_ `HappyStk`
	(HappyAbsSyn7  happy_var_3) `HappyStk`
	_ `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "M")) `HappyStk`
	happyRest)
	 = HappyAbsSyn7
		 (ECoreMeasure happy_var_1 happy_var_3
	) `HappyStk` happyRest

happyReduce_75 = happyReduce 4 35 happyReduction_75
happyReduction_75 (_ `HappyStk`
	(HappyAbsSyn7  happy_var_3) `HappyStk`
	_ `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "M")) `HappyStk`
	happyRest)
	 = HappyAbsSyn7
		 (ECoreMeasure happy_var_1 happy_var_3
	) `HappyStk` happyRest

happyReduce_76 = happySpecReduce_1  36 happyReduction_76
happyReduction_76 (HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn19
		 (NCoreMeasure (annotation happy_var_1) happy_var_1
	)
happyReduction_76 _  = notHappyAtAll 

happyReduce_77 = happySpecReduce_3  37 happyReduction_77
happyReduction_77 _
	_
	(HappyAbsSyn7  happy_var_1)
	 =  HappyAbsSyn19
		 (NCoreReset (annotation happy_var_1) happy_var_1
	)
happyReduction_77 _ _ _  = notHappyAtAll 

happyReduce_78 = happySpecReduce_2  38 happyReduction_78
happyReduction_78 (HappyAbsSyn7  happy_var_2)
	(HappyTerminal (TokenReservedId happy_var_1 "print"))
	 =  HappyAbsSyn19
		 (NCorePrint happy_var_1 happy_var_2
	)
happyReduction_78 _ _  = notHappyAtAll 

happyReduce_79 = happySpecReduce_2  39 happyReduction_79
happyReduction_79 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_79 _ _  = notHappyAtAll 

happyReduce_80 = happySpecReduce_1  39 happyReduction_80
happyReduction_80 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_80 _  = notHappyAtAll 

happyReduce_81 = happySpecReduce_1  39 happyReduction_81
happyReduction_81 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_81 _  = notHappyAtAll 

happyReduce_82 = happySpecReduce_1  39 happyReduction_82
happyReduction_82 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_82 _  = notHappyAtAll 

happyReduce_83 = happySpecReduce_2  39 happyReduction_83
happyReduction_83 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_83 _ _  = notHappyAtAll 

happyReduce_84 = happySpecReduce_2  39 happyReduction_84
happyReduction_84 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_84 _ _  = notHappyAtAll 

happyReduce_85 = happySpecReduce_2  39 happyReduction_85
happyReduction_85 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_85 _ _  = notHappyAtAll 

happyReduce_86 = happySpecReduce_2  39 happyReduction_86
happyReduction_86 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_86 _ _  = notHappyAtAll 

happyReduce_87 = happySpecReduce_2  39 happyReduction_87
happyReduction_87 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_87 _ _  = notHappyAtAll 

happyReduce_88 = happySpecReduce_2  39 happyReduction_88
happyReduction_88 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_88 _ _  = notHappyAtAll 

happyReduce_89 = happySpecReduce_2  39 happyReduction_89
happyReduction_89 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_89 _ _  = notHappyAtAll 

happyReduce_90 = happySpecReduce_2  39 happyReduction_90
happyReduction_90 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_90 _ _  = notHappyAtAll 

happyReduce_91 = happySpecReduce_1  40 happyReduction_91
happyReduction_91 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn40
		 (Just happy_var_1
	)
happyReduction_91 _  = notHappyAtAll 

happyReduce_92 = happySpecReduce_1  40 happyReduction_92
happyReduction_92 _
	 =  HappyAbsSyn40
		 (Nothing
	)

happyReduce_93 = happySpecReduce_2  41 happyReduction_93
happyReduction_93 _
	_
	 =  HappyAbsSyn41
		 (UnknownArray
	)

happyReduce_94 = happySpecReduce_3  41 happyReduction_94
happyReduction_94 _
	(HappyTerminal happy_var_2)
	_
	 =  HappyAbsSyn41
		 (FixedArray (tokenNaturalV happy_var_2)
	)
happyReduction_94 _ _ _  = notHappyAtAll 

happyReduce_95 = happySpecReduce_1  42 happyReduction_95
happyReduction_95 (HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn42
		 (happy_var_1
	)
happyReduction_95 _  = notHappyAtAll 

happyReduce_96 = happySpecReduce_1  42 happyReduction_96
happyReduction_96 (HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn42
		 (happy_var_1
	)
happyReduction_96 _  = notHappyAtAll 

happyReduce_97 = happySpecReduce_1  43 happyReduction_97
happyReduction_97 (HappyTerminal (TokenReservedId happy_var_1 "int"))
	 =  HappyAbsSyn42
		 (intType happy_var_1
	)
happyReduction_97 _  = notHappyAtAll 

happyReduce_98 = happySpecReduce_1  43 happyReduction_98
happyReduction_98 (HappyTerminal (TokenReservedId happy_var_1 "qbit"))
	 =  HappyAbsSyn42
		 (qbitType happy_var_1
	)
happyReduction_98 _  = notHappyAtAll 

happyReduce_99 = happySpecReduce_1  43 happyReduction_99
happyReduction_99 (HappyTerminal (TokenReservedId happy_var_1 "bool"))
	 =  HappyAbsSyn42
		 (boolType happy_var_1
	)
happyReduction_99 _  = notHappyAtAll 

happyReduce_100 = happySpecReduce_1  43 happyReduction_100
happyReduction_100 (HappyTerminal (TokenReservedId happy_var_1 "unit"))
	 =  HappyAbsSyn42
		 (unitType happy_var_1
	)
happyReduction_100 _  = notHappyAtAll 

happyReduce_101 = happySpecReduce_2  44 happyReduction_101
happyReduction_101 (HappyAbsSyn41  happy_var_2)
	(HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn42
		 (Type (annotation happy_var_1) happy_var_2 [happy_var_1]
	)
happyReduction_101 _ _  = notHappyAtAll 

happyReduce_102 = happySpecReduce_1  45 happyReduction_102
happyReduction_102 (HappyTerminal happy_var_1)
	 =  HappyAbsSyn45
		 (\t->(t, happy_var_1, Nothing)
	)
happyReduction_102 _  = notHappyAtAll 

happyReduce_103 = happySpecReduce_2  45 happyReduction_103
happyReduction_103 (HappyAbsSyn41  happy_var_2)
	(HappyTerminal happy_var_1)
	 =  HappyAbsSyn45
		 (\t->(Type (annotation happy_var_1) happy_var_2 [t], happy_var_1, Nothing)
	)
happyReduction_103 _ _  = notHappyAtAll 

happyReduce_104 = happySpecReduce_3  45 happyReduction_104
happyReduction_104 (HappyAbsSyn7  happy_var_3)
	_
	(HappyTerminal happy_var_1)
	 =  HappyAbsSyn45
		 (\t->(t, happy_var_1, Just happy_var_3)
	)
happyReduction_104 _ _ _  = notHappyAtAll 

happyReduce_105 = happyReduce 4 45 happyReduction_105
happyReduction_105 ((HappyAbsSyn7  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn41  happy_var_2) `HappyStk`
	(HappyTerminal happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn45
		 (\t->(Type (annotation happy_var_1) happy_var_2 [t], happy_var_1, Just happy_var_4)
	) `HappyStk` happyRest

happyReduce_106 = happySpecReduce_1  46 happyReduction_106
happyReduction_106 (HappyAbsSyn45  happy_var_1)
	 =  HappyAbsSyn46
		 ([happy_var_1]
	)
happyReduction_106 _  = notHappyAtAll 

happyReduce_107 = happySpecReduce_3  46 happyReduction_107
happyReduction_107 (HappyAbsSyn45  happy_var_3)
	_
	(HappyAbsSyn46  happy_var_1)
	 =  HappyAbsSyn46
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_107 _ _ _  = notHappyAtAll 

happyReduce_108 = happySpecReduce_2  47 happyReduction_108
happyReduction_108 (HappyAbsSyn46  happy_var_2)
	(HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn19
		 (let args = fmap (\f->let (a, b, c) = f happy_var_1 in (a, tokenIdentV b, c)) happy_var_2 in NDefvar (annotation happy_var_1) args
	)
happyReduction_108 _ _  = notHappyAtAll 

happyReduce_109 = happySpecReduce_1  48 happyReduction_109
happyReduction_109 (HappyAbsSyn50  happy_var_1)
	 =  HappyAbsSyn48
		 ([happy_var_1]
	)
happyReduction_109 _  = notHappyAtAll 

happyReduce_110 = happySpecReduce_3  48 happyReduction_110
happyReduction_110 (HappyAbsSyn50  happy_var_3)
	_
	(HappyAbsSyn48  happy_var_1)
	 =  HappyAbsSyn48
		 (happy_var_1 ++ [happy_var_3]
	)
happyReduction_110 _ _ _  = notHappyAtAll 

happyReduce_111 = happySpecReduce_1  49 happyReduction_111
happyReduction_111 (HappyAbsSyn48  happy_var_1)
	 =  HappyAbsSyn48
		 (happy_var_1
	)
happyReduction_111 _  = notHappyAtAll 

happyReduce_112 = happySpecReduce_0  49 happyReduction_112
happyReduction_112  =  HappyAbsSyn48
		 ([]
	)

happyReduce_113 = happySpecReduce_2  50 happyReduction_113
happyReduction_113 (HappyTerminal happy_var_2)
	(HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn50
		 ((happy_var_1, tokenIdentV happy_var_2)
	)
happyReduction_113 _ _  = notHappyAtAll 

happyReduce_114 = happySpecReduce_1  50 happyReduction_114
happyReduction_114 (HappyAbsSyn50  happy_var_1)
	 =  HappyAbsSyn50
		 (happy_var_1
	)
happyReduction_114 _  = notHappyAtAll 

happyReduce_115 = happySpecReduce_3  50 happyReduction_115
happyReduction_115 (HappyAbsSyn42  happy_var_3)
	_
	(HappyTerminal happy_var_1)
	 =  HappyAbsSyn50
		 ((happy_var_3, tokenIdentV happy_var_1)
	)
happyReduction_115 _ _ _  = notHappyAtAll 

happyReduce_116 = happySpecReduce_3  51 happyReduction_116
happyReduction_116 (HappyAbsSyn41  happy_var_3)
	(HappyTerminal happy_var_2)
	(HappyAbsSyn42  happy_var_1)
	 =  HappyAbsSyn50
		 ((Type (annotation happy_var_1) happy_var_3 [happy_var_1], tokenIdentV happy_var_2)
	)
happyReduction_116 _ _ _  = notHappyAtAll 

happyReduce_117 = happyReduce 8 52 happyReduction_117
happyReduction_117 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_7) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn48  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyTerminal happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "procedure")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NProcedure happy_var_1 (unitType happy_var_1) (tokenIdentV happy_var_2) happy_var_4 happy_var_7
	) `HappyStk` happyRest

happyReduce_118 = happyReduce 10 52 happyReduction_118
happyReduction_118 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_9) `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn42  happy_var_7) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn48  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyTerminal happy_var_2) `HappyStk`
	(HappyTerminal (TokenReservedId happy_var_1 "procedure")) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NProcedure happy_var_1 happy_var_7 (tokenIdentV happy_var_2) happy_var_4 happy_var_9
	) `HappyStk` happyRest

happyReduce_119 = happyReduce 8 52 happyReduction_119
happyReduction_119 (_ `HappyStk`
	(HappyAbsSyn4  happy_var_7) `HappyStk`
	_ `HappyStk`
	_ `HappyStk`
	(HappyAbsSyn48  happy_var_4) `HappyStk`
	_ `HappyStk`
	(HappyTerminal happy_var_2) `HappyStk`
	(HappyAbsSyn42  happy_var_1) `HappyStk`
	happyRest)
	 = HappyAbsSyn19
		 (NProcedure (annotation happy_var_1) happy_var_1 (tokenIdentV happy_var_2) happy_var_4 happy_var_7
	) `HappyStk` happyRest

happyReduce_120 = happySpecReduce_2  53 happyReduction_120
happyReduction_120 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_120 _ _  = notHappyAtAll 

happyReduce_121 = happySpecReduce_1  54 happyReduction_121
happyReduction_121 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_121 _  = notHappyAtAll 

happyReduce_122 = happySpecReduce_2  54 happyReduction_122
happyReduction_122 _
	(HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_122 _ _  = notHappyAtAll 

happyReduce_123 = happySpecReduce_1  54 happyReduction_123
happyReduction_123 (HappyAbsSyn19  happy_var_1)
	 =  HappyAbsSyn19
		 (happy_var_1
	)
happyReduction_123 _  = notHappyAtAll 

happyNewToken action sts stk [] =
	action 106 106 notHappyAtAll (HappyState action) sts stk []

happyNewToken action sts stk (tk:tks) =
	let cont i = action i i tk (HappyState action) sts stk tks in
	case tk of {
	TokenReservedId happy_dollar_dollar "if" -> cont 55;
	TokenReservedId happy_dollar_dollar "else" -> cont 56;
	TokenReservedId happy_dollar_dollar "for" -> cont 57;
	TokenReservedId happy_dollar_dollar "in" -> cont 58;
	TokenReservedId happy_dollar_dollar "while" -> cont 59;
	TokenReservedId happy_dollar_dollar "procedure" -> cont 60;
	TokenReservedId happy_dollar_dollar "int" -> cont 61;
	TokenReservedId happy_dollar_dollar "qbit" -> cont 62;
	TokenReservedId happy_dollar_dollar "M" -> cont 63;
	TokenReservedId happy_dollar_dollar "print" -> cont 64;
	TokenReservedId happy_dollar_dollar "defgate" -> cont 65;
	TokenReservedId happy_dollar_dollar "pass" -> cont 66;
	TokenReservedId happy_dollar_dollar "return" -> cont 67;
	TokenReservedId happy_dollar_dollar "ctrl" -> cont 68;
	TokenReservedId happy_dollar_dollar "nctrl" -> cont 69;
	TokenReservedId happy_dollar_dollar "inv" -> cont 70;
	TokenReservedId happy_dollar_dollar "bool" -> cont 71;
	TokenReservedId happy_dollar_dollar "true" -> cont 72;
	TokenReservedId happy_dollar_dollar "false" -> cont 73;
	TokenReservedId happy_dollar_dollar "let" -> cont 74;
	TokenReservedId happy_dollar_dollar "const" -> cont 75;
	TokenReservedId happy_dollar_dollar "unit" -> cont 76;
	TokenReservedOp happy_dollar_dollar "|0>" -> cont 77;
	TokenReservedOp happy_dollar_dollar "=" -> cont 78;
	TokenReservedOp happy_dollar_dollar "==" -> cont 79;
	TokenReservedOp happy_dollar_dollar "+" -> cont 80;
	TokenReservedOp happy_dollar_dollar "-" -> cont 81;
	TokenReservedOp happy_dollar_dollar "*" -> cont 82;
	TokenReservedOp happy_dollar_dollar "/" -> cont 83;
	TokenReservedOp happy_dollar_dollar "<" -> cont 84;
	TokenReservedOp happy_dollar_dollar ">" -> cont 85;
	TokenReservedOp happy_dollar_dollar "<=" -> cont 86;
	TokenReservedOp happy_dollar_dollar ">=" -> cont 87;
	TokenReservedOp happy_dollar_dollar "!=" -> cont 88;
	TokenReservedOp happy_dollar_dollar "&&" -> cont 89;
	TokenReservedOp happy_dollar_dollar "||" -> cont 90;
	TokenReservedOp happy_dollar_dollar "," -> cont 91;
	TokenReservedOp happy_dollar_dollar ";" -> cont 92;
	TokenReservedOp happy_dollar_dollar "(" -> cont 93;
	TokenReservedOp happy_dollar_dollar ")" -> cont 94;
	TokenReservedOp happy_dollar_dollar "[" -> cont 95;
	TokenReservedOp happy_dollar_dollar "]" -> cont 96;
	TokenReservedOp happy_dollar_dollar "{" -> cont 97;
	TokenReservedOp happy_dollar_dollar "}" -> cont 98;
	TokenReservedOp happy_dollar_dollar ":" -> cont 99;
	TokenReservedOp happy_dollar_dollar "->" -> cont 100;
	TokenReservedOp happy_dollar_dollar "." -> cont 101;
	TokenNatural _ _ -> cont 102;
	TokenFloat _ _ -> cont 103;
	TokenImagPart _ _ -> cont 104;
	TokenIdent _ _ -> cont 105;
	_ -> happyError' ((tk:tks), [])
	}

happyError_ explist 106 tk tks = happyError' (tks, explist)
happyError_ explist _ tk tks = happyError' ((tk:tks), explist)

newtype HappyIdentity a = HappyIdentity a
happyIdentity = HappyIdentity
happyRunIdentity (HappyIdentity a) = a

instance Prelude.Functor HappyIdentity where
    fmap f (HappyIdentity a) = HappyIdentity (f a)

instance Applicative HappyIdentity where
    pure  = HappyIdentity
    (<*>) = ap
instance Prelude.Monad HappyIdentity where
    return = pure
    (HappyIdentity p) >>= q = q p

happyThen :: () => HappyIdentity a -> (a -> HappyIdentity b) -> HappyIdentity b
happyThen = (Prelude.>>=)
happyReturn :: () => a -> HappyIdentity a
happyReturn = (Prelude.return)
happyThen1 m k tks = (Prelude.>>=) m (\a -> k a tks)
happyReturn1 :: () => a -> b -> HappyIdentity a
happyReturn1 = \a tks -> (Prelude.return) a
happyError' :: () => ([(ISQv2Token)], [Prelude.String]) -> HappyIdentity a
happyError' = HappyIdentity Prelude.. (\(tokens, _) -> parseError tokens)
isqv2 tks = happyRunIdentity happySomeParser where
 happySomeParser = happyThen (happyParse action_0 tks) (\x -> case x of {HappyAbsSyn4 z -> happyReturn z; _other -> notHappyAtAll })

happySeq = happyDontSeq


parseError :: [ISQv2Token] -> a
parseError xs = error $ "Parse error at token "++ show (head xs)
{-# LINE 1 "templates/GenericTemplate.hs" #-}
-- $Id: GenericTemplate.hs,v 1.26 2005/01/14 14:47:22 simonmar Exp $










































data Happy_IntList = HappyCons Prelude.Int Happy_IntList








































infixr 9 `HappyStk`
data HappyStk a = HappyStk a (HappyStk a)

-----------------------------------------------------------------------------
-- starting the parse

happyParse start_state = happyNewToken start_state notHappyAtAll notHappyAtAll

-----------------------------------------------------------------------------
-- Accepting the parse

-- If the current token is ERROR_TOK, it means we've just accepted a partial
-- parse (a %partial parser).  We must ignore the saved token on the top of
-- the stack in this case.
happyAccept (1) tk st sts (_ `HappyStk` ans `HappyStk` _) =
        happyReturn1 ans
happyAccept j tk st sts (HappyStk ans _) = 
         (happyReturn1 ans)

-----------------------------------------------------------------------------
-- Arrays only: do the next action









































indexShortOffAddr arr off = arr Happy_Data_Array.! off


{-# INLINE happyLt #-}
happyLt x y = (x Prelude.< y)






readArrayBit arr bit =
    Bits.testBit (indexShortOffAddr arr (bit `Prelude.div` 16)) (bit `Prelude.mod` 16)






-----------------------------------------------------------------------------
-- HappyState data type (not arrays)



newtype HappyState b c = HappyState
        (Prelude.Int ->                    -- token number
         Prelude.Int ->                    -- token number (yes, again)
         b ->                           -- token semantic value
         HappyState b c ->              -- current state
         [HappyState b c] ->            -- state stack
         c)



-----------------------------------------------------------------------------
-- Shifting a token

happyShift new_state (1) tk st sts stk@(x `HappyStk` _) =
     let i = (case x of { HappyErrorToken (i) -> i }) in
--     trace "shifting the error token" $
     new_state i i tk (HappyState (new_state)) ((st):(sts)) (stk)

happyShift new_state i tk st sts stk =
     happyNewToken new_state ((st):(sts)) ((HappyTerminal (tk))`HappyStk`stk)

-- happyReduce is specialised for the common cases.

happySpecReduce_0 i fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happySpecReduce_0 nt fn j tk st@((HappyState (action))) sts stk
     = action nt j tk st ((st):(sts)) (fn `HappyStk` stk)

happySpecReduce_1 i fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happySpecReduce_1 nt fn j tk _ sts@(((st@(HappyState (action))):(_))) (v1`HappyStk`stk')
     = let r = fn v1 in
       happySeq r (action nt j tk st sts (r `HappyStk` stk'))

happySpecReduce_2 i fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happySpecReduce_2 nt fn j tk _ ((_):(sts@(((st@(HappyState (action))):(_))))) (v1`HappyStk`v2`HappyStk`stk')
     = let r = fn v1 v2 in
       happySeq r (action nt j tk st sts (r `HappyStk` stk'))

happySpecReduce_3 i fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happySpecReduce_3 nt fn j tk _ ((_):(((_):(sts@(((st@(HappyState (action))):(_))))))) (v1`HappyStk`v2`HappyStk`v3`HappyStk`stk')
     = let r = fn v1 v2 v3 in
       happySeq r (action nt j tk st sts (r `HappyStk` stk'))

happyReduce k i fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happyReduce k nt fn j tk st sts stk
     = case happyDrop (k Prelude.- ((1) :: Prelude.Int)) sts of
         sts1@(((st1@(HappyState (action))):(_))) ->
                let r = fn stk in  -- it doesn't hurt to always seq here...
                happyDoSeq r (action nt j tk st1 sts1 r)

happyMonadReduce k nt fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happyMonadReduce k nt fn j tk st sts stk =
      case happyDrop k ((st):(sts)) of
        sts1@(((st1@(HappyState (action))):(_))) ->
          let drop_stk = happyDropStk k stk in
          happyThen1 (fn stk tk) (\r -> action nt j tk st1 sts1 (r `HappyStk` drop_stk))

happyMonad2Reduce k nt fn (1) tk st sts stk
     = happyFail [] (1) tk st sts stk
happyMonad2Reduce k nt fn j tk st sts stk =
      case happyDrop k ((st):(sts)) of
        sts1@(((st1@(HappyState (action))):(_))) ->
         let drop_stk = happyDropStk k stk





             _ = nt :: Prelude.Int
             new_state = action

          in
          happyThen1 (fn stk tk) (\r -> happyNewToken new_state sts1 (r `HappyStk` drop_stk))

happyDrop (0) l = l
happyDrop n ((_):(t)) = happyDrop (n Prelude.- ((1) :: Prelude.Int)) t

happyDropStk (0) l = l
happyDropStk n (x `HappyStk` xs) = happyDropStk (n Prelude.- ((1)::Prelude.Int)) xs

-----------------------------------------------------------------------------
-- Moving to a new state after a reduction









happyGoto action j tk st = action j j tk (HappyState action)


-----------------------------------------------------------------------------
-- Error recovery (ERROR_TOK is the error token)

-- parse error if we are in recovery and we fail again
happyFail explist (1) tk old_st _ stk@(x `HappyStk` _) =
     let i = (case x of { HappyErrorToken (i) -> i }) in
--      trace "failing" $ 
        happyError_ explist i tk

{-  We don't need state discarding for our restricted implementation of
    "error".  In fact, it can cause some bogus parses, so I've disabled it
    for now --SDM

-- discard a state
happyFail  ERROR_TOK tk old_st CONS(HAPPYSTATE(action),sts) 
                                                (saved_tok `HappyStk` _ `HappyStk` stk) =
--      trace ("discarding state, depth " ++ show (length stk))  $
        DO_ACTION(action,ERROR_TOK,tk,sts,(saved_tok`HappyStk`stk))
-}

-- Enter error recovery: generate an error token,
--                       save the old token and carry on.
happyFail explist i tk (HappyState (action)) sts stk =
--      trace "entering error recovery" $
        action (1) (1) tk (HappyState (action)) sts ((HappyErrorToken (i)) `HappyStk` stk)

-- Internal happy errors:

notHappyAtAll :: a
notHappyAtAll = Prelude.error "Internal Happy error\n"

-----------------------------------------------------------------------------
-- Hack to get the typechecker to accept our action functions







-----------------------------------------------------------------------------
-- Seq-ing.  If the --strict flag is given, then Happy emits 
--      happySeq = happyDoSeq
-- otherwise it emits
--      happySeq = happyDontSeq

happyDoSeq, happyDontSeq :: a -> b -> b
happyDoSeq   a b = a `Prelude.seq` b
happyDontSeq a b = b

-----------------------------------------------------------------------------
-- Don't inline any functions from the template.  GHC has a nasty habit
-- of deciding to inline happyGoto everywhere, which increases the size of
-- the generated parser quite a bit.









{-# NOINLINE happyShift #-}
{-# NOINLINE happySpecReduce_0 #-}
{-# NOINLINE happySpecReduce_1 #-}
{-# NOINLINE happySpecReduce_2 #-}
{-# NOINLINE happySpecReduce_3 #-}
{-# NOINLINE happyReduce #-}
{-# NOINLINE happyMonadReduce #-}
{-# NOINLINE happyGoto #-}
{-# NOINLINE happyFail #-}

-- end of Happy Template.
