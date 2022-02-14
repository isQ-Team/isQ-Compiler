module ISQ.Lang.ISQv2Grammar where
import ISQ.Lang.ISQv2Tokenizer
type Ident = String
type ASTBlock ann = [AST ann]
data BuiltinType = Ref | Qbit | Int | Bool | Double | Complex | FixedArray Int | UnknownArray | UserType | IntRange String deriving Show
data Type ann = Type { annotationType :: ann, ty :: BuiltinType, subTypes :: [Type ann]} deriving Show

intType ann = Type ann Int []
qbitType ann = Type ann Qbit []
boolType ann = Type ann Bool []
doubleType ann = Type ann Double []
complexType ann = Type ann Complex []


data CmpType = Equal | NEqual | Greater | Less | GreaterEq | LessEq deriving Show
data BinaryOperator = Add | Sub | Mul | Div | Cmp CmpType deriving Show
data UnaryOperator = Neg | Positive deriving Show
data Expr ann = 
       EIdent { annotationExpr :: ann, identName :: String}
     | EBinary { annotationExpr :: ann, binaryOp :: BinaryOperator, binaryLhs :: Expr ann, binaryRhs :: Expr ann}
     | EUnary { annotationExpr :: ann, unaryOp :: UnaryOperator, unaryOperand :: Expr ann}
     | ESubscript { annotationExpr :: ann, usedExpr :: Expr ann, subscript :: Expr ann}
     | ECall { annotationExpr :: ann, callee :: Expr ann, callArgs :: [Expr ann]}
     | EIntLit { annotationExpr :: ann, intLitVal :: Int}
     | EFloatingLit { annotationExpr :: ann, floatingLitVal :: Double}
     | EImagLit { annotationExpr :: ann, imagLitVal :: Double}
     | ERange {annotationExpr :: ann, rangeLo :: Maybe (Expr ann), rangeHi :: Maybe (Expr ann), rangeStep :: Maybe (Expr ann)}
     | EDeref {annotationExpr :: ann, derefVal :: Expr ann}
     | ECoreMeasure { annotationExpr :: ann, measureOperand :: Expr ann }
     | EList { annotationExpr :: ann, exprListElems :: Expr ann }
     | EImplicitCast { annotationExpr :: ann, castedExpr :: Expr ann}
     deriving Show
instance Annotated Expr where
  annotation = annotationExpr
instance Annotated Type where
  annotation = annotationType
data AST ann = 
       NIf { annotationAST :: ann, thenBlock :: [AST ann], elseBlock :: [AST ann]}
     | NFor { annotationAST :: ann, forVar :: Ident, forRange :: Expr ann, body :: ASTBlock ann}
     | NPass { annotationAST :: ann }
     | NWhile { annotationAST :: ann, condition :: Expr ann,  body :: ASTBlock ann}
     | NCall { annotationAST :: ann, callExpr :: Expr ann}
     | NDefvar { annotationAST :: ann, definitions :: [(Type ann, Ident, Maybe (Expr ann))]}
     | NAssign { annotationAST :: ann, assignLhs :: Expr ann, assignRhs :: Expr ann}
     | NGatedef { annotationAST :: ann, gateName :: String, gateRhs :: Expr ann}
     | NCoreUnitary { annotationAST :: ann, unitaryName :: String, unitaryOperands :: Expr ann}
     | NCoreReset { annotationAST :: ann, resetOperands :: Expr ann}
     | NCorePrint { annotationAST :: ann, printOperands :: Expr ann}
     | NCoreMeasure {annotationAST :: ann, measExpr :: Expr ann}
     deriving Show

instance Annotated AST where
  annotation = annotationAST