{
module ISQ.Lang.ISQv2Parser where

import ISQ.Lang.ISQv2Tokenizer


}
%name isqv2
%tokentype { ISQv2Token }
%error { parseError }

%token 
    if { TokenReservedId $$ "if" }
    else { TokenReservedId $$ "else" }
    for { TokenReservedId $$ "for" }
    in { TokenReservedId $$ "in" }
    while { TokenReservedId $$ "while" }
    procedure { TokenReservedId $$ "procedure" }
    int { TokenReservedId $$ "int" }
    qbit { TokenReservedId $$ "qbit" }
    M { TokenReservedId $$ "measure" }
    print { TokenReservedId $$ "print" }
    defgate { TokenReservedId $$ "defgate" }
    pass { TokenReservedId $$ "pass" }
    return { TokenReservedId $$ "return" }
    ctrl { TokenReservedId $$ "ctrl" }
    nctrl { TokenReservedId $$ "nctrl" }
    inv { TokenReservedId $$ "inv" }
    '|0>' { TokenReservedOp $$ "|0>" }
    '=' { TokenReservedOp $$ "=" }
    '==' { TokenReservedOp $$ "==" }
    '+' { TokenReservedOp $$ "+" }
    '-' { TokenReservedOp $$ "-" }
    '*' { TokenReservedOp $$ "*" }
    '/' { TokenReservedOp $$ "/" }
    '<' { TokenReservedOp $$ "<" }
    '>' { TokenReservedOp $$ ">" }
    '<=' { TokenReservedOp $$ "<=" }
    '>=' { TokenReservedOp $$ ">=" }
    '!=' { TokenReservedOp $$ "!=" }
    '&&' { TokenReservedOp $$ "&&" }
    '||' { TokenReservedOp $$ "||" }
    ',' { TokenReservedOp $$ "," }
    ';' { TokenReservedOp $$ ";" }
    '(' { TokenReservedOp $$ "(" }
    ')' { TokenReservedOp $$ ")" }
    '[' { TokenReservedOp $$ "[" }
    ']' { TokenReservedOp $$ "]" }
    '{' { TokenReservedOp $$ "{" }
    '}' { TokenReservedOp $$ "}" }
    '.' { TokenReservedOp $$ "." }
    ':' { TokenReservedOp $$ ":" }
    NATURAL { TokenNatural _ _ }
    FLOAT { TokenFloat _ _ }
    IMAGPART { TokenImagPart _ _ }
    IDENTIFIER { TokenIdent _ _ }


-- Level 7
%left '==' '!='
-- Level 6
%nonassoc '>' '<' '>=' '<='
-- Level 4
%left '+' '-'
-- Level 3
%left '*' '/'
-- Level 2
%left NEG POS
%%

-- TODO: first-class function
CallExpr : IDENTIFIER '(' ExprList ')' { ECall (annotation $1) (tokenIdentV $3)}
MaybeExpr : Expr {Just $1}
          | {- empty -} {Nothing}
RangeExpr : MaybeExpr ':' MaybeExpr ':' MaybeExpr { ERange $2 $1 $3 $5 }

Expr : '(' Expr ')' { $2 }
     |  Expr '+' Expr { EBinary $2 Add $1 $3 }
     |  Expr '-' Expr { EBinary $2 Sub $1 $3 }
     |  Expr '*' Expr { EBinary $2 Mul $1 $3 }
     |  Expr '/' Expr { EBinary $2 Div $1 $3 }
     |  Expr '==' Expr { EBinary $2 (Cmp Equal) $1 $3 }
     |  Expr '!=' Expr { EBinary $2 (Cmp NEqual) $1 $3 }
     |  Expr '>' Expr { EBinary $2 (Cmp Greater) $1 $3 }
     |  Expr '<' Expr { EBinary $2 (Cmp Less) $1 $3 }
     |  Expr '>=' Expr { EBinary $2 (Cmp GreaterEq) $1 $3 }
     |  Expr '<=' Expr { EBinary $2 (Cmp LessEq) $1 $3 }
     | '-' Expr %prec NEG { EUnary $1 Neg $2 }
     | '+' Expr %prec POS { EUnary $1 Positive $2 }
     | INT { EIntLit (annotation $1) (tokenNaturalV $1) }
     | FLOAT { EFloatingLit (annotation $1) (tokenFloatV $1) }
     | IMAGPART { EImagLit (annotation $1) (tokenImagPartV $2) }
     | CallExpr { $1 }
     | Expr '[' Expr ']' { ESubscript $2 $1 $3 }
     | RangeExpr { $1 }
     | '[' ExprListNonEmpty ']' { EList $1 $3 }
     -- isQ Core (isQ v1) grammar.
     -- TODO: should they be allowed only in compatible mode?
     | ISQCore_MeasureExpr { ECoreMeasure $1 $3 }

ExprList : Expr { [$1] }
         | ExprList ',' Expr { $1 ++ $3 }
         | {- empty -} { [] }
ExprListNonEmpty : Expr { [$1] }
                 | ExprListNonEmpty ',' Expr { $1 ++ $3 }

IdentListNonEmpty : IDENTIFIER { [$1] }
                  | IdentListNonEmpty ',' IDENTIFIER { $1 ++ $3 }
ForStatement : for IDENTIFIER in RangeExpr '{' StatementList '}' {NFor $1 $2 $4 $6 }
WhileStatement : while Expr '{' StatementList '}' {NWhile $1 $2 $4}
IfStatement : if Expr '{' StatementList '}' {NIf $1 $2 $4 []}
            | if Expr '{' StatementList '}' else '{' StatementList '}'  {NIf $1 $2 $4 $8}
PassStatement : pass { NPass $1 }
DefvarStatement : Type IDENTIFIER {NDefvar (annotation $1) [($1, $2)]}
                | Type IdentListNonEmpty { NDefVar }
CallStatement : CallExpr
AssignStatement : Expr '=' Expr
GatedefStatement : defgate IDENTIFIER '=' Expr
ISQCore_UnitaryStatement = IDENTIFIER '<' Expr '>'
ISQCore_MeasureExpr : M '<' Expr '>' { ECoreMeasure $1 $3 }
                | M '(' Expr ')' { ECoreMeasure $1 $3 }
ISQCore_MeasureStatement : ISQCore_MeasureExpr { NCoreMeasure (annotation $1) $1}
ISQCore_ResetStatement : Expr '=' '|0>'
ISQCore_PrintStatement : print Expr


Statement : ForStatement
          | WhileStatement
          | IfStatement
          | PassStatement
          | DefvarStatement
          | CallStatement
          | AssignStatement
          | GatedefStatement
          | ISQCore_UnitaryStatement
          | ISQCore_MeasureStatement
          | ISQCore_ResetStatement
          | ISQCore_PrintStatement


ArrayTypeDecorator : '[' ']' { UnknownArray }
                   | '[' NATURAL ']' { FixedArray (tokenNaturalV $2)}

FullType : SimpleType { $1 }
     | Type ArrayTypeDecorator { $1 }
SimpleType : int { intType $1 }
           | qbit { qbitType $1 }
           | bool { boolType $1 }

ISQCore_CStyleVarDefTerm : IDENTIFIER { \t->(t, $1, Nothing) }
                         | IDENTIFIER ArrayTypeDecorator { \t->(Type (annotation $1) $2 [t], Nothing)}
                         | IDENTIFIER '=' Expr { \t->(t, $1, Just $3) }
                         | IDENTIFIER ArrayTypeDecorator '=' Expr { \t->(Type (annotation $1) $2 [t], Just $4)}
ISQCore_CStyleVarDefList : ISQCore_CStyleVarDefTerm {[$1]}
                         | ISQCore_CStyleVarDefList, ISQCore_CStyleVarDefTerm {$1 ++ [$2]}
ISQCore_CStyleVarDef: SimpleType ISQCore_CStyleVarDefList { fmap (\f->f $1) $2 }

