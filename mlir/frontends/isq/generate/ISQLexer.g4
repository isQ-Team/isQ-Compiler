lexer grammar ISQLexer;

//换行，注释
WhiteSpace: [ \t]+ -> skip;

NewLine: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~[\r\n]* -> skip;

// 关键字
If: 'if';

Then: 'then';

Else: 'else';

Fi: 'fi';

For: 'for';

To: 'to';

While: 'while';

Do: 'do';

Od: 'od';

Procedure: 'procedure';

Main: 'main';

Int: 'int';

Qbit: 'qbit';

H: 'H';

X: 'X';

Y: 'Y';

Z: 'Z';

S: 'S';

T: 'T';

CZ: 'CZ';

CX: 'CX';

CNOT: 'CNOT';

M: 'M';

Print: 'print';

Defgate: 'Defgate';

Pass: 'pass';

Return: 'return';

// 操作符

Assign: '=';

Plus: '+';

Minus: '-';

Mult: '*';

Div: '/';

Less: '<';

Greater: '>';

Comma: ',';

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Semi: ';';

Equal: '==';

LessEqual: '<=';

GreaterEqual: '>=';

KetZero: '|0>';

// 标识符和数字
Identifier: IdentifierAlpha (IdentifierAlpha | Digit)*;

Number: ('0' | NoneZeroDigit Digit*) (Dot Digit+)? ([i-j])?;

// 定义一些常用fragment
fragment IdentifierAlpha: [a-zA-Z_];

fragment Digit: [0-9];

fragment NoneZeroDigit: [1-9];

fragment Dot: '.';