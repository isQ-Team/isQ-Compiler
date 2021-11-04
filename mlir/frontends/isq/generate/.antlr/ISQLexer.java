// Generated from /Users/huazhelou/Documents/quantum/llvm/mlir/generate/ISQLexer.g4 by ANTLR 4.8
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class ISQLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		WhiteSpace=1, NewLine=2, BlockComment=3, LineComment=4, If=5, Then=6, 
		Else=7, Fi=8, For=9, To=10, While=11, Do=12, Od=13, Procedure=14, Main=15, 
		Int=16, Qbit=17, H=18, X=19, Y=20, Z=21, S=22, T=23, CZ=24, CX=25, CNOT=26, 
		M=27, Print=28, Defgate=29, Pass=30, Return=31, Assign=32, Plus=33, Minus=34, 
		Mult=35, Div=36, Less=37, Greater=38, Comma=39, LeftParen=40, RightParen=41, 
		LeftBracket=42, RightBracket=43, LeftBrace=44, RightBrace=45, Semi=46, 
		Equal=47, LessEqual=48, GreaterEqual=49, KetZero=50, Identifier=51, Number=52;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"WhiteSpace", "NewLine", "BlockComment", "LineComment", "If", "Then", 
			"Else", "Fi", "For", "To", "While", "Do", "Od", "Procedure", "Main", 
			"Int", "Qbit", "H", "X", "Y", "Z", "S", "T", "CZ", "CX", "CNOT", "M", 
			"Print", "Defgate", "Pass", "Return", "Assign", "Plus", "Minus", "Mult", 
			"Div", "Less", "Greater", "Comma", "LeftParen", "RightParen", "LeftBracket", 
			"RightBracket", "LeftBrace", "RightBrace", "Semi", "Equal", "LessEqual", 
			"GreaterEqual", "KetZero", "Identifier", "Number", "IdentifierAlpha", 
			"Digit", "NoneZeroDigit", "Dot"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, null, null, null, null, "'if'", "'then'", "'else'", "'fi'", "'for'", 
			"'to'", "'while'", "'do'", "'od'", "'procedure'", "'main'", "'int'", 
			"'qbit'", "'H'", "'X'", "'Y'", "'Z'", "'S'", "'T'", "'CZ'", "'CX'", "'CNOT'", 
			"'M'", "'print'", "'Defgate'", "'pass'", "'return'", "'='", "'+'", "'-'", 
			"'*'", "'/'", "'<'", "'>'", "','", "'('", "')'", "'['", "']'", "'{'", 
			"'}'", "';'", "'=='", "'<='", "'>='", "'|0>'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "WhiteSpace", "NewLine", "BlockComment", "LineComment", "If", "Then", 
			"Else", "Fi", "For", "To", "While", "Do", "Od", "Procedure", "Main", 
			"Int", "Qbit", "H", "X", "Y", "Z", "S", "T", "CZ", "CX", "CNOT", "M", 
			"Print", "Defgate", "Pass", "Return", "Assign", "Plus", "Minus", "Mult", 
			"Div", "Less", "Greater", "Comma", "LeftParen", "RightParen", "LeftBracket", 
			"RightBracket", "LeftBrace", "RightBrace", "Semi", "Equal", "LessEqual", 
			"GreaterEqual", "KetZero", "Identifier", "Number"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public ISQLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "ISQLexer.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\66\u015a\b\1\4\2"+
		"\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4"+
		"\13\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22"+
		"\t\22\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31"+
		"\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t"+
		" \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t"+
		"+\4,\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64"+
		"\t\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\3\2\6\2u\n\2\r\2\16\2"+
		"v\3\2\3\2\3\3\3\3\5\3}\n\3\3\3\5\3\u0080\n\3\3\3\3\3\3\4\3\4\3\4\3\4\7"+
		"\4\u0088\n\4\f\4\16\4\u008b\13\4\3\4\3\4\3\4\3\4\3\4\3\5\3\5\3\5\3\5\7"+
		"\5\u0096\n\5\f\5\16\5\u0099\13\5\3\5\3\5\3\6\3\6\3\6\3\7\3\7\3\7\3\7\3"+
		"\7\3\b\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\n\3\n\3\n\3\n\3\13\3\13\3\13\3\f"+
		"\3\f\3\f\3\f\3\f\3\f\3\r\3\r\3\r\3\16\3\16\3\16\3\17\3\17\3\17\3\17\3"+
		"\17\3\17\3\17\3\17\3\17\3\17\3\20\3\20\3\20\3\20\3\20\3\21\3\21\3\21\3"+
		"\21\3\22\3\22\3\22\3\22\3\22\3\23\3\23\3\24\3\24\3\25\3\25\3\26\3\26\3"+
		"\27\3\27\3\30\3\30\3\31\3\31\3\31\3\32\3\32\3\32\3\33\3\33\3\33\3\33\3"+
		"\33\3\34\3\34\3\35\3\35\3\35\3\35\3\35\3\35\3\36\3\36\3\36\3\36\3\36\3"+
		"\36\3\36\3\36\3\37\3\37\3\37\3\37\3\37\3 \3 \3 \3 \3 \3 \3 \3!\3!\3\""+
		"\3\"\3#\3#\3$\3$\3%\3%\3&\3&\3\'\3\'\3(\3(\3)\3)\3*\3*\3+\3+\3,\3,\3-"+
		"\3-\3.\3.\3/\3/\3\60\3\60\3\60\3\61\3\61\3\61\3\62\3\62\3\62\3\63\3\63"+
		"\3\63\3\63\3\64\3\64\3\64\7\64\u0139\n\64\f\64\16\64\u013c\13\64\3\65"+
		"\3\65\3\65\7\65\u0141\n\65\f\65\16\65\u0144\13\65\5\65\u0146\n\65\3\65"+
		"\3\65\6\65\u014a\n\65\r\65\16\65\u014b\5\65\u014e\n\65\3\65\5\65\u0151"+
		"\n\65\3\66\3\66\3\67\3\67\38\38\39\39\3\u0089\2:\3\3\5\4\7\5\t\6\13\7"+
		"\r\b\17\t\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25"+
		")\26+\27-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'M(O"+
		")Q*S+U,W-Y.[/]\60_\61a\62c\63e\64g\65i\66k\2m\2o\2q\2\3\2\b\4\2\13\13"+
		"\"\"\4\2\f\f\17\17\3\2kl\5\2C\\aac|\3\2\62;\3\2\63;\2\u0161\2\3\3\2\2"+
		"\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3"+
		"\2\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2"+
		"\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2"+
		"\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2"+
		"\2\2\2\63\3\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3"+
		"\2\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2"+
		"\2\2K\3\2\2\2\2M\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2"+
		"W\3\2\2\2\2Y\3\2\2\2\2[\3\2\2\2\2]\3\2\2\2\2_\3\2\2\2\2a\3\2\2\2\2c\3"+
		"\2\2\2\2e\3\2\2\2\2g\3\2\2\2\2i\3\2\2\2\3t\3\2\2\2\5\177\3\2\2\2\7\u0083"+
		"\3\2\2\2\t\u0091\3\2\2\2\13\u009c\3\2\2\2\r\u009f\3\2\2\2\17\u00a4\3\2"+
		"\2\2\21\u00a9\3\2\2\2\23\u00ac\3\2\2\2\25\u00b0\3\2\2\2\27\u00b3\3\2\2"+
		"\2\31\u00b9\3\2\2\2\33\u00bc\3\2\2\2\35\u00bf\3\2\2\2\37\u00c9\3\2\2\2"+
		"!\u00ce\3\2\2\2#\u00d2\3\2\2\2%\u00d7\3\2\2\2\'\u00d9\3\2\2\2)\u00db\3"+
		"\2\2\2+\u00dd\3\2\2\2-\u00df\3\2\2\2/\u00e1\3\2\2\2\61\u00e3\3\2\2\2\63"+
		"\u00e6\3\2\2\2\65\u00e9\3\2\2\2\67\u00ee\3\2\2\29\u00f0\3\2\2\2;\u00f6"+
		"\3\2\2\2=\u00fe\3\2\2\2?\u0103\3\2\2\2A\u010a\3\2\2\2C\u010c\3\2\2\2E"+
		"\u010e\3\2\2\2G\u0110\3\2\2\2I\u0112\3\2\2\2K\u0114\3\2\2\2M\u0116\3\2"+
		"\2\2O\u0118\3\2\2\2Q\u011a\3\2\2\2S\u011c\3\2\2\2U\u011e\3\2\2\2W\u0120"+
		"\3\2\2\2Y\u0122\3\2\2\2[\u0124\3\2\2\2]\u0126\3\2\2\2_\u0128\3\2\2\2a"+
		"\u012b\3\2\2\2c\u012e\3\2\2\2e\u0131\3\2\2\2g\u0135\3\2\2\2i\u0145\3\2"+
		"\2\2k\u0152\3\2\2\2m\u0154\3\2\2\2o\u0156\3\2\2\2q\u0158\3\2\2\2su\t\2"+
		"\2\2ts\3\2\2\2uv\3\2\2\2vt\3\2\2\2vw\3\2\2\2wx\3\2\2\2xy\b\2\2\2y\4\3"+
		"\2\2\2z|\7\17\2\2{}\7\f\2\2|{\3\2\2\2|}\3\2\2\2}\u0080\3\2\2\2~\u0080"+
		"\7\f\2\2\177z\3\2\2\2\177~\3\2\2\2\u0080\u0081\3\2\2\2\u0081\u0082\b\3"+
		"\2\2\u0082\6\3\2\2\2\u0083\u0084\7\61\2\2\u0084\u0085\7,\2\2\u0085\u0089"+
		"\3\2\2\2\u0086\u0088\13\2\2\2\u0087\u0086\3\2\2\2\u0088\u008b\3\2\2\2"+
		"\u0089\u008a\3\2\2\2\u0089\u0087\3\2\2\2\u008a\u008c\3\2\2\2\u008b\u0089"+
		"\3\2\2\2\u008c\u008d\7,\2\2\u008d\u008e\7\61\2\2\u008e\u008f\3\2\2\2\u008f"+
		"\u0090\b\4\2\2\u0090\b\3\2\2\2\u0091\u0092\7\61\2\2\u0092\u0093\7\61\2"+
		"\2\u0093\u0097\3\2\2\2\u0094\u0096\n\3\2\2\u0095\u0094\3\2\2\2\u0096\u0099"+
		"\3\2\2\2\u0097\u0095\3\2\2\2\u0097\u0098\3\2\2\2\u0098\u009a\3\2\2\2\u0099"+
		"\u0097\3\2\2\2\u009a\u009b\b\5\2\2\u009b\n\3\2\2\2\u009c\u009d\7k\2\2"+
		"\u009d\u009e\7h\2\2\u009e\f\3\2\2\2\u009f\u00a0\7v\2\2\u00a0\u00a1\7j"+
		"\2\2\u00a1\u00a2\7g\2\2\u00a2\u00a3\7p\2\2\u00a3\16\3\2\2\2\u00a4\u00a5"+
		"\7g\2\2\u00a5\u00a6\7n\2\2\u00a6\u00a7\7u\2\2\u00a7\u00a8\7g\2\2\u00a8"+
		"\20\3\2\2\2\u00a9\u00aa\7h\2\2\u00aa\u00ab\7k\2\2\u00ab\22\3\2\2\2\u00ac"+
		"\u00ad\7h\2\2\u00ad\u00ae\7q\2\2\u00ae\u00af\7t\2\2\u00af\24\3\2\2\2\u00b0"+
		"\u00b1\7v\2\2\u00b1\u00b2\7q\2\2\u00b2\26\3\2\2\2\u00b3\u00b4\7y\2\2\u00b4"+
		"\u00b5\7j\2\2\u00b5\u00b6\7k\2\2\u00b6\u00b7\7n\2\2\u00b7\u00b8\7g\2\2"+
		"\u00b8\30\3\2\2\2\u00b9\u00ba\7f\2\2\u00ba\u00bb\7q\2\2\u00bb\32\3\2\2"+
		"\2\u00bc\u00bd\7q\2\2\u00bd\u00be\7f\2\2\u00be\34\3\2\2\2\u00bf\u00c0"+
		"\7r\2\2\u00c0\u00c1\7t\2\2\u00c1\u00c2\7q\2\2\u00c2\u00c3\7e\2\2\u00c3"+
		"\u00c4\7g\2\2\u00c4\u00c5\7f\2\2\u00c5\u00c6\7w\2\2\u00c6\u00c7\7t\2\2"+
		"\u00c7\u00c8\7g\2\2\u00c8\36\3\2\2\2\u00c9\u00ca\7o\2\2\u00ca\u00cb\7"+
		"c\2\2\u00cb\u00cc\7k\2\2\u00cc\u00cd\7p\2\2\u00cd \3\2\2\2\u00ce\u00cf"+
		"\7k\2\2\u00cf\u00d0\7p\2\2\u00d0\u00d1\7v\2\2\u00d1\"\3\2\2\2\u00d2\u00d3"+
		"\7s\2\2\u00d3\u00d4\7d\2\2\u00d4\u00d5\7k\2\2\u00d5\u00d6\7v\2\2\u00d6"+
		"$\3\2\2\2\u00d7\u00d8\7J\2\2\u00d8&\3\2\2\2\u00d9\u00da\7Z\2\2\u00da("+
		"\3\2\2\2\u00db\u00dc\7[\2\2\u00dc*\3\2\2\2\u00dd\u00de\7\\\2\2\u00de,"+
		"\3\2\2\2\u00df\u00e0\7U\2\2\u00e0.\3\2\2\2\u00e1\u00e2\7V\2\2\u00e2\60"+
		"\3\2\2\2\u00e3\u00e4\7E\2\2\u00e4\u00e5\7\\\2\2\u00e5\62\3\2\2\2\u00e6"+
		"\u00e7\7E\2\2\u00e7\u00e8\7Z\2\2\u00e8\64\3\2\2\2\u00e9\u00ea\7E\2\2\u00ea"+
		"\u00eb\7P\2\2\u00eb\u00ec\7Q\2\2\u00ec\u00ed\7V\2\2\u00ed\66\3\2\2\2\u00ee"+
		"\u00ef\7O\2\2\u00ef8\3\2\2\2\u00f0\u00f1\7r\2\2\u00f1\u00f2\7t\2\2\u00f2"+
		"\u00f3\7k\2\2\u00f3\u00f4\7p\2\2\u00f4\u00f5\7v\2\2\u00f5:\3\2\2\2\u00f6"+
		"\u00f7\7F\2\2\u00f7\u00f8\7g\2\2\u00f8\u00f9\7h\2\2\u00f9\u00fa\7i\2\2"+
		"\u00fa\u00fb\7c\2\2\u00fb\u00fc\7v\2\2\u00fc\u00fd\7g\2\2\u00fd<\3\2\2"+
		"\2\u00fe\u00ff\7r\2\2\u00ff\u0100\7c\2\2\u0100\u0101\7u\2\2\u0101\u0102"+
		"\7u\2\2\u0102>\3\2\2\2\u0103\u0104\7t\2\2\u0104\u0105\7g\2\2\u0105\u0106"+
		"\7v\2\2\u0106\u0107\7w\2\2\u0107\u0108\7t\2\2\u0108\u0109\7p\2\2\u0109"+
		"@\3\2\2\2\u010a\u010b\7?\2\2\u010bB\3\2\2\2\u010c\u010d\7-\2\2\u010dD"+
		"\3\2\2\2\u010e\u010f\7/\2\2\u010fF\3\2\2\2\u0110\u0111\7,\2\2\u0111H\3"+
		"\2\2\2\u0112\u0113\7\61\2\2\u0113J\3\2\2\2\u0114\u0115\7>\2\2\u0115L\3"+
		"\2\2\2\u0116\u0117\7@\2\2\u0117N\3\2\2\2\u0118\u0119\7.\2\2\u0119P\3\2"+
		"\2\2\u011a\u011b\7*\2\2\u011bR\3\2\2\2\u011c\u011d\7+\2\2\u011dT\3\2\2"+
		"\2\u011e\u011f\7]\2\2\u011fV\3\2\2\2\u0120\u0121\7_\2\2\u0121X\3\2\2\2"+
		"\u0122\u0123\7}\2\2\u0123Z\3\2\2\2\u0124\u0125\7\177\2\2\u0125\\\3\2\2"+
		"\2\u0126\u0127\7=\2\2\u0127^\3\2\2\2\u0128\u0129\7?\2\2\u0129\u012a\7"+
		"?\2\2\u012a`\3\2\2\2\u012b\u012c\7>\2\2\u012c\u012d\7?\2\2\u012db\3\2"+
		"\2\2\u012e\u012f\7@\2\2\u012f\u0130\7?\2\2\u0130d\3\2\2\2\u0131\u0132"+
		"\7~\2\2\u0132\u0133\7\62\2\2\u0133\u0134\7@\2\2\u0134f\3\2\2\2\u0135\u013a"+
		"\5k\66\2\u0136\u0139\5k\66\2\u0137\u0139\5m\67\2\u0138\u0136\3\2\2\2\u0138"+
		"\u0137\3\2\2\2\u0139\u013c\3\2\2\2\u013a\u0138\3\2\2\2\u013a\u013b\3\2"+
		"\2\2\u013bh\3\2\2\2\u013c\u013a\3\2\2\2\u013d\u0146\7\62\2\2\u013e\u0142"+
		"\5o8\2\u013f\u0141\5m\67\2\u0140\u013f\3\2\2\2\u0141\u0144\3\2\2\2\u0142"+
		"\u0140\3\2\2\2\u0142\u0143\3\2\2\2\u0143\u0146\3\2\2\2\u0144\u0142\3\2"+
		"\2\2\u0145\u013d\3\2\2\2\u0145\u013e\3\2\2\2\u0146\u014d\3\2\2\2\u0147"+
		"\u0149\5q9\2\u0148\u014a\5m\67\2\u0149\u0148\3\2\2\2\u014a\u014b\3\2\2"+
		"\2\u014b\u0149\3\2\2\2\u014b\u014c\3\2\2\2\u014c\u014e\3\2\2\2\u014d\u0147"+
		"\3\2\2\2\u014d\u014e\3\2\2\2\u014e\u0150\3\2\2\2\u014f\u0151\t\4\2\2\u0150"+
		"\u014f\3\2\2\2\u0150\u0151\3\2\2\2\u0151j\3\2\2\2\u0152\u0153\t\5\2\2"+
		"\u0153l\3\2\2\2\u0154\u0155\t\6\2\2\u0155n\3\2\2\2\u0156\u0157\t\7\2\2"+
		"\u0157p\3\2\2\2\u0158\u0159\7\60\2\2\u0159r\3\2\2\2\17\2v|\177\u0089\u0097"+
		"\u0138\u013a\u0142\u0145\u014b\u014d\u0150\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}