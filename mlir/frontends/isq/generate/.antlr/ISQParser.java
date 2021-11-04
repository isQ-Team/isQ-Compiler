// Generated from /Users/huazhelou/Documents/quantum/llvm/mlir/generate/ISQParser.g4 by ANTLR 4.8
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class ISQParser extends Parser {
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
	public static final int
		RULE_program = 0, RULE_gateDefclause = 1, RULE_matrixContents = 2, RULE_cNumber = 3, 
		RULE_numberExpr = 4, RULE_varType = 5, RULE_defclause = 6, RULE_idlist = 7, 
		RULE_programBody = 8, RULE_procedureBlock = 9, RULE_callParas = 10, RULE_procedureMain = 11, 
		RULE_procedureBody = 12, RULE_statement = 13, RULE_statementBlock = 14, 
		RULE_uGate = 15, RULE_variable = 16, RULE_variableList = 17, RULE_binopPlus = 18, 
		RULE_binopMult = 19, RULE_expression = 20, RULE_multexp = 21, RULE_atomexp = 22, 
		RULE_mExpression = 23, RULE_association = 24, RULE_qbitInitStatement = 25, 
		RULE_qbitUnitaryStatement = 26, RULE_cintAssign = 27, RULE_regionBody = 28, 
		RULE_ifStatement = 29, RULE_forStatement = 30, RULE_whileStatement = 31, 
		RULE_callStatement = 32, RULE_printStatement = 33, RULE_passStatement = 34, 
		RULE_returnStatement = 35;
	private static String[] makeRuleNames() {
		return new String[] {
			"program", "gateDefclause", "matrixContents", "cNumber", "numberExpr", 
			"varType", "defclause", "idlist", "programBody", "procedureBlock", "callParas", 
			"procedureMain", "procedureBody", "statement", "statementBlock", "uGate", 
			"variable", "variableList", "binopPlus", "binopMult", "expression", "multexp", 
			"atomexp", "mExpression", "association", "qbitInitStatement", "qbitUnitaryStatement", 
			"cintAssign", "regionBody", "ifStatement", "forStatement", "whileStatement", 
			"callStatement", "printStatement", "passStatement", "returnStatement"
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

	@Override
	public String getGrammarFileName() { return "ISQParser.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public ISQParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class ProgramContext extends ParserRuleContext {
		public ProgramBodyContext programBody() {
			return getRuleContext(ProgramBodyContext.class,0);
		}
		public List<GateDefclauseContext> gateDefclause() {
			return getRuleContexts(GateDefclauseContext.class);
		}
		public GateDefclauseContext gateDefclause(int i) {
			return getRuleContext(GateDefclauseContext.class,i);
		}
		public List<DefclauseContext> defclause() {
			return getRuleContexts(DefclauseContext.class);
		}
		public DefclauseContext defclause(int i) {
			return getRuleContext(DefclauseContext.class,i);
		}
		public ProgramContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_program; }
	}

	public final ProgramContext program() throws RecognitionException {
		ProgramContext _localctx = new ProgramContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_program);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(75);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==Defgate) {
				{
				{
				setState(72);
				gateDefclause();
				}
				}
				setState(77);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(79); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(78);
					defclause();
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(81); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			setState(83);
			programBody();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class GateDefclauseContext extends ParserRuleContext {
		public TerminalNode Defgate() { return getToken(ISQParser.Defgate, 0); }
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode Assign() { return getToken(ISQParser.Assign, 0); }
		public TerminalNode LeftBracket() { return getToken(ISQParser.LeftBracket, 0); }
		public MatrixContentsContext matrixContents() {
			return getRuleContext(MatrixContentsContext.class,0);
		}
		public TerminalNode RightBracket() { return getToken(ISQParser.RightBracket, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public GateDefclauseContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_gateDefclause; }
	}

	public final GateDefclauseContext gateDefclause() throws RecognitionException {
		GateDefclauseContext _localctx = new GateDefclauseContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_gateDefclause);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(85);
			match(Defgate);
			setState(86);
			match(Identifier);
			setState(87);
			match(Assign);
			setState(88);
			match(LeftBracket);
			setState(89);
			matrixContents();
			setState(90);
			match(RightBracket);
			setState(91);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MatrixContentsContext extends ParserRuleContext {
		public MatrixContentsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_matrixContents; }
	 
		public MatrixContentsContext() { }
		public void copyFrom(MatrixContentsContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class MatrixdefContext extends MatrixContentsContext {
		public CNumberContext cNumber() {
			return getRuleContext(CNumberContext.class,0);
		}
		public MatrixContentsContext matrixContents() {
			return getRuleContext(MatrixContentsContext.class,0);
		}
		public TerminalNode Comma() { return getToken(ISQParser.Comma, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public MatrixdefContext(MatrixContentsContext ctx) { copyFrom(ctx); }
	}
	public static class MatrixvaldefContext extends MatrixContentsContext {
		public CNumberContext cNumber() {
			return getRuleContext(CNumberContext.class,0);
		}
		public MatrixvaldefContext(MatrixContentsContext ctx) { copyFrom(ctx); }
	}

	public final MatrixContentsContext matrixContents() throws RecognitionException {
		MatrixContentsContext _localctx = new MatrixContentsContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_matrixContents);
		int _la;
		try {
			setState(98);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
			case 1:
				_localctx = new MatrixvaldefContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(93);
				cNumber();
				}
				break;
			case 2:
				_localctx = new MatrixdefContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(94);
				cNumber();
				setState(95);
				_la = _input.LA(1);
				if ( !(_la==Comma || _la==Semi) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(96);
				matrixContents();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CNumberContext extends ParserRuleContext {
		public NumberExprContext numberExpr() {
			return getRuleContext(NumberExprContext.class,0);
		}
		public TerminalNode Minus() { return getToken(ISQParser.Minus, 0); }
		public CNumberContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_cNumber; }
	}

	public final CNumberContext cNumber() throws RecognitionException {
		CNumberContext _localctx = new CNumberContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_cNumber);
		try {
			setState(103);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case Number:
				enterOuterAlt(_localctx, 1);
				{
				setState(100);
				numberExpr();
				}
				break;
			case Minus:
				enterOuterAlt(_localctx, 2);
				{
				setState(101);
				match(Minus);
				setState(102);
				numberExpr();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class NumberExprContext extends ParserRuleContext {
		public List<TerminalNode> Number() { return getTokens(ISQParser.Number); }
		public TerminalNode Number(int i) {
			return getToken(ISQParser.Number, i);
		}
		public TerminalNode Plus() { return getToken(ISQParser.Plus, 0); }
		public TerminalNode Minus() { return getToken(ISQParser.Minus, 0); }
		public NumberExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_numberExpr; }
	}

	public final NumberExprContext numberExpr() throws RecognitionException {
		NumberExprContext _localctx = new NumberExprContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_numberExpr);
		try {
			setState(112);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,4,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(105);
				match(Number);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(106);
				match(Number);
				setState(107);
				match(Plus);
				setState(108);
				match(Number);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(109);
				match(Number);
				setState(110);
				match(Minus);
				setState(111);
				match(Number);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarTypeContext extends ParserRuleContext {
		public TerminalNode Qbit() { return getToken(ISQParser.Qbit, 0); }
		public TerminalNode Int() { return getToken(ISQParser.Int, 0); }
		public VarTypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_varType; }
	}

	public final VarTypeContext varType() throws RecognitionException {
		VarTypeContext _localctx = new VarTypeContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_varType);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(114);
			_la = _input.LA(1);
			if ( !(_la==Int || _la==Qbit) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class DefclauseContext extends ParserRuleContext {
		public VarTypeContext varType() {
			return getRuleContext(VarTypeContext.class,0);
		}
		public IdlistContext idlist() {
			return getRuleContext(IdlistContext.class,0);
		}
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public DefclauseContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_defclause; }
	}

	public final DefclauseContext defclause() throws RecognitionException {
		DefclauseContext _localctx = new DefclauseContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_defclause);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(116);
			varType();
			setState(117);
			idlist(0);
			setState(118);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class IdlistContext extends ParserRuleContext {
		public IdlistContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_idlist; }
	 
		public IdlistContext() { }
		public void copyFrom(IdlistContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class IdlistdefContext extends IdlistContext {
		public List<IdlistContext> idlist() {
			return getRuleContexts(IdlistContext.class);
		}
		public IdlistContext idlist(int i) {
			return getRuleContext(IdlistContext.class,i);
		}
		public TerminalNode Comma() { return getToken(ISQParser.Comma, 0); }
		public IdlistdefContext(IdlistContext ctx) { copyFrom(ctx); }
	}
	public static class ArraydefContext extends IdlistContext {
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode LeftBracket() { return getToken(ISQParser.LeftBracket, 0); }
		public TerminalNode Number() { return getToken(ISQParser.Number, 0); }
		public TerminalNode RightBracket() { return getToken(ISQParser.RightBracket, 0); }
		public ArraydefContext(IdlistContext ctx) { copyFrom(ctx); }
	}
	public static class SingleiddefContext extends IdlistContext {
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public SingleiddefContext(IdlistContext ctx) { copyFrom(ctx); }
	}

	public final IdlistContext idlist() throws RecognitionException {
		return idlist(0);
	}

	private IdlistContext idlist(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		IdlistContext _localctx = new IdlistContext(_ctx, _parentState);
		IdlistContext _prevctx = _localctx;
		int _startState = 14;
		enterRecursionRule(_localctx, 14, RULE_idlist, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(126);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				{
				_localctx = new SingleiddefContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(121);
				match(Identifier);
				}
				break;
			case 2:
				{
				_localctx = new ArraydefContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(122);
				match(Identifier);
				setState(123);
				match(LeftBracket);
				setState(124);
				match(Number);
				setState(125);
				match(RightBracket);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(133);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new IdlistdefContext(new IdlistContext(_parentctx, _parentState));
					pushNewRecursionContext(_localctx, _startState, RULE_idlist);
					setState(128);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(129);
					match(Comma);
					setState(130);
					idlist(2);
					}
					} 
				}
				setState(135);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ProgramBodyContext extends ParserRuleContext {
		public ProcedureMainContext procedureMain() {
			return getRuleContext(ProcedureMainContext.class,0);
		}
		public List<ProcedureBlockContext> procedureBlock() {
			return getRuleContexts(ProcedureBlockContext.class);
		}
		public ProcedureBlockContext procedureBlock(int i) {
			return getRuleContext(ProcedureBlockContext.class,i);
		}
		public ProgramBodyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_programBody; }
	}

	public final ProgramBodyContext programBody() throws RecognitionException {
		ProgramBodyContext _localctx = new ProgramBodyContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_programBody);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(139);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(136);
					procedureBlock();
					}
					} 
				}
				setState(141);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			}
			setState(142);
			procedureMain();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ProcedureBlockContext extends ParserRuleContext {
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public TerminalNode LeftBrace() { return getToken(ISQParser.LeftBrace, 0); }
		public ProcedureBodyContext procedureBody() {
			return getRuleContext(ProcedureBodyContext.class,0);
		}
		public TerminalNode RightBrace() { return getToken(ISQParser.RightBrace, 0); }
		public TerminalNode Procedure() { return getToken(ISQParser.Procedure, 0); }
		public TerminalNode Int() { return getToken(ISQParser.Int, 0); }
		public CallParasContext callParas() {
			return getRuleContext(CallParasContext.class,0);
		}
		public ProcedureBlockContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_procedureBlock; }
	}

	public final ProcedureBlockContext procedureBlock() throws RecognitionException {
		ProcedureBlockContext _localctx = new ProcedureBlockContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_procedureBlock);
		int _la;
		try {
			setState(161);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,8,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(144);
				_la = _input.LA(1);
				if ( !(_la==Procedure || _la==Int) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(145);
				match(Identifier);
				setState(146);
				match(LeftParen);
				setState(147);
				match(RightParen);
				setState(148);
				match(LeftBrace);
				setState(149);
				procedureBody();
				setState(150);
				match(RightBrace);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(152);
				_la = _input.LA(1);
				if ( !(_la==Procedure || _la==Int) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(153);
				match(Identifier);
				setState(154);
				match(LeftParen);
				setState(155);
				callParas(0);
				setState(156);
				match(RightParen);
				setState(157);
				match(LeftBrace);
				setState(158);
				procedureBody();
				setState(159);
				match(RightBrace);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CallParasContext extends ParserRuleContext {
		public VarTypeContext varType() {
			return getRuleContext(VarTypeContext.class,0);
		}
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode LeftBracket() { return getToken(ISQParser.LeftBracket, 0); }
		public TerminalNode RightBracket() { return getToken(ISQParser.RightBracket, 0); }
		public List<CallParasContext> callParas() {
			return getRuleContexts(CallParasContext.class);
		}
		public CallParasContext callParas(int i) {
			return getRuleContext(CallParasContext.class,i);
		}
		public TerminalNode Comma() { return getToken(ISQParser.Comma, 0); }
		public CallParasContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_callParas; }
	}

	public final CallParasContext callParas() throws RecognitionException {
		return callParas(0);
	}

	private CallParasContext callParas(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		CallParasContext _localctx = new CallParasContext(_ctx, _parentState);
		CallParasContext _prevctx = _localctx;
		int _startState = 20;
		enterRecursionRule(_localctx, 20, RULE_callParas, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(172);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,9,_ctx) ) {
			case 1:
				{
				setState(164);
				varType();
				setState(165);
				match(Identifier);
				}
				break;
			case 2:
				{
				setState(167);
				varType();
				setState(168);
				match(Identifier);
				setState(169);
				match(LeftBracket);
				setState(170);
				match(RightBracket);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(179);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new CallParasContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_callParas);
					setState(174);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(175);
					match(Comma);
					setState(176);
					callParas(2);
					}
					} 
				}
				setState(181);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ProcedureMainContext extends ParserRuleContext {
		public TerminalNode Procedure() { return getToken(ISQParser.Procedure, 0); }
		public TerminalNode Main() { return getToken(ISQParser.Main, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public TerminalNode LeftBrace() { return getToken(ISQParser.LeftBrace, 0); }
		public ProcedureBodyContext procedureBody() {
			return getRuleContext(ProcedureBodyContext.class,0);
		}
		public TerminalNode RightBrace() { return getToken(ISQParser.RightBrace, 0); }
		public ProcedureMainContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_procedureMain; }
	}

	public final ProcedureMainContext procedureMain() throws RecognitionException {
		ProcedureMainContext _localctx = new ProcedureMainContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_procedureMain);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(182);
			match(Procedure);
			setState(183);
			match(Main);
			setState(184);
			match(LeftParen);
			setState(185);
			match(RightParen);
			setState(186);
			match(LeftBrace);
			setState(187);
			procedureBody();
			setState(188);
			match(RightBrace);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ProcedureBodyContext extends ParserRuleContext {
		public StatementBlockContext statementBlock() {
			return getRuleContext(StatementBlockContext.class,0);
		}
		public ReturnStatementContext returnStatement() {
			return getRuleContext(ReturnStatementContext.class,0);
		}
		public ProcedureBodyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_procedureBody; }
	}

	public final ProcedureBodyContext procedureBody() throws RecognitionException {
		ProcedureBodyContext _localctx = new ProcedureBodyContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_procedureBody);
		try {
			setState(194);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,11,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(190);
				statementBlock();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(191);
				statementBlock();
				setState(192);
				returnStatement();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class StatementContext extends ParserRuleContext {
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
	 
		public StatementContext() { }
		public void copyFrom(StatementContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class QbitinitdefContext extends StatementContext {
		public QbitInitStatementContext qbitInitStatement() {
			return getRuleContext(QbitInitStatementContext.class,0);
		}
		public QbitinitdefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class CinassigndefContext extends StatementContext {
		public CintAssignContext cintAssign() {
			return getRuleContext(CintAssignContext.class,0);
		}
		public CinassigndefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class FordefContext extends StatementContext {
		public ForStatementContext forStatement() {
			return getRuleContext(ForStatementContext.class,0);
		}
		public FordefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class PrintdefContext extends StatementContext {
		public PrintStatementContext printStatement() {
			return getRuleContext(PrintStatementContext.class,0);
		}
		public PrintdefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class CalldefContext extends StatementContext {
		public CallStatementContext callStatement() {
			return getRuleContext(CallStatementContext.class,0);
		}
		public CalldefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class QbitunitarydefContext extends StatementContext {
		public QbitUnitaryStatementContext qbitUnitaryStatement() {
			return getRuleContext(QbitUnitaryStatementContext.class,0);
		}
		public QbitunitarydefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class WhiledefContext extends StatementContext {
		public WhileStatementContext whileStatement() {
			return getRuleContext(WhileStatementContext.class,0);
		}
		public WhiledefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class VardefContext extends StatementContext {
		public DefclauseContext defclause() {
			return getRuleContext(DefclauseContext.class,0);
		}
		public VardefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class PassdefContext extends StatementContext {
		public PassStatementContext passStatement() {
			return getRuleContext(PassStatementContext.class,0);
		}
		public PassdefContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class IfdefContext extends StatementContext {
		public IfStatementContext ifStatement() {
			return getRuleContext(IfStatementContext.class,0);
		}
		public IfdefContext(StatementContext ctx) { copyFrom(ctx); }
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_statement);
		try {
			setState(206);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,12,_ctx) ) {
			case 1:
				_localctx = new QbitinitdefContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(196);
				qbitInitStatement();
				}
				break;
			case 2:
				_localctx = new QbitunitarydefContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(197);
				qbitUnitaryStatement();
				}
				break;
			case 3:
				_localctx = new CinassigndefContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(198);
				cintAssign();
				}
				break;
			case 4:
				_localctx = new IfdefContext(_localctx);
				enterOuterAlt(_localctx, 4);
				{
				setState(199);
				ifStatement();
				}
				break;
			case 5:
				_localctx = new CalldefContext(_localctx);
				enterOuterAlt(_localctx, 5);
				{
				setState(200);
				callStatement();
				}
				break;
			case 6:
				_localctx = new WhiledefContext(_localctx);
				enterOuterAlt(_localctx, 6);
				{
				setState(201);
				whileStatement();
				}
				break;
			case 7:
				_localctx = new FordefContext(_localctx);
				enterOuterAlt(_localctx, 7);
				{
				setState(202);
				forStatement();
				}
				break;
			case 8:
				_localctx = new PrintdefContext(_localctx);
				enterOuterAlt(_localctx, 8);
				{
				setState(203);
				printStatement();
				}
				break;
			case 9:
				_localctx = new PassdefContext(_localctx);
				enterOuterAlt(_localctx, 9);
				{
				setState(204);
				passStatement();
				}
				break;
			case 10:
				_localctx = new VardefContext(_localctx);
				enterOuterAlt(_localctx, 10);
				{
				setState(205);
				defclause();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class StatementBlockContext extends ParserRuleContext {
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public StatementBlockContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statementBlock; }
	}

	public final StatementBlockContext statementBlock() throws RecognitionException {
		StatementBlockContext _localctx = new StatementBlockContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_statementBlock);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(209); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(208);
				statement();
				}
				}
				setState(211); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << If) | (1L << For) | (1L << While) | (1L << Int) | (1L << Qbit) | (1L << H) | (1L << X) | (1L << Y) | (1L << Z) | (1L << S) | (1L << T) | (1L << CZ) | (1L << CX) | (1L << CNOT) | (1L << Print) | (1L << Pass) | (1L << Identifier) | (1L << Number))) != 0) );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class UGateContext extends ParserRuleContext {
		public TerminalNode H() { return getToken(ISQParser.H, 0); }
		public TerminalNode X() { return getToken(ISQParser.X, 0); }
		public TerminalNode Y() { return getToken(ISQParser.Y, 0); }
		public TerminalNode Z() { return getToken(ISQParser.Z, 0); }
		public TerminalNode S() { return getToken(ISQParser.S, 0); }
		public TerminalNode T() { return getToken(ISQParser.T, 0); }
		public TerminalNode CZ() { return getToken(ISQParser.CZ, 0); }
		public TerminalNode CX() { return getToken(ISQParser.CX, 0); }
		public TerminalNode CNOT() { return getToken(ISQParser.CNOT, 0); }
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public UGateContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_uGate; }
	}

	public final UGateContext uGate() throws RecognitionException {
		UGateContext _localctx = new UGateContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_uGate);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(213);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << H) | (1L << X) | (1L << Y) | (1L << Z) | (1L << S) | (1L << T) | (1L << CZ) | (1L << CX) | (1L << CNOT) | (1L << Identifier))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VariableContext extends ParserRuleContext {
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode Number() { return getToken(ISQParser.Number, 0); }
		public TerminalNode LeftBracket() { return getToken(ISQParser.LeftBracket, 0); }
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode RightBracket() { return getToken(ISQParser.RightBracket, 0); }
		public VariableContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variable; }
	}

	public final VariableContext variable() throws RecognitionException {
		VariableContext _localctx = new VariableContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_variable);
		try {
			setState(222);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,14,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(215);
				match(Identifier);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(216);
				match(Number);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(217);
				match(Identifier);
				setState(218);
				match(LeftBracket);
				setState(219);
				variable();
				setState(220);
				match(RightBracket);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VariableListContext extends ParserRuleContext {
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public List<VariableListContext> variableList() {
			return getRuleContexts(VariableListContext.class);
		}
		public VariableListContext variableList(int i) {
			return getRuleContext(VariableListContext.class,i);
		}
		public TerminalNode Comma() { return getToken(ISQParser.Comma, 0); }
		public VariableListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variableList; }
	}

	public final VariableListContext variableList() throws RecognitionException {
		return variableList(0);
	}

	private VariableListContext variableList(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		VariableListContext _localctx = new VariableListContext(_ctx, _parentState);
		VariableListContext _prevctx = _localctx;
		int _startState = 34;
		enterRecursionRule(_localctx, 34, RULE_variableList, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(225);
			variable();
			}
			_ctx.stop = _input.LT(-1);
			setState(232);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,15,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new VariableListContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_variableList);
					setState(227);
					if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
					setState(228);
					match(Comma);
					setState(229);
					variableList(3);
					}
					} 
				}
				setState(234);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,15,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class BinopPlusContext extends ParserRuleContext {
		public TerminalNode Plus() { return getToken(ISQParser.Plus, 0); }
		public TerminalNode Minus() { return getToken(ISQParser.Minus, 0); }
		public BinopPlusContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_binopPlus; }
	}

	public final BinopPlusContext binopPlus() throws RecognitionException {
		BinopPlusContext _localctx = new BinopPlusContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_binopPlus);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(235);
			_la = _input.LA(1);
			if ( !(_la==Plus || _la==Minus) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class BinopMultContext extends ParserRuleContext {
		public TerminalNode Mult() { return getToken(ISQParser.Mult, 0); }
		public TerminalNode Div() { return getToken(ISQParser.Div, 0); }
		public BinopMultContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_binopMult; }
	}

	public final BinopMultContext binopMult() throws RecognitionException {
		BinopMultContext _localctx = new BinopMultContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_binopMult);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(237);
			_la = _input.LA(1);
			if ( !(_la==Mult || _la==Div) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExpressionContext extends ParserRuleContext {
		public List<MultexpContext> multexp() {
			return getRuleContexts(MultexpContext.class);
		}
		public MultexpContext multexp(int i) {
			return getRuleContext(MultexpContext.class,i);
		}
		public List<BinopPlusContext> binopPlus() {
			return getRuleContexts(BinopPlusContext.class);
		}
		public BinopPlusContext binopPlus(int i) {
			return getRuleContext(BinopPlusContext.class,i);
		}
		public ExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expression; }
	}

	public final ExpressionContext expression() throws RecognitionException {
		ExpressionContext _localctx = new ExpressionContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_expression);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(239);
			multexp();
			setState(245);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==Plus || _la==Minus) {
				{
				{
				setState(240);
				binopPlus();
				setState(241);
				multexp();
				}
				}
				setState(247);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MultexpContext extends ParserRuleContext {
		public List<AtomexpContext> atomexp() {
			return getRuleContexts(AtomexpContext.class);
		}
		public AtomexpContext atomexp(int i) {
			return getRuleContext(AtomexpContext.class,i);
		}
		public List<BinopMultContext> binopMult() {
			return getRuleContexts(BinopMultContext.class);
		}
		public BinopMultContext binopMult(int i) {
			return getRuleContext(BinopMultContext.class,i);
		}
		public MultexpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_multexp; }
	}

	public final MultexpContext multexp() throws RecognitionException {
		MultexpContext _localctx = new MultexpContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_multexp);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(248);
			atomexp();
			setState(254);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==Mult || _la==Div) {
				{
				{
				setState(249);
				binopMult();
				setState(250);
				atomexp();
				}
				}
				setState(256);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AtomexpContext extends ParserRuleContext {
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public AtomexpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_atomexp; }
	}

	public final AtomexpContext atomexp() throws RecognitionException {
		AtomexpContext _localctx = new AtomexpContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_atomexp);
		try {
			setState(262);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case Identifier:
			case Number:
				enterOuterAlt(_localctx, 1);
				{
				setState(257);
				variable();
				}
				break;
			case LeftParen:
				enterOuterAlt(_localctx, 2);
				{
				setState(258);
				match(LeftParen);
				setState(259);
				expression();
				setState(260);
				match(RightParen);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MExpressionContext extends ParserRuleContext {
		public TerminalNode M() { return getToken(ISQParser.M, 0); }
		public TerminalNode Less() { return getToken(ISQParser.Less, 0); }
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode Greater() { return getToken(ISQParser.Greater, 0); }
		public MExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mExpression; }
	}

	public final MExpressionContext mExpression() throws RecognitionException {
		MExpressionContext _localctx = new MExpressionContext(_ctx, getState());
		enterRule(_localctx, 46, RULE_mExpression);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(264);
			match(M);
			setState(265);
			match(Less);
			setState(266);
			variable();
			setState(267);
			match(Greater);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AssociationContext extends ParserRuleContext {
		public TerminalNode Equal() { return getToken(ISQParser.Equal, 0); }
		public TerminalNode GreaterEqual() { return getToken(ISQParser.GreaterEqual, 0); }
		public TerminalNode LessEqual() { return getToken(ISQParser.LessEqual, 0); }
		public TerminalNode Greater() { return getToken(ISQParser.Greater, 0); }
		public TerminalNode Less() { return getToken(ISQParser.Less, 0); }
		public AssociationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_association; }
	}

	public final AssociationContext association() throws RecognitionException {
		AssociationContext _localctx = new AssociationContext(_ctx, getState());
		enterRule(_localctx, 48, RULE_association);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(269);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << Less) | (1L << Greater) | (1L << Equal) | (1L << LessEqual) | (1L << GreaterEqual))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class QbitInitStatementContext extends ParserRuleContext {
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode Assign() { return getToken(ISQParser.Assign, 0); }
		public TerminalNode KetZero() { return getToken(ISQParser.KetZero, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public QbitInitStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_qbitInitStatement; }
	}

	public final QbitInitStatementContext qbitInitStatement() throws RecognitionException {
		QbitInitStatementContext _localctx = new QbitInitStatementContext(_ctx, getState());
		enterRule(_localctx, 50, RULE_qbitInitStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(271);
			variable();
			setState(272);
			match(Assign);
			setState(273);
			match(KetZero);
			setState(274);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class QbitUnitaryStatementContext extends ParserRuleContext {
		public UGateContext uGate() {
			return getRuleContext(UGateContext.class,0);
		}
		public TerminalNode Less() { return getToken(ISQParser.Less, 0); }
		public VariableListContext variableList() {
			return getRuleContext(VariableListContext.class,0);
		}
		public TerminalNode Greater() { return getToken(ISQParser.Greater, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public QbitUnitaryStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_qbitUnitaryStatement; }
	}

	public final QbitUnitaryStatementContext qbitUnitaryStatement() throws RecognitionException {
		QbitUnitaryStatementContext _localctx = new QbitUnitaryStatementContext(_ctx, getState());
		enterRule(_localctx, 52, RULE_qbitUnitaryStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(276);
			uGate();
			setState(277);
			match(Less);
			setState(278);
			variableList(0);
			setState(279);
			match(Greater);
			setState(280);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CintAssignContext extends ParserRuleContext {
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode Assign() { return getToken(ISQParser.Assign, 0); }
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public MExpressionContext mExpression() {
			return getRuleContext(MExpressionContext.class,0);
		}
		public CallStatementContext callStatement() {
			return getRuleContext(CallStatementContext.class,0);
		}
		public CintAssignContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_cintAssign; }
	}

	public final CintAssignContext cintAssign() throws RecognitionException {
		CintAssignContext _localctx = new CintAssignContext(_ctx, getState());
		enterRule(_localctx, 54, RULE_cintAssign);
		try {
			setState(296);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,19,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(282);
				variable();
				setState(283);
				match(Assign);
				setState(284);
				expression();
				setState(285);
				match(Semi);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(287);
				variable();
				setState(288);
				match(Assign);
				setState(289);
				mExpression();
				setState(290);
				match(Semi);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(292);
				variable();
				setState(293);
				match(Assign);
				setState(294);
				callStatement();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class RegionBodyContext extends ParserRuleContext {
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public TerminalNode LeftBrace() { return getToken(ISQParser.LeftBrace, 0); }
		public StatementBlockContext statementBlock() {
			return getRuleContext(StatementBlockContext.class,0);
		}
		public TerminalNode RightBrace() { return getToken(ISQParser.RightBrace, 0); }
		public RegionBodyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_regionBody; }
	}

	public final RegionBodyContext regionBody() throws RecognitionException {
		RegionBodyContext _localctx = new RegionBodyContext(_ctx, getState());
		enterRule(_localctx, 56, RULE_regionBody);
		try {
			setState(303);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case If:
			case For:
			case While:
			case Int:
			case Qbit:
			case H:
			case X:
			case Y:
			case Z:
			case S:
			case T:
			case CZ:
			case CX:
			case CNOT:
			case Print:
			case Pass:
			case Identifier:
			case Number:
				enterOuterAlt(_localctx, 1);
				{
				setState(298);
				statement();
				}
				break;
			case LeftBrace:
				enterOuterAlt(_localctx, 2);
				{
				setState(299);
				match(LeftBrace);
				setState(300);
				statementBlock();
				setState(301);
				match(RightBrace);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class IfStatementContext extends ParserRuleContext {
		public TerminalNode If() { return getToken(ISQParser.If, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public AssociationContext association() {
			return getRuleContext(AssociationContext.class,0);
		}
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public List<RegionBodyContext> regionBody() {
			return getRuleContexts(RegionBodyContext.class);
		}
		public RegionBodyContext regionBody(int i) {
			return getRuleContext(RegionBodyContext.class,i);
		}
		public TerminalNode Else() { return getToken(ISQParser.Else, 0); }
		public IfStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ifStatement; }
	}

	public final IfStatementContext ifStatement() throws RecognitionException {
		IfStatementContext _localctx = new IfStatementContext(_ctx, getState());
		enterRule(_localctx, 58, RULE_ifStatement);
		try {
			setState(323);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,21,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(305);
				match(If);
				setState(306);
				match(LeftParen);
				setState(307);
				expression();
				setState(308);
				association();
				setState(309);
				expression();
				setState(310);
				match(RightParen);
				setState(311);
				regionBody();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(313);
				match(If);
				setState(314);
				match(LeftParen);
				setState(315);
				expression();
				setState(316);
				association();
				setState(317);
				expression();
				setState(318);
				match(RightParen);
				setState(319);
				regionBody();
				setState(320);
				match(Else);
				setState(321);
				regionBody();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ForStatementContext extends ParserRuleContext {
		public TerminalNode For() { return getToken(ISQParser.For, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode Assign() { return getToken(ISQParser.Assign, 0); }
		public List<VariableContext> variable() {
			return getRuleContexts(VariableContext.class);
		}
		public VariableContext variable(int i) {
			return getRuleContext(VariableContext.class,i);
		}
		public TerminalNode To() { return getToken(ISQParser.To, 0); }
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public RegionBodyContext regionBody() {
			return getRuleContext(RegionBodyContext.class,0);
		}
		public ForStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_forStatement; }
	}

	public final ForStatementContext forStatement() throws RecognitionException {
		ForStatementContext _localctx = new ForStatementContext(_ctx, getState());
		enterRule(_localctx, 60, RULE_forStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(325);
			match(For);
			setState(326);
			match(LeftParen);
			setState(327);
			match(Identifier);
			setState(328);
			match(Assign);
			setState(329);
			variable();
			setState(330);
			match(To);
			setState(331);
			variable();
			setState(332);
			match(RightParen);
			setState(333);
			regionBody();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class WhileStatementContext extends ParserRuleContext {
		public TerminalNode While() { return getToken(ISQParser.While, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public AssociationContext association() {
			return getRuleContext(AssociationContext.class,0);
		}
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public RegionBodyContext regionBody() {
			return getRuleContext(RegionBodyContext.class,0);
		}
		public WhileStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_whileStatement; }
	}

	public final WhileStatementContext whileStatement() throws RecognitionException {
		WhileStatementContext _localctx = new WhileStatementContext(_ctx, getState());
		enterRule(_localctx, 62, RULE_whileStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(335);
			match(While);
			setState(336);
			match(LeftParen);
			setState(337);
			expression();
			setState(338);
			association();
			setState(339);
			expression();
			setState(340);
			match(RightParen);
			setState(341);
			regionBody();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CallStatementContext extends ParserRuleContext {
		public TerminalNode Identifier() { return getToken(ISQParser.Identifier, 0); }
		public TerminalNode LeftParen() { return getToken(ISQParser.LeftParen, 0); }
		public TerminalNode RightParen() { return getToken(ISQParser.RightParen, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public VariableListContext variableList() {
			return getRuleContext(VariableListContext.class,0);
		}
		public CallStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_callStatement; }
	}

	public final CallStatementContext callStatement() throws RecognitionException {
		CallStatementContext _localctx = new CallStatementContext(_ctx, getState());
		enterRule(_localctx, 64, RULE_callStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(343);
			match(Identifier);
			setState(344);
			match(LeftParen);
			setState(347);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case RightParen:
				{
				}
				break;
			case Identifier:
			case Number:
				{
				setState(346);
				variableList(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(349);
			match(RightParen);
			setState(350);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PrintStatementContext extends ParserRuleContext {
		public TerminalNode Print() { return getToken(ISQParser.Print, 0); }
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public PrintStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_printStatement; }
	}

	public final PrintStatementContext printStatement() throws RecognitionException {
		PrintStatementContext _localctx = new PrintStatementContext(_ctx, getState());
		enterRule(_localctx, 66, RULE_printStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(352);
			match(Print);
			setState(353);
			variable();
			setState(354);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PassStatementContext extends ParserRuleContext {
		public TerminalNode Pass() { return getToken(ISQParser.Pass, 0); }
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public PassStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_passStatement; }
	}

	public final PassStatementContext passStatement() throws RecognitionException {
		PassStatementContext _localctx = new PassStatementContext(_ctx, getState());
		enterRule(_localctx, 68, RULE_passStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(356);
			match(Pass);
			setState(357);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ReturnStatementContext extends ParserRuleContext {
		public TerminalNode Return() { return getToken(ISQParser.Return, 0); }
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public TerminalNode Semi() { return getToken(ISQParser.Semi, 0); }
		public ReturnStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_returnStatement; }
	}

	public final ReturnStatementContext returnStatement() throws RecognitionException {
		ReturnStatementContext _localctx = new ReturnStatementContext(_ctx, getState());
		enterRule(_localctx, 70, RULE_returnStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(359);
			match(Return);
			setState(360);
			variable();
			setState(361);
			match(Semi);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 7:
			return idlist_sempred((IdlistContext)_localctx, predIndex);
		case 10:
			return callParas_sempred((CallParasContext)_localctx, predIndex);
		case 17:
			return variableList_sempred((VariableListContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean idlist_sempred(IdlistContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean callParas_sempred(CallParasContext _localctx, int predIndex) {
		switch (predIndex) {
		case 1:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean variableList_sempred(VariableListContext _localctx, int predIndex) {
		switch (predIndex) {
		case 2:
			return precpred(_ctx, 2);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\66\u016e\4\2\t\2"+
		"\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\3\2\7\2L\n\2\f\2\16\2O\13\2\3\2\6\2R\n\2"+
		"\r\2\16\2S\3\2\3\2\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4"+
		"\5\4e\n\4\3\5\3\5\3\5\5\5j\n\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\5\6s\n\6\3"+
		"\7\3\7\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3\t\3\t\5\t\u0081\n\t\3\t\3\t\3"+
		"\t\7\t\u0086\n\t\f\t\16\t\u0089\13\t\3\n\7\n\u008c\n\n\f\n\16\n\u008f"+
		"\13\n\3\n\3\n\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3"+
		"\13\3\13\3\13\3\13\3\13\3\13\5\13\u00a4\n\13\3\f\3\f\3\f\3\f\3\f\3\f\3"+
		"\f\3\f\3\f\5\f\u00af\n\f\3\f\3\f\3\f\7\f\u00b4\n\f\f\f\16\f\u00b7\13\f"+
		"\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\16\3\16\3\16\3\16\5\16\u00c5\n\16\3"+
		"\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\5\17\u00d1\n\17\3\20"+
		"\6\20\u00d4\n\20\r\20\16\20\u00d5\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3"+
		"\22\3\22\5\22\u00e1\n\22\3\23\3\23\3\23\3\23\3\23\3\23\7\23\u00e9\n\23"+
		"\f\23\16\23\u00ec\13\23\3\24\3\24\3\25\3\25\3\26\3\26\3\26\3\26\7\26\u00f6"+
		"\n\26\f\26\16\26\u00f9\13\26\3\27\3\27\3\27\3\27\7\27\u00ff\n\27\f\27"+
		"\16\27\u0102\13\27\3\30\3\30\3\30\3\30\3\30\5\30\u0109\n\30\3\31\3\31"+
		"\3\31\3\31\3\31\3\32\3\32\3\33\3\33\3\33\3\33\3\33\3\34\3\34\3\34\3\34"+
		"\3\34\3\34\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\35\3\35"+
		"\3\35\3\35\5\35\u012b\n\35\3\36\3\36\3\36\3\36\3\36\5\36\u0132\n\36\3"+
		"\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3"+
		"\37\3\37\3\37\3\37\5\37\u0146\n\37\3 \3 \3 \3 \3 \3 \3 \3 \3 \3 \3!\3"+
		"!\3!\3!\3!\3!\3!\3!\3\"\3\"\3\"\3\"\5\"\u015e\n\"\3\"\3\"\3\"\3#\3#\3"+
		"#\3#\3$\3$\3$\3%\3%\3%\3%\3%\2\5\20\26$&\2\4\6\b\n\f\16\20\22\24\26\30"+
		"\32\34\36 \"$&(*,.\60\62\64\668:<>@BDFH\2\t\4\2))\60\60\3\2\22\23\4\2"+
		"\20\20\22\22\4\2\24\34\65\65\3\2#$\3\2%&\4\2\'(\61\63\2\u016b\2M\3\2\2"+
		"\2\4W\3\2\2\2\6d\3\2\2\2\bi\3\2\2\2\nr\3\2\2\2\ft\3\2\2\2\16v\3\2\2\2"+
		"\20\u0080\3\2\2\2\22\u008d\3\2\2\2\24\u00a3\3\2\2\2\26\u00ae\3\2\2\2\30"+
		"\u00b8\3\2\2\2\32\u00c4\3\2\2\2\34\u00d0\3\2\2\2\36\u00d3\3\2\2\2 \u00d7"+
		"\3\2\2\2\"\u00e0\3\2\2\2$\u00e2\3\2\2\2&\u00ed\3\2\2\2(\u00ef\3\2\2\2"+
		"*\u00f1\3\2\2\2,\u00fa\3\2\2\2.\u0108\3\2\2\2\60\u010a\3\2\2\2\62\u010f"+
		"\3\2\2\2\64\u0111\3\2\2\2\66\u0116\3\2\2\28\u012a\3\2\2\2:\u0131\3\2\2"+
		"\2<\u0145\3\2\2\2>\u0147\3\2\2\2@\u0151\3\2\2\2B\u0159\3\2\2\2D\u0162"+
		"\3\2\2\2F\u0166\3\2\2\2H\u0169\3\2\2\2JL\5\4\3\2KJ\3\2\2\2LO\3\2\2\2M"+
		"K\3\2\2\2MN\3\2\2\2NQ\3\2\2\2OM\3\2\2\2PR\5\16\b\2QP\3\2\2\2RS\3\2\2\2"+
		"SQ\3\2\2\2ST\3\2\2\2TU\3\2\2\2UV\5\22\n\2V\3\3\2\2\2WX\7\37\2\2XY\7\65"+
		"\2\2YZ\7\"\2\2Z[\7,\2\2[\\\5\6\4\2\\]\7-\2\2]^\7\60\2\2^\5\3\2\2\2_e\5"+
		"\b\5\2`a\5\b\5\2ab\t\2\2\2bc\5\6\4\2ce\3\2\2\2d_\3\2\2\2d`\3\2\2\2e\7"+
		"\3\2\2\2fj\5\n\6\2gh\7$\2\2hj\5\n\6\2if\3\2\2\2ig\3\2\2\2j\t\3\2\2\2k"+
		"s\7\66\2\2lm\7\66\2\2mn\7#\2\2ns\7\66\2\2op\7\66\2\2pq\7$\2\2qs\7\66\2"+
		"\2rk\3\2\2\2rl\3\2\2\2ro\3\2\2\2s\13\3\2\2\2tu\t\3\2\2u\r\3\2\2\2vw\5"+
		"\f\7\2wx\5\20\t\2xy\7\60\2\2y\17\3\2\2\2z{\b\t\1\2{\u0081\7\65\2\2|}\7"+
		"\65\2\2}~\7,\2\2~\177\7\66\2\2\177\u0081\7-\2\2\u0080z\3\2\2\2\u0080|"+
		"\3\2\2\2\u0081\u0087\3\2\2\2\u0082\u0083\f\3\2\2\u0083\u0084\7)\2\2\u0084"+
		"\u0086\5\20\t\4\u0085\u0082\3\2\2\2\u0086\u0089\3\2\2\2\u0087\u0085\3"+
		"\2\2\2\u0087\u0088\3\2\2\2\u0088\21\3\2\2\2\u0089\u0087\3\2\2\2\u008a"+
		"\u008c\5\24\13\2\u008b\u008a\3\2\2\2\u008c\u008f\3\2\2\2\u008d\u008b\3"+
		"\2\2\2\u008d\u008e\3\2\2\2\u008e\u0090\3\2\2\2\u008f\u008d\3\2\2\2\u0090"+
		"\u0091\5\30\r\2\u0091\23\3\2\2\2\u0092\u0093\t\4\2\2\u0093\u0094\7\65"+
		"\2\2\u0094\u0095\7*\2\2\u0095\u0096\7+\2\2\u0096\u0097\7.\2\2\u0097\u0098"+
		"\5\32\16\2\u0098\u0099\7/\2\2\u0099\u00a4\3\2\2\2\u009a\u009b\t\4\2\2"+
		"\u009b\u009c\7\65\2\2\u009c\u009d\7*\2\2\u009d\u009e\5\26\f\2\u009e\u009f"+
		"\7+\2\2\u009f\u00a0\7.\2\2\u00a0\u00a1\5\32\16\2\u00a1\u00a2\7/\2\2\u00a2"+
		"\u00a4\3\2\2\2\u00a3\u0092\3\2\2\2\u00a3\u009a\3\2\2\2\u00a4\25\3\2\2"+
		"\2\u00a5\u00a6\b\f\1\2\u00a6\u00a7\5\f\7\2\u00a7\u00a8\7\65\2\2\u00a8"+
		"\u00af\3\2\2\2\u00a9\u00aa\5\f\7\2\u00aa\u00ab\7\65\2\2\u00ab\u00ac\7"+
		",\2\2\u00ac\u00ad\7-\2\2\u00ad\u00af\3\2\2\2\u00ae\u00a5\3\2\2\2\u00ae"+
		"\u00a9\3\2\2\2\u00af\u00b5\3\2\2\2\u00b0\u00b1\f\3\2\2\u00b1\u00b2\7)"+
		"\2\2\u00b2\u00b4\5\26\f\4\u00b3\u00b0\3\2\2\2\u00b4\u00b7\3\2\2\2\u00b5"+
		"\u00b3\3\2\2\2\u00b5\u00b6\3\2\2\2\u00b6\27\3\2\2\2\u00b7\u00b5\3\2\2"+
		"\2\u00b8\u00b9\7\20\2\2\u00b9\u00ba\7\21\2\2\u00ba\u00bb\7*\2\2\u00bb"+
		"\u00bc\7+\2\2\u00bc\u00bd\7.\2\2\u00bd\u00be\5\32\16\2\u00be\u00bf\7/"+
		"\2\2\u00bf\31\3\2\2\2\u00c0\u00c5\5\36\20\2\u00c1\u00c2\5\36\20\2\u00c2"+
		"\u00c3\5H%\2\u00c3\u00c5\3\2\2\2\u00c4\u00c0\3\2\2\2\u00c4\u00c1\3\2\2"+
		"\2\u00c5\33\3\2\2\2\u00c6\u00d1\5\64\33\2\u00c7\u00d1\5\66\34\2\u00c8"+
		"\u00d1\58\35\2\u00c9\u00d1\5<\37\2\u00ca\u00d1\5B\"\2\u00cb\u00d1\5@!"+
		"\2\u00cc\u00d1\5> \2\u00cd\u00d1\5D#\2\u00ce\u00d1\5F$\2\u00cf\u00d1\5"+
		"\16\b\2\u00d0\u00c6\3\2\2\2\u00d0\u00c7\3\2\2\2\u00d0\u00c8\3\2\2\2\u00d0"+
		"\u00c9\3\2\2\2\u00d0\u00ca\3\2\2\2\u00d0\u00cb\3\2\2\2\u00d0\u00cc\3\2"+
		"\2\2\u00d0\u00cd\3\2\2\2\u00d0\u00ce\3\2\2\2\u00d0\u00cf\3\2\2\2\u00d1"+
		"\35\3\2\2\2\u00d2\u00d4\5\34\17\2\u00d3\u00d2\3\2\2\2\u00d4\u00d5\3\2"+
		"\2\2\u00d5\u00d3\3\2\2\2\u00d5\u00d6\3\2\2\2\u00d6\37\3\2\2\2\u00d7\u00d8"+
		"\t\5\2\2\u00d8!\3\2\2\2\u00d9\u00e1\7\65\2\2\u00da\u00e1\7\66\2\2\u00db"+
		"\u00dc\7\65\2\2\u00dc\u00dd\7,\2\2\u00dd\u00de\5\"\22\2\u00de\u00df\7"+
		"-\2\2\u00df\u00e1\3\2\2\2\u00e0\u00d9\3\2\2\2\u00e0\u00da\3\2\2\2\u00e0"+
		"\u00db\3\2\2\2\u00e1#\3\2\2\2\u00e2\u00e3\b\23\1\2\u00e3\u00e4\5\"\22"+
		"\2\u00e4\u00ea\3\2\2\2\u00e5\u00e6\f\4\2\2\u00e6\u00e7\7)\2\2\u00e7\u00e9"+
		"\5$\23\5\u00e8\u00e5\3\2\2\2\u00e9\u00ec\3\2\2\2\u00ea\u00e8\3\2\2\2\u00ea"+
		"\u00eb\3\2\2\2\u00eb%\3\2\2\2\u00ec\u00ea\3\2\2\2\u00ed\u00ee\t\6\2\2"+
		"\u00ee\'\3\2\2\2\u00ef\u00f0\t\7\2\2\u00f0)\3\2\2\2\u00f1\u00f7\5,\27"+
		"\2\u00f2\u00f3\5&\24\2\u00f3\u00f4\5,\27\2\u00f4\u00f6\3\2\2\2\u00f5\u00f2"+
		"\3\2\2\2\u00f6\u00f9\3\2\2\2\u00f7\u00f5\3\2\2\2\u00f7\u00f8\3\2\2\2\u00f8"+
		"+\3\2\2\2\u00f9\u00f7\3\2\2\2\u00fa\u0100\5.\30\2\u00fb\u00fc\5(\25\2"+
		"\u00fc\u00fd\5.\30\2\u00fd\u00ff\3\2\2\2\u00fe\u00fb\3\2\2\2\u00ff\u0102"+
		"\3\2\2\2\u0100\u00fe\3\2\2\2\u0100\u0101\3\2\2\2\u0101-\3\2\2\2\u0102"+
		"\u0100\3\2\2\2\u0103\u0109\5\"\22\2\u0104\u0105\7*\2\2\u0105\u0106\5*"+
		"\26\2\u0106\u0107\7+\2\2\u0107\u0109\3\2\2\2\u0108\u0103\3\2\2\2\u0108"+
		"\u0104\3\2\2\2\u0109/\3\2\2\2\u010a\u010b\7\35\2\2\u010b\u010c\7\'\2\2"+
		"\u010c\u010d\5\"\22\2\u010d\u010e\7(\2\2\u010e\61\3\2\2\2\u010f\u0110"+
		"\t\b\2\2\u0110\63\3\2\2\2\u0111\u0112\5\"\22\2\u0112\u0113\7\"\2\2\u0113"+
		"\u0114\7\64\2\2\u0114\u0115\7\60\2\2\u0115\65\3\2\2\2\u0116\u0117\5 \21"+
		"\2\u0117\u0118\7\'\2\2\u0118\u0119\5$\23\2\u0119\u011a\7(\2\2\u011a\u011b"+
		"\7\60\2\2\u011b\67\3\2\2\2\u011c\u011d\5\"\22\2\u011d\u011e\7\"\2\2\u011e"+
		"\u011f\5*\26\2\u011f\u0120\7\60\2\2\u0120\u012b\3\2\2\2\u0121\u0122\5"+
		"\"\22\2\u0122\u0123\7\"\2\2\u0123\u0124\5\60\31\2\u0124\u0125\7\60\2\2"+
		"\u0125\u012b\3\2\2\2\u0126\u0127\5\"\22\2\u0127\u0128\7\"\2\2\u0128\u0129"+
		"\5B\"\2\u0129\u012b\3\2\2\2\u012a\u011c\3\2\2\2\u012a\u0121\3\2\2\2\u012a"+
		"\u0126\3\2\2\2\u012b9\3\2\2\2\u012c\u0132\5\34\17\2\u012d\u012e\7.\2\2"+
		"\u012e\u012f\5\36\20\2\u012f\u0130\7/\2\2\u0130\u0132\3\2\2\2\u0131\u012c"+
		"\3\2\2\2\u0131\u012d\3\2\2\2\u0132;\3\2\2\2\u0133\u0134\7\7\2\2\u0134"+
		"\u0135\7*\2\2\u0135\u0136\5*\26\2\u0136\u0137\5\62\32\2\u0137\u0138\5"+
		"*\26\2\u0138\u0139\7+\2\2\u0139\u013a\5:\36\2\u013a\u0146\3\2\2\2\u013b"+
		"\u013c\7\7\2\2\u013c\u013d\7*\2\2\u013d\u013e\5*\26\2\u013e\u013f\5\62"+
		"\32\2\u013f\u0140\5*\26\2\u0140\u0141\7+\2\2\u0141\u0142\5:\36\2\u0142"+
		"\u0143\7\t\2\2\u0143\u0144\5:\36\2\u0144\u0146\3\2\2\2\u0145\u0133\3\2"+
		"\2\2\u0145\u013b\3\2\2\2\u0146=\3\2\2\2\u0147\u0148\7\13\2\2\u0148\u0149"+
		"\7*\2\2\u0149\u014a\7\65\2\2\u014a\u014b\7\"\2\2\u014b\u014c\5\"\22\2"+
		"\u014c\u014d\7\f\2\2\u014d\u014e\5\"\22\2\u014e\u014f\7+\2\2\u014f\u0150"+
		"\5:\36\2\u0150?\3\2\2\2\u0151\u0152\7\r\2\2\u0152\u0153\7*\2\2\u0153\u0154"+
		"\5*\26\2\u0154\u0155\5\62\32\2\u0155\u0156\5*\26\2\u0156\u0157\7+\2\2"+
		"\u0157\u0158\5:\36\2\u0158A\3\2\2\2\u0159\u015a\7\65\2\2\u015a\u015d\7"+
		"*\2\2\u015b\u015e\3\2\2\2\u015c\u015e\5$\23\2\u015d\u015b\3\2\2\2\u015d"+
		"\u015c\3\2\2\2\u015e\u015f\3\2\2\2\u015f\u0160\7+\2\2\u0160\u0161\7\60"+
		"\2\2\u0161C\3\2\2\2\u0162\u0163\7\36\2\2\u0163\u0164\5\"\22\2\u0164\u0165"+
		"\7\60\2\2\u0165E\3\2\2\2\u0166\u0167\7 \2\2\u0167\u0168\7\60\2\2\u0168"+
		"G\3\2\2\2\u0169\u016a\7!\2\2\u016a\u016b\5\"\22\2\u016b\u016c\7\60\2\2"+
		"\u016cI\3\2\2\2\31MSdir\u0080\u0087\u008d\u00a3\u00ae\u00b5\u00c4\u00d0"+
		"\u00d5\u00e0\u00ea\u00f7\u0100\u0108\u012a\u0131\u0145\u015d";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}