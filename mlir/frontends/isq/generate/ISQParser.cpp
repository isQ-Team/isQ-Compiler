
// Generated from ISQParser.g4 by ANTLR 4.7.2


#include "ISQParserListener.h"
#include "ISQParserVisitor.h"

#include "ISQParser.h"


using namespace antlrcpp;
using namespace antlr4;

ISQParser::ISQParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

ISQParser::~ISQParser() {
  delete _interpreter;
}

std::string ISQParser::getGrammarFileName() const {
  return "ISQParser.g4";
}

const std::vector<std::string>& ISQParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& ISQParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgramContext ------------------------------------------------------------------

ISQParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::ProgramBodyContext* ISQParser::ProgramContext::programBody() {
  return getRuleContext<ISQParser::ProgramBodyContext>(0);
}

std::vector<ISQParser::GateDefclauseContext *> ISQParser::ProgramContext::gateDefclause() {
  return getRuleContexts<ISQParser::GateDefclauseContext>();
}

ISQParser::GateDefclauseContext* ISQParser::ProgramContext::gateDefclause(size_t i) {
  return getRuleContext<ISQParser::GateDefclauseContext>(i);
}

std::vector<ISQParser::DefclauseContext *> ISQParser::ProgramContext::defclause() {
  return getRuleContexts<ISQParser::DefclauseContext>();
}

ISQParser::DefclauseContext* ISQParser::ProgramContext::defclause(size_t i) {
  return getRuleContext<ISQParser::DefclauseContext>(i);
}


size_t ISQParser::ProgramContext::getRuleIndex() const {
  return ISQParser::RuleProgram;
}

void ISQParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void ISQParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


antlrcpp::Any ISQParser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ProgramContext* ISQParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, ISQParser::RuleProgram);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(75);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ISQParser::Defgate) {
      setState(72);
      gateDefclause();
      setState(77);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(79); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(78);
              defclause();
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(81); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
    setState(83);
    programBody();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GateDefclauseContext ------------------------------------------------------------------

ISQParser::GateDefclauseContext::GateDefclauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::GateDefclauseContext::Defgate() {
  return getToken(ISQParser::Defgate, 0);
}

tree::TerminalNode* ISQParser::GateDefclauseContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::GateDefclauseContext::Assign() {
  return getToken(ISQParser::Assign, 0);
}

tree::TerminalNode* ISQParser::GateDefclauseContext::LeftBracket() {
  return getToken(ISQParser::LeftBracket, 0);
}

ISQParser::MatrixContentsContext* ISQParser::GateDefclauseContext::matrixContents() {
  return getRuleContext<ISQParser::MatrixContentsContext>(0);
}

tree::TerminalNode* ISQParser::GateDefclauseContext::RightBracket() {
  return getToken(ISQParser::RightBracket, 0);
}

tree::TerminalNode* ISQParser::GateDefclauseContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::GateDefclauseContext::getRuleIndex() const {
  return ISQParser::RuleGateDefclause;
}

void ISQParser::GateDefclauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGateDefclause(this);
}

void ISQParser::GateDefclauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGateDefclause(this);
}


antlrcpp::Any ISQParser::GateDefclauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitGateDefclause(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::GateDefclauseContext* ISQParser::gateDefclause() {
  GateDefclauseContext *_localctx = _tracker.createInstance<GateDefclauseContext>(_ctx, getState());
  enterRule(_localctx, 2, ISQParser::RuleGateDefclause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(85);
    match(ISQParser::Defgate);
    setState(86);
    match(ISQParser::Identifier);
    setState(87);
    match(ISQParser::Assign);
    setState(88);
    match(ISQParser::LeftBracket);
    setState(89);
    matrixContents();
    setState(90);
    match(ISQParser::RightBracket);
    setState(91);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatrixContentsContext ------------------------------------------------------------------

ISQParser::MatrixContentsContext::MatrixContentsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t ISQParser::MatrixContentsContext::getRuleIndex() const {
  return ISQParser::RuleMatrixContents;
}

void ISQParser::MatrixContentsContext::copyFrom(MatrixContentsContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- MatrixdefContext ------------------------------------------------------------------

ISQParser::CNumberContext* ISQParser::MatrixdefContext::cNumber() {
  return getRuleContext<ISQParser::CNumberContext>(0);
}

ISQParser::MatrixContentsContext* ISQParser::MatrixdefContext::matrixContents() {
  return getRuleContext<ISQParser::MatrixContentsContext>(0);
}

tree::TerminalNode* ISQParser::MatrixdefContext::Comma() {
  return getToken(ISQParser::Comma, 0);
}

tree::TerminalNode* ISQParser::MatrixdefContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}

ISQParser::MatrixdefContext::MatrixdefContext(MatrixContentsContext *ctx) { copyFrom(ctx); }

void ISQParser::MatrixdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatrixdef(this);
}
void ISQParser::MatrixdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatrixdef(this);
}

antlrcpp::Any ISQParser::MatrixdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitMatrixdef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- MatrixvaldefContext ------------------------------------------------------------------

ISQParser::CNumberContext* ISQParser::MatrixvaldefContext::cNumber() {
  return getRuleContext<ISQParser::CNumberContext>(0);
}

ISQParser::MatrixvaldefContext::MatrixvaldefContext(MatrixContentsContext *ctx) { copyFrom(ctx); }

void ISQParser::MatrixvaldefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatrixvaldef(this);
}
void ISQParser::MatrixvaldefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatrixvaldef(this);
}

antlrcpp::Any ISQParser::MatrixvaldefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitMatrixvaldef(this);
  else
    return visitor->visitChildren(this);
}
ISQParser::MatrixContentsContext* ISQParser::matrixContents() {
  MatrixContentsContext *_localctx = _tracker.createInstance<MatrixContentsContext>(_ctx, getState());
  enterRule(_localctx, 4, ISQParser::RuleMatrixContents);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(98);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<MatrixContentsContext *>(_tracker.createInstance<ISQParser::MatrixvaldefContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(93);
      cNumber();
      break;
    }

    case 2: {
      _localctx = dynamic_cast<MatrixContentsContext *>(_tracker.createInstance<ISQParser::MatrixdefContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(94);
      cNumber();
      setState(95);
      _la = _input->LA(1);
      if (!(_la == ISQParser::Comma

      || _la == ISQParser::Semi)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(96);
      matrixContents();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CNumberContext ------------------------------------------------------------------

ISQParser::CNumberContext::CNumberContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::NumberExprContext* ISQParser::CNumberContext::numberExpr() {
  return getRuleContext<ISQParser::NumberExprContext>(0);
}

tree::TerminalNode* ISQParser::CNumberContext::Minus() {
  return getToken(ISQParser::Minus, 0);
}


size_t ISQParser::CNumberContext::getRuleIndex() const {
  return ISQParser::RuleCNumber;
}

void ISQParser::CNumberContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCNumber(this);
}

void ISQParser::CNumberContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCNumber(this);
}


antlrcpp::Any ISQParser::CNumberContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCNumber(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::CNumberContext* ISQParser::cNumber() {
  CNumberContext *_localctx = _tracker.createInstance<CNumberContext>(_ctx, getState());
  enterRule(_localctx, 6, ISQParser::RuleCNumber);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(103);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ISQParser::Number: {
        enterOuterAlt(_localctx, 1);
        setState(100);
        numberExpr();
        break;
      }

      case ISQParser::Minus: {
        enterOuterAlt(_localctx, 2);
        setState(101);
        match(ISQParser::Minus);
        setState(102);
        numberExpr();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumberExprContext ------------------------------------------------------------------

ISQParser::NumberExprContext::NumberExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> ISQParser::NumberExprContext::Number() {
  return getTokens(ISQParser::Number);
}

tree::TerminalNode* ISQParser::NumberExprContext::Number(size_t i) {
  return getToken(ISQParser::Number, i);
}

tree::TerminalNode* ISQParser::NumberExprContext::Plus() {
  return getToken(ISQParser::Plus, 0);
}

tree::TerminalNode* ISQParser::NumberExprContext::Minus() {
  return getToken(ISQParser::Minus, 0);
}


size_t ISQParser::NumberExprContext::getRuleIndex() const {
  return ISQParser::RuleNumberExpr;
}

void ISQParser::NumberExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumberExpr(this);
}

void ISQParser::NumberExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumberExpr(this);
}


antlrcpp::Any ISQParser::NumberExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitNumberExpr(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::NumberExprContext* ISQParser::numberExpr() {
  NumberExprContext *_localctx = _tracker.createInstance<NumberExprContext>(_ctx, getState());
  enterRule(_localctx, 8, ISQParser::RuleNumberExpr);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(112);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(105);
      match(ISQParser::Number);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(106);
      match(ISQParser::Number);
      setState(107);
      match(ISQParser::Plus);
      setState(108);
      match(ISQParser::Number);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(109);
      match(ISQParser::Number);
      setState(110);
      match(ISQParser::Minus);
      setState(111);
      match(ISQParser::Number);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarTypeContext ------------------------------------------------------------------

ISQParser::VarTypeContext::VarTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::VarTypeContext::Qbit() {
  return getToken(ISQParser::Qbit, 0);
}

tree::TerminalNode* ISQParser::VarTypeContext::Int() {
  return getToken(ISQParser::Int, 0);
}


size_t ISQParser::VarTypeContext::getRuleIndex() const {
  return ISQParser::RuleVarType;
}

void ISQParser::VarTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarType(this);
}

void ISQParser::VarTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarType(this);
}


antlrcpp::Any ISQParser::VarTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitVarType(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::VarTypeContext* ISQParser::varType() {
  VarTypeContext *_localctx = _tracker.createInstance<VarTypeContext>(_ctx, getState());
  enterRule(_localctx, 10, ISQParser::RuleVarType);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    _la = _input->LA(1);
    if (!(_la == ISQParser::Int

    || _la == ISQParser::Qbit)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DefclauseContext ------------------------------------------------------------------

ISQParser::DefclauseContext::DefclauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VarTypeContext* ISQParser::DefclauseContext::varType() {
  return getRuleContext<ISQParser::VarTypeContext>(0);
}

ISQParser::IdlistContext* ISQParser::DefclauseContext::idlist() {
  return getRuleContext<ISQParser::IdlistContext>(0);
}

tree::TerminalNode* ISQParser::DefclauseContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::DefclauseContext::getRuleIndex() const {
  return ISQParser::RuleDefclause;
}

void ISQParser::DefclauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDefclause(this);
}

void ISQParser::DefclauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDefclause(this);
}


antlrcpp::Any ISQParser::DefclauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitDefclause(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::DefclauseContext* ISQParser::defclause() {
  DefclauseContext *_localctx = _tracker.createInstance<DefclauseContext>(_ctx, getState());
  enterRule(_localctx, 12, ISQParser::RuleDefclause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(116);
    varType();
    setState(117);
    idlist(0);
    setState(118);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdlistContext ------------------------------------------------------------------

ISQParser::IdlistContext::IdlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t ISQParser::IdlistContext::getRuleIndex() const {
  return ISQParser::RuleIdlist;
}

void ISQParser::IdlistContext::copyFrom(IdlistContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- IdlistdefContext ------------------------------------------------------------------

std::vector<ISQParser::IdlistContext *> ISQParser::IdlistdefContext::idlist() {
  return getRuleContexts<ISQParser::IdlistContext>();
}

ISQParser::IdlistContext* ISQParser::IdlistdefContext::idlist(size_t i) {
  return getRuleContext<ISQParser::IdlistContext>(i);
}

tree::TerminalNode* ISQParser::IdlistdefContext::Comma() {
  return getToken(ISQParser::Comma, 0);
}

ISQParser::IdlistdefContext::IdlistdefContext(IdlistContext *ctx) { copyFrom(ctx); }

void ISQParser::IdlistdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdlistdef(this);
}
void ISQParser::IdlistdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdlistdef(this);
}

antlrcpp::Any ISQParser::IdlistdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitIdlistdef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArraydefContext ------------------------------------------------------------------

tree::TerminalNode* ISQParser::ArraydefContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::ArraydefContext::LeftBracket() {
  return getToken(ISQParser::LeftBracket, 0);
}

tree::TerminalNode* ISQParser::ArraydefContext::Number() {
  return getToken(ISQParser::Number, 0);
}

tree::TerminalNode* ISQParser::ArraydefContext::RightBracket() {
  return getToken(ISQParser::RightBracket, 0);
}

ISQParser::ArraydefContext::ArraydefContext(IdlistContext *ctx) { copyFrom(ctx); }

void ISQParser::ArraydefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArraydef(this);
}
void ISQParser::ArraydefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArraydef(this);
}

antlrcpp::Any ISQParser::ArraydefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitArraydef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- SingleiddefContext ------------------------------------------------------------------

tree::TerminalNode* ISQParser::SingleiddefContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

ISQParser::SingleiddefContext::SingleiddefContext(IdlistContext *ctx) { copyFrom(ctx); }

void ISQParser::SingleiddefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingleiddef(this);
}
void ISQParser::SingleiddefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingleiddef(this);
}

antlrcpp::Any ISQParser::SingleiddefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitSingleiddef(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::IdlistContext* ISQParser::idlist() {
   return idlist(0);
}

ISQParser::IdlistContext* ISQParser::idlist(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  ISQParser::IdlistContext *_localctx = _tracker.createInstance<IdlistContext>(_ctx, parentState);
  ISQParser::IdlistContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 14;
  enterRecursionRule(_localctx, 14, ISQParser::RuleIdlist, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(126);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<SingleiddefContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(121);
      match(ISQParser::Identifier);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<ArraydefContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(122);
      match(ISQParser::Identifier);
      setState(123);
      match(ISQParser::LeftBracket);
      setState(124);
      match(ISQParser::Number);
      setState(125);
      match(ISQParser::RightBracket);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(133);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<IdlistdefContext>(_tracker.createInstance<IdlistContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleIdlist);
        setState(128);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(129);
        match(ISQParser::Comma);
        setState(130);
        idlist(2); 
      }
      setState(135);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ProgramBodyContext ------------------------------------------------------------------

ISQParser::ProgramBodyContext::ProgramBodyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::ProcedureMainContext* ISQParser::ProgramBodyContext::procedureMain() {
  return getRuleContext<ISQParser::ProcedureMainContext>(0);
}

std::vector<ISQParser::ProcedureBlockContext *> ISQParser::ProgramBodyContext::procedureBlock() {
  return getRuleContexts<ISQParser::ProcedureBlockContext>();
}

ISQParser::ProcedureBlockContext* ISQParser::ProgramBodyContext::procedureBlock(size_t i) {
  return getRuleContext<ISQParser::ProcedureBlockContext>(i);
}


size_t ISQParser::ProgramBodyContext::getRuleIndex() const {
  return ISQParser::RuleProgramBody;
}

void ISQParser::ProgramBodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgramBody(this);
}

void ISQParser::ProgramBodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgramBody(this);
}


antlrcpp::Any ISQParser::ProgramBodyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitProgramBody(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ProgramBodyContext* ISQParser::programBody() {
  ProgramBodyContext *_localctx = _tracker.createInstance<ProgramBodyContext>(_ctx, getState());
  enterRule(_localctx, 16, ISQParser::RuleProgramBody);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(139);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(136);
        procedureBlock(); 
      }
      setState(141);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    }
    setState(142);
    procedureMain();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProcedureBlockContext ------------------------------------------------------------------

ISQParser::ProcedureBlockContext::ProcedureBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::LeftBrace() {
  return getToken(ISQParser::LeftBrace, 0);
}

ISQParser::ProcedureBodyContext* ISQParser::ProcedureBlockContext::procedureBody() {
  return getRuleContext<ISQParser::ProcedureBodyContext>(0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::RightBrace() {
  return getToken(ISQParser::RightBrace, 0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::Procedure() {
  return getToken(ISQParser::Procedure, 0);
}

tree::TerminalNode* ISQParser::ProcedureBlockContext::Int() {
  return getToken(ISQParser::Int, 0);
}

ISQParser::CallParasContext* ISQParser::ProcedureBlockContext::callParas() {
  return getRuleContext<ISQParser::CallParasContext>(0);
}


size_t ISQParser::ProcedureBlockContext::getRuleIndex() const {
  return ISQParser::RuleProcedureBlock;
}

void ISQParser::ProcedureBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProcedureBlock(this);
}

void ISQParser::ProcedureBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProcedureBlock(this);
}


antlrcpp::Any ISQParser::ProcedureBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitProcedureBlock(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ProcedureBlockContext* ISQParser::procedureBlock() {
  ProcedureBlockContext *_localctx = _tracker.createInstance<ProcedureBlockContext>(_ctx, getState());
  enterRule(_localctx, 18, ISQParser::RuleProcedureBlock);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(161);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(144);
      _la = _input->LA(1);
      if (!(_la == ISQParser::Procedure

      || _la == ISQParser::Int)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(145);
      match(ISQParser::Identifier);
      setState(146);
      match(ISQParser::LeftParen);
      setState(147);
      match(ISQParser::RightParen);
      setState(148);
      match(ISQParser::LeftBrace);
      setState(149);
      procedureBody();
      setState(150);
      match(ISQParser::RightBrace);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(152);
      _la = _input->LA(1);
      if (!(_la == ISQParser::Procedure

      || _la == ISQParser::Int)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(153);
      match(ISQParser::Identifier);
      setState(154);
      match(ISQParser::LeftParen);
      setState(155);
      callParas(0);
      setState(156);
      match(ISQParser::RightParen);
      setState(157);
      match(ISQParser::LeftBrace);
      setState(158);
      procedureBody();
      setState(159);
      match(ISQParser::RightBrace);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CallParasContext ------------------------------------------------------------------

ISQParser::CallParasContext::CallParasContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VarTypeContext* ISQParser::CallParasContext::varType() {
  return getRuleContext<ISQParser::VarTypeContext>(0);
}

tree::TerminalNode* ISQParser::CallParasContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::CallParasContext::LeftBracket() {
  return getToken(ISQParser::LeftBracket, 0);
}

tree::TerminalNode* ISQParser::CallParasContext::RightBracket() {
  return getToken(ISQParser::RightBracket, 0);
}

std::vector<ISQParser::CallParasContext *> ISQParser::CallParasContext::callParas() {
  return getRuleContexts<ISQParser::CallParasContext>();
}

ISQParser::CallParasContext* ISQParser::CallParasContext::callParas(size_t i) {
  return getRuleContext<ISQParser::CallParasContext>(i);
}

tree::TerminalNode* ISQParser::CallParasContext::Comma() {
  return getToken(ISQParser::Comma, 0);
}


size_t ISQParser::CallParasContext::getRuleIndex() const {
  return ISQParser::RuleCallParas;
}

void ISQParser::CallParasContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCallParas(this);
}

void ISQParser::CallParasContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCallParas(this);
}


antlrcpp::Any ISQParser::CallParasContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCallParas(this);
  else
    return visitor->visitChildren(this);
}


ISQParser::CallParasContext* ISQParser::callParas() {
   return callParas(0);
}

ISQParser::CallParasContext* ISQParser::callParas(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  ISQParser::CallParasContext *_localctx = _tracker.createInstance<CallParasContext>(_ctx, parentState);
  ISQParser::CallParasContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 20;
  enterRecursionRule(_localctx, 20, ISQParser::RuleCallParas, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(172);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx)) {
    case 1: {
      setState(164);
      varType();
      setState(165);
      match(ISQParser::Identifier);
      break;
    }

    case 2: {
      setState(167);
      varType();
      setState(168);
      match(ISQParser::Identifier);
      setState(169);
      match(ISQParser::LeftBracket);
      setState(170);
      match(ISQParser::RightBracket);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(179);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<CallParasContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleCallParas);
        setState(174);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(175);
        match(ISQParser::Comma);
        setState(176);
        callParas(2); 
      }
      setState(181);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ProcedureMainContext ------------------------------------------------------------------

ISQParser::ProcedureMainContext::ProcedureMainContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::ProcedureMainContext::Procedure() {
  return getToken(ISQParser::Procedure, 0);
}

tree::TerminalNode* ISQParser::ProcedureMainContext::Main() {
  return getToken(ISQParser::Main, 0);
}

tree::TerminalNode* ISQParser::ProcedureMainContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

tree::TerminalNode* ISQParser::ProcedureMainContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

tree::TerminalNode* ISQParser::ProcedureMainContext::LeftBrace() {
  return getToken(ISQParser::LeftBrace, 0);
}

ISQParser::ProcedureBodyContext* ISQParser::ProcedureMainContext::procedureBody() {
  return getRuleContext<ISQParser::ProcedureBodyContext>(0);
}

tree::TerminalNode* ISQParser::ProcedureMainContext::RightBrace() {
  return getToken(ISQParser::RightBrace, 0);
}


size_t ISQParser::ProcedureMainContext::getRuleIndex() const {
  return ISQParser::RuleProcedureMain;
}

void ISQParser::ProcedureMainContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProcedureMain(this);
}

void ISQParser::ProcedureMainContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProcedureMain(this);
}


antlrcpp::Any ISQParser::ProcedureMainContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitProcedureMain(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ProcedureMainContext* ISQParser::procedureMain() {
  ProcedureMainContext *_localctx = _tracker.createInstance<ProcedureMainContext>(_ctx, getState());
  enterRule(_localctx, 22, ISQParser::RuleProcedureMain);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(182);
    match(ISQParser::Procedure);
    setState(183);
    match(ISQParser::Main);
    setState(184);
    match(ISQParser::LeftParen);
    setState(185);
    match(ISQParser::RightParen);
    setState(186);
    match(ISQParser::LeftBrace);
    setState(187);
    procedureBody();
    setState(188);
    match(ISQParser::RightBrace);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProcedureBodyContext ------------------------------------------------------------------

ISQParser::ProcedureBodyContext::ProcedureBodyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::StatementBlockContext* ISQParser::ProcedureBodyContext::statementBlock() {
  return getRuleContext<ISQParser::StatementBlockContext>(0);
}

ISQParser::ReturnStatementContext* ISQParser::ProcedureBodyContext::returnStatement() {
  return getRuleContext<ISQParser::ReturnStatementContext>(0);
}


size_t ISQParser::ProcedureBodyContext::getRuleIndex() const {
  return ISQParser::RuleProcedureBody;
}

void ISQParser::ProcedureBodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProcedureBody(this);
}

void ISQParser::ProcedureBodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProcedureBody(this);
}


antlrcpp::Any ISQParser::ProcedureBodyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitProcedureBody(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ProcedureBodyContext* ISQParser::procedureBody() {
  ProcedureBodyContext *_localctx = _tracker.createInstance<ProcedureBodyContext>(_ctx, getState());
  enterRule(_localctx, 24, ISQParser::RuleProcedureBody);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(194);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(190);
      statementBlock();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(191);
      statementBlock();
      setState(192);
      returnStatement();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

ISQParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t ISQParser::StatementContext::getRuleIndex() const {
  return ISQParser::RuleStatement;
}

void ISQParser::StatementContext::copyFrom(StatementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- QbitinitdefContext ------------------------------------------------------------------

ISQParser::QbitInitStatementContext* ISQParser::QbitinitdefContext::qbitInitStatement() {
  return getRuleContext<ISQParser::QbitInitStatementContext>(0);
}

ISQParser::QbitinitdefContext::QbitinitdefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::QbitinitdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQbitinitdef(this);
}
void ISQParser::QbitinitdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQbitinitdef(this);
}

antlrcpp::Any ISQParser::QbitinitdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitQbitinitdef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CinassigndefContext ------------------------------------------------------------------

ISQParser::CintAssignContext* ISQParser::CinassigndefContext::cintAssign() {
  return getRuleContext<ISQParser::CintAssignContext>(0);
}

ISQParser::CinassigndefContext::CinassigndefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::CinassigndefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCinassigndef(this);
}
void ISQParser::CinassigndefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCinassigndef(this);
}

antlrcpp::Any ISQParser::CinassigndefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCinassigndef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- FordefContext ------------------------------------------------------------------

ISQParser::ForStatementContext* ISQParser::FordefContext::forStatement() {
  return getRuleContext<ISQParser::ForStatementContext>(0);
}

ISQParser::FordefContext::FordefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::FordefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFordef(this);
}
void ISQParser::FordefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFordef(this);
}

antlrcpp::Any ISQParser::FordefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitFordef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PrintdefContext ------------------------------------------------------------------

ISQParser::PrintStatementContext* ISQParser::PrintdefContext::printStatement() {
  return getRuleContext<ISQParser::PrintStatementContext>(0);
}

ISQParser::PrintdefContext::PrintdefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::PrintdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrintdef(this);
}
void ISQParser::PrintdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrintdef(this);
}

antlrcpp::Any ISQParser::PrintdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitPrintdef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CalldefContext ------------------------------------------------------------------

ISQParser::CallStatementContext* ISQParser::CalldefContext::callStatement() {
  return getRuleContext<ISQParser::CallStatementContext>(0);
}

ISQParser::CalldefContext::CalldefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::CalldefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCalldef(this);
}
void ISQParser::CalldefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCalldef(this);
}

antlrcpp::Any ISQParser::CalldefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCalldef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- QbitunitarydefContext ------------------------------------------------------------------

ISQParser::QbitUnitaryStatementContext* ISQParser::QbitunitarydefContext::qbitUnitaryStatement() {
  return getRuleContext<ISQParser::QbitUnitaryStatementContext>(0);
}

ISQParser::QbitunitarydefContext::QbitunitarydefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::QbitunitarydefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQbitunitarydef(this);
}
void ISQParser::QbitunitarydefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQbitunitarydef(this);
}

antlrcpp::Any ISQParser::QbitunitarydefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitQbitunitarydef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- WhiledefContext ------------------------------------------------------------------

ISQParser::WhileStatementContext* ISQParser::WhiledefContext::whileStatement() {
  return getRuleContext<ISQParser::WhileStatementContext>(0);
}

ISQParser::WhiledefContext::WhiledefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::WhiledefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhiledef(this);
}
void ISQParser::WhiledefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhiledef(this);
}

antlrcpp::Any ISQParser::WhiledefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitWhiledef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- VardefContext ------------------------------------------------------------------

ISQParser::DefclauseContext* ISQParser::VardefContext::defclause() {
  return getRuleContext<ISQParser::DefclauseContext>(0);
}

ISQParser::VardefContext::VardefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::VardefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVardef(this);
}
void ISQParser::VardefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVardef(this);
}

antlrcpp::Any ISQParser::VardefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitVardef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PassdefContext ------------------------------------------------------------------

ISQParser::PassStatementContext* ISQParser::PassdefContext::passStatement() {
  return getRuleContext<ISQParser::PassStatementContext>(0);
}

ISQParser::PassdefContext::PassdefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::PassdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPassdef(this);
}
void ISQParser::PassdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPassdef(this);
}

antlrcpp::Any ISQParser::PassdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitPassdef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IfdefContext ------------------------------------------------------------------

ISQParser::IfStatementContext* ISQParser::IfdefContext::ifStatement() {
  return getRuleContext<ISQParser::IfStatementContext>(0);
}

ISQParser::IfdefContext::IfdefContext(StatementContext *ctx) { copyFrom(ctx); }

void ISQParser::IfdefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfdef(this);
}
void ISQParser::IfdefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfdef(this);
}

antlrcpp::Any ISQParser::IfdefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitIfdef(this);
  else
    return visitor->visitChildren(this);
}
ISQParser::StatementContext* ISQParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 26, ISQParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(206);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::QbitinitdefContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(196);
      qbitInitStatement();
      break;
    }

    case 2: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::QbitunitarydefContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(197);
      qbitUnitaryStatement();
      break;
    }

    case 3: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::CinassigndefContext>(_localctx));
      enterOuterAlt(_localctx, 3);
      setState(198);
      cintAssign();
      break;
    }

    case 4: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::IfdefContext>(_localctx));
      enterOuterAlt(_localctx, 4);
      setState(199);
      ifStatement();
      break;
    }

    case 5: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::CalldefContext>(_localctx));
      enterOuterAlt(_localctx, 5);
      setState(200);
      callStatement();
      break;
    }

    case 6: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::WhiledefContext>(_localctx));
      enterOuterAlt(_localctx, 6);
      setState(201);
      whileStatement();
      break;
    }

    case 7: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::FordefContext>(_localctx));
      enterOuterAlt(_localctx, 7);
      setState(202);
      forStatement();
      break;
    }

    case 8: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::PrintdefContext>(_localctx));
      enterOuterAlt(_localctx, 8);
      setState(203);
      printStatement();
      break;
    }

    case 9: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::PassdefContext>(_localctx));
      enterOuterAlt(_localctx, 9);
      setState(204);
      passStatement();
      break;
    }

    case 10: {
      _localctx = dynamic_cast<StatementContext *>(_tracker.createInstance<ISQParser::VardefContext>(_localctx));
      enterOuterAlt(_localctx, 10);
      setState(205);
      defclause();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementBlockContext ------------------------------------------------------------------

ISQParser::StatementBlockContext::StatementBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ISQParser::StatementContext *> ISQParser::StatementBlockContext::statement() {
  return getRuleContexts<ISQParser::StatementContext>();
}

ISQParser::StatementContext* ISQParser::StatementBlockContext::statement(size_t i) {
  return getRuleContext<ISQParser::StatementContext>(i);
}


size_t ISQParser::StatementBlockContext::getRuleIndex() const {
  return ISQParser::RuleStatementBlock;
}

void ISQParser::StatementBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatementBlock(this);
}

void ISQParser::StatementBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatementBlock(this);
}


antlrcpp::Any ISQParser::StatementBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitStatementBlock(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::StatementBlockContext* ISQParser::statementBlock() {
  StatementBlockContext *_localctx = _tracker.createInstance<StatementBlockContext>(_ctx, getState());
  enterRule(_localctx, 28, ISQParser::RuleStatementBlock);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(209); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(208);
      statement();
      setState(211); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << ISQParser::If)
      | (1ULL << ISQParser::For)
      | (1ULL << ISQParser::While)
      | (1ULL << ISQParser::Int)
      | (1ULL << ISQParser::Qbit)
      | (1ULL << ISQParser::H)
      | (1ULL << ISQParser::X)
      | (1ULL << ISQParser::Y)
      | (1ULL << ISQParser::Z)
      | (1ULL << ISQParser::S)
      | (1ULL << ISQParser::T)
      | (1ULL << ISQParser::CZ)
      | (1ULL << ISQParser::CX)
      | (1ULL << ISQParser::CNOT)
      | (1ULL << ISQParser::Print)
      | (1ULL << ISQParser::Pass)
      | (1ULL << ISQParser::Identifier)
      | (1ULL << ISQParser::Number))) != 0));
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UGateContext ------------------------------------------------------------------

ISQParser::UGateContext::UGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::UGateContext::H() {
  return getToken(ISQParser::H, 0);
}

tree::TerminalNode* ISQParser::UGateContext::X() {
  return getToken(ISQParser::X, 0);
}

tree::TerminalNode* ISQParser::UGateContext::Y() {
  return getToken(ISQParser::Y, 0);
}

tree::TerminalNode* ISQParser::UGateContext::Z() {
  return getToken(ISQParser::Z, 0);
}

tree::TerminalNode* ISQParser::UGateContext::S() {
  return getToken(ISQParser::S, 0);
}

tree::TerminalNode* ISQParser::UGateContext::T() {
  return getToken(ISQParser::T, 0);
}

tree::TerminalNode* ISQParser::UGateContext::CZ() {
  return getToken(ISQParser::CZ, 0);
}

tree::TerminalNode* ISQParser::UGateContext::CX() {
  return getToken(ISQParser::CX, 0);
}

tree::TerminalNode* ISQParser::UGateContext::CNOT() {
  return getToken(ISQParser::CNOT, 0);
}

tree::TerminalNode* ISQParser::UGateContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}


size_t ISQParser::UGateContext::getRuleIndex() const {
  return ISQParser::RuleUGate;
}

void ISQParser::UGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUGate(this);
}

void ISQParser::UGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUGate(this);
}


antlrcpp::Any ISQParser::UGateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitUGate(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::UGateContext* ISQParser::uGate() {
  UGateContext *_localctx = _tracker.createInstance<UGateContext>(_ctx, getState());
  enterRule(_localctx, 30, ISQParser::RuleUGate);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(213);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << ISQParser::H)
      | (1ULL << ISQParser::X)
      | (1ULL << ISQParser::Y)
      | (1ULL << ISQParser::Z)
      | (1ULL << ISQParser::S)
      | (1ULL << ISQParser::T)
      | (1ULL << ISQParser::CZ)
      | (1ULL << ISQParser::CX)
      | (1ULL << ISQParser::CNOT)
      | (1ULL << ISQParser::Identifier))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VariableContext ------------------------------------------------------------------

ISQParser::VariableContext::VariableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::VariableContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::VariableContext::Number() {
  return getToken(ISQParser::Number, 0);
}

tree::TerminalNode* ISQParser::VariableContext::LeftBracket() {
  return getToken(ISQParser::LeftBracket, 0);
}

ISQParser::VariableContext* ISQParser::VariableContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::VariableContext::RightBracket() {
  return getToken(ISQParser::RightBracket, 0);
}


size_t ISQParser::VariableContext::getRuleIndex() const {
  return ISQParser::RuleVariable;
}

void ISQParser::VariableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVariable(this);
}

void ISQParser::VariableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVariable(this);
}


antlrcpp::Any ISQParser::VariableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitVariable(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::VariableContext* ISQParser::variable() {
  VariableContext *_localctx = _tracker.createInstance<VariableContext>(_ctx, getState());
  enterRule(_localctx, 32, ISQParser::RuleVariable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(222);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(215);
      match(ISQParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(216);
      match(ISQParser::Number);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(217);
      match(ISQParser::Identifier);
      setState(218);
      match(ISQParser::LeftBracket);
      setState(219);
      variable();
      setState(220);
      match(ISQParser::RightBracket);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VariableListContext ------------------------------------------------------------------

ISQParser::VariableListContext::VariableListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VariableContext* ISQParser::VariableListContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

std::vector<ISQParser::VariableListContext *> ISQParser::VariableListContext::variableList() {
  return getRuleContexts<ISQParser::VariableListContext>();
}

ISQParser::VariableListContext* ISQParser::VariableListContext::variableList(size_t i) {
  return getRuleContext<ISQParser::VariableListContext>(i);
}

tree::TerminalNode* ISQParser::VariableListContext::Comma() {
  return getToken(ISQParser::Comma, 0);
}


size_t ISQParser::VariableListContext::getRuleIndex() const {
  return ISQParser::RuleVariableList;
}

void ISQParser::VariableListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVariableList(this);
}

void ISQParser::VariableListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVariableList(this);
}


antlrcpp::Any ISQParser::VariableListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitVariableList(this);
  else
    return visitor->visitChildren(this);
}


ISQParser::VariableListContext* ISQParser::variableList() {
   return variableList(0);
}

ISQParser::VariableListContext* ISQParser::variableList(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  ISQParser::VariableListContext *_localctx = _tracker.createInstance<VariableListContext>(_ctx, parentState);
  ISQParser::VariableListContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 34;
  enterRecursionRule(_localctx, 34, ISQParser::RuleVariableList, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(225);
    variable();
    _ctx->stop = _input->LT(-1);
    setState(232);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<VariableListContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleVariableList);
        setState(227);

        if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
        setState(228);
        match(ISQParser::Comma);
        setState(229);
        variableList(3); 
      }
      setState(234);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- BinopPlusContext ------------------------------------------------------------------

ISQParser::BinopPlusContext::BinopPlusContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::BinopPlusContext::Plus() {
  return getToken(ISQParser::Plus, 0);
}

tree::TerminalNode* ISQParser::BinopPlusContext::Minus() {
  return getToken(ISQParser::Minus, 0);
}


size_t ISQParser::BinopPlusContext::getRuleIndex() const {
  return ISQParser::RuleBinopPlus;
}

void ISQParser::BinopPlusContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinopPlus(this);
}

void ISQParser::BinopPlusContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinopPlus(this);
}


antlrcpp::Any ISQParser::BinopPlusContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitBinopPlus(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::BinopPlusContext* ISQParser::binopPlus() {
  BinopPlusContext *_localctx = _tracker.createInstance<BinopPlusContext>(_ctx, getState());
  enterRule(_localctx, 36, ISQParser::RuleBinopPlus);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(235);
    _la = _input->LA(1);
    if (!(_la == ISQParser::Plus

    || _la == ISQParser::Minus)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinopMultContext ------------------------------------------------------------------

ISQParser::BinopMultContext::BinopMultContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::BinopMultContext::Mult() {
  return getToken(ISQParser::Mult, 0);
}

tree::TerminalNode* ISQParser::BinopMultContext::Div() {
  return getToken(ISQParser::Div, 0);
}


size_t ISQParser::BinopMultContext::getRuleIndex() const {
  return ISQParser::RuleBinopMult;
}

void ISQParser::BinopMultContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinopMult(this);
}

void ISQParser::BinopMultContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinopMult(this);
}


antlrcpp::Any ISQParser::BinopMultContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitBinopMult(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::BinopMultContext* ISQParser::binopMult() {
  BinopMultContext *_localctx = _tracker.createInstance<BinopMultContext>(_ctx, getState());
  enterRule(_localctx, 38, ISQParser::RuleBinopMult);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(237);
    _la = _input->LA(1);
    if (!(_la == ISQParser::Mult

    || _la == ISQParser::Div)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

ISQParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ISQParser::MultexpContext *> ISQParser::ExpressionContext::multexp() {
  return getRuleContexts<ISQParser::MultexpContext>();
}

ISQParser::MultexpContext* ISQParser::ExpressionContext::multexp(size_t i) {
  return getRuleContext<ISQParser::MultexpContext>(i);
}

std::vector<ISQParser::BinopPlusContext *> ISQParser::ExpressionContext::binopPlus() {
  return getRuleContexts<ISQParser::BinopPlusContext>();
}

ISQParser::BinopPlusContext* ISQParser::ExpressionContext::binopPlus(size_t i) {
  return getRuleContext<ISQParser::BinopPlusContext>(i);
}


size_t ISQParser::ExpressionContext::getRuleIndex() const {
  return ISQParser::RuleExpression;
}

void ISQParser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void ISQParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any ISQParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ExpressionContext* ISQParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 40, ISQParser::RuleExpression);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(239);
    multexp();
    setState(245);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ISQParser::Plus

    || _la == ISQParser::Minus) {
      setState(240);
      binopPlus();
      setState(241);
      multexp();
      setState(247);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultexpContext ------------------------------------------------------------------

ISQParser::MultexpContext::MultexpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ISQParser::AtomexpContext *> ISQParser::MultexpContext::atomexp() {
  return getRuleContexts<ISQParser::AtomexpContext>();
}

ISQParser::AtomexpContext* ISQParser::MultexpContext::atomexp(size_t i) {
  return getRuleContext<ISQParser::AtomexpContext>(i);
}

std::vector<ISQParser::BinopMultContext *> ISQParser::MultexpContext::binopMult() {
  return getRuleContexts<ISQParser::BinopMultContext>();
}

ISQParser::BinopMultContext* ISQParser::MultexpContext::binopMult(size_t i) {
  return getRuleContext<ISQParser::BinopMultContext>(i);
}


size_t ISQParser::MultexpContext::getRuleIndex() const {
  return ISQParser::RuleMultexp;
}

void ISQParser::MultexpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultexp(this);
}

void ISQParser::MultexpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultexp(this);
}


antlrcpp::Any ISQParser::MultexpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitMultexp(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::MultexpContext* ISQParser::multexp() {
  MultexpContext *_localctx = _tracker.createInstance<MultexpContext>(_ctx, getState());
  enterRule(_localctx, 42, ISQParser::RuleMultexp);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(248);
    atomexp();
    setState(254);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ISQParser::Mult

    || _la == ISQParser::Div) {
      setState(249);
      binopMult();
      setState(250);
      atomexp();
      setState(256);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AtomexpContext ------------------------------------------------------------------

ISQParser::AtomexpContext::AtomexpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VariableContext* ISQParser::AtomexpContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::AtomexpContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

ISQParser::ExpressionContext* ISQParser::AtomexpContext::expression() {
  return getRuleContext<ISQParser::ExpressionContext>(0);
}

tree::TerminalNode* ISQParser::AtomexpContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}


size_t ISQParser::AtomexpContext::getRuleIndex() const {
  return ISQParser::RuleAtomexp;
}

void ISQParser::AtomexpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAtomexp(this);
}

void ISQParser::AtomexpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAtomexp(this);
}


antlrcpp::Any ISQParser::AtomexpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitAtomexp(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::AtomexpContext* ISQParser::atomexp() {
  AtomexpContext *_localctx = _tracker.createInstance<AtomexpContext>(_ctx, getState());
  enterRule(_localctx, 44, ISQParser::RuleAtomexp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(262);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ISQParser::Identifier:
      case ISQParser::Number: {
        enterOuterAlt(_localctx, 1);
        setState(257);
        variable();
        break;
      }

      case ISQParser::LeftParen: {
        enterOuterAlt(_localctx, 2);
        setState(258);
        match(ISQParser::LeftParen);
        setState(259);
        expression();
        setState(260);
        match(ISQParser::RightParen);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MExpressionContext ------------------------------------------------------------------

ISQParser::MExpressionContext::MExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::MExpressionContext::M() {
  return getToken(ISQParser::M, 0);
}

tree::TerminalNode* ISQParser::MExpressionContext::Less() {
  return getToken(ISQParser::Less, 0);
}

ISQParser::VariableContext* ISQParser::MExpressionContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::MExpressionContext::Greater() {
  return getToken(ISQParser::Greater, 0);
}


size_t ISQParser::MExpressionContext::getRuleIndex() const {
  return ISQParser::RuleMExpression;
}

void ISQParser::MExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMExpression(this);
}

void ISQParser::MExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMExpression(this);
}


antlrcpp::Any ISQParser::MExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitMExpression(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::MExpressionContext* ISQParser::mExpression() {
  MExpressionContext *_localctx = _tracker.createInstance<MExpressionContext>(_ctx, getState());
  enterRule(_localctx, 46, ISQParser::RuleMExpression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(264);
    match(ISQParser::M);
    setState(265);
    match(ISQParser::Less);
    setState(266);
    variable();
    setState(267);
    match(ISQParser::Greater);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssociationContext ------------------------------------------------------------------

ISQParser::AssociationContext::AssociationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::AssociationContext::Equal() {
  return getToken(ISQParser::Equal, 0);
}

tree::TerminalNode* ISQParser::AssociationContext::GreaterEqual() {
  return getToken(ISQParser::GreaterEqual, 0);
}

tree::TerminalNode* ISQParser::AssociationContext::LessEqual() {
  return getToken(ISQParser::LessEqual, 0);
}

tree::TerminalNode* ISQParser::AssociationContext::Greater() {
  return getToken(ISQParser::Greater, 0);
}

tree::TerminalNode* ISQParser::AssociationContext::Less() {
  return getToken(ISQParser::Less, 0);
}


size_t ISQParser::AssociationContext::getRuleIndex() const {
  return ISQParser::RuleAssociation;
}

void ISQParser::AssociationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssociation(this);
}

void ISQParser::AssociationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssociation(this);
}


antlrcpp::Any ISQParser::AssociationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitAssociation(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::AssociationContext* ISQParser::association() {
  AssociationContext *_localctx = _tracker.createInstance<AssociationContext>(_ctx, getState());
  enterRule(_localctx, 48, ISQParser::RuleAssociation);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(269);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << ISQParser::Less)
      | (1ULL << ISQParser::Greater)
      | (1ULL << ISQParser::Equal)
      | (1ULL << ISQParser::LessEqual)
      | (1ULL << ISQParser::GreaterEqual))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QbitInitStatementContext ------------------------------------------------------------------

ISQParser::QbitInitStatementContext::QbitInitStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VariableContext* ISQParser::QbitInitStatementContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::QbitInitStatementContext::Assign() {
  return getToken(ISQParser::Assign, 0);
}

tree::TerminalNode* ISQParser::QbitInitStatementContext::KetZero() {
  return getToken(ISQParser::KetZero, 0);
}

tree::TerminalNode* ISQParser::QbitInitStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::QbitInitStatementContext::getRuleIndex() const {
  return ISQParser::RuleQbitInitStatement;
}

void ISQParser::QbitInitStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQbitInitStatement(this);
}

void ISQParser::QbitInitStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQbitInitStatement(this);
}


antlrcpp::Any ISQParser::QbitInitStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitQbitInitStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::QbitInitStatementContext* ISQParser::qbitInitStatement() {
  QbitInitStatementContext *_localctx = _tracker.createInstance<QbitInitStatementContext>(_ctx, getState());
  enterRule(_localctx, 50, ISQParser::RuleQbitInitStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(271);
    variable();
    setState(272);
    match(ISQParser::Assign);
    setState(273);
    match(ISQParser::KetZero);
    setState(274);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QbitUnitaryStatementContext ------------------------------------------------------------------

ISQParser::QbitUnitaryStatementContext::QbitUnitaryStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::UGateContext* ISQParser::QbitUnitaryStatementContext::uGate() {
  return getRuleContext<ISQParser::UGateContext>(0);
}

tree::TerminalNode* ISQParser::QbitUnitaryStatementContext::Less() {
  return getToken(ISQParser::Less, 0);
}

ISQParser::VariableListContext* ISQParser::QbitUnitaryStatementContext::variableList() {
  return getRuleContext<ISQParser::VariableListContext>(0);
}

tree::TerminalNode* ISQParser::QbitUnitaryStatementContext::Greater() {
  return getToken(ISQParser::Greater, 0);
}

tree::TerminalNode* ISQParser::QbitUnitaryStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::QbitUnitaryStatementContext::getRuleIndex() const {
  return ISQParser::RuleQbitUnitaryStatement;
}

void ISQParser::QbitUnitaryStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQbitUnitaryStatement(this);
}

void ISQParser::QbitUnitaryStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQbitUnitaryStatement(this);
}


antlrcpp::Any ISQParser::QbitUnitaryStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitQbitUnitaryStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::QbitUnitaryStatementContext* ISQParser::qbitUnitaryStatement() {
  QbitUnitaryStatementContext *_localctx = _tracker.createInstance<QbitUnitaryStatementContext>(_ctx, getState());
  enterRule(_localctx, 52, ISQParser::RuleQbitUnitaryStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(276);
    uGate();
    setState(277);
    match(ISQParser::Less);
    setState(278);
    variableList(0);
    setState(279);
    match(ISQParser::Greater);
    setState(280);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CintAssignContext ------------------------------------------------------------------

ISQParser::CintAssignContext::CintAssignContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::VariableContext* ISQParser::CintAssignContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::CintAssignContext::Assign() {
  return getToken(ISQParser::Assign, 0);
}

ISQParser::ExpressionContext* ISQParser::CintAssignContext::expression() {
  return getRuleContext<ISQParser::ExpressionContext>(0);
}

tree::TerminalNode* ISQParser::CintAssignContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}

ISQParser::MExpressionContext* ISQParser::CintAssignContext::mExpression() {
  return getRuleContext<ISQParser::MExpressionContext>(0);
}

ISQParser::CallStatementContext* ISQParser::CintAssignContext::callStatement() {
  return getRuleContext<ISQParser::CallStatementContext>(0);
}


size_t ISQParser::CintAssignContext::getRuleIndex() const {
  return ISQParser::RuleCintAssign;
}

void ISQParser::CintAssignContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCintAssign(this);
}

void ISQParser::CintAssignContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCintAssign(this);
}


antlrcpp::Any ISQParser::CintAssignContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCintAssign(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::CintAssignContext* ISQParser::cintAssign() {
  CintAssignContext *_localctx = _tracker.createInstance<CintAssignContext>(_ctx, getState());
  enterRule(_localctx, 54, ISQParser::RuleCintAssign);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(296);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(282);
      variable();
      setState(283);
      match(ISQParser::Assign);
      setState(284);
      expression();
      setState(285);
      match(ISQParser::Semi);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(287);
      variable();
      setState(288);
      match(ISQParser::Assign);
      setState(289);
      mExpression();
      setState(290);
      match(ISQParser::Semi);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(292);
      variable();
      setState(293);
      match(ISQParser::Assign);
      setState(294);
      callStatement();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RegionBodyContext ------------------------------------------------------------------

ISQParser::RegionBodyContext::RegionBodyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ISQParser::StatementContext* ISQParser::RegionBodyContext::statement() {
  return getRuleContext<ISQParser::StatementContext>(0);
}

tree::TerminalNode* ISQParser::RegionBodyContext::LeftBrace() {
  return getToken(ISQParser::LeftBrace, 0);
}

ISQParser::StatementBlockContext* ISQParser::RegionBodyContext::statementBlock() {
  return getRuleContext<ISQParser::StatementBlockContext>(0);
}

tree::TerminalNode* ISQParser::RegionBodyContext::RightBrace() {
  return getToken(ISQParser::RightBrace, 0);
}


size_t ISQParser::RegionBodyContext::getRuleIndex() const {
  return ISQParser::RuleRegionBody;
}

void ISQParser::RegionBodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRegionBody(this);
}

void ISQParser::RegionBodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRegionBody(this);
}


antlrcpp::Any ISQParser::RegionBodyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitRegionBody(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::RegionBodyContext* ISQParser::regionBody() {
  RegionBodyContext *_localctx = _tracker.createInstance<RegionBodyContext>(_ctx, getState());
  enterRule(_localctx, 56, ISQParser::RuleRegionBody);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(303);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ISQParser::If:
      case ISQParser::For:
      case ISQParser::While:
      case ISQParser::Int:
      case ISQParser::Qbit:
      case ISQParser::H:
      case ISQParser::X:
      case ISQParser::Y:
      case ISQParser::Z:
      case ISQParser::S:
      case ISQParser::T:
      case ISQParser::CZ:
      case ISQParser::CX:
      case ISQParser::CNOT:
      case ISQParser::Print:
      case ISQParser::Pass:
      case ISQParser::Identifier:
      case ISQParser::Number: {
        enterOuterAlt(_localctx, 1);
        setState(298);
        statement();
        break;
      }

      case ISQParser::LeftBrace: {
        enterOuterAlt(_localctx, 2);
        setState(299);
        match(ISQParser::LeftBrace);
        setState(300);
        statementBlock();
        setState(301);
        match(ISQParser::RightBrace);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfStatementContext ------------------------------------------------------------------

ISQParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::IfStatementContext::If() {
  return getToken(ISQParser::If, 0);
}

tree::TerminalNode* ISQParser::IfStatementContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

std::vector<ISQParser::ExpressionContext *> ISQParser::IfStatementContext::expression() {
  return getRuleContexts<ISQParser::ExpressionContext>();
}

ISQParser::ExpressionContext* ISQParser::IfStatementContext::expression(size_t i) {
  return getRuleContext<ISQParser::ExpressionContext>(i);
}

ISQParser::AssociationContext* ISQParser::IfStatementContext::association() {
  return getRuleContext<ISQParser::AssociationContext>(0);
}

tree::TerminalNode* ISQParser::IfStatementContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

std::vector<ISQParser::RegionBodyContext *> ISQParser::IfStatementContext::regionBody() {
  return getRuleContexts<ISQParser::RegionBodyContext>();
}

ISQParser::RegionBodyContext* ISQParser::IfStatementContext::regionBody(size_t i) {
  return getRuleContext<ISQParser::RegionBodyContext>(i);
}

tree::TerminalNode* ISQParser::IfStatementContext::Else() {
  return getToken(ISQParser::Else, 0);
}


size_t ISQParser::IfStatementContext::getRuleIndex() const {
  return ISQParser::RuleIfStatement;
}

void ISQParser::IfStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStatement(this);
}

void ISQParser::IfStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStatement(this);
}


antlrcpp::Any ISQParser::IfStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitIfStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::IfStatementContext* ISQParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 58, ISQParser::RuleIfStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(323);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(305);
      match(ISQParser::If);
      setState(306);
      match(ISQParser::LeftParen);
      setState(307);
      expression();
      setState(308);
      association();
      setState(309);
      expression();
      setState(310);
      match(ISQParser::RightParen);
      setState(311);
      regionBody();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(313);
      match(ISQParser::If);
      setState(314);
      match(ISQParser::LeftParen);
      setState(315);
      expression();
      setState(316);
      association();
      setState(317);
      expression();
      setState(318);
      match(ISQParser::RightParen);
      setState(319);
      regionBody();
      setState(320);
      match(ISQParser::Else);
      setState(321);
      regionBody();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ForStatementContext ------------------------------------------------------------------

ISQParser::ForStatementContext::ForStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::ForStatementContext::For() {
  return getToken(ISQParser::For, 0);
}

tree::TerminalNode* ISQParser::ForStatementContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

tree::TerminalNode* ISQParser::ForStatementContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::ForStatementContext::Assign() {
  return getToken(ISQParser::Assign, 0);
}

std::vector<ISQParser::VariableContext *> ISQParser::ForStatementContext::variable() {
  return getRuleContexts<ISQParser::VariableContext>();
}

ISQParser::VariableContext* ISQParser::ForStatementContext::variable(size_t i) {
  return getRuleContext<ISQParser::VariableContext>(i);
}

tree::TerminalNode* ISQParser::ForStatementContext::To() {
  return getToken(ISQParser::To, 0);
}

tree::TerminalNode* ISQParser::ForStatementContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

ISQParser::RegionBodyContext* ISQParser::ForStatementContext::regionBody() {
  return getRuleContext<ISQParser::RegionBodyContext>(0);
}


size_t ISQParser::ForStatementContext::getRuleIndex() const {
  return ISQParser::RuleForStatement;
}

void ISQParser::ForStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterForStatement(this);
}

void ISQParser::ForStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitForStatement(this);
}


antlrcpp::Any ISQParser::ForStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitForStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ForStatementContext* ISQParser::forStatement() {
  ForStatementContext *_localctx = _tracker.createInstance<ForStatementContext>(_ctx, getState());
  enterRule(_localctx, 60, ISQParser::RuleForStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(325);
    match(ISQParser::For);
    setState(326);
    match(ISQParser::LeftParen);
    setState(327);
    match(ISQParser::Identifier);
    setState(328);
    match(ISQParser::Assign);
    setState(329);
    variable();
    setState(330);
    match(ISQParser::To);
    setState(331);
    variable();
    setState(332);
    match(ISQParser::RightParen);
    setState(333);
    regionBody();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhileStatementContext ------------------------------------------------------------------

ISQParser::WhileStatementContext::WhileStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::WhileStatementContext::While() {
  return getToken(ISQParser::While, 0);
}

tree::TerminalNode* ISQParser::WhileStatementContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

std::vector<ISQParser::ExpressionContext *> ISQParser::WhileStatementContext::expression() {
  return getRuleContexts<ISQParser::ExpressionContext>();
}

ISQParser::ExpressionContext* ISQParser::WhileStatementContext::expression(size_t i) {
  return getRuleContext<ISQParser::ExpressionContext>(i);
}

ISQParser::AssociationContext* ISQParser::WhileStatementContext::association() {
  return getRuleContext<ISQParser::AssociationContext>(0);
}

tree::TerminalNode* ISQParser::WhileStatementContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

ISQParser::RegionBodyContext* ISQParser::WhileStatementContext::regionBody() {
  return getRuleContext<ISQParser::RegionBodyContext>(0);
}


size_t ISQParser::WhileStatementContext::getRuleIndex() const {
  return ISQParser::RuleWhileStatement;
}

void ISQParser::WhileStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhileStatement(this);
}

void ISQParser::WhileStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhileStatement(this);
}


antlrcpp::Any ISQParser::WhileStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitWhileStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::WhileStatementContext* ISQParser::whileStatement() {
  WhileStatementContext *_localctx = _tracker.createInstance<WhileStatementContext>(_ctx, getState());
  enterRule(_localctx, 62, ISQParser::RuleWhileStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(335);
    match(ISQParser::While);
    setState(336);
    match(ISQParser::LeftParen);
    setState(337);
    expression();
    setState(338);
    association();
    setState(339);
    expression();
    setState(340);
    match(ISQParser::RightParen);
    setState(341);
    regionBody();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CallStatementContext ------------------------------------------------------------------

ISQParser::CallStatementContext::CallStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::CallStatementContext::Identifier() {
  return getToken(ISQParser::Identifier, 0);
}

tree::TerminalNode* ISQParser::CallStatementContext::LeftParen() {
  return getToken(ISQParser::LeftParen, 0);
}

tree::TerminalNode* ISQParser::CallStatementContext::RightParen() {
  return getToken(ISQParser::RightParen, 0);
}

tree::TerminalNode* ISQParser::CallStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}

ISQParser::VariableListContext* ISQParser::CallStatementContext::variableList() {
  return getRuleContext<ISQParser::VariableListContext>(0);
}


size_t ISQParser::CallStatementContext::getRuleIndex() const {
  return ISQParser::RuleCallStatement;
}

void ISQParser::CallStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCallStatement(this);
}

void ISQParser::CallStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCallStatement(this);
}


antlrcpp::Any ISQParser::CallStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitCallStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::CallStatementContext* ISQParser::callStatement() {
  CallStatementContext *_localctx = _tracker.createInstance<CallStatementContext>(_ctx, getState());
  enterRule(_localctx, 64, ISQParser::RuleCallStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(343);
    match(ISQParser::Identifier);
    setState(344);
    match(ISQParser::LeftParen);
    setState(347);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ISQParser::RightParen: {
        break;
      }

      case ISQParser::Identifier:
      case ISQParser::Number: {
        setState(346);
        variableList(0);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(349);
    match(ISQParser::RightParen);
    setState(350);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrintStatementContext ------------------------------------------------------------------

ISQParser::PrintStatementContext::PrintStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::PrintStatementContext::Print() {
  return getToken(ISQParser::Print, 0);
}

ISQParser::VariableContext* ISQParser::PrintStatementContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::PrintStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::PrintStatementContext::getRuleIndex() const {
  return ISQParser::RulePrintStatement;
}

void ISQParser::PrintStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrintStatement(this);
}

void ISQParser::PrintStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrintStatement(this);
}


antlrcpp::Any ISQParser::PrintStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitPrintStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::PrintStatementContext* ISQParser::printStatement() {
  PrintStatementContext *_localctx = _tracker.createInstance<PrintStatementContext>(_ctx, getState());
  enterRule(_localctx, 66, ISQParser::RulePrintStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(352);
    match(ISQParser::Print);
    setState(353);
    variable();
    setState(354);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PassStatementContext ------------------------------------------------------------------

ISQParser::PassStatementContext::PassStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::PassStatementContext::Pass() {
  return getToken(ISQParser::Pass, 0);
}

tree::TerminalNode* ISQParser::PassStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::PassStatementContext::getRuleIndex() const {
  return ISQParser::RulePassStatement;
}

void ISQParser::PassStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPassStatement(this);
}

void ISQParser::PassStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPassStatement(this);
}


antlrcpp::Any ISQParser::PassStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitPassStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::PassStatementContext* ISQParser::passStatement() {
  PassStatementContext *_localctx = _tracker.createInstance<PassStatementContext>(_ctx, getState());
  enterRule(_localctx, 68, ISQParser::RulePassStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(356);
    match(ISQParser::Pass);
    setState(357);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnStatementContext ------------------------------------------------------------------

ISQParser::ReturnStatementContext::ReturnStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ISQParser::ReturnStatementContext::Return() {
  return getToken(ISQParser::Return, 0);
}

ISQParser::VariableContext* ISQParser::ReturnStatementContext::variable() {
  return getRuleContext<ISQParser::VariableContext>(0);
}

tree::TerminalNode* ISQParser::ReturnStatementContext::Semi() {
  return getToken(ISQParser::Semi, 0);
}


size_t ISQParser::ReturnStatementContext::getRuleIndex() const {
  return ISQParser::RuleReturnStatement;
}

void ISQParser::ReturnStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnStatement(this);
}

void ISQParser::ReturnStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ISQParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnStatement(this);
}


antlrcpp::Any ISQParser::ReturnStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ISQParserVisitor*>(visitor))
    return parserVisitor->visitReturnStatement(this);
  else
    return visitor->visitChildren(this);
}

ISQParser::ReturnStatementContext* ISQParser::returnStatement() {
  ReturnStatementContext *_localctx = _tracker.createInstance<ReturnStatementContext>(_ctx, getState());
  enterRule(_localctx, 70, ISQParser::RuleReturnStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(359);
    match(ISQParser::Return);
    setState(360);
    variable();
    setState(361);
    match(ISQParser::Semi);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool ISQParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 7: return idlistSempred(dynamic_cast<IdlistContext *>(context), predicateIndex);
    case 10: return callParasSempred(dynamic_cast<CallParasContext *>(context), predicateIndex);
    case 17: return variableListSempred(dynamic_cast<VariableListContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool ISQParser::idlistSempred(IdlistContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool ISQParser::callParasSempred(CallParasContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool ISQParser::variableListSempred(VariableListContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 2);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> ISQParser::_decisionToDFA;
atn::PredictionContextCache ISQParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN ISQParser::_atn;
std::vector<uint16_t> ISQParser::_serializedATN;

std::vector<std::string> ISQParser::_ruleNames = {
  "program", "gateDefclause", "matrixContents", "cNumber", "numberExpr", 
  "varType", "defclause", "idlist", "programBody", "procedureBlock", "callParas", 
  "procedureMain", "procedureBody", "statement", "statementBlock", "uGate", 
  "variable", "variableList", "binopPlus", "binopMult", "expression", "multexp", 
  "atomexp", "mExpression", "association", "qbitInitStatement", "qbitUnitaryStatement", 
  "cintAssign", "regionBody", "ifStatement", "forStatement", "whileStatement", 
  "callStatement", "printStatement", "passStatement", "returnStatement"
};

std::vector<std::string> ISQParser::_literalNames = {
  "", "", "", "", "", "'if'", "'then'", "'else'", "'fi'", "'for'", "'to'", 
  "'while'", "'do'", "'od'", "'procedure'", "'main'", "'int'", "'qbit'", 
  "'H'", "'X'", "'Y'", "'Z'", "'S'", "'T'", "'CZ'", "'CX'", "'CNOT'", "'M'", 
  "'print'", "'Defgate'", "'pass'", "'return'", "'='", "'+'", "'-'", "'*'", 
  "'/'", "'<'", "'>'", "','", "'('", "')'", "'['", "']'", "'{'", "'}'", 
  "';'", "'=='", "'<='", "'>='", "'|0>'"
};

std::vector<std::string> ISQParser::_symbolicNames = {
  "", "WhiteSpace", "NewLine", "BlockComment", "LineComment", "If", "Then", 
  "Else", "Fi", "For", "To", "While", "Do", "Od", "Procedure", "Main", "Int", 
  "Qbit", "H", "X", "Y", "Z", "S", "T", "CZ", "CX", "CNOT", "M", "Print", 
  "Defgate", "Pass", "Return", "Assign", "Plus", "Minus", "Mult", "Div", 
  "Less", "Greater", "Comma", "LeftParen", "RightParen", "LeftBracket", 
  "RightBracket", "LeftBrace", "RightBrace", "Semi", "Equal", "LessEqual", 
  "GreaterEqual", "KetZero", "Identifier", "Number"
};

dfa::Vocabulary ISQParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> ISQParser::_tokenNames;

ISQParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x36, 0x16e, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x3, 0x2, 0x7, 0x2, 0x4c, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x4f, 
    0xb, 0x2, 0x3, 0x2, 0x6, 0x2, 0x52, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 0x53, 
    0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 
    0x3, 0x4, 0x5, 0x4, 0x65, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x5, 
    0x5, 0x6a, 0xa, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x73, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0x81, 0xa, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x7, 0x9, 0x86, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x89, 0xb, 0x9, 
    0x3, 0xa, 0x7, 0xa, 0x8c, 0xa, 0xa, 0xc, 0xa, 0xe, 0xa, 0x8f, 0xb, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0xa4, 0xa, 
    0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0xaf, 0xa, 0xc, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xc, 0x7, 0xc, 0xb4, 0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0xb7, 0xb, 0xc, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0xc5, 0xa, 
    0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 0xd1, 0xa, 0xf, 0x3, 0x10, 
    0x6, 0x10, 0xd4, 0xa, 0x10, 0xd, 0x10, 0xe, 0x10, 0xd5, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x5, 0x12, 0xe1, 0xa, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0xe9, 0xa, 0x13, 0xc, 0x13, 
    0xe, 0x13, 0xec, 0xb, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x7, 0x16, 0xf6, 0xa, 0x16, 
    0xc, 0x16, 0xe, 0x16, 0xf9, 0xb, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 
    0x3, 0x17, 0x7, 0x17, 0xff, 0xa, 0x17, 0xc, 0x17, 0xe, 0x17, 0x102, 
    0xb, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x5, 
    0x18, 0x109, 0xa, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 
    0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x3, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 
    0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x12b, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x5, 0x1e, 0x132, 0xa, 0x1e, 0x3, 0x1f, 
    0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 
    0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 
    0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 0x146, 0xa, 0x1f, 
    0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 
    0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x15e, 0xa, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 
    0x3, 0x25, 0x2, 0x5, 0x10, 0x16, 0x24, 0x26, 0x2, 0x4, 0x6, 0x8, 0xa, 
    0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 
    0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 
    0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x2, 0x9, 0x4, 0x2, 0x29, 
    0x29, 0x30, 0x30, 0x3, 0x2, 0x12, 0x13, 0x4, 0x2, 0x10, 0x10, 0x12, 
    0x12, 0x4, 0x2, 0x14, 0x1c, 0x35, 0x35, 0x3, 0x2, 0x23, 0x24, 0x3, 0x2, 
    0x25, 0x26, 0x4, 0x2, 0x27, 0x28, 0x31, 0x33, 0x2, 0x16b, 0x2, 0x4d, 
    0x3, 0x2, 0x2, 0x2, 0x4, 0x57, 0x3, 0x2, 0x2, 0x2, 0x6, 0x64, 0x3, 0x2, 
    0x2, 0x2, 0x8, 0x69, 0x3, 0x2, 0x2, 0x2, 0xa, 0x72, 0x3, 0x2, 0x2, 0x2, 
    0xc, 0x74, 0x3, 0x2, 0x2, 0x2, 0xe, 0x76, 0x3, 0x2, 0x2, 0x2, 0x10, 
    0x80, 0x3, 0x2, 0x2, 0x2, 0x12, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x14, 0xa3, 
    0x3, 0x2, 0x2, 0x2, 0x16, 0xae, 0x3, 0x2, 0x2, 0x2, 0x18, 0xb8, 0x3, 
    0x2, 0x2, 0x2, 0x1a, 0xc4, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xd0, 0x3, 0x2, 
    0x2, 0x2, 0x1e, 0xd3, 0x3, 0x2, 0x2, 0x2, 0x20, 0xd7, 0x3, 0x2, 0x2, 
    0x2, 0x22, 0xe0, 0x3, 0x2, 0x2, 0x2, 0x24, 0xe2, 0x3, 0x2, 0x2, 0x2, 
    0x26, 0xed, 0x3, 0x2, 0x2, 0x2, 0x28, 0xef, 0x3, 0x2, 0x2, 0x2, 0x2a, 
    0xf1, 0x3, 0x2, 0x2, 0x2, 0x2c, 0xfa, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x108, 
    0x3, 0x2, 0x2, 0x2, 0x30, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x32, 0x10f, 0x3, 
    0x2, 0x2, 0x2, 0x34, 0x111, 0x3, 0x2, 0x2, 0x2, 0x36, 0x116, 0x3, 0x2, 
    0x2, 0x2, 0x38, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x131, 0x3, 0x2, 0x2, 
    0x2, 0x3c, 0x145, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x147, 0x3, 0x2, 0x2, 0x2, 
    0x40, 0x151, 0x3, 0x2, 0x2, 0x2, 0x42, 0x159, 0x3, 0x2, 0x2, 0x2, 0x44, 
    0x162, 0x3, 0x2, 0x2, 0x2, 0x46, 0x166, 0x3, 0x2, 0x2, 0x2, 0x48, 0x169, 
    0x3, 0x2, 0x2, 0x2, 0x4a, 0x4c, 0x5, 0x4, 0x3, 0x2, 0x4b, 0x4a, 0x3, 
    0x2, 0x2, 0x2, 0x4c, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x4d, 0x4b, 0x3, 0x2, 
    0x2, 0x2, 0x4d, 0x4e, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x51, 0x3, 0x2, 0x2, 
    0x2, 0x4f, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x50, 0x52, 0x5, 0xe, 0x8, 0x2, 
    0x51, 0x50, 0x3, 0x2, 0x2, 0x2, 0x52, 0x53, 0x3, 0x2, 0x2, 0x2, 0x53, 
    0x51, 0x3, 0x2, 0x2, 0x2, 0x53, 0x54, 0x3, 0x2, 0x2, 0x2, 0x54, 0x55, 
    0x3, 0x2, 0x2, 0x2, 0x55, 0x56, 0x5, 0x12, 0xa, 0x2, 0x56, 0x3, 0x3, 
    0x2, 0x2, 0x2, 0x57, 0x58, 0x7, 0x1f, 0x2, 0x2, 0x58, 0x59, 0x7, 0x35, 
    0x2, 0x2, 0x59, 0x5a, 0x7, 0x22, 0x2, 0x2, 0x5a, 0x5b, 0x7, 0x2c, 0x2, 
    0x2, 0x5b, 0x5c, 0x5, 0x6, 0x4, 0x2, 0x5c, 0x5d, 0x7, 0x2d, 0x2, 0x2, 
    0x5d, 0x5e, 0x7, 0x30, 0x2, 0x2, 0x5e, 0x5, 0x3, 0x2, 0x2, 0x2, 0x5f, 
    0x65, 0x5, 0x8, 0x5, 0x2, 0x60, 0x61, 0x5, 0x8, 0x5, 0x2, 0x61, 0x62, 
    0x9, 0x2, 0x2, 0x2, 0x62, 0x63, 0x5, 0x6, 0x4, 0x2, 0x63, 0x65, 0x3, 
    0x2, 0x2, 0x2, 0x64, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x64, 0x60, 0x3, 0x2, 
    0x2, 0x2, 0x65, 0x7, 0x3, 0x2, 0x2, 0x2, 0x66, 0x6a, 0x5, 0xa, 0x6, 
    0x2, 0x67, 0x68, 0x7, 0x24, 0x2, 0x2, 0x68, 0x6a, 0x5, 0xa, 0x6, 0x2, 
    0x69, 0x66, 0x3, 0x2, 0x2, 0x2, 0x69, 0x67, 0x3, 0x2, 0x2, 0x2, 0x6a, 
    0x9, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x73, 0x7, 0x36, 0x2, 0x2, 0x6c, 0x6d, 
    0x7, 0x36, 0x2, 0x2, 0x6d, 0x6e, 0x7, 0x23, 0x2, 0x2, 0x6e, 0x73, 0x7, 
    0x36, 0x2, 0x2, 0x6f, 0x70, 0x7, 0x36, 0x2, 0x2, 0x70, 0x71, 0x7, 0x24, 
    0x2, 0x2, 0x71, 0x73, 0x7, 0x36, 0x2, 0x2, 0x72, 0x6b, 0x3, 0x2, 0x2, 
    0x2, 0x72, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x72, 0x6f, 0x3, 0x2, 0x2, 0x2, 
    0x73, 0xb, 0x3, 0x2, 0x2, 0x2, 0x74, 0x75, 0x9, 0x3, 0x2, 0x2, 0x75, 
    0xd, 0x3, 0x2, 0x2, 0x2, 0x76, 0x77, 0x5, 0xc, 0x7, 0x2, 0x77, 0x78, 
    0x5, 0x10, 0x9, 0x2, 0x78, 0x79, 0x7, 0x30, 0x2, 0x2, 0x79, 0xf, 0x3, 
    0x2, 0x2, 0x2, 0x7a, 0x7b, 0x8, 0x9, 0x1, 0x2, 0x7b, 0x81, 0x7, 0x35, 
    0x2, 0x2, 0x7c, 0x7d, 0x7, 0x35, 0x2, 0x2, 0x7d, 0x7e, 0x7, 0x2c, 0x2, 
    0x2, 0x7e, 0x7f, 0x7, 0x36, 0x2, 0x2, 0x7f, 0x81, 0x7, 0x2d, 0x2, 0x2, 
    0x80, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x80, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x81, 
    0x87, 0x3, 0x2, 0x2, 0x2, 0x82, 0x83, 0xc, 0x3, 0x2, 0x2, 0x83, 0x84, 
    0x7, 0x29, 0x2, 0x2, 0x84, 0x86, 0x5, 0x10, 0x9, 0x4, 0x85, 0x82, 0x3, 
    0x2, 0x2, 0x2, 0x86, 0x89, 0x3, 0x2, 0x2, 0x2, 0x87, 0x85, 0x3, 0x2, 
    0x2, 0x2, 0x87, 0x88, 0x3, 0x2, 0x2, 0x2, 0x88, 0x11, 0x3, 0x2, 0x2, 
    0x2, 0x89, 0x87, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x8c, 0x5, 0x14, 0xb, 0x2, 
    0x8b, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x8d, 
    0x8b, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x90, 
    0x3, 0x2, 0x2, 0x2, 0x8f, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x90, 0x91, 0x5, 
    0x18, 0xd, 0x2, 0x91, 0x13, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 0x9, 0x4, 
    0x2, 0x2, 0x93, 0x94, 0x7, 0x35, 0x2, 0x2, 0x94, 0x95, 0x7, 0x2a, 0x2, 
    0x2, 0x95, 0x96, 0x7, 0x2b, 0x2, 0x2, 0x96, 0x97, 0x7, 0x2e, 0x2, 0x2, 
    0x97, 0x98, 0x5, 0x1a, 0xe, 0x2, 0x98, 0x99, 0x7, 0x2f, 0x2, 0x2, 0x99, 
    0xa4, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x9b, 0x9, 0x4, 0x2, 0x2, 0x9b, 0x9c, 
    0x7, 0x35, 0x2, 0x2, 0x9c, 0x9d, 0x7, 0x2a, 0x2, 0x2, 0x9d, 0x9e, 0x5, 
    0x16, 0xc, 0x2, 0x9e, 0x9f, 0x7, 0x2b, 0x2, 0x2, 0x9f, 0xa0, 0x7, 0x2e, 
    0x2, 0x2, 0xa0, 0xa1, 0x5, 0x1a, 0xe, 0x2, 0xa1, 0xa2, 0x7, 0x2f, 0x2, 
    0x2, 0xa2, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xa3, 0x92, 0x3, 0x2, 0x2, 0x2, 
    0xa3, 0x9a, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x15, 0x3, 0x2, 0x2, 0x2, 0xa5, 
    0xa6, 0x8, 0xc, 0x1, 0x2, 0xa6, 0xa7, 0x5, 0xc, 0x7, 0x2, 0xa7, 0xa8, 
    0x7, 0x35, 0x2, 0x2, 0xa8, 0xaf, 0x3, 0x2, 0x2, 0x2, 0xa9, 0xaa, 0x5, 
    0xc, 0x7, 0x2, 0xaa, 0xab, 0x7, 0x35, 0x2, 0x2, 0xab, 0xac, 0x7, 0x2c, 
    0x2, 0x2, 0xac, 0xad, 0x7, 0x2d, 0x2, 0x2, 0xad, 0xaf, 0x3, 0x2, 0x2, 
    0x2, 0xae, 0xa5, 0x3, 0x2, 0x2, 0x2, 0xae, 0xa9, 0x3, 0x2, 0x2, 0x2, 
    0xaf, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 0xc, 0x3, 0x2, 0x2, 0xb1, 
    0xb2, 0x7, 0x29, 0x2, 0x2, 0xb2, 0xb4, 0x5, 0x16, 0xc, 0x4, 0xb3, 0xb0, 
    0x3, 0x2, 0x2, 0x2, 0xb4, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb3, 0x3, 
    0x2, 0x2, 0x2, 0xb5, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb6, 0x17, 0x3, 0x2, 
    0x2, 0x2, 0xb7, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x7, 0x10, 0x2, 
    0x2, 0xb9, 0xba, 0x7, 0x11, 0x2, 0x2, 0xba, 0xbb, 0x7, 0x2a, 0x2, 0x2, 
    0xbb, 0xbc, 0x7, 0x2b, 0x2, 0x2, 0xbc, 0xbd, 0x7, 0x2e, 0x2, 0x2, 0xbd, 
    0xbe, 0x5, 0x1a, 0xe, 0x2, 0xbe, 0xbf, 0x7, 0x2f, 0x2, 0x2, 0xbf, 0x19, 
    0x3, 0x2, 0x2, 0x2, 0xc0, 0xc5, 0x5, 0x1e, 0x10, 0x2, 0xc1, 0xc2, 0x5, 
    0x1e, 0x10, 0x2, 0xc2, 0xc3, 0x5, 0x48, 0x25, 0x2, 0xc3, 0xc5, 0x3, 
    0x2, 0x2, 0x2, 0xc4, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc1, 0x3, 0x2, 
    0x2, 0x2, 0xc5, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xd1, 0x5, 0x34, 0x1b, 
    0x2, 0xc7, 0xd1, 0x5, 0x36, 0x1c, 0x2, 0xc8, 0xd1, 0x5, 0x38, 0x1d, 
    0x2, 0xc9, 0xd1, 0x5, 0x3c, 0x1f, 0x2, 0xca, 0xd1, 0x5, 0x42, 0x22, 
    0x2, 0xcb, 0xd1, 0x5, 0x40, 0x21, 0x2, 0xcc, 0xd1, 0x5, 0x3e, 0x20, 
    0x2, 0xcd, 0xd1, 0x5, 0x44, 0x23, 0x2, 0xce, 0xd1, 0x5, 0x46, 0x24, 
    0x2, 0xcf, 0xd1, 0x5, 0xe, 0x8, 0x2, 0xd0, 0xc6, 0x3, 0x2, 0x2, 0x2, 
    0xd0, 0xc7, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xc8, 0x3, 0x2, 0x2, 0x2, 0xd0, 
    0xc9, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xca, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xcb, 
    0x3, 0x2, 0x2, 0x2, 0xd0, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xcd, 0x3, 
    0x2, 0x2, 0x2, 0xd0, 0xce, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xcf, 0x3, 0x2, 
    0x2, 0x2, 0xd1, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd4, 0x5, 0x1c, 0xf, 
    0x2, 0xd3, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x3, 0x2, 0x2, 0x2, 
    0xd5, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x3, 0x2, 0x2, 0x2, 0xd6, 
    0x1f, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd8, 0x9, 0x5, 0x2, 0x2, 0xd8, 0x21, 
    0x3, 0x2, 0x2, 0x2, 0xd9, 0xe1, 0x7, 0x35, 0x2, 0x2, 0xda, 0xe1, 0x7, 
    0x36, 0x2, 0x2, 0xdb, 0xdc, 0x7, 0x35, 0x2, 0x2, 0xdc, 0xdd, 0x7, 0x2c, 
    0x2, 0x2, 0xdd, 0xde, 0x5, 0x22, 0x12, 0x2, 0xde, 0xdf, 0x7, 0x2d, 0x2, 
    0x2, 0xdf, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xd9, 0x3, 0x2, 0x2, 0x2, 
    0xe0, 0xda, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xdb, 0x3, 0x2, 0x2, 0x2, 0xe1, 
    0x23, 0x3, 0x2, 0x2, 0x2, 0xe2, 0xe3, 0x8, 0x13, 0x1, 0x2, 0xe3, 0xe4, 
    0x5, 0x22, 0x12, 0x2, 0xe4, 0xea, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe6, 0xc, 
    0x4, 0x2, 0x2, 0xe6, 0xe7, 0x7, 0x29, 0x2, 0x2, 0xe7, 0xe9, 0x5, 0x24, 
    0x13, 0x5, 0xe8, 0xe5, 0x3, 0x2, 0x2, 0x2, 0xe9, 0xec, 0x3, 0x2, 0x2, 
    0x2, 0xea, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xea, 0xeb, 0x3, 0x2, 0x2, 0x2, 
    0xeb, 0x25, 0x3, 0x2, 0x2, 0x2, 0xec, 0xea, 0x3, 0x2, 0x2, 0x2, 0xed, 
    0xee, 0x9, 0x6, 0x2, 0x2, 0xee, 0x27, 0x3, 0x2, 0x2, 0x2, 0xef, 0xf0, 
    0x9, 0x7, 0x2, 0x2, 0xf0, 0x29, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf7, 0x5, 
    0x2c, 0x17, 0x2, 0xf2, 0xf3, 0x5, 0x26, 0x14, 0x2, 0xf3, 0xf4, 0x5, 
    0x2c, 0x17, 0x2, 0xf4, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf2, 0x3, 0x2, 
    0x2, 0x2, 0xf6, 0xf9, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xf5, 0x3, 0x2, 0x2, 
    0x2, 0xf7, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xf8, 0x2b, 0x3, 0x2, 0x2, 0x2, 
    0xf9, 0xf7, 0x3, 0x2, 0x2, 0x2, 0xfa, 0x100, 0x5, 0x2e, 0x18, 0x2, 0xfb, 
    0xfc, 0x5, 0x28, 0x15, 0x2, 0xfc, 0xfd, 0x5, 0x2e, 0x18, 0x2, 0xfd, 
    0xff, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xfb, 0x3, 0x2, 0x2, 0x2, 0xff, 0x102, 
    0x3, 0x2, 0x2, 0x2, 0x100, 0xfe, 0x3, 0x2, 0x2, 0x2, 0x100, 0x101, 0x3, 
    0x2, 0x2, 0x2, 0x101, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x102, 0x100, 0x3, 0x2, 
    0x2, 0x2, 0x103, 0x109, 0x5, 0x22, 0x12, 0x2, 0x104, 0x105, 0x7, 0x2a, 
    0x2, 0x2, 0x105, 0x106, 0x5, 0x2a, 0x16, 0x2, 0x106, 0x107, 0x7, 0x2b, 
    0x2, 0x2, 0x107, 0x109, 0x3, 0x2, 0x2, 0x2, 0x108, 0x103, 0x3, 0x2, 
    0x2, 0x2, 0x108, 0x104, 0x3, 0x2, 0x2, 0x2, 0x109, 0x2f, 0x3, 0x2, 0x2, 
    0x2, 0x10a, 0x10b, 0x7, 0x1d, 0x2, 0x2, 0x10b, 0x10c, 0x7, 0x27, 0x2, 
    0x2, 0x10c, 0x10d, 0x5, 0x22, 0x12, 0x2, 0x10d, 0x10e, 0x7, 0x28, 0x2, 
    0x2, 0x10e, 0x31, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x110, 0x9, 0x8, 0x2, 0x2, 
    0x110, 0x33, 0x3, 0x2, 0x2, 0x2, 0x111, 0x112, 0x5, 0x22, 0x12, 0x2, 
    0x112, 0x113, 0x7, 0x22, 0x2, 0x2, 0x113, 0x114, 0x7, 0x34, 0x2, 0x2, 
    0x114, 0x115, 0x7, 0x30, 0x2, 0x2, 0x115, 0x35, 0x3, 0x2, 0x2, 0x2, 
    0x116, 0x117, 0x5, 0x20, 0x11, 0x2, 0x117, 0x118, 0x7, 0x27, 0x2, 0x2, 
    0x118, 0x119, 0x5, 0x24, 0x13, 0x2, 0x119, 0x11a, 0x7, 0x28, 0x2, 0x2, 
    0x11a, 0x11b, 0x7, 0x30, 0x2, 0x2, 0x11b, 0x37, 0x3, 0x2, 0x2, 0x2, 
    0x11c, 0x11d, 0x5, 0x22, 0x12, 0x2, 0x11d, 0x11e, 0x7, 0x22, 0x2, 0x2, 
    0x11e, 0x11f, 0x5, 0x2a, 0x16, 0x2, 0x11f, 0x120, 0x7, 0x30, 0x2, 0x2, 
    0x120, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x121, 0x122, 0x5, 0x22, 0x12, 0x2, 
    0x122, 0x123, 0x7, 0x22, 0x2, 0x2, 0x123, 0x124, 0x5, 0x30, 0x19, 0x2, 
    0x124, 0x125, 0x7, 0x30, 0x2, 0x2, 0x125, 0x12b, 0x3, 0x2, 0x2, 0x2, 
    0x126, 0x127, 0x5, 0x22, 0x12, 0x2, 0x127, 0x128, 0x7, 0x22, 0x2, 0x2, 
    0x128, 0x129, 0x5, 0x42, 0x22, 0x2, 0x129, 0x12b, 0x3, 0x2, 0x2, 0x2, 
    0x12a, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x12a, 0x121, 0x3, 0x2, 0x2, 0x2, 
    0x12a, 0x126, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x39, 0x3, 0x2, 0x2, 0x2, 0x12c, 
    0x132, 0x5, 0x1c, 0xf, 0x2, 0x12d, 0x12e, 0x7, 0x2e, 0x2, 0x2, 0x12e, 
    0x12f, 0x5, 0x1e, 0x10, 0x2, 0x12f, 0x130, 0x7, 0x2f, 0x2, 0x2, 0x130, 
    0x132, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12c, 0x3, 0x2, 0x2, 0x2, 0x131, 
    0x12d, 0x3, 0x2, 0x2, 0x2, 0x132, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x133, 0x134, 
    0x7, 0x7, 0x2, 0x2, 0x134, 0x135, 0x7, 0x2a, 0x2, 0x2, 0x135, 0x136, 
    0x5, 0x2a, 0x16, 0x2, 0x136, 0x137, 0x5, 0x32, 0x1a, 0x2, 0x137, 0x138, 
    0x5, 0x2a, 0x16, 0x2, 0x138, 0x139, 0x7, 0x2b, 0x2, 0x2, 0x139, 0x13a, 
    0x5, 0x3a, 0x1e, 0x2, 0x13a, 0x146, 0x3, 0x2, 0x2, 0x2, 0x13b, 0x13c, 
    0x7, 0x7, 0x2, 0x2, 0x13c, 0x13d, 0x7, 0x2a, 0x2, 0x2, 0x13d, 0x13e, 
    0x5, 0x2a, 0x16, 0x2, 0x13e, 0x13f, 0x5, 0x32, 0x1a, 0x2, 0x13f, 0x140, 
    0x5, 0x2a, 0x16, 0x2, 0x140, 0x141, 0x7, 0x2b, 0x2, 0x2, 0x141, 0x142, 
    0x5, 0x3a, 0x1e, 0x2, 0x142, 0x143, 0x7, 0x9, 0x2, 0x2, 0x143, 0x144, 
    0x5, 0x3a, 0x1e, 0x2, 0x144, 0x146, 0x3, 0x2, 0x2, 0x2, 0x145, 0x133, 
    0x3, 0x2, 0x2, 0x2, 0x145, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x146, 0x3d, 0x3, 
    0x2, 0x2, 0x2, 0x147, 0x148, 0x7, 0xb, 0x2, 0x2, 0x148, 0x149, 0x7, 
    0x2a, 0x2, 0x2, 0x149, 0x14a, 0x7, 0x35, 0x2, 0x2, 0x14a, 0x14b, 0x7, 
    0x22, 0x2, 0x2, 0x14b, 0x14c, 0x5, 0x22, 0x12, 0x2, 0x14c, 0x14d, 0x7, 
    0xc, 0x2, 0x2, 0x14d, 0x14e, 0x5, 0x22, 0x12, 0x2, 0x14e, 0x14f, 0x7, 
    0x2b, 0x2, 0x2, 0x14f, 0x150, 0x5, 0x3a, 0x1e, 0x2, 0x150, 0x3f, 0x3, 
    0x2, 0x2, 0x2, 0x151, 0x152, 0x7, 0xd, 0x2, 0x2, 0x152, 0x153, 0x7, 
    0x2a, 0x2, 0x2, 0x153, 0x154, 0x5, 0x2a, 0x16, 0x2, 0x154, 0x155, 0x5, 
    0x32, 0x1a, 0x2, 0x155, 0x156, 0x5, 0x2a, 0x16, 0x2, 0x156, 0x157, 0x7, 
    0x2b, 0x2, 0x2, 0x157, 0x158, 0x5, 0x3a, 0x1e, 0x2, 0x158, 0x41, 0x3, 
    0x2, 0x2, 0x2, 0x159, 0x15a, 0x7, 0x35, 0x2, 0x2, 0x15a, 0x15d, 0x7, 
    0x2a, 0x2, 0x2, 0x15b, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x15e, 0x5, 
    0x24, 0x13, 0x2, 0x15d, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15c, 0x3, 
    0x2, 0x2, 0x2, 0x15e, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x160, 0x7, 
    0x2b, 0x2, 0x2, 0x160, 0x161, 0x7, 0x30, 0x2, 0x2, 0x161, 0x43, 0x3, 
    0x2, 0x2, 0x2, 0x162, 0x163, 0x7, 0x1e, 0x2, 0x2, 0x163, 0x164, 0x5, 
    0x22, 0x12, 0x2, 0x164, 0x165, 0x7, 0x30, 0x2, 0x2, 0x165, 0x45, 0x3, 
    0x2, 0x2, 0x2, 0x166, 0x167, 0x7, 0x20, 0x2, 0x2, 0x167, 0x168, 0x7, 
    0x30, 0x2, 0x2, 0x168, 0x47, 0x3, 0x2, 0x2, 0x2, 0x169, 0x16a, 0x7, 
    0x21, 0x2, 0x2, 0x16a, 0x16b, 0x5, 0x22, 0x12, 0x2, 0x16b, 0x16c, 0x7, 
    0x30, 0x2, 0x2, 0x16c, 0x49, 0x3, 0x2, 0x2, 0x2, 0x19, 0x4d, 0x53, 0x64, 
    0x69, 0x72, 0x80, 0x87, 0x8d, 0xa3, 0xae, 0xb5, 0xc4, 0xd0, 0xd5, 0xe0, 
    0xea, 0xf7, 0x100, 0x108, 0x12a, 0x131, 0x145, 0x15d, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

ISQParser::Initializer ISQParser::_init;
