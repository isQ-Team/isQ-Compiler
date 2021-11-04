
// Generated from ISQLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  ISQLexer : public antlr4::Lexer {
public:
  enum {
    WhiteSpace = 1, NewLine = 2, BlockComment = 3, LineComment = 4, If = 5, 
    Then = 6, Else = 7, Fi = 8, For = 9, To = 10, While = 11, Do = 12, Od = 13, 
    Procedure = 14, Main = 15, Int = 16, Qbit = 17, H = 18, X = 19, Y = 20, 
    Z = 21, S = 22, T = 23, CZ = 24, CX = 25, CNOT = 26, M = 27, Print = 28, 
    Defgate = 29, Pass = 30, Return = 31, Assign = 32, Plus = 33, Minus = 34, 
    Mult = 35, Div = 36, Less = 37, Greater = 38, Comma = 39, LeftParen = 40, 
    RightParen = 41, LeftBracket = 42, RightBracket = 43, LeftBrace = 44, 
    RightBrace = 45, Semi = 46, Equal = 47, LessEqual = 48, GreaterEqual = 49, 
    KetZero = 50, Identifier = 51, Number = 52
  };

  ISQLexer(antlr4::CharStream *input);
  ~ISQLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

