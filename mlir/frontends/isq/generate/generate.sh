rm -f *.cpp
rm -f *.h

alias antlr4='java -jar /usr/local/lib/antlr-4.7.2-complete.jar'
alias grun='java org.antlr.v4.gui.TestRig'

antlr4 -Dlanguage=Cpp ISQLexer.g4 -visitor
antlr4 -Dlanguage=Cpp ISQParser.g4 -visitor
