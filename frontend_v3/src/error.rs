// TODO: better error.

use crate::{lang::location::Span, resolve::{symboltable::SymbolInfo, package::{Module, Symbol}}};

pub enum ISQFrontendError{
    GeneralError(&'static str, Span),
    RedefinedSymbol{
        symbol_name: String,
        defined_here: Span,
        first_defined_here: Span
    },
    NameClashWithModule{
        symbol_name: String,
        defined_here: Span,
        module_path: String,
    },
    UndefinedSymbol{
        symbol_name: String,
        used_here: Span,
    },
    NameIsModule{
        usage_name: String,
        qualified_name: String,
        used_here: Span,
    },
    NotAModule{
        qualified_name: String,
        used_here: Span,
        defined_here: Span
    }
}
impl ISQFrontendError{
    pub fn undefined_symbol_error(symbol_name: &str, used_here: Span)->Self{
        Self::UndefinedSymbol { symbol_name: symbol_name.to_owned(), used_here }
    }
    pub fn undefined_module_symbol_error(module: &Module, entry: &str, used_here: Span)->Self{
        Self::UndefinedSymbol { symbol_name: format!("{}::{}", module.qualified_name(), entry), used_here }
    }
    pub fn name_is_module_error(qualified: &str, module: &Module, used_here: Span)->Self{
        Self::NameIsModule { usage_name: qualified.to_owned(), qualified_name: module.qualified_name(), used_here }
    }
    pub fn not_a_module_error(symbol: &Symbol, used_here: Span)->Self{
        Self::NotAModule { qualified_name: symbol.qualified_name(), used_here, defined_here: symbol.definition_location() } 
    }
    pub fn redefined_symbol_with_definition_error(symbol_name: &str, defined_here: Span, clashed_with: &SymbolInfo)->ISQFrontendError{
        let clashed_with = (*clashed_with, match clashed_with{
            SymbolInfo::GlobalModule(module) => {
                Span::empty()
            },
            SymbolInfo::GlobalSymbol(symbol) => {
                symbol.definition_location()
            },
            SymbolInfo::Local(local) => panic!(),
        });
        Self::redefined_symbol_error(symbol_name, defined_here, &clashed_with)
    }
    pub fn redefined_symbol_error(symbol_name: &str, defined_here: Span, clashed_with: &(SymbolInfo, Span))->ISQFrontendError{
        use crate::resolve::symboltable::SymbolInfo::*;
        let symbol_name = symbol_name.to_owned();
        match clashed_with.0{
            GlobalModule(module) => {
                ISQFrontendError::NameClashWithModule{
                    symbol_name,
                    defined_here,
                    module_path: module.qualified_name()
                }
            },
            GlobalSymbol(sym) => {
                ISQFrontendError::RedefinedSymbol{
                    symbol_name,
                    defined_here,
                    first_defined_here: sym.definition_location()
                }
            },
            Local(id, )=>{
                ISQFrontendError::RedefinedSymbol{
                    symbol_name,
                    defined_here,
                    first_defined_here: clashed_with.1
                }
            }
        }
    }
}


pub type FResult<T> = core::result::Result<T, ISQFrontendError>;