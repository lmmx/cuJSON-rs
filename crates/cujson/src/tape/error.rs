//! Errors produced while building or navigating a tape.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyInput,
    InvalidUtf8,
    UnbalancedBrackets,
    UnterminatedString,
    InvalidEscape,
    InvalidNumber,
    IndexOutOfRange,
    KeyNotFound,
    TypeMismatch(&'static str),
    InvalidPointer,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyInput => write!(f, "empty input"),
            Error::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Error::UnbalancedBrackets => write!(f, "unbalanced brackets"),
            Error::UnterminatedString => write!(f, "unterminated string"),
            Error::InvalidEscape => write!(f, "invalid escape sequence"),
            Error::InvalidNumber => write!(f, "invalid number"),
            Error::IndexOutOfRange => write!(f, "array index out of range"),
            Error::KeyNotFound => write!(f, "object key not found"),
            Error::TypeMismatch(expected) => write!(f, "expected {expected}"),
            Error::InvalidPointer => write!(f, "invalid JSON pointer"),
        }
    }
}

impl std::error::Error for Error {}
