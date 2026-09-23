//! Errors produced while building or navigating a tape.

use std::fmt;

pub(crate) const NOT_SINGLE_VALUE: &str =
    "input is not a single JSON value (for JSON Lines, use parse_lines)";

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
    NotSingleValue,
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
            Error::NotSingleValue => write!(f, "{NOT_SINGLE_VALUE}"),
        }
    }
}

impl std::error::Error for Error {}
