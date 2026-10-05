//! Tokenizer for Soufflé .dl source.
//!
//! Hand-rolled lexer covering the subset of Soufflé we
//! parse for static analysis. We track 1-indexed line
//! numbers on every token so the AST and downstream
//! findings can carry source locations.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// Identifier: relation, variable, type name. Soufflé
    /// distinguishes case but the parser handles
    /// disambiguation; the lexer treats all alphanumeric
    /// identifiers uniformly.
    Ident(String),
    /// String literal — without the surrounding quotes.
    String(String),
    /// Integer literal — kept as the source spelling so
    /// we can defer signed/unsigned interpretation to
    /// the parser.
    Int(i128),
    /// Punctuation and keyword tokens.
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Dot,
    Colon,
    Semicolon,
    /// `:-` rule arrow.
    ColonDash,
    /// `<:` subtype arrow used in `.type X <: symbol`.
    SubtypeArrow,
    Pipe,
    /// `=` (comparison/binding).
    Eq,
    /// `!=`.
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    /// Negation `!` (when followed by an atom).
    Bang,
    /// `$` constructor prefix: `$CallTool(...)`.
    Dollar,
    /// `@` functor prefix: `@json_get_str(...)`.
    At,
    /// Unsigned literal `5u`, `0xFFu`, `0b101u`. Stored
    /// as the underlying value with the `u` suffix
    /// already stripped.
    Uint(u64),
    /// `true` keyword body constraint.
    KwTrue,
    /// `false` keyword body constraint.
    KwFalse,
    /// `nil` constant for empty records.
    KwNil,
    /// `as` keyword for type casts: `as(x, MyType)`.
    KwAs,
    /// Keyword-like tokens (recognized after lexing
    /// so directives keep their leading dot).
    DotDecl,
    DotInput,
    DotOutput,
    DotType,
    DotFunctor,
    DotPrintsize,
    KwNot,
    KwStateful,
    /// End of input.
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "ident({})", s),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Int(n) => write!(f, "{}", n),
            t => write!(f, "{:?}", t),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned {
    pub token: Token,
    pub line: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum LexError {
    #[error("unterminated string literal at line {line}")]
    UnterminatedString { line: u32 },
    #[error("unterminated block comment at line {line}")]
    UnterminatedBlockComment { line: u32 },
    #[error("unexpected character {ch:?} at line {line}")]
    UnexpectedChar { ch: char, line: u32 },
    #[error("invalid integer literal {value:?} at line {line}")]
    InvalidInt { value: String, line: u32 },
}

/// Parse a number literal starting at the current
/// position. Handles decimal, hexadecimal (`0x...`),
/// binary (`0b...`), and the optional `u` suffix that
/// produces an unsigned literal. Soufflé's IPv4 literal
/// form (`a.b.c.d`) and float literals are not yet
/// supported.
fn read_number_literal(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    line: u32,
) -> Result<Token, LexError> {
    let mut s = String::new();
    let mut digits = String::new();
    let (radix, prefix_consumed) = match chars.peek() {
        Some(&(_, '0')) => {
            // Look ahead for `0x` / `0b`. If the next
            // char isn't a radix marker, the leading `0`
            // is part of the decimal digit sequence.
            chars.next();
            s.push('0');
            match chars.peek().map(|(_, c)| *c) {
                Some('x') | Some('X') => {
                    chars.next();
                    s.push('x');
                    (16, true)
                }
                Some('b') | Some('B') => {
                    chars.next();
                    s.push('b');
                    (2, true)
                }
                _ => {
                    digits.push('0');
                    (10, false)
                }
            }
        }
        _ => (10, false),
    };
    while let Some(&(_, ch)) = chars.peek() {
        let is_digit = match radix {
            16 => ch.is_ascii_hexdigit(),
            2 => ch == '0' || ch == '1',
            _ => ch.is_ascii_digit(),
        };
        if is_digit {
            digits.push(ch);
            s.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    if prefix_consumed && digits.is_empty() {
        return Err(LexError::InvalidInt { value: s, line });
    }
    // Optional unsigned suffix.
    let unsigned = if matches!(chars.peek(), Some(&(_, 'u')) | Some(&(_, 'U'))) {
        chars.next();
        true
    } else {
        false
    };
    if unsigned {
        let n = if radix == 10 {
            digits.parse::<u64>().map_err(|_| LexError::InvalidInt {
                value: s.clone(),
                line,
            })?
        } else {
            u64::from_str_radix(&digits, radix as u32).map_err(|_| LexError::InvalidInt {
                value: s.clone(),
                line,
            })?
        };
        Ok(Token::Uint(n))
    } else if radix == 10 {
        // Decimal: keep as i128 then narrow to Int.
        let n = digits.parse::<i128>().map_err(|_| LexError::InvalidInt {
            value: s.clone(),
            line,
        })?;
        Ok(Token::Int(n))
    } else {
        // Hex/binary without suffix → still produce Int.
        let n = i128::from_str_radix(&digits, radix as u32).map_err(|_| LexError::InvalidInt {
            value: s.clone(),
            line,
        })?;
        Ok(Token::Int(n))
    }
}

pub fn tokenize(input: &str) -> Result<Vec<Spanned>, LexError> {
    let mut out = Vec::new();
    let mut chars = input.char_indices().peekable();
    let mut line: u32 = 1;
    while let Some(&(_, c)) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' => {
                chars.next();
            }
            '\n' => {
                line += 1;
                chars.next();
            }
            '/' => {
                let (_, _) = chars.next().unwrap();
                match chars.peek().map(|(_, c)| *c) {
                    Some('/') => {
                        chars.next();
                        while let Some(&(_, ch)) = chars.peek() {
                            if ch == '\n' {
                                break;
                            }
                            chars.next();
                        }
                    }
                    Some('*') => {
                        chars.next();
                        let start_line = line;
                        let mut closed = false;
                        while let Some((_, ch)) = chars.next() {
                            if ch == '\n' {
                                line += 1;
                            }
                            if ch == '*' {
                                if let Some(&(_, '/')) = chars.peek() {
                                    chars.next();
                                    closed = true;
                                    break;
                                }
                            }
                        }
                        if !closed {
                            return Err(LexError::UnterminatedBlockComment { line: start_line });
                        }
                    }
                    _ => {
                        out.push(Spanned {
                            token: Token::Slash,
                            line,
                        });
                    }
                }
            }
            '"' => {
                chars.next();
                let start_line = line;
                let mut s = String::new();
                let mut closed = false;
                while let Some(&(_, ch)) = chars.peek() {
                    if ch == '"' {
                        chars.next();
                        closed = true;
                        break;
                    } else if ch == '\\' {
                        chars.next();
                        if let Some((_, esc)) = chars.next() {
                            let real = match esc {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                '\\' => '\\',
                                '"' => '"',
                                other => other,
                            };
                            s.push(real);
                        }
                    } else if ch == '\n' {
                        return Err(LexError::UnterminatedString { line: start_line });
                    } else {
                        s.push(ch);
                        chars.next();
                    }
                }
                if !closed {
                    return Err(LexError::UnterminatedString { line: start_line });
                }
                out.push(Spanned {
                    token: Token::String(s),
                    line: start_line,
                });
            }
            '.' => {
                chars.next();
                // A `.<keyword>` token is a directive
                // only when it occurs at the start of a
                // statement — i.e., no prior token on
                // the same line. Inside an expression
                // (e.g., after an Ident on the same
                // line), `.` is sugar-level member
                // access; we emit a plain `Dot` and let
                // the parser consume the following
                // identifier as a field name.
                let directive_eligible = match out.last() {
                    None => true,
                    Some(t) => t.line != line,
                };
                if directive_eligible {
                    if let Some(&(_, ch)) = chars.peek() {
                        if ch.is_ascii_alphabetic() {
                            let mut name = String::new();
                            while let Some(&(_, c)) = chars.peek() {
                                if c.is_ascii_alphanumeric() || c == '_' {
                                    name.push(c);
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                            let tok = match name.as_str() {
                                "decl" => Token::DotDecl,
                                "input" => Token::DotInput,
                                "output" => Token::DotOutput,
                                "type" => Token::DotType,
                                "functor" => Token::DotFunctor,
                                "printsize" => Token::DotPrintsize,
                                _ => {
                                    return Err(LexError::UnexpectedChar { ch: '.', line });
                                }
                            };
                            out.push(Spanned { token: tok, line });
                            continue;
                        }
                    }
                }
                out.push(Spanned {
                    token: Token::Dot,
                    line,
                });
            }
            '(' => {
                chars.next();
                out.push(Spanned {
                    token: Token::LParen,
                    line,
                });
            }
            ')' => {
                chars.next();
                out.push(Spanned {
                    token: Token::RParen,
                    line,
                });
            }
            '[' => {
                chars.next();
                out.push(Spanned {
                    token: Token::LBracket,
                    line,
                });
            }
            ']' => {
                chars.next();
                out.push(Spanned {
                    token: Token::RBracket,
                    line,
                });
            }
            '{' => {
                chars.next();
                out.push(Spanned {
                    token: Token::LBrace,
                    line,
                });
            }
            '}' => {
                chars.next();
                out.push(Spanned {
                    token: Token::RBrace,
                    line,
                });
            }
            ',' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Comma,
                    line,
                });
            }
            ';' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Semicolon,
                    line,
                });
            }
            ':' => {
                chars.next();
                match chars.peek().map(|(_, c)| *c) {
                    Some('-') => {
                        chars.next();
                        out.push(Spanned {
                            token: Token::ColonDash,
                            line,
                        });
                    }
                    _ => {
                        out.push(Spanned {
                            token: Token::Colon,
                            line,
                        });
                    }
                }
            }
            '|' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Pipe,
                    line,
                });
            }
            '=' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Eq,
                    line,
                });
            }
            '!' => {
                chars.next();
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                    out.push(Spanned {
                        token: Token::Ne,
                        line,
                    });
                } else {
                    out.push(Spanned {
                        token: Token::Bang,
                        line,
                    });
                }
            }
            '<' => {
                chars.next();
                match chars.peek().map(|(_, c)| *c) {
                    Some('=') => {
                        chars.next();
                        out.push(Spanned {
                            token: Token::Le,
                            line,
                        });
                    }
                    Some(':') => {
                        chars.next();
                        out.push(Spanned {
                            token: Token::SubtypeArrow,
                            line,
                        });
                    }
                    _ => {
                        out.push(Spanned {
                            token: Token::Lt,
                            line,
                        });
                    }
                }
            }
            '>' => {
                chars.next();
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                    out.push(Spanned {
                        token: Token::Ge,
                        line,
                    });
                } else {
                    out.push(Spanned {
                        token: Token::Gt,
                        line,
                    });
                }
            }
            '+' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Plus,
                    line,
                });
            }
            '-' => {
                chars.next();
                if let Some(&(_, ch)) = chars.peek() {
                    if ch.is_ascii_digit() {
                        let mut s = String::from('-');
                        while let Some(&(_, c)) = chars.peek() {
                            if c.is_ascii_digit() {
                                s.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        let n = s
                            .parse::<i128>()
                            .map_err(|_| LexError::InvalidInt { value: s, line })?;
                        out.push(Spanned {
                            token: Token::Int(n),
                            line,
                        });
                        continue;
                    }
                }
                out.push(Spanned {
                    token: Token::Minus,
                    line,
                });
            }
            '*' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Star,
                    line,
                });
            }
            '%' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Percent,
                    line,
                });
            }
            '$' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Dollar,
                    line,
                });
            }
            '@' => {
                chars.next();
                out.push(Spanned {
                    token: Token::At,
                    line,
                });
            }
            c if c.is_ascii_digit() => {
                let token = read_number_literal(&mut chars, line)?;
                out.push(Spanned { token, line });
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::new();
                while let Some(&(_, ch)) = chars.peek() {
                    if ch.is_ascii_alphanumeric() || ch == '_' {
                        s.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let tok = match s.as_str() {
                    "not" => Token::KwNot,
                    "stateful" => Token::KwStateful,
                    "true" => Token::KwTrue,
                    "false" => Token::KwFalse,
                    "nil" => Token::KwNil,
                    "as" => Token::KwAs,
                    _ => Token::Ident(s),
                };
                out.push(Spanned { token: tok, line });
            }
            other => {
                return Err(LexError::UnexpectedChar { ch: other, line });
            }
        }
    }
    out.push(Spanned {
        token: Token::Eof,
        line,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<Token> {
        tokenize(s).unwrap().into_iter().map(|t| t.token).collect()
    }

    #[test]
    fn rule_with_constructor_and_functor() {
        let src = r#"IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _)."#;
        let ts = tokens(src);
        assert!(matches!(ts[0], Token::Ident(ref s) if s == "IsTool"));
        assert!(ts.contains(&Token::ColonDash));
        assert!(ts.iter().any(|t| matches!(t, Token::Dollar)));
    }

    #[test]
    fn negation_and_comparisons() {
        let src = r#"R(x) :- !S(x), x > 0, y != "z"."#;
        let ts = tokens(src);
        assert!(ts.contains(&Token::Bang));
        assert!(ts.contains(&Token::Gt));
        assert!(ts.contains(&Token::Ne));
    }

    #[test]
    fn directive_keywords() {
        // Directives are recognized only at line start —
        // realistic Soufflé places each on its own line.
        let src = ".decl R(x: symbol)\n.input R\n.output R";
        let ts = tokens(src);
        assert_eq!(ts[0], Token::DotDecl);
        assert!(ts.contains(&Token::DotInput));
        assert!(ts.contains(&Token::DotOutput));
    }

    #[test]
    fn comments_skipped() {
        let src = "// comment\n/* block\nblock */ R(x).";
        let ts = tokens(src);
        assert!(matches!(ts[0], Token::Ident(ref s) if s == "R"));
    }

    #[test]
    fn string_with_escape() {
        let src = r#"R("a\"b\n")."#;
        let ts = tokens(src);
        assert_eq!(ts[2], Token::String("a\"b\n".to_string()));
    }

    #[test]
    fn line_numbers_advance() {
        let src = "R(x).\nS(y).";
        let spans = tokenize(src).unwrap();
        let s_pos = spans
            .iter()
            .position(|t| matches!(&t.token, Token::Ident(s) if s == "S"))
            .unwrap();
        assert_eq!(spans[s_pos].line, 2);
    }

    #[test]
    fn negative_integer_literal() {
        let src = "R(-3).";
        let ts = tokens(src);
        assert!(ts.contains(&Token::Int(-3)));
    }
}
