use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

use sasy_policy_analysis::analysis::lexer::LexError;
use sasy_policy_analysis::analysis::{
    parse_with_options, BridgeParseError, ParseError, ParseOptions,
};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorSpan {
    start_line: u32,
    end_line: u32,
}

#[derive(Serialize)]
struct ErrorOutput<'a> {
    error: String,
    file: &'a str,
    line: u32,
    span: ErrorSpan,
}

fn parse_error_line(error: &BridgeParseError) -> u32 {
    match error {
        BridgeParseError::Parse(ParseError::Lex(error)) => match error {
            LexError::UnterminatedString { line }
            | LexError::UnterminatedBlockComment { line }
            | LexError::UnexpectedChar { line, .. }
            | LexError::InvalidInt { line, .. } => *line,
        },
        BridgeParseError::Parse(ParseError::Unexpected { line, .. }) => *line,
        BridgeParseError::Eof { line, .. }
        | BridgeParseError::DuplicateType { line, .. }
        | BridgeParseError::DuplicateRelation { line, .. } => *line,
        BridgeParseError::Parse(ParseError::Eof { .. })
        | BridgeParseError::Parse(ParseError::DuplicateType { .. }) => 1,
    }
}

fn dump_program(source: &str, file: &str) -> Result<String, BridgeParseError> {
    let program = parse_with_options(
        source,
        file,
        ParseOptions {
            expand_body_disjunctions: false,
            reject_duplicate_relations: true,
            full_rule_spans: true,
            preserve_aggregates: true,
        },
    )?;
    Ok(serde_json::to_string_pretty(&program)
        .expect("serializing the Program AST to JSON cannot fail"))
}

fn error_json(error: impl ToString, file: &str, line: u32) -> String {
    let output = ErrorOutput {
        error: error.to_string(),
        file,
        line,
        span: ErrorSpan {
            start_line: line,
            end_line: line,
        },
    };
    serde_json::to_string_pretty(&output)
        .expect("serializing the fixed error object to JSON cannot fail")
}

fn print_error(error: impl ToString, file: &str, line: u32) {
    println!("{}", error_json(error, file, line));
}

fn path_argument() -> Result<OsString, &'static str> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("missing .dl file path argument")?;
    if args.next().is_some() {
        return Err("expected exactly one .dl file path argument");
    }
    Ok(path)
}

fn main() -> ExitCode {
    let path = match path_argument() {
        Ok(path) => path,
        Err(error) => {
            print_error(error, "", 1);
            return ExitCode::FAILURE;
        }
    };
    let file = Path::new(&path).to_string_lossy();
    let source = match std::fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            print_error(error, &file, 1);
            return ExitCode::FAILURE;
        }
    };
    match dump_program(&source, &file) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            let line = parse_error_line(&error);
            print_error(error, &file, line);
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_is_json_and_contains_example_01_block_rule() {
        let source = r#".decl request(path:symbol)
.decl block(path:symbol)
block(p) :- request(p), p = "/admin".
"#;

        let output = dump_program(source, "01-admin-path/policy.dl").unwrap();
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();
        let has_block_rule = json["rules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|rule| rule["head"]["relation"] == "block");

        assert!(has_block_rule, "expected a block rule in {output}");
    }

    #[test]
    fn dump_preserves_full_multiline_rule_span() {
        let output = dump_program(
            "block(R) :-\n    http_host(R, \"api.example.com\"),\n    uri_path(R, \"/admin\").\n",
            "policy.dl",
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(json["rules"][0]["span"]["start_line"], 1);
        assert_eq!(json["rules"][0]["span"]["end_line"], 3);
    }

    #[test]
    fn duplicate_type_error_uses_duplicate_declaration_line() {
        let error = parse_with_options(
            ".type Request <: symbol\n.type Request <: symbol\n",
            "policy.dl",
            ParseOptions::default(),
        )
        .unwrap_err();
        let line = parse_error_line(&error);
        let json: serde_json::Value =
            serde_json::from_str(&error_json(&error, "policy.dl", line)).unwrap();

        assert_eq!(json["line"], 2);
        assert_eq!(json["span"]["start_line"], 2);
        assert_eq!(json["span"]["end_line"], 2);
    }

    #[test]
    fn eof_error_after_trailing_newline_uses_next_line() {
        let error = parse_with_options(
            "block(R) :- (http_host(R, \"x\")\n",
            "policy.dl",
            ParseOptions::default(),
        )
        .unwrap_err();
        let line = parse_error_line(&error);
        let json: serde_json::Value =
            serde_json::from_str(&error_json(&error, "policy.dl", line)).unwrap();

        assert_eq!(json["line"], 2);
        assert_eq!(json["span"]["start_line"], 2);
        assert_eq!(json["span"]["end_line"], 2);
    }

    #[test]
    fn body_disjunction_is_dumped_explicitly_without_rule_duplication() {
        let source = concat!(
            ".decl block(r: symbol)\n",
            ".decl http_host(r: symbol, value: symbol)\n",
            ".decl uri_path(r: symbol, value: symbol)\n",
            "block(R) :- (http_host(R, \"api.example.com\"); ",
            "uri_path(R, \"/admin\")).\n",
        );

        let output = dump_program(source, "policy.dl").unwrap();
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();
        let rules = json["rules"].as_array().unwrap();

        assert_eq!(rules.len(), 1);
        assert!(rules[0]["body"][0].get("Disjunction").is_some());
    }

    #[test]
    fn negated_grouped_disjunction_is_dumped_with_spans() {
        let source = "block(R) :- !(helper_a(R); helper_b(R)).\n";
        let output = dump_program(source, "policy.dl").unwrap();
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();
        let negation = &json["rules"][0]["body"][0]["Negation"];
        let disjunction = &negation["literal"]["Disjunction"];

        assert_eq!(negation["span"]["start_line"], 1);
        assert_eq!(negation["span"]["end_line"], 1);
        assert_eq!(disjunction["span"]["start_line"], 1);
        assert_eq!(disjunction["span"]["end_line"], 1);
        assert_eq!(disjunction["alternatives"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn duplicate_relation_is_rejected_before_ast_dump() {
        let source = concat!(
            ".decl http_host(r: symbol, value: symbol)\n",
            ".decl http_host(r: symbol, value: symbol)\n",
        );

        let error = dump_program(source, "policy.dl").unwrap_err();
        let line = parse_error_line(&error);
        let output = error_json(&error, "policy.dl", line);
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert!(matches!(
            error,
            BridgeParseError::DuplicateRelation {
                name,
                line: 2
            } if name == "http_host"
        ));
        assert_eq!(json["line"], 2);
        assert_eq!(
            json["error"],
            "duplicate relation declaration at line 2: http_host"
        );
    }

    #[test]
    fn dump_is_byte_stable_across_repeated_parses() {
        let source = concat!(
            ".type Request <: symbol\n",
            ".decl z_field(r: Request, value: symbol)\n",
            ".decl a_field(r: Request, value: symbol)\n",
            ".decl block(r: Request)\n",
            "block(R) :- z_field(R, \"z\"), a_field(R, \"a\").\n",
        );
        let outputs: Vec<String> = (0..5)
            .map(|_| dump_program(source, "policy.dl").unwrap())
            .collect();

        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(outputs[0].find("\"a_field\"").unwrap() < outputs[0].find("\"z_field\"").unwrap());
    }

    #[test]
    fn first_duplicate_declaration_diagnostic_is_deterministic() {
        let source = concat!(
            ".decl Alpha(r: symbol)\n",
            ".decl Beta(r: symbol)\n",
            ".decl Beta(r: symbol)\n",
            ".decl Alpha(r: symbol)\n",
        );
        let outputs: Vec<String> = (0..5)
            .map(|_| {
                let error = dump_program(source, "policy.dl").unwrap_err();
                let line = parse_error_line(&error);
                error_json(error, "policy.dl", line)
            })
            .collect();

        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
        let json: serde_json::Value = serde_json::from_str(&outputs[0]).unwrap();
        assert_eq!(json["line"], 3);
        assert_eq!(
            json["error"],
            "duplicate relation declaration at line 3: Beta"
        );
    }

    #[test]
    fn aggregate_survives_as_explicit_bridge_node() {
        let source = concat!(
            ".functor __sasy_bridge_preserved_aggregate(value: symbol): symbol\n",
            "block(R) :-\n",
            "    Existing = @__sasy_bridge_preserved_aggregate(\"kept\"),\n",
            "    Count = count : { source_ip(R, _) },\n",
            "    Count > 0.\n",
        );
        let output = dump_program(source, "policy.dl").unwrap();
        let json: serde_json::Value = serde_json::from_str(&output).unwrap();
        let existing = &json["rules"][0]["body"][0]["Compare"]["right"]["Functor"];
        let aggregate = &json["rules"][0]["body"][1]["Compare"]["right"]["Aggregate"];

        assert_eq!(existing["name"], "__sasy_bridge_preserved_aggregate");
        assert_eq!(aggregate["raw"], "count : { source_ip(R, _) }");
        assert_eq!(aggregate["span"]["start_line"], 4);
        assert_eq!(aggregate["span"]["end_line"], 4);
    }

    #[test]
    fn arithmetic_survives_as_explicit_ast_node() {
        for expression in ["Path + 1", "Path-1", "Path - 1"] {
            let source = format!("block(R) :- uri_path(R, Path), {expression} = 2.\n");
            let output = dump_program(&source, "policy.dl").unwrap();
            let json: serde_json::Value = serde_json::from_str(&output).unwrap();
            let arithmetic = &json["rules"][0]["body"][1]["Compare"]["left"]["Arith"];

            let expected_op = if expression.contains('+') {
                "Add"
            } else {
                "Sub"
            };
            assert_eq!(arithmetic["op"], expected_op, "{expression}");
            assert_eq!(arithmetic["left"]["Var"], "Path", "{expression}");
            assert_eq!(arithmetic["right"]["NumberLit"], 1, "{expression}");
        }
    }
}
