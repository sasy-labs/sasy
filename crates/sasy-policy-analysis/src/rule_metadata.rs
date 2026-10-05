//! Rule metadata extraction from Soufflé `.dl` source files.
//!
//! Parses `.dl` files to extract annotations from comments
//! above rules. Annotations follow the format:
//!
//! ```datalog
//! // @deny_message: Access restricted to specific agent
//! // @suggestion: Use the appropriate agent for this API
//! Unauthorized(a) :- ...
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use tracing::debug;

/// Metadata associated with a policy rule.
#[derive(Debug, Clone, Default)]
pub struct RuleMetadata {
    /// Human-readable denial message.
    pub deny_message: Option<String>,
    /// Suggestions for how to fix the issue.
    pub suggestions: Vec<String>,
    /// Rule category/tag.
    pub category: Option<String>,
    /// Custom key-value annotations.
    pub custom: HashMap<String, String>,
    /// Source location (file:line).
    pub source_location: Option<String>,
    /// The relation name this rule produces
    /// (e.g., "Unauthorized", "IsAuthorized").
    pub relation_name: Option<String>,
}

/// Parser for extracting rule annotations from policy source.
pub struct RuleMetadataParser {
    line_metadata: HashMap<u32, RuleMetadata>,
    relation_metadata: HashMap<String, Vec<RuleMetadata>>,
    source_path: String,
}

impl RuleMetadataParser {
    /// Parse a policy source file and extract annotations.
    pub fn parse_file(path: &Path) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let source_path = path.display().to_string();
        Ok(Self::parse_content(&content, source_path))
    }

    /// Parse policy source content.
    pub fn parse_content(content: &str, source_path: String) -> Self {
        let mut line_metadata: HashMap<u32, RuleMetadata> = HashMap::new();
        let mut relation_metadata: HashMap<String, Vec<RuleMetadata>> = HashMap::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut current_annotations: Vec<(String, String)> = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = (line_idx + 1) as u32;
            let trimmed = line.trim();

            // Check for annotation comment
            if trimmed.starts_with("// @") {
                let annotation_part = trimmed.trim_start_matches("//").trim();
                if let Some(annotation) = Self::parse_annotation_line(annotation_part) {
                    current_annotations.push(annotation);
                }
                continue;
            }

            // Check for multi-line comment annotation
            if trimmed.starts_with("/* @") || trimmed.starts_with("* @") {
                let clean = trimmed
                    .trim_start_matches("/*")
                    .trim_start_matches('*')
                    .trim();
                if let Some(annotation) = Self::parse_annotation_line(clean) {
                    current_annotations.push(annotation);
                }
                continue;
            }

            // Check if this line starts a rule
            if !current_annotations.is_empty() && Self::is_rule_head(trimmed) {
                let relation_name = Self::extract_relation_name(trimmed);
                let source_loc = format!("{}:{}", source_path, line_num);
                let mut metadata = Self::build_metadata(&current_annotations);
                metadata.source_location = Some(source_loc);
                metadata.relation_name = relation_name.clone();

                debug!(
                    "Found rule '{}' at line {} with {} annotations",
                    relation_name.as_deref().unwrap_or("unknown"),
                    line_num,
                    current_annotations.len()
                );

                if let Some(rel_name) = &relation_name {
                    relation_metadata
                        .entry(rel_name.clone())
                        .or_default()
                        .push(metadata.clone());
                }

                line_metadata.insert(line_num, metadata);
                current_annotations.clear();
            }

            // Non-annotation, non-empty lines clear annotations
            if !trimmed.is_empty()
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("/*")
                && !trimmed.starts_with('*')
            {
                current_annotations.clear();
            }
        }

        Self {
            line_metadata,
            relation_metadata,
            source_path,
        }
    }

    /// Extract the relation name from a rule head.
    fn extract_relation_name(line: &str) -> Option<String> {
        let line = line.trim();
        if let Some(paren_pos) = line.find('(') {
            let name = line[..paren_pos].trim();
            if !name.is_empty()
                && name
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Parse a single annotation line.
    pub(crate) fn parse_annotation_line(line: &str) -> Option<(String, String)> {
        let line = line.trim();
        if !line.starts_with('@') {
            return None;
        }

        let after_at = &line[1..];
        if let Some(colon_pos) = after_at.find(':') {
            let key = after_at[..colon_pos].trim().to_lowercase();
            let value = after_at[colon_pos + 1..].trim().to_string();
            Some((key, value))
        } else {
            let key = after_at.trim().to_lowercase();
            Some((key, String::new()))
        }
    }

    /// Check if a line looks like a Soufflé rule head.
    pub(crate) fn is_rule_head(line: &str) -> bool {
        let line = line.trim();
        if line.starts_with("input") || line.starts_with("output") {
            return false;
        }
        if let Some(paren_pos) = line.find('(') {
            let name = &line[..paren_pos];
            if name
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                return line.contains(":-");
            }
        }
        false
    }

    /// Build RuleMetadata from collected annotations.
    fn build_metadata(annotations: &[(String, String)]) -> RuleMetadata {
        let mut metadata = RuleMetadata::default();

        for (key, value) in annotations {
            match key.as_str() {
                "deny_message" | "message" | "reason" => {
                    metadata.deny_message = Some(value.clone());
                }
                "suggestion" | "fix" | "hint" => {
                    metadata.suggestions.push(value.clone());
                }
                "category" | "tag" => {
                    metadata.category = Some(value.clone());
                }
                _ => {
                    metadata.custom.insert(key.clone(), value.clone());
                }
            }
        }

        metadata
    }

    /// Get metadata for a rule at a specific line number.
    pub fn get_metadata(&self, line_num: u32) -> Option<&RuleMetadata> {
        self.line_metadata.get(&line_num)
    }

    /// Get metadata for a rule within a line range.
    pub fn get_metadata_for_range(&self, start_line: u32, end_line: u32) -> Option<&RuleMetadata> {
        if let Some(metadata) = self.line_metadata.get(&start_line) {
            return Some(metadata);
        }
        for line in start_line..=end_line {
            if let Some(metadata) = self.line_metadata.get(&line) {
                return Some(metadata);
            }
        }
        None
    }

    /// Get all metadata for rules producing a relation.
    pub fn get_metadata_for_relation(&self, relation_name: &str) -> Option<&Vec<RuleMetadata>> {
        self.relation_metadata.get(relation_name)
    }

    /// Get the first metadata entry for a relation.
    pub fn get_first_metadata_for_relation(&self, relation_name: &str) -> Option<&RuleMetadata> {
        self.relation_metadata
            .get(relation_name)
            .and_then(|v| v.first())
    }

    /// Get the source file path.
    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    /// Get count of rules with metadata.
    pub fn rule_count(&self) -> usize {
        self.line_metadata.len()
    }

    /// Get count of relations with metadata.
    pub fn relation_count(&self) -> usize {
        self.relation_metadata.len()
    }
}

/// Container for loaded rule metadata with hot reload.
pub struct RuleMetadataStore {
    parser: Option<RuleMetadataParser>,
    policy_path: Option<String>,
}

impl RuleMetadataStore {
    pub fn new() -> Self {
        Self {
            parser: None,
            policy_path: None,
        }
    }

    /// Load metadata from a policy file.
    pub fn load(&mut self, path: &Path) -> Result<(), std::io::Error> {
        let parser = RuleMetadataParser::parse_file(path)?;
        debug!(
            "Loaded {} rule metadata entries for {} relations \
             from {}",
            parser.rule_count(),
            parser.relation_count(),
            path.display()
        );
        self.policy_path = Some(path.display().to_string());
        self.parser = Some(parser);
        Ok(())
    }

    /// Get metadata for a rule at a specific source range.
    pub fn get_metadata(&self, start_line: u32, end_line: u32) -> Option<&RuleMetadata> {
        self.parser
            .as_ref()
            .and_then(|p| p.get_metadata_for_range(start_line, end_line))
    }

    /// Get all metadata for rules producing a relation.
    pub fn get_metadata_for_relation(&self, relation_name: &str) -> Option<&Vec<RuleMetadata>> {
        self.parser
            .as_ref()
            .and_then(|p| p.get_metadata_for_relation(relation_name))
    }

    /// Get the first metadata entry for a relation.
    pub fn get_first_metadata_for_relation(&self, relation_name: &str) -> Option<&RuleMetadata> {
        self.parser
            .as_ref()
            .and_then(|p| p.get_first_metadata_for_relation(relation_name))
    }

    /// Check if metadata is loaded.
    pub fn is_loaded(&self) -> bool {
        self.parser.is_some()
    }
}

impl Default for RuleMetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annotation_line() {
        assert_eq!(
            RuleMetadataParser::parse_annotation_line("@deny_message: Access denied"),
            Some(("deny_message".into(), "Access denied".into()))
        );

        assert_eq!(
            RuleMetadataParser::parse_annotation_line("@suggestion: Use FDAHandler agent"),
            Some(("suggestion".into(), "Use FDAHandler agent".into()))
        );

        assert_eq!(
            RuleMetadataParser::parse_annotation_line("@deprecated"),
            Some(("deprecated".into(), String::new()))
        );

        assert_eq!(
            RuleMetadataParser::parse_annotation_line("not an annotation"),
            None
        );
    }

    #[test]
    fn test_is_rule_head() {
        assert!(RuleMetadataParser::is_rule_head(
            "Unauthorized(a) :- Actions(_, a)."
        ));
        assert!(RuleMetadataParser::is_rule_head("IsAuthorized(action) :-"));
        assert!(!RuleMetadataParser::is_rule_head(
            "input relation Edge(a, b)"
        ));
        assert!(!RuleMetadataParser::is_rule_head(
            "output relation Authorized(idx)"
        ));
        assert!(!RuleMetadataParser::is_rule_head("// A comment"));
    }

    #[test]
    fn test_parse_content() {
        let content = r#"
// Graph rules
Depends(a, b) :- Edge(a, b).

// @deny_message: FDA API access restricted
// @suggestion: Use FDAHandler agent for FDA API requests
// @category: api_restriction
Unauthorized(a) :- Actions(_, a), queries(a, "api.openfda.gov").

// @deny_message: OpenAI access allowed
IsAuthorized(a) :- Actions(_, a), queries(a, "openai.com").
"#;

        let parser = RuleMetadataParser::parse_content(content, "test.dl".into());

        let meta = parser.get_metadata(8);
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.deny_message, Some("FDA API access restricted".into()));
        assert_eq!(meta.suggestions.len(), 1);
        assert_eq!(
            meta.suggestions[0],
            "Use FDAHandler agent for FDA API requests"
        );
        assert_eq!(meta.category, Some("api_restriction".into()));

        let meta = parser.get_metadata(11);
        assert!(meta.is_some());

        let unauth = parser.get_metadata_for_relation("Unauthorized");
        assert!(unauth.is_some());
        assert_eq!(unauth.unwrap().len(), 1);

        let is_auth = parser.get_metadata_for_relation("IsAuthorized");
        assert!(is_auth.is_some());

        let missing = parser.get_metadata_for_relation("NonExistentRelation");
        assert!(missing.is_none());
    }

    #[test]
    fn test_extract_relation_name() {
        assert_eq!(
            RuleMetadataParser::extract_relation_name("Unauthorized(a) :- Actions(_, a)."),
            Some("Unauthorized".into())
        );
        assert_eq!(
            RuleMetadataParser::extract_relation_name("IsAuthorized(action) :-"),
            Some("IsAuthorized".into())
        );
        assert_eq!(
            RuleMetadataParser::extract_relation_name("function_call(x) :-"),
            None
        );
    }
}
