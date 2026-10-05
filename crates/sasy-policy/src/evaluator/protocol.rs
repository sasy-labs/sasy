//! Shared validation for requests sent to evaluator adapters.

use super::types::EvalRequest;
use super::EvaluatorError;

fn contains_embedded_nul(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(text) => text.contains('\0'),
        serde_json::Value::Array(items) => items.iter().any(contains_embedded_nul),
        serde_json::Value::Object(fields) => fields
            .iter()
            .any(|(name, value)| name.contains('\0') || contains_embedded_nul(value)),
        _ => false,
    }
}

pub(crate) fn validate_transportable_request(request: &EvalRequest) -> Result<(), EvaluatorError> {
    let value = serde_json::to_value(request).map_err(|error| {
        EvaluatorError::IpcError(format!("Serialize evaluator request: {error}"))
    })?;
    if contains_embedded_nul(&value) {
        return Err(EvaluatorError::IpcError(
            "evaluator request contains an embedded NUL character".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::types::PolicyMetadataFact;

    #[test]
    fn evaluator_request_rejects_embedded_nul_before_transport() {
        let request = EvalRequest::SetMetadata {
            facts: vec![PolicyMetadataFact {
                rel: "safe".into(),
                a: "contains\0nul".into(),
                b: "safe".into(),
            }],
        };

        let error = validate_transportable_request(&request).unwrap_err();
        assert!(error.to_string().contains("embedded NUL"));
    }

    #[test]
    fn evaluator_request_accepts_unicode_scalar_strings() {
        let request = EvalRequest::SetMetadata {
            facts: vec![PolicyMetadataFact {
                rel: "metadata".into(),
                a: "snow-雪".into(),
                b: "emoji-😀".into(),
            }],
        };

        validate_transportable_request(&request).unwrap();
    }
}
