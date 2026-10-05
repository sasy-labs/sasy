//! Small SHA-256 / hex helpers shared across the crate's content-addressing
//! paths (souffle binary cache key, upload dedup hash, replay workdir id).

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// Lowercase hex-encode a byte slice (two zero-padded chars per byte).
pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// SHA-256 of a string, lowercase-hex encoded.
pub(crate) fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex(&h.finalize())
}

/// The content hash an upload is stored and deduped under.
///
/// The digest is over an unambiguous encoding: the domain tag
/// `sasy-policy-upload-v2`, then each field as its byte length in a
/// fixed-width little-endian `u64` followed by the field's bytes, in the
/// order (backend, magic_set, policy_source, functor_source). Because every
/// field carries its own length, no two different field tuples can produce
/// the same byte string, so no two different uploads can share a hash.
///
/// Without the lengths, any two field tuples whose concatenation is cut in a
/// different place would collide: `(P + "\n" + F, "")` and `(P, F + "\n")`
/// join to the same bytes, so one caller's record would answer for another's
/// content.
pub fn upload_content_hash(
    backend: &str,
    magic_set: &str,
    policy_source: &str,
    functor_source: &str,
) -> String {
    let mut h = Sha256::new();
    h.update(b"sasy-policy-upload-v2");
    for field in [backend, magic_set, policy_source, functor_source] {
        let bytes = field.as_bytes();
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    hex(&h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the v2 formula to an exact digest, so any change to the domain
    /// tag, the field order, or the length framing has to be deliberate.
    ///
    /// The input is `backend="souffle"`, `magic_set=""`, `policy_source="P"`,
    /// `functor_source="F"`; the pre-image is the tag followed by
    /// `07 00 00 00 00 00 00 00 "souffle"`, `00 * 8`,
    /// `01 00 00 00 00 00 00 00 "P"`, `01 00 00 00 00 00 00 00 "F"`.
    #[test]
    fn upload_content_hash_pins_the_v2_formula() {
        assert_eq!(
            upload_content_hash("souffle", "", "P", "F"),
            "44bc93c76fd2c0b562650c08f2fc9af9ad6d06762f5a8601978adf5035282d06"
        );
    }

    /// An upload whose functor source ends in a newline, against an upload
    /// that moved that newline into the policy source and left the functor
    /// source empty. Joined without lengths both would hash the same byte
    /// string; the per-field lengths keep them apart.
    #[test]
    fn upload_content_hash_separates_shifted_field_boundaries() {
        let policy = "P";
        let functor = "F";
        let admin = upload_content_hash("souffle", "", policy, &format!("{functor}\n"));
        let non_admin = upload_content_hash("souffle", "", &format!("{policy}\n{functor}"), "");
        assert_ne!(
            admin, non_admin,
            "two different uploads must not share a content hash"
        );
    }

    /// The same fields in the same order still hash the same, so dedup works.
    #[test]
    fn upload_content_hash_is_stable_for_identical_input() {
        assert_eq!(
            upload_content_hash("souffle", "m", "policy", "functor"),
            upload_content_hash("souffle", "m", "policy", "functor")
        );
    }
}
