//! Secret scrubbing: a string goes in, and the same string comes back with
//! anything that looks like a credential replaced by `[redacted]`, along with
//! a count of what was replaced.
//!
//! Where it runs in this release: `sasy-policy` scrubs the prompt and the
//! context of an `LlmQuery` frame before the log preview, the cache key, or
//! the model provider sees them. That is the LLM oracle — the one place the
//! engine sends recorded content off the host — and it is this crate's only
//! caller outside its own tests.
//!
//! **Nothing scrubs at ingest here.** The graph store keeps message content in
//! the clear — RocksDB on disk, and the optional Neo4j mirror — and whatever
//! an SDK sends is what gets persisted, so a credential that made it into a
//! recorded string is a credential on disk. The shape to watch for is an
//! exception text such as `... for url 'https://host/path?key=<the key>'`
//! recorded as message content. Two things would close that and neither is
//! part of this crate: SDKs redacting at capture, and a server scrubbing every
//! string it is about to store.
//!
//! Three kinds of thing are removed:
//!
//! 1. **URL query parameters** with a credential-bearing name (`key`,
//!    `access_token`, `password`, …) lose their value to `[redacted]`. The
//!    name, the other parameters, the path and the host stay, so the graph
//!    still shows which endpoint was called. Found as whole fields, as
//!    `https://…` runs inside free text, and as bare `name=…` runs with no
//!    scheme in front of them. The bare reading is split in two: an
//!    unambiguous name (`password`, `client_secret`, …) is a credential
//!    after any delimiter, while an ambiguous one (`key`, `token`, `code`,
//!    …) is only read as a parameter inside a query string, so
//!    `session=requests.Session()` in a recorded file stays as written.
//!    Outside a query string an unambiguous name is read in every form it is
//!    written in — `name=value`, `"name": value`, YAML `name: value`, TOML
//!    `name = value`, an environment-variable name that ends in a credential
//!    suffix (`DB_PASSWORD=…`, `POSTGRES_PASSWORD: …`,
//!    `aws_secret_access_key = …`), and the `-u user:pass`
//!    flag — and the value is taken only when it is a credential literal
//!    (see [`read_value`] and [`is_credential_literal`]), so ordinary source
//!    keeps its text and its tail. A password in a `scheme://user:pass@host`
//!    authority goes too, the user kept.
//! 2. **Header values** for the authentication headers (`authorization`,
//!    `x-api-key`, `cookie`, …), in `Name: value` lines inside text, in
//!    quoted `"name": "value"` pairs inside a JSON blob, and wherever the
//!    name stands inside a longer string — a `curl -H "authorization: …"`
//!    command, a `> authorization: …` trace line, a log line with a
//!    timestamp in front of it. What counts as the value depends on the
//!    header: a scheme and a token for `authorization`, a string carrying an
//!    `=` for a cookie, a single token for the rest, and a placeholder for
//!    none of them — so prose that mentions a cookie is left as prose. When a
//!    quote stands where the value does — `authorization: "Basic …"` in a
//!    YAML config or a JavaScript object literal — the value is read by the
//!    literal grammar instead: its content goes and its quotes stay.
//! 3. **Token shapes** — OpenAI, Google, AWS, GitHub, GitLab, Slack, npm,
//!    PyPI, JWTs, bearer tokens, PEM private-key blocks — replaced whole by
//!    `[redacted:<shape>]`.
//!
//! Scrubbing is idempotent, it reports how many replacements it made, and it
//! never logs a value. It is best effort: it recognises shapes, so a secret in
//! a shape this module does not know passes through untouched.
//!
//! The rules themselves live in data. `tests/fixtures/redaction/rules.json`
//! holds the names and the shapes, and this crate has a test comparing its own
//! constants to that file; `tests/fixtures/redaction/corpus.json` holds the strings that pin
//! what the rules do to them, and the tests run the whole corpus. Some shape
//! patterns here are written differently from the reference patterns the rule
//! file records: the PEM body, because this engine does not backtrack and so
//! needs no counted bound, and every shape whose reference pattern carries a
//! lookahead, because this engine has none — those lookaheads are applied to
//! each match in code instead, see [`SHAPE_CHECKS`]. The rule file records both
//! deviations, in a shape's `linear_engine_pattern` and `checks` fields.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::{Captures, Regex, RegexSet};
use sasy_common::observability::{Computation, Edge, Event, Tool};

/// What a redacted value is replaced by.
pub const REDACTED: &str = "[redacted]";

/// Every character Unicode calls whitespace, as a regex character-class body.
///
/// A macro rather than a constant so it can be pasted into `concat!` where a
/// pattern is built at compile time. The rule file carries the same list:
/// see [`SPACE_CHARS`] for why `\s` is not used.
macro_rules! space_chars {
    () => {
        r"\t\n\u000b\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
    };
}

/// The space class the shared rule set pins, as a string.
///
/// `\s` means something different in each of the three regex engines — this
/// one reads it as Unicode whitespace, Python narrows it to the five ASCII
/// spaces under `re.ASCII`, JavaScript counts U+FEFF as one — so wherever a
/// URL run, a URL userinfo field or a parameter value has to stop at a space,
/// the class is written out character for character instead and no engine's
/// own reading decides. U+FEFF is deliberately absent: it is not Unicode
/// whitespace.
///
/// The patterns paste the macro rather than this constant, because they are
/// built at compile time by `concat!`; the constant is what the comparison
/// test holds against the shared rule set.
#[cfg_attr(not(test), allow(dead_code))]
const SPACE_CHARS: &str = space_chars!();

/// Parameter names that mean a credential wherever they stand. Nothing else
/// is called `client_secret` or `refresh_token`, so a run under one of these
/// names is scrubbed after any delimiter, in a query string or not:
/// `--password=…` on a command line is as much a credential as `?password=…`
/// in a URL.
const UNAMBIGUOUS_PARAMS: &[&str] = &[
    "access_token",
    "api-key",
    "api_key",
    "apikey",
    "auth_token",
    "client_secret",
    "id_token",
    "passwd",
    "password",
    "pwd",
    "refresh_token",
    "secret",
    "sessionid",
    "signature",
    "x-api-key",
];

/// Parameter names that mean a credential *in a query string* and ordinary
/// words anywhere else. `session=requests.Session()`, `if code==200` and a
/// `key=value` line in a config file are all normal content, and a rule that
/// rewrote them would corrupt the record it was protecting — a Bash command
/// whose URL is cut off is a policy fail-open, not a redaction. So these
/// names are only read as parameters after a query-string delimiter.
const QUERY_STRING_ONLY_PARAMS: &[&str] = &["auth", "code", "key", "session", "sig", "token"];

/// Header names whose value is a credential. Compared case-insensitively.
const SENSITIVE_HEADERS: &[&str] = &[
    "api-key",
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "x-goog-api-key",
    "x-subscription-token",
];

/// Token shapes, in the order they are applied. The order is part of the rule
/// set: `jwt` before `bearer` so `Bearer <jwt>` keeps the word `Bearer`, and
/// the PEM block first so its base64 body is never picked apart.
///
/// The PEM body is written differently here than the reference pattern in the
/// rule file, in two ways.
///
/// It is tempered: the body may hold only base64 and the punctuation a PEM
/// header line uses, and a `-` in it may not open a second `-----` run.
/// Without that an unclosed `-----BEGIN … PRIVATE KEY-----` would swallow
/// every character up to the next block's `-----END`, ordinary prose
/// included, and replace the lot with one marker. The reference pattern writes
/// the second half of that as a lookahead; this engine has none, so it says
/// the same thing as "one to four `-` followed by a body character".
///
/// The body also admits the two-character escapes `\n`, `\r` and `\t`, in all
/// three. That is what a key written with the Write tool looks like by the
/// time it is recorded: tool arguments arrive JSON-encoded, so every line
/// break in the body is a backslash and an `n`, not a newline. Without them
/// the body match died at the first line break, no other rule reached a bare
/// base64 body, and the whole private key was stored.
///
/// The other difference is the bound. The SDKs bound the body at 8000
/// characters because their engines
/// backtrack — an unbounded run restarts a full forward scan at every
/// candidate start. This engine does not backtrack, and a counted repetition
/// is not free to it: it expands into that many states, which on hostile
/// input costs three orders of magnitude more time than the unbounded body,
/// and `redact_events` runs on a request-handling thread. So the body is
/// unbounded here. The practical difference is at one edge: a private-key
/// block whose body runs past 8000 characters is redacted here and not by a
/// bounded pattern.
///
/// The JWT segments are unbounded too, and a bound would buy nothing: a JWT
/// segment class cannot be followed by the `.` that has to come next, so the
/// run never backtracks.
const TOKEN_SHAPES: &[(&str, &str)] = &[
    (
        "pem-private-key",
        concat!(
            r"-----BEGIN[^-]*PRIVATE KEY-----",
            r"(?:[A-Za-z0-9+/=:, \t\n\r\f\x0b]|\\?\\[nrt]|-{1,4}[A-Za-z0-9+/=:, \t\n\r\f\x0b])*?",
            r"-----END[^-]*PRIVATE KEY-----",
        ),
    ),
    (
        "jwt",
        r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    ),
    ("bearer", r"Bearer [A-Za-z0-9._~+/=-]{20,}"),
    ("openai-key", r"sk-[A-Za-z0-9_-]{16,}"),
    ("google-api-key", r"AIza[0-9A-Za-z_-]{30,}"),
    ("aws-access-key-id", r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    ("github-pat", r"github_pat_[A-Za-z0-9_]{60,}"),
    ("github-token", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    ("gitlab-token", r"glpat-[A-Za-z0-9_-]{20}"),
    ("slack-token", r"xox[abpr]-[A-Za-z0-9-]{10,}"),
    ("npm-token", r"npm_[A-Za-z0-9]{36}"),
    ("pypi-token", r"pypi-AgEI[A-Za-z0-9_-]{50,}"),
];

/// Trailing punctuation a URL run swallows but does not own: sentence
/// punctuation, and the bracket that closes a markdown link, a parenthesis or
/// a JSON array. Trimming the closing bracket is what keeps
/// `[docs](https://h/v1?api_key=…)` a markdown link after scrubbing.
/// Idempotence does not depend on the trim: a field whose value already
/// begins `[redacted` is left alone, so a second pass over `?key=[redacted]`
/// cannot produce `[redacted]]`.
const URL_TRAILING_PUNCTUATION: &[char] = &['.', ',', ';', ':', '!', '?', ')', ']', '}', '>'];

/// Both tiers, compared ASCII-case-insensitively after percent-decoding. The
/// structural URL pass — a run with a scheme, or a `host/path?…` form — is
/// already inside a query string, so it judges every one of these names by
/// name alone.
static QUERY_PARAM_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    UNAMBIGUOUS_PARAMS
        .iter()
        .chain(QUERY_STRING_ONLY_PARAMS.iter())
        .copied()
        .collect()
});

/// One quoted string, in either quote character. A backslash escape does not
/// end it: a JSON blob holds a value containing a quote as `\"`, and a cookie
/// is allowed to be quoted (RFC 6265), so `"session=\"abc\""` is one value
/// and not three.
const QUOTED_VALUE: &str = r#"(?:"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*')"#;

/// A credential header written as a quoted `"name": value` pair.
///
/// This is the shape a header takes once it has been serialised into a blob
/// rather than a line: an OpenTelemetry `http.request.header.authorization`
/// attribute inside `attributes_json`, or a headers dictionary printed into a
/// tool result. [`HEADER_NAME_RE`] reads the value where it stands and cannot
/// see an array, which is what the serialised form of a header is.
/// The optional dotted prefix is what lets `"http.request.header.cookie"`
/// answer to `cookie`. Either quote character is accepted on each side, so a
/// Python `repr` of a dictionary is read as well as JSON.
///
/// The value is a quoted string *or* an array of them: OpenTelemetry defines
/// `http.request.header.<key>` as a string array, so the serialisation that
/// actually arrives is `{"http.request.header.authorization": ["<the
/// credential>"]}`. An array is condemned whole, however many elements it
/// has. Brackets are excluded from the array's filler class, so an array that
/// is never closed fails fast instead of scanning to the end of the text from
/// every candidate start.
static HEADER_PAIR_RE: LazyLock<Regex> = LazyLock::new(|| {
    let array = format!(r#"\[(?:[^\[\]"']|{QUOTED_VALUE})*\]"#);
    Regex::new(&format!(
        r#"(["'](?:[A-Za-z0-9_.-]*\.)?(?:{})["'][ \t]*:[ \t]*)({array}|{QUOTED_VALUE})"#,
        ascii_fold(SENSITIVE_HEADERS)
    ))
    .expect("header-pair pattern is valid")
});

/// What may stand to the left of a header name that is not at the start of a
/// line: the start of the string, or a character a header name cannot contain
/// — a quote, whitespace, an `=`, a `(` or a `,`, and the `[`, `{` and `|`
/// that open a bracketed log line, a JavaScript object literal and a YAML
/// block scalar. Written as one fragment so every implementation compiles
/// the same thing. The two-character escapes `\n`, `\r` and `\t` stand
/// alongside those characters: a raw header block inside a JSON-encoded file
/// body or tool result arrives with `\` and `n` between its lines rather than
/// a newline, and without the escape alternative no header after the first
/// line of such a block was read at all.
const HEADER_BOUNDARY: &str = r#"(?:^|\\[nrt]|["'\s=(,\[{|])"#;

/// What ends a header value read where it stands: the end of the line, in
/// either line ending, the quote that closes the string the header sits
/// inside, and the backslash that escapes that quote in JSON-encoded text.
/// Stopping at the backslash is what keeps
/// `{"command": "curl -H \"authorization: x\" …"}` valid JSON: the escape and
/// the quote it protects are left where they are. The `\r` is what keeps a
/// CRLF header block's line endings: a value that ran to the `\n` deleted the
/// carriage return with the credential. The `]` and `}` are what close a
/// bracketed log line and an object literal, so `[authorization: Bearer …]`
/// keeps its closing bracket.
const HEADER_VALUE_TERMINATORS: &str = "\"'`\\\r\n]}";

/// What stands between a header name and its colon: the same optional
/// `\?["']?` the [`NAME_VALUE_SEPARATOR`] carries, so a header name is read
/// where a JSON blob has been encoded into a string a second time and the
/// name arrives as `\"authorization\"`. Without it the seven header-only
/// names (authorization, proxy-authorization, cookie, set-cookie,
/// x-auth-token, x-goog-api-key, x-subscription-token) would be unreachable
/// in double-encoded text — a request-headers JSON file, or a tool result
/// quoting one back — and the credential would be stored whole. Written as
/// one fragment so every implementation compiles the same thing.
const HEADER_NAME_SEPARATOR: &str = r#"\\?["']?[ \t]*:"#;

/// A credential header, in the two readings that are not a serialised map: a
/// line of its own (`Authorization: …`), and the name standing inside a
/// longer string — a recorded `curl -H "authorization: …"`, a verbose curl
/// trace (`> Authorization: …`), a log line with a timestamp in front of it.
///
/// [`HEADER_PAIR_RE`] wants the name in quotes, so without this reading the
/// single most common way a credential reaches this redactor from a coding
/// agent — a recorded shell command — would be missed. A line of its own is
/// the same shape: [`HEADER_BOUNDARY`] admits the line break before it, both
/// the newline character and the two-character `\n` a JSON-encoded body
/// carries in its place.
///
/// One pattern finds the name; the value is read and judged in code, because
/// what a header value may be depends on the header.
static HEADER_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"{}({}){HEADER_NAME_SEPARATOR}",
        ascii_spaces(HEADER_BOUNDARY),
        ascii_fold(SENSITIVE_HEADERS)
    ))
    .expect("header-name pattern is valid")
});

/// Headers whose value is `<scheme> <token>` or a single token.
const SCHEME_HEADERS: &[&str] = &["authorization", "proxy-authorization"];

/// Headers whose value is a cookie string: it runs to the end of the line,
/// and it has to contain an `=` to be one at all. That is what leaves
/// "We store a cookie: the session identifier, nothing else." alone.
const COOKIE_HEADERS: &[&str] = &["cookie", "set-cookie"];

/// Whether a lone value after a header name is a word and not a secret.
///
/// One shape: made solely of letters, digits, `-`, `/`, `.` and `_`, and
/// carrying no digit at all. `application/json`, `not-set`, `role-based`,
/// `user-provided` and `N/A` are what a config template, a README table and a
/// filled-in manifest write after one of the nine header names, and the
/// literal grammar took every one of them, because a `/` and a `-` are both
/// outside its plain-word class. `abc123def456`,
/// `sk_live_ex4mple0key0for0tests00` and `12345` carry digits and are still
/// credentials.
///
/// The cost is a limit: a real key of letters alone standing after a header
/// name is kept. The two-token `<scheme> <token>` shape does not come through
/// here at all, so `Basic dXNlcjpwYXNz` still goes.
pub fn is_header_word(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_alphabetic() || matches!(c, '.' | '_' | '/' | '-'))
}

/// Whether a header value's token is a credential.
///
/// It has to be made only of the characters an unquoted literal may hold —
/// `role,` and `str,` are words of a sentence, not tokens of a header — it
/// must not be a word by [`is_header_word`], and it has to be a credential by
/// the literal grammar. Every reading that takes a single token applies this,
/// so `Authorization: required`, `x-api-key: application/json` and
/// `def check(authorization: str):` are left as written.
fn is_credential_token(token: &str) -> bool {
    if token.is_empty() || !token.chars().all(is_literal_value_char) {
        return false;
    }
    if is_header_word(token) {
        return false;
    }
    is_credential_literal(token)
}

/// How much of `value` is the credential. `0` means none of it.
///
/// `value` is what stands after the colon, already cut at the end of the line
/// or at the quote that closes it. What counts as a credential depends on the
/// header: an `Authorization` value is a scheme and a token or a bare token,
/// a cookie is a string containing an `=`, and everything else is a single
/// token. Whatever follows the credential on the same line — the next field
/// of a log line — is kept.
fn header_credential_span(name: &str, value: &str) -> usize {
    let stripped = value.trim_end();
    if stripped.is_empty() || stripped.starts_with("[redacted") {
        return 0;
    }
    if COOKIE_HEADERS.contains(&name) {
        return if stripped.contains('=') {
            stripped.len()
        } else {
            0
        };
    }
    let tokens: Vec<&str> = stripped.split_whitespace().collect();
    let (credential, end) = if SCHEME_HEADERS.contains(&name) {
        // There are two shapes here and no third: `<scheme> <token>` and a
        // bare token. The scheme word is ASCII letters, so
        // `Authorization: École abc123def` is not a scheme-and-token value.
        //
        // A value with a third token past the shape is a sentence unless the
        // candidate is a credential by the literal grammar. That is what
        // keeps the log line (`authorization: Bearer eyJ… host:
        // api.example.com`, where the token holds digits, and the `host:`
        // field has to survive) while leaving `There are three kinds of
        // authorization: role, attribute and policy.`, `def
        // check(authorization: str, cookie: str) -> bool:` and
        // `Authorization: Bearer token required for all endpoints.`
        // untouched. Inside the two-token shape — and only there — the value
        // is a credential whatever it looks like, because `Basic
        // dXNlcjpwYXNz` is base64 that may hold no digit at all: the literal
        // grammar's short-all-letters limit is about an ambiguous name, and a
        // header name is not ambiguous. A lone word has no scheme in front of
        // it to say so, and is tested like any other value.
        let (credential, end, shape) =
            if tokens.len() >= 2 && tokens[0].chars().all(|c| c.is_ascii_alphabetic()) {
                let start = stripped[tokens[0].len()..]
                    .find(tokens[1])
                    .expect("the second token stands after the first")
                    + tokens[0].len();
                (tokens[1], start + tokens[1].len(), 2)
            } else {
                (tokens[0], tokens[0].len(), 1)
            };
        // The single-token reading takes whatever word stands after the
        // colon, so it is the one that has to ask whether that word is a
        // credential at all: without the test `def check(authorization:
        // str):` loses its `):` and `{ authorization: authHeader }` loses
        // its reference. Only the two-token shape is exempt, because
        // `Basic dXNlcjpwYXNz` is base64 that may hold no digit at all.
        if (shape == 1 || tokens.len() > shape) && !is_credential_token(credential) {
            return 0;
        }
        (credential, end)
    } else {
        if tokens.len() != 1 {
            return 0;
        }
        if !is_credential_token(tokens[0]) {
            return 0;
        }
        (tokens[0], tokens[0].len())
    };
    if PLACEHOLDER_VALUES.contains(&credential)
        || PLACEHOLDER_PREFIXES
            .iter()
            .any(|prefix| credential.starts_with(prefix))
        || credential.starts_with("[redacted")
    {
        return 0;
    }
    end
}

/// An `https?://…` run inside free text. Whitespace, quotes and angle brackets
/// end the run — which is exactly how the leaked httpx message
/// (`… for url 'https://…?key=…'`) delimits its URL. Case-insensitive: an
/// uppercase `HTTPS://` is the same URL. The backslash ends the
/// run too: in JSON-encoded text the URL is followed by `\"`, and a run that
/// swallowed the escape left the blob unparseable and the rest of the command
/// — `-d @.env` — cut off.
///
/// The space class is written out character for character rather than as
/// `\s`, because the three engines read `\s` differently: this one reads it
/// as Unicode whitespace, Python narrows it to the five ASCII spaces under
/// `re.ASCII`, and JavaScript counts U+FEFF as one. A URL followed by a
/// no-break space — what a browser, a PDF or French typography emits — has
/// to end at that space everywhere, so no engine's own `\s` decides.
const URL_IN_TEXT: &str = concat!(r#"https?://[^"#, space_chars!(), r#""'<>`\\]+"#);

static URL_IN_TEXT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&URL_IN_TEXT.replacen("https?", &ascii_fold(&["https?"]), 1))
        .expect("url pattern is valid")
});

/// A `scheme://user:password@host` authority, in any scheme.
///
/// The scheme is bounded and preceded by a character it cannot contain, so a
/// megabyte of letters with no `://` in it is not rescanned from every
/// position: a git remote in a `fatal: unable to access …` line, a
/// `postgresql://` connection string, an `amqp://` broker URL. The user is
/// kept — it says which account the call was made as — and the password goes.
///
/// The user part is `{0,256}`, not `{1,256}`: `redis://:s3cr3tredis@cache`
/// and `amqp://:pw@broker` are the ordinary spelling for a server that takes
/// a password and no account name, and a user part that had to be non-empty
/// stored those passwords whole.
///
/// The two host-and-user runs stop at every Unicode space, spelled out the
/// same way [`URL_IN_TEXT`] spells it.
const URL_USERINFO: &str = concat!(
    r#"(^|[^A-Za-z0-9+.:-])([A-Za-z][A-Za-z0-9+.-]{0,31}://)([^"#,
    space_chars!(),
    r#"/:@"'`\\]{0,256}):([^"#,
    space_chars!(),
    r#"/@"'`\\]{1,256})@"#
);

static URL_USERINFO_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(URL_USERINFO).expect("userinfo pattern is valid"));

/// What may stand to the left of an unambiguous parameter name: the start of
/// the string, or a character that cannot be part of a name. A one- or
/// two-hyphen flag cluster may stand in between, which is what makes
/// `--password=…` and `-Dpassword=…` — a long CLI flag and a JVM property,
/// two of the likeliest places a password reaches a recorded command — read
/// as parameters. `_` and `.` are part of a name, so `OPENAI_API_KEY=…` is
/// not read as an `api_key` parameter; its value is caught by the token
/// shapes when it is a real key.
///
/// The two-character escapes `\n`, `\r` and `\t` are a delimiter of their
/// own, alongside the real characters. A file body reaches this code JSON
/// encoded — a Write's `content`, an Edit's replacement, a Read result — and
/// there a line break is the two characters `\` and `n`, not one newline.
/// Without the escape alternative the backslash is the delimiter, `n` has to
/// begin the name, and every name on the second or later line of a written or
/// read `.env` file went unread.
const PARAM_DELIMITER: &str = r"(?:^|\\[nrt]|[^A-Za-z0-9_.-])(?:-{1,2}[A-Za-z]?)?";

/// What marks a query string: the `?` that opens one, or one of the
/// separators between its fields. `&amp;` is the same separator in a URL that
/// has been HTML-escaped — a fetched page, an error body — and `;` is the
/// legacy one.
const QUERY_STRING_DELIMITER: &str = r"(?:\?|&amp;|&|;)";

/// What a parameter value may contain. A backslash is excluded along with the
/// quotes: in JSON-encoded text the value is followed by `\"`, and a value
/// that swallowed the backslash would leave the blob unparseable and the rest
/// of the field — the destination URL of a recorded command, say — cut off.
/// A parameter value ends at every Unicode space, spelled out the same way
/// [`URL_IN_TEXT`] spells it.
const PARAM_VALUE: &str = concat!(r#"[^"#, space_chars!(), r#""'<>`&#\\]*"#);

/// The separators a query string is split on, in a URL or in a bare run.
const QUERY_SEPARATOR: &str = "&amp;|[&;]";

/// The separator between an unambiguous name and its value, in every form one
/// takes outside a query string: `name=value` on a command line,
/// `"name": value` in JSON, `name: value` in YAML, `name = value` in a TOML
/// file or an assignment. An optional quote closes a quoted name. What
/// follows is read by [`read_value`], not by the pattern: a regular
/// expression that ran to the next space is exactly what corrupted
/// `password=os.environ['PW']`.
///
/// The optional backslash in front of the closing quote is what reads a name
/// in double-encoded text: an OAuth response body quoted back inside a tool
/// result, or a span attribute holding JSON that itself holds JSON, writes
/// the name as `\"access_token\"`. Without it the separator never matched and
/// the token was stored whole.
const NAME_VALUE_SEPARATOR: &str = r#"\\?["']?[ \t]*[:=][ \t]*"#;

/// The same separator with the colon dropped, for the names that read under
/// an `=` and nothing else: see [`ENV_BARE_NAMES`].
const NAME_EQUALS_SEPARATOR: &str = r#"\\?["']?[ \t]*=[ \t]*"#;

/// What marks a query string for a name of the *unambiguous* tier: the `?`
/// that opens one and the separators between its fields, but not the `;`. A
/// `;` separates fields in a legacy URL and statements in a connection string
/// or a shell line, and reading `Server=db;password=s3cr3tpw;Encrypt=true` as
/// a query string would take `Encrypt=true` with the password. So under a `;`
/// an unambiguous name is read by the literal grammar like any other
/// assignment.
const UNAMBIGUOUS_QUERY_DELIMITER: &str = r"(?:\?|&amp;|&)";

/// The `user:password` flag that curl, wget and git all take: `-u user:pass`,
/// `--user user:pass`, `--user=user:pass`. The user part is kept — it says
/// which account the call was made as — and what follows the first `:` is
/// read by the literal grammar, so `-u "$USER:$PASS"` and `--user admin:` are
/// left alone. The delimiter in front of the flag counts the two-character
/// escapes as well, for the reason [`PARAM_DELIMITER`] does: a command on the
/// second line of a JSON-encoded body begins after a backslash and an `n`.
const USER_PASSWORD_FLAG: &str =
    r#"(?:^|\\[nrt]|[^A-Za-z0-9_.-])(?:-u|--user)[= ]["']?[^\s:"'`]*:"#;

/// Suffixes that make an environment-variable name a credential. This is how
/// a secret reaches a recorded string most often of all: a `.env` file the
/// agent wrote, a `docker run -e` line, a Makefile, an `export` in a shell
/// command, a docker-compose `environment:` block, a GitHub Actions `env:`
/// mapping. A Kubernetes pod `env:` list is a different shape and is not
/// read: it writes `- name: DB_PASSWORD` on one line and `value: s3cr3tpg`
/// on the next, so the name and the value never stand on the same line and
/// no rule here carries state across a line break. A corpus row pins such a
/// list untouched.
/// `API_KEY` on its own is not enough — the name has to *end* in one of
/// these, so `MAX_TOKENS` (which ends in `TOKENS`, not `_TOKEN`) and
/// `TOKEN_BUDGET` are content.
const ENV_NAME_SUFFIXES: &[&str] = &[
    "PASSWORD",
    "PASSWD",
    "_SECRET",
    "_SECRET_KEY",
    "_ACCESS_KEY",
    "_API_KEY",
    "_PRIVATE_KEY",
    "_AUTH_TOKEN",
    "_TOKEN",
];

/// Environment-variable names that carry no prefix at all. Every suffix above
/// needs at least one character in front of it, which left the three
/// commonest bare spellings — Django's and Flask's `SECRET_KEY`, a cloud
/// provider's `ACCESS_KEY`, a service's `PRIVATE_KEY` — read by no rule and
/// stored whole, while `APP_SECRET_KEY` and `AWS_SECRET_ACCESS_KEY` lost
/// their values. These three are listed rather than made into an optional
/// prefix, because the bare form of every other suffix is either an
/// unambiguous name already (`SECRET`, `API_KEY`, `AUTH_TOKEN`, `PASSWORD`)
/// or the ambiguous `TOKEN`, which is read only inside a query string.
///
/// They read under the `=` forms only, which is why they are a pattern of
/// their own rather than an alternative inside [`ENV_NAME_RE`]: joined to the
/// full separator they read a colon too, and a colon after `private_key` or
/// `secret_key` is a type annotation far more often than a value —
/// `def sign(private_key: EllipticCurvePrivateKey, …)` lost its annotation.
const ENV_BARE_NAMES: &[&str] = &["SECRET_KEY", "ACCESS_KEY", "PRIVATE_KEY"];

/// Rewrite `\s` as the six ASCII space characters.
///
/// The shared patterns are written with `\s`, which several engines read as
/// `[ \t\n\r\f\v]` and nothing else. This engine reads `\s` as Unicode
/// whitespace, so a pattern carried over verbatim would treat a no-break
/// space as a terminator where those engines do not. The pinned constant
/// keeps the shared spelling; what is compiled here is its ASCII reading.
fn ascii_spaces(pattern: &str) -> String {
    pattern.replace(r"\s", r" \t\n\r\x0B\x0C")
}

/// The names of a rule, joined into one alternation folded ASCII-only.
///
/// `(?i)` on its own folds by Unicode simple case folding in this engine, and
/// `\u{17f}` folds onto `s`, so `--pa\u{17f}sword=` would match. Python
/// compiles these patterns with `re.ASCII` and TypeScript without the `u`
/// flag, and neither folds it. `(?i-u:…)` is the ASCII-only reading: the
/// names are plain ASCII literals, so turning Unicode mode off inside the
/// group cannot make the pattern match half a character.
fn ascii_fold(names: &[&str]) -> String {
    format!("(?i-u:{})", names.join("|"))
}

/// A `name=value` run with no scheme in front of it.
///
/// Plenty of recorded strings hold a query string without its scheme: an
/// OpenTelemetry span records `url.path` and `url.query` as separate
/// attributes, and an error message may quote only the part of the request it
/// failed on. [`URL_IN_TEXT_RE`] needs a scheme to find a URL, so these
/// passes judge a bare parameter by its name where it stands — the
/// unambiguous names after any delimiter, the ambiguous ones only inside a
/// query string.
///
/// This pattern matches the name and its separator and nothing else: the
/// value that follows is read by [`read_value`] and judged by
/// [`is_credential_literal`], so what is not a credential literal is left
/// exactly as written and the text after the value — the rest of the command,
/// the closing bracket, the next field — is never touched.
static UNAMBIGUOUS_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "{PARAM_DELIMITER}(?:{}){NAME_VALUE_SEPARATOR}",
        ascii_fold(UNAMBIGUOUS_PARAMS)
    ))
    .expect("unambiguous parameter pattern is valid")
});

/// A dotted credential name under a GLUED `=`.
///
/// `spring.datasource.password=s3cr3tpassword` is Spring Boot's own spelling,
/// and `db.password=`, `mail.smtp.password=`, `-Dspring.datasource.password=`
/// and `helm install --set postgres.password=` are the same name in a
/// .properties file, a JVM property and a Helm value. [`PARAM_DELIMITER`]
/// excludes the `.`, so no other rule reads any of them.
///
/// One shape and no more: the `=` is glued to the name, with no space and no
/// quote in between. The exclusion of the `.` is load-bearing for the SPACED
/// assignment reading — `self.password = get_pw()`, `config.api_key =
/// settings.API_KEY`, `self.password = other_password` are ordinary source and
/// stay unread — and the colon form of a dotted name
/// (`spring.datasource.password: hunter2hunter2`) stays unread with it. Both
/// are stated limits. What follows the `=` is read by [`read_value`] like
/// every other value, so an expansion or a call is left as written.
static DOTTED_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!("\\.(?:{})=", ascii_fold(UNAMBIGUOUS_PARAMS)))
        .expect("dotted parameter pattern is valid")
});

/// A snake_case name ending in one of [`ENV_NAME_SUFFIXES`], followed by the
/// same separator the unambiguous names take. The value is read by the
/// literal grammar, so `DB_PASSWORD=$PGPASS` and `DB_PASSWORD=` are left
/// alone and `DB_PASSWORD=hunter2` is not.
///
/// The separator is not a bare `=`: a docker-compose `environment:` block, a
/// GitHub Actions `env:` mapping and a Helm values
/// file all write `POSTGRES_PASSWORD: s3cr3tpg` with a colon, which is the
/// single likeliest place a password reaches a recorded Write or Read, and a
/// TOML file writes `NAME = value` with spaces.
///
/// The name is compared ASCII-case-insensitively, so the AWS CLI's own
/// spelling of the file it writes — `aws_secret_access_key = …` in
/// `~/.aws/credentials`, which no other rule reaches, because the `_` before
/// `secret` is not a delimiter — is read as well as `AWS_SECRET_ACCESS_KEY=`.
///
/// The delimiter is [`PARAM_DELIMITER`], the one the unambiguous names use.
/// Its own class excludes the `-`, so a name at the start of a git diff's
/// removed line had no legal delimiter in front of it and the `-` line of a
/// rotated credential — exactly where the old secret lives — was stored whole
/// while the `+` line was scrubbed. The flag cluster makes the `-` a
/// delimiter.
static ENV_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "{PARAM_DELIMITER}(?:[A-Za-z][A-Za-z0-9_]*{}){NAME_VALUE_SEPARATOR}",
        ascii_fold(ENV_NAME_SUFFIXES)
    ))
    .expect("environment-variable name pattern is valid")
});

/// The [`ENV_BARE_NAMES`] under the `=` forms only, applied in the same pass
/// as [`ENV_NAME_RE`] and right after it.
static ENV_BARE_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "{PARAM_DELIMITER}(?:{}){NAME_EQUALS_SEPARATOR}",
        ascii_fold(ENV_BARE_NAMES)
    ))
    .expect("bare environment-variable name pattern is valid")
});

static USER_PASSWORD_FLAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&ascii_spaces(USER_PASSWORD_FLAG)).expect("user:password flag pattern is valid")
});

/// An unambiguous name inside a query string. Here the value is whatever the
/// field holds, as it is for the ambiguous tier: a URL field ends at the next
/// separator, so it cannot eat the syntax around it, and the short
/// all-letters value the literal grammar leaves alone is a credential here.
static QUERY_UNAMBIGUOUS_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "({UNAMBIGUOUS_QUERY_DELIMITER})({})=({})",
        ascii_fold(UNAMBIGUOUS_PARAMS),
        PARAM_VALUE
    ))
    .expect("query-string unambiguous parameter pattern is valid")
});

static QUERY_STRING_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "({QUERY_STRING_DELIMITER})({})=({PARAM_VALUE})",
        ascii_fold(QUERY_STRING_ONLY_PARAMS)
    ))
    .expect("query-string parameter pattern is valid")
});

static QUERY_SEPARATOR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!("(?i){QUERY_SEPARATOR}")).expect("query separator pattern is valid")
});

// ── The credential-literal grammar (one reading, every name-based rule) ──
//
// Outside a query string, a name is not enough. `password=None`,
// `password=os.environ['PW']`, `--password=*)` and `api_key=API_KEY)` are all
// ordinary source, and a rule that ran to the next space rewrote them and ate
// what followed — a recorded command whose tail is gone is a policy reading
// an action that was never run. So every name-based rule outside a query
// string reads its value with [`read_value`] and judges it with
// [`is_credential_literal`]; the text outside the value is kept
// byte-for-byte, so the enclosing syntax survives.

/// What an unquoted credential literal may be made of. Everything a base64,
/// hex, URL-safe-base64 or percent-encoded secret needs, and nothing that
/// opens a shell construct. [`is_literal_value_char`] is this class, read
/// character by character; a test compares the two.
pub const LITERAL_VALUE_CHARS: &str = "A-Za-z0-9_.+/=~%@:-";

/// What ends an unquoted literal. A character that is neither a literal
/// character nor a terminator (`$`, `[`, `{`, `*`) means the run is not a
/// plain literal at all — it is an expansion, a subscript or a glob — and
/// nothing is redacted.
const LITERAL_TERMINATORS: &str = " \t;&|)]},<>#\"'`\\\r\n";

/// Values that name a secret without being one. None of these is redacted:
/// rewriting them turns readable content into a false positive a reader has
/// to chase, and `[redacted]` again would break idempotence. Sorted, because
/// the shared rule set lists them sorted.
const PLACEHOLDER_VALUES: &[&str] = &[
    "...",
    "None",
    REDACTED,
    "false",
    "nil",
    "none",
    "null",
    "true",
    "undefined",
];

/// Prefixes that mark a reference to a secret rather than the secret: a shell
/// or Make expansion, a Windows variable, a placeholder in angle brackets, a
/// template hole, a glob.
const PLACEHOLDER_PREFIXES: &[&str] = &["$", "%", "<", "{{", "*"];

/// The quotes a value may be written in.
const QUOTE_CHARACTERS: &str = "\"'`";

/// A JSON number, whole. Anchored, so it says whether a bare value *is* one.
/// A bare value of this shape under a *JSON member* colon is the one case
/// where the bare marker would leave the blob unparseable; there the marker is
/// written as a JSON string instead.
static JSON_NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$").expect("number pattern is valid")
});

/// Whether a character may stand inside an unquoted literal — the character
/// class [`LITERAL_VALUE_CHARS`] names.
fn is_literal_value_char(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            '_' | '.' | '+' | '/' | '=' | '~' | '%' | '@' | ':' | '-'
        )
}

/// What [`read_value`] found where a value was expected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ValueKind {
    /// There is no value here, or what is here is not a plain literal, and
    /// the caller must leave the text alone.
    Nothing,
    /// A quoted string. The end index is past the closing quote.
    Quoted,
    /// A quoted string in double-encoded text, opened by `\"` and closed by
    /// the next `\"`. The end index is past the closing escaped quote.
    EscapedQuoted,
    /// An unquoted literal. The end index is past its last character.
    Bare,
}

/// Whether what stands between two quotes could be a secret.
///
/// One rule: it carries an ASCII letter or an ASCII digit. Every real
/// credential does. What this refuses is the text a reader splices over when a
/// credential name ends a string and the next quote it finds is the string's
/// own closer — `", "` between two JSON members, `"], "`, `"}, "` — so
/// `{"stdout": "Enter password: ", "exit_code": 1}` and
/// `{"content": "DB_PASSWORD=", "path": ".env"}` come back as written and
/// still parse.
fn quoted_content_is_a_credential(inner: &str) -> bool {
    inner
        .chars()
        .any(|character| character.is_ascii_alphanumeric())
}

/// Read the value that starts at `start`. Returns the end index and the kind.
///
/// A quoted string is read to the matching unescaped quote *on the same
/// line*: a backslash escapes the next character, and both the backslash and
/// the character it escapes stay in the output. An unquoted value is a run of
/// [`LITERAL_VALUE_CHARS`] that ends at one of [`LITERAL_TERMINATORS`] or at
/// the end of the input; a run stopped by anything else is not a literal.
///
/// Two rules hold of both quoted readings, the raw one and the double-encoded
/// one, and they are what keeps the reader from splicing across a tool result.
/// A quoted value never crosses a line break: the raw reading stops at a CR or
/// an LF, and the double-encoded reading stops at those and at the
/// two-character `\n` and `\r` as well, so a quote with no closer before the
/// break opens no value at all. And a quoted value whose content carries no
/// ASCII letter and no digit is not a credential — `", "`, `"}, "`, pure
/// punctuation — so `{"stdout": "Enter password: ", "exit_code": 1}` reads no
/// value here and the blob still parses. A real secret has a letter or a digit
/// in it and does not span lines; a PEM key does, and the PEM shape, not this
/// reader, is what redacts one.
fn read_value(text: &str, start: usize) -> (usize, ValueKind) {
    if start >= text.len() {
        return (start, ValueKind::Nothing);
    }
    let first = text[start..].chars().next().expect("a character");
    if first == '\\'
        && text[start + 1..]
            .chars()
            .next()
            .is_some_and(|next| QUOTE_CHARACTERS.contains(next))
    {
        // A quoted string in double-encoded text: an HTTP response body
        // quoted back inside a tool result, or a span attribute holding a
        // JSON blob that itself holds JSON. The quote that opens the value
        // is written `\"`, and the value ends at the next one.
        let quote = text[start + 1..].chars().next().expect("a character");
        let mut index = start + 1 + quote.len_utf8();
        while index < text.len() {
            let character = text[index..].chars().next().expect("a character");
            if character == '\r' || character == '\n' {
                return (start, ValueKind::Nothing);
            }
            if character == '\\' {
                if text[index + 1..]
                    .chars()
                    .next()
                    .is_some_and(|next| next == quote)
                {
                    if quoted_content_is_a_credential(&text[start + 1 + quote.len_utf8()..index]) {
                        return (index + 1 + quote.len_utf8(), ValueKind::EscapedQuoted);
                    }
                    return (start, ValueKind::Nothing);
                }
                if text[index + 1..]
                    .chars()
                    .next()
                    .is_some_and(|next| matches!(next, 'n' | 'r' | '\r' | '\n'))
                {
                    // The line break of the encoded text, written with a
                    // backslash. A value does not cross it either.
                    return (start, ValueKind::Nothing);
                }
                index += 1;
                index += text[index..]
                    .chars()
                    .next()
                    .map_or(0, |character| character.len_utf8());
                continue;
            }
            index += character.len_utf8();
        }
        return (start, ValueKind::Nothing);
    }
    if QUOTE_CHARACTERS.contains(first) {
        let content_start = start + first.len_utf8();
        let mut index = content_start;
        while index < text.len() {
            let character = text[index..].chars().next().expect("a character");
            if character == '\\' {
                if text[index + 1..]
                    .chars()
                    .next()
                    .is_some_and(|next| next == '\r' || next == '\n')
                {
                    return (start, ValueKind::Nothing);
                }
                index += 1;
                index += text[index..]
                    .chars()
                    .next()
                    .map_or(0, |character| character.len_utf8());
                continue;
            }
            if character == '\r' || character == '\n' {
                return (start, ValueKind::Nothing);
            }
            if character == first {
                if quoted_content_is_a_credential(&text[content_start..index]) {
                    return (index + character.len_utf8(), ValueKind::Quoted);
                }
                return (start, ValueKind::Nothing);
            }
            index += character.len_utf8();
        }
        return (start, ValueKind::Nothing);
    }
    if first == '=' {
        // The separator matched the first character of a two-character
        // operator: `password == x`, `===`, Make's `:=`, Go's short
        // assignment. What stands here is the rest of the operator, not a
        // value. Base64 padding is only ever at the end of a run, so no
        // credential literal begins with an `=` either.
        return (start, ValueKind::Nothing);
    }
    let end = text[start..]
        .find(|character: char| !is_literal_value_char(character))
        .map_or(text.len(), |offset| start + offset);
    if end < text.len() {
        let stopper = text[end..].chars().next().expect("a character");
        if !LITERAL_TERMINATORS.contains(stopper) {
            return (start, ValueKind::Nothing);
        }
    }
    if end == start {
        return (start, ValueKind::Nothing);
    }
    (end, ValueKind::Bare)
}

/// Whether an unquoted value is a secret rather than a name for one.
///
/// A reference or a placeholder is never a secret. What is left counts as one
/// when it carries a digit, or a character outside `[A-Za-z_.]`, or is at
/// least 16 characters long. The cost of that last clause is a documented
/// limit: a short all-letters value — `pw`, `secret`, `API_KEY` as a name
/// being passed on — is left as written, because a rule that took it would
/// rewrite ordinary source.
fn is_credential_literal(value: &str) -> bool {
    if value.is_empty() || PLACEHOLDER_VALUES.contains(&value) {
        return false;
    }
    if PLACEHOLDER_PREFIXES
        .iter()
        .any(|prefix| value.starts_with(prefix))
        || value.starts_with("[redacted")
    {
        return false;
    }
    if value.chars().any(|character| character.is_ascii_digit()) {
        return true;
    }
    if value
        .chars()
        .any(|character| !matches!(character, 'A'..='Z' | 'a'..='z' | '_' | '.'))
    {
        return true;
    }
    if value.contains('.') {
        // A dotted run of ordinary word characters with no digit in it is an
        // attribute path, not a secret: `settings.DATABASE_PASSWORD`,
        // `config.database.password`, `process.env.DB_PASSWORD`,
        // `self._resolve_password`. The length clause below would take every
        // one of them, which is the reference case the literal grammar exists
        // to leave alone. `correcthorsebatterystaple` carries no dot and
        // still goes.
        return false;
    }
    value.chars().count() >= 16
}

/// Whether a name read backwards from a separator is environment-style:
/// upper case, digits and underscores, the spelling a `.env` file, a compose
/// file, a Dockerfile and a shell export use. There an unquoted word after
/// the `=` *is* the value, so the reference reading does not apply to it.
fn is_environment_style_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    match bytes.next() {
        Some(first) if first.is_ascii_uppercase() => {}
        _ => return false,
    }
    bytes.all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Whether the value at `start` follows `name =` in source.
///
/// One shape: the `=` has whitespace on both sides of it and the name in
/// front of it is not an environment-style upper-case name.
/// `password = DEFAULT_ADMIN_PASSWORD`, `const password = getDatabasePassword;`
/// and `api_key = SETTINGS_DEFAULT_API_KEY` are source, and every one of them
/// is spaced. A glued `=` is a literal whatever the name's case and whatever
/// the value looks like: `mysql --password=MyVerySecretPassword`,
/// `docker login -u ci --password=DockerHubAccessToken`,
/// `export password=MyVerySecretPassword` and `DB_PASSWORD=DefaultAdminPassword`
/// are a flag, a shell line and a .env line, and the spacing is what tells
/// them apart — the same discriminator [`is_empty_equals_value`] uses. Every
/// colon form is refused here by asking for the `=`. A leading `-` of a flag
/// is not part of the name.
fn is_source_assignment(text: &str, start: usize) -> bool {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut spaced_after = false;
    while index > 0 && (bytes[index - 1] == b' ' || bytes[index - 1] == b'\t') {
        spaced_after = true;
        index -= 1;
    }
    if index == 0 || bytes[index - 1] != b'=' || !spaced_after {
        return false;
    }
    index -= 1;
    let mut spaced_before = false;
    while index > 0 && (bytes[index - 1] == b' ' || bytes[index - 1] == b'\t') {
        spaced_before = true;
        index -= 1;
    }
    if !spaced_before {
        return false;
    }
    if index > 0 && (bytes[index - 1] == b'"' || bytes[index - 1] == b'\'') {
        index -= 1;
        if index > 0 && bytes[index - 1] == b'\\' {
            index -= 1;
        }
    }
    let end = index;
    while index > 0
        && (bytes[index - 1].is_ascii_alphanumeric()
            || matches!(bytes[index - 1], b'_' | b'.' | b'-'))
    {
        index -= 1;
    }
    let name = text[index..end].trim_start_matches('-');
    !name.is_empty() && !is_environment_style_name(name)
}

/// Whether the `=` in front of `start` was given no value at all.
///
/// One shape, for the `=` separator only: no whitespace before the `=` and
/// whitespace after it. `password= host=db.internal` is a logfmt field that
/// was left empty and then the next field, and `mysql --password= --host=db`
/// is a flag that was left empty and then the next flag; reading on from
/// there rewrites the tail of a recorded command. `a = b` is spaced on both
/// sides and reads as before, `a=b` has no space at all and reads as before,
/// and the `:` separator is untouched, because `password: host=db` is a YAML
/// value.
fn is_empty_equals_value(text: &str, start: usize) -> bool {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut spaced_after = false;
    while index > 0 && (bytes[index - 1] == b' ' || bytes[index - 1] == b'\t') {
        spaced_after = true;
        index -= 1;
    }
    if index == 0 || bytes[index - 1] != b'=' || !spaced_after {
        return false;
    }
    index -= 1;
    index == 0 || !(bytes[index - 1] == b' ' || bytes[index - 1] == b'\t')
}

/// Whether a value is a name for a secret rather than the secret.
///
/// One shape: a legal identifier of letters and underscores only that carries
/// an underscore or a change of case — `DEFAULT_ADMIN_PASSWORD`,
/// `getDatabasePassword`, `ClientSecretName`. A constant, a function and a
/// variable are all spelled that way and none of them is a credential,
/// whatever their length. A run of one case with no underscore
/// (`correcthorsebatterystaple`) is not one of these, and the length clause
/// still takes it.
fn is_identifier_reference(value: &str) -> bool {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphabetic() || byte == b'_')
    {
        return false;
    }
    if value.contains('_') {
        return true;
    }
    value.bytes().any(|byte| byte.is_ascii_lowercase())
        && value.bytes().any(|byte| byte.is_ascii_uppercase())
}

/// Read the value at `start` and say what should stand in its place.
///
/// Returns the index just past what the caller should skip — the end of the
/// value when it is replaced, and `start` when it is not, which is where the
/// search resumes — and the replacement, or `None` when the grammar leaves
/// the value alone.
fn literal_replacement(text: &str, start: usize) -> (usize, Option<String>) {
    if is_empty_equals_value(text, start) {
        return (start, None);
    }
    let (end, kind) = read_value(text, start);
    match kind {
        ValueKind::Nothing => (start, None),
        ValueKind::Quoted => {
            let value = &text[start..end];
            let quote = &value[..1];
            let inner = &value[1..value.len() - 1];
            if inner.is_empty() || inner.starts_with("[redacted") {
                return (start, None);
            }
            (end, Some(format!("{quote}{REDACTED}{quote}")))
        }
        ValueKind::EscapedQuoted => {
            let value = &text[start..end];
            let quote = &value[1..2];
            let inner = &value[2..value.len() - 2];
            if inner.is_empty() || inner.starts_with("[redacted") {
                return (start, None);
            }
            (end, Some(format!("\\{quote}{REDACTED}\\{quote}")))
        }
        ValueKind::Bare => {
            let value = &text[start..end];
            if is_identifier_reference(value) && is_source_assignment(text, start) {
                // A name for the secret standing in source, whatever its
                // length: `password = DEFAULT_ADMIN_PASSWORD`, `const
                // password = getDatabasePassword;`. The reference test wins
                // over the length here and only here — under an
                // environment-style name, and in every colon form, an
                // unquoted word is the value itself.
                return (start, None);
            }
            if !is_credential_literal(value) {
                return (start, None);
            }
            if is_json_number(value) {
                if let Some(quote) = json_member_quote(text, start) {
                    // `{"password": 123456}`: a bare marker where a number
                    // stood leaves the blob unparseable, and `@json_get_str`
                    // in the policy functors stops at the first unescaped
                    // quote, so a broken blob blinds the policy. The marker
                    // is written as a JSON string instead.
                    return (end, Some(format!("{quote}{REDACTED}{quote}")));
                }
            }
            (end, Some(REDACTED.to_string()))
        }
    }
}

/// Whether `value` is a JSON number and nothing else.
fn is_json_number(value: &str) -> bool {
    JSON_NUMBER_RE.is_match(value)
}

/// The quoting the marker at `start` needs, or `None` for none.
///
/// One shape says the colon in front of the value is a JSON member colon:
/// the name was closed by a quote character standing immediately before the
/// separator — `"password": 12345`, `'password': 123`, and the double-encoded
/// `\"password\": 12345`, whose closing quote is written with a backslash in
/// front of it. The marker is then written with the name's own closing form,
/// so the blob parses as it did.
///
/// Every other colon is not a mapping: a YAML or ini line, a logfmt field and
/// the `-u user:pass` flag all read the same inside a JSON string as outside
/// one, and they take the bare marker. Writing quotes there is what turned
/// `{"content": "POSTGRES_PASSWORD: 12345"}` into text `json.loads` rejects.
///
/// The scan reads bytes rather than characters: every byte it compares is
/// ASCII, and a UTF-8 continuation byte is never equal to one, so a multi-byte
/// character in front of the separator can never be read as a quote.
fn json_member_quote(text: &str, start: usize) -> Option<&'static str> {
    let bytes = text.as_bytes();
    let mut index = start;
    while index > 0 && (bytes[index - 1] == b' ' || bytes[index - 1] == b'\t') {
        index -= 1;
    }
    if index == 0 || bytes[index - 1] != b':' {
        return None;
    }
    index -= 1;
    while index > 0 && (bytes[index - 1] == b' ' || bytes[index - 1] == b'\t') {
        index -= 1;
    }
    if index == 0 {
        return None;
    }
    match bytes[index - 1] {
        b'"' if index >= 2 && bytes[index - 2] == b'\\' => Some("\\\""),
        b'"' => Some("\""),
        b'\'' => Some("'"),
        _ => None,
    }
}

/// Redact the value that follows every match of `pattern`.
///
/// `pattern` matches the name and its separator and nothing else. The search
/// resumes past the value, so a `[redacted]` this pass wrote is never read as
/// the next value.
///
/// The output is built once, left to right, rather than rewritten per
/// replacement: a string of nothing but credentials — a `.env` file, a
/// recorded environment — holds tens of thousands of them, and copying the
/// whole text once each is quadratic on a request-handling thread. Reading
/// the matches out of the unmodified text is the same reading: every pattern
/// here begins with the boundary character it matches, so a match never
/// depends on text before the point the search resumes, and that text is
/// exactly what the replacements have already passed.
/// The name a pattern matched, as a run of name characters that ends where
/// the separator begins.
static MATCHED_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"([A-Za-z0-9_.-]+)[\\"']*[ \t]*[:=][ \t]*$"#)
        .expect("matched-name pattern is valid")
});

/// Whether the name a pattern just matched is one of the nine header names.
///
/// A leading `.` is stripped alongside a leading `-`: the dotted pattern's
/// whole match is `.api-key=`, so without the strip the name read back is
/// `.api-key`, the header-word exemption is skipped, and the same word value
/// is content under one spelling of the name (`api-key: user-provided` kept)
/// and a credential under another (`gateway.api-key=user-provided`
/// rewritten). `apikey` without the hyphen is not one of the nine header
/// names under any spelling, so it takes the exemption under neither:
/// `apikey: not-set` and `svc.apikey=not-set` both lose their value.
///
/// A leading escaped line break is dropped before the name is read. The
/// delimiter these patterns carry matches an escaped break — `\n`, `\r`,
/// `\t` — as one of the places a name may start, and the backslash is not a
/// name character while `n`, `r` and `t` are: so the name read back out of
/// `\napi-key=` would be `napi-key`, which is not one of the nine, the
/// exemption would be skipped, and one file body would lose `not-set` when it
/// reached the recorder JSON-encoded and keep it when it arrived raw.
fn matched_name_is_a_header(matched: &str) -> bool {
    let matched = ["\\n", "\\r", "\\t"]
        .iter()
        .find_map(|escape| matched.strip_prefix(escape))
        .unwrap_or(matched);
    let Some(found) = MATCHED_NAME_RE.captures(matched) else {
        return false;
    };
    let name = found[1].trim_start_matches(['-', '.']).to_ascii_lowercase();
    SENSITIVE_HEADERS.contains(&name.as_str())
}

fn redact_named_values(text: &str, pattern: &Regex) -> (String, usize) {
    let mut out = String::new();
    let mut count = 0usize;
    let mut copied = 0usize;
    let mut position = 0usize;
    while let Some(found) = pattern.find_at(text, position) {
        let start = found.end();
        if matched_name_is_a_header(found.as_str()) {
            let (end, kind) = read_value(text, start);
            if kind == ValueKind::Bare && is_header_word(&text[start..end]) {
                // The same reading the header lines take: two of the nine
                // header names — `api-key` and `x-api-key` — are
                // unambiguous parameter names as well, so the name-based rule
                // reaches them first and has to say the same thing about a
                // lone word. `x-api-key: application/json` and
                // `api-key: not-set` are what a config template and a
                // filled-in manifest write.
                position = start;
                continue;
            }
        }
        let (resume, replacement) = literal_replacement(text, start);
        position = resume;
        let Some(replacement) = replacement else {
            continue;
        };
        out.push_str(&text[copied..start]);
        out.push_str(&replacement);
        copied = resume;
        count += 1;
    }
    if count == 0 {
        return (text.to_string(), 0);
    }
    out.push_str(&text[copied..]);
    (out, count)
}

/// What must stand to the left of a token shape: the start of the string, one
/// of the two-character escapes `\n`, `\r` and `\t`, or a character the token
/// itself could not contain. Without the boundary a shape matches inside a
/// longer word — `sk-` is the worst of them, since `-` and `_` keep the run
/// going, so `risk-assessment-summary.md` reads as an OpenAI key and is
/// destroyed before it is stored. Without the escape a shape on any line but
/// the first of a JSON-encoded body — a Read result, a Write's `content` —
/// stands behind the letter of the escape, which the boundary refuses, and a
/// private key read or written through a tool was stored whole while its raw
/// twin redacted. The escape stands first in the alternation so it wins over
/// the single-character reading of the backslash. What the boundary consumed
/// is not part of the secret, so it is put back where it was.
const SHAPE_BOUNDARY: &str = r"(^|\\[nrt]|[^A-Za-z0-9])";

/// Conditions the server checks in code, per shape, because its engine has
/// no lookahead. The two SDKs write each of these as a lookahead inside the
/// shape's `pattern`; the rule set carries the same list in the shape's
/// `checks`, and the pattern the server compiles is that shape's
/// `linear_engine_pattern` — the SDK pattern with the lookaheads removed.
///
/// * `right_boundary` — the character after the match may not be
///   `[A-Za-z0-9]`. A fixed-length shape has no other way to refuse the head
///   of a longer word, so without it `ASIAPACIFICREGION2026` reads as an AWS
///   key id and the last characters of the word survive the replacement.
/// * `digit_in_tail` — the matched shape must hold an ASCII digit. Only the
///   OpenAI shape carries it, and its `sk-` prefix has no digit, so this is
///   a digit in the run that follows the prefix.
/// * `not_file_extension` — the match may not be followed by a `.` and one
///   to four letters. The OpenAI run accepts the `-` and `_` that fill an
///   ordinary branch name or file name, so without these two checks
///   `sk-refactor-authentication` and `docs/sk-quickstart-guide-2026.md` are
///   destroyed as keys.
const SHAPE_CHECKS: &[(&str, &[&str])] = &[
    ("openai-key", &["digit_in_tail", "not_file_extension"]),
    ("aws-access-key-id", &["right_boundary"]),
    ("gitlab-token", &["right_boundary"]),
    ("npm-token", &["right_boundary"]),
];

/// The checks for one shape, or nothing when it carries none.
fn shape_checks(shape: &str) -> &'static [&'static str] {
    SHAPE_CHECKS
        .iter()
        .find(|(name, _)| *name == shape)
        .map(|(_, checks)| *checks)
        .unwrap_or(&[])
}

/// Whether one match of `shape` passes the checks that shape carries.
/// `value` is the matched shape without the boundary character, and `after`
/// is the text that follows it.
fn shape_match_passes(shape: &str, value: &str, after: &str) -> bool {
    shape_checks(shape).iter().all(|check| match *check {
        "right_boundary" => !after
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric()),
        "digit_in_tail" => value.chars().any(|c| c.is_ascii_digit()),
        "not_file_extension" => {
            let mut following = after.chars();
            !(following.next() == Some('.')
                && following.next().is_some_and(|c| c.is_ascii_alphabetic()))
        }
        other => unreachable!("unknown shape check {other}"),
    })
}

/// One compiled regex per shape, in [`TOKEN_SHAPES`] order, each with
/// [`SHAPE_BOUNDARY`] in front of it.
static TOKEN_RULES: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    TOKEN_SHAPES
        .iter()
        .map(|(shape, pattern)| {
            (
                *shape,
                Regex::new(&format!("{SHAPE_BOUNDARY}{pattern}")).expect("shape pattern is valid"),
            )
        })
        .collect()
});

/// Prefilter: one pass says which shapes occur at all, so a string with no
/// secrets in it costs a single scan instead of one per shape.
static TOKEN_SET: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new(
        TOKEN_SHAPES
            .iter()
            .map(|(_, pattern)| format!("{SHAPE_BOUNDARY}{pattern}")),
    )
    .expect("shape patterns are valid")
});

/// Percent-decode and ASCII-lowercase a query-parameter name, so `%5FKEY`,
/// `API_KEY` and `api_key` all answer to the same entry.
fn decode_param_name(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && i + 2 < chars.len() {
            if let (Some(hi), Some(lo)) = (chars[i + 1].to_digit(16), chars[i + 2].to_digit(16)) {
                out.push(char::from((hi * 16 + lo) as u8));
                i += 3;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out.to_ascii_lowercase()
}

/// Whether a query-string field's name means a credential.
///
/// Both tiers count inside a query string — the run is a query string
/// already — and so does a name whose last hyphen-separated segment is
/// `signature` or `sig`. That last clause is what catches a pre-signed URL:
/// the credential in `?X-Amz-Signature=…` or `?X-Goog-Signature=…` is the
/// signature itself, and the vendor prefix in front of it means neither
/// tier's list can name it.
fn is_sensitive_query_name(name: &str) -> bool {
    QUERY_PARAM_SET.contains(name) || matches!(name.rsplit('-').next(), Some("signature" | "sig"))
}

/// Scrub one `name=value` field of a query string, judged by its name.
///
/// A value that is empty, or already `[redacted]`, is left as it stands — so
/// scrubbing twice changes nothing and counts nothing.
fn redact_query_field(field: &str) -> (String, usize) {
    let Some(eq) = field.find('=') else {
        return (field.to_string(), 0);
    };
    if !is_sensitive_query_name(&decode_param_name(&field[..eq])) {
        return (field.to_string(), 0);
    }
    let value = &field[eq + 1..];
    if value.is_empty() || value.starts_with("[redacted") {
        return (field.to_string(), 0);
    }
    (format!("{}{REDACTED}", &field[..=eq]), 1)
}

/// Scrub the password out of a `scheme://user:password@host` authority.
///
/// A git remote in a `fatal: unable to access …` line and a database
/// connection string both carry the credential here rather than in a query
/// parameter, and neither has a name to judge it by — the position is the
/// name. The scheme, the user and the host are kept. A reference
/// (`$PGPASSWORD`) and an already-scrubbed marker are left alone.
fn redact_url_userinfo(text: &str) -> (String, usize) {
    let mut count = 0usize;
    let out = URL_USERINFO_RE
        .replace_all(text, |caps: &Captures| {
            let password = &caps[4];
            if PLACEHOLDER_PREFIXES
                .iter()
                .any(|prefix| password.starts_with(prefix))
                || password.starts_with("[redacted")
            {
                return caps[0].to_string();
            }
            count += 1;
            format!("{}{}{}:{REDACTED}@", &caps[1], &caps[2], &caps[3])
        })
        .into_owned();
    (out, count)
}

/// Replace credential-bearing query-parameter values in one URL.
///
/// Structural rather than pattern-based: the query is split into fields and
/// each is judged by its name, so a secret in a shape nothing recognises is
/// still removed as long as it sits under a known parameter name. Both tiers
/// of name count here — the run is a query string already.
///
/// Fields are separated by `&`, by `;` (the legacy separator) or by `&amp;`
/// (the same URL after HTML escaping, which is how one arrives out of a
/// fetched page or an error body). The separators come back exactly as they
/// were written. A password in the authority — `scheme://user:password@host`
/// — goes as well, the user kept.
pub fn redact_url(url: &str) -> (String, usize) {
    let (url, userinfo_count) = redact_url_userinfo(url);
    let url = url.as_str();
    let Some(question) = url.find('?') else {
        return (url.to_string(), userinfo_count);
    };
    let fragment = url[question..]
        .find('#')
        .map(|i| question + i)
        .unwrap_or(url.len());
    let head = &url[..=question];
    let query = &url[question + 1..fragment];
    let tail = &url[fragment..];

    let mut count = userinfo_count;
    let mut rebuilt = String::with_capacity(query.len());
    let mut position = 0usize;
    for separator in QUERY_SEPARATOR_RE.find_iter(query) {
        let (field, found) = redact_query_field(&query[position..separator.start()]);
        rebuilt.push_str(&field);
        rebuilt.push_str(separator.as_str());
        count += found;
        position = separator.end();
    }
    let (field, found) = redact_query_field(&query[position..]);
    rebuilt.push_str(&field);
    count += found;

    if count == 0 {
        return (url.to_string(), 0);
    }
    (format!("{head}{rebuilt}{tail}"), count)
}

/// Scrub `name=…` runs that the scheme-anchored URL pass could not see.
///
/// Two readings, one per tier. An unambiguous name is a credential after any
/// delimiter and in any of the assignment forms, but only when what follows
/// it is a credential literal. An ambiguous name is only read inside a query
/// string, where the value is whatever the field holds — a URL field cannot
/// eat the syntax around it. A value already reduced to `[redacted]`, or
/// empty, is left alone, so a parameter inside a URL the pass before handled
/// is not counted twice and a second scrub is a no-op.
fn redact_bare_query_params(text: &str) -> (String, usize) {
    let mut count = 0usize;
    let mut out = text.to_string();
    for pattern in [&*QUERY_UNAMBIGUOUS_PARAM_RE, &*QUERY_STRING_PARAM_RE] {
        let replaced = pattern
            .replace_all(&out, |caps: &Captures| {
                let value = &caps[3];
                let trimmed = value.trim_end_matches(URL_TRAILING_PUNCTUATION);
                if trimmed.is_empty() || value.starts_with("[redacted") {
                    return caps[0].to_string();
                }
                count += 1;
                format!(
                    "{}{}={REDACTED}{}",
                    &caps[1],
                    &caps[2],
                    &value[trimmed.len()..]
                )
            })
            .into_owned();
        out = replaced;
    }
    for pattern in [
        &*UNAMBIGUOUS_PARAM_RE,
        &*DOTTED_PARAM_RE,
        &*ENV_NAME_RE,
        &*ENV_BARE_NAME_RE,
        &*USER_PASSWORD_FLAG_RE,
    ] {
        let (replaced, found) = redact_named_values(&out, pattern);
        out = replaced;
        count += found;
    }
    (out, count)
}

/// Scrub one string. Returns the scrubbed text and the number of replacements.
///
/// Headers go first — a credential header's value is condemned whole, which
/// is the safest reading — as a `Name: value` line, then as a quoted
/// `"name": "value"` pair, then wherever else the name stands; then URLs,
/// then bare query parameters, then token shapes.
pub fn redact_text(text: &str) -> (String, usize) {
    if text.is_empty() {
        return (text.to_string(), 0);
    }
    let mut count = 0usize;
    let (out, found) = redact_header_pairs(text);
    count += found;
    let (out, found) = redact_headers_in_text(&out);
    count += found;
    let (out, found) = redact_urls_in_text(&out);
    count += found;
    let (out, found) = redact_url_userinfo(&out);
    count += found;
    let (out, found) = redact_bare_query_params(&out);
    count += found;
    let (out, found) = redact_token_shapes(&out);
    count += found;
    (out, count)
}

/// Scrub `"authorization": "…"` pairs inside a serialised blob.
///
/// The value keeps its own punctuation: a quoted string comes back quoted the
/// same way, an array comes back as a one-element array. A value that is
/// already `[redacted]`, or empty, is left as it stands, so a second scrub
/// changes nothing and counts nothing.
fn redact_header_pairs(text: &str) -> (String, usize) {
    let mut count = 0usize;
    let out = HEADER_PAIR_RE
        .replace_all(text, |caps: &Captures| {
            let token = &caps[2];
            let inner = &token[1..token.len() - 1];
            // A bare marker standing where a value belongs reads back as a
            // one-element array holding a bareword, so without this guard a
            // second pass over `"authorization": [redacted]` wrote
            // `["[redacted]"]` and a third changed nothing again. The quoted
            // branch is already idempotent because its rewrite of an
            // already-scrubbed value is the value; the array branch is not,
            // and takes the same guard the quoted values take elsewhere.
            if token.starts_with("[redacted") {
                return caps[0].to_string();
            }
            let replacement = if token.starts_with('[') {
                if inner.trim().is_empty() {
                    return caps[0].to_string();
                }
                format!("[\"{REDACTED}\"]")
            } else {
                if inner.is_empty() {
                    return caps[0].to_string();
                }
                let quote = &token[..1];
                format!("{quote}{REDACTED}{quote}")
            };
            if replacement == token {
                return caps[0].to_string();
            }
            count += 1;
            format!("{}{replacement}", &caps[1])
        })
        .into_owned();
    (out, count)
}

/// Whether a quoted value starts at `start`, escaped quote included.
fn opens_quoted_value(text: &str, start: usize) -> bool {
    let mut characters = text[start..].chars();
    match characters.next() {
        None => false,
        Some(first) if QUOTE_CHARACTERS.contains(first) => true,
        Some('\\') => characters
            .next()
            .is_some_and(|next| QUOTE_CHARACTERS.contains(next)),
        Some(_) => false,
    }
}

/// Scrub a credential header, on a line of its own or inside a string.
///
/// The value ends where the line does, at the quote that closes the string
/// the header sits inside, or — when nothing separates the name from its
/// value, as in `-H authorization:abc123def` — where the literal grammar says
/// it does, so the rest of the command survives. A reference
/// (`-H Authorization:$GH_TOKEN`) and a placeholder
/// (`Authorization: Bearer <YOUR_TOKEN>`) are left as written.
fn redact_headers_in_text(text: &str) -> (String, usize) {
    let mut out = String::new();
    let mut count = 0usize;
    let mut copied = 0usize;
    let mut position = 0usize;
    while let Some(caps) = HEADER_NAME_RE.captures_at(text, position) {
        let name = caps[1].to_ascii_lowercase();
        let start = caps.get(0).expect("the whole match").end();
        // Every branch below sets `position` before it loops: past what it
        // wrote when it wrote something, and to the value position when it
        // left the value alone, so the next name is found and this one is not
        // found again.
        if text[start..].starts_with([' ', '\t']) {
            // A value separated from the name by a space runs to the end of
            // the line, and how much of it is the credential depends on the
            // header.
            let value_start = start
                + text[start..]
                    .find(|character: char| character != ' ' && character != '\t')
                    .unwrap_or(text.len() - start);
            if opens_quoted_value(text, value_start) {
                // The value is quoted and the name is not, which is how a
                // header is written in every YAML config (a Prometheus scrape
                // job, a Grafana datasource, an Ansible `uri` task, an Actions
                // workflow), in a JavaScript object literal and in TOML. The
                // scan below stops at the quote that *opens* such a value,
                // leaving an empty span and the credential in the record, so
                // the literal grammar reads it instead: a quoted value is a
                // credential literal, its content goes and its quotes stay.
                let (resume, replacement) = literal_replacement(text, value_start);
                position = resume;
                let Some(replacement) = replacement else {
                    position = value_start;
                    continue;
                };
                out.push_str(&text[copied..value_start]);
                out.push_str(&replacement);
                copied = resume;
                count += 1;
                continue;
            }
            let value_end = value_start
                + text[value_start..]
                    .find(|character: char| HEADER_VALUE_TERMINATORS.contains(character))
                    .unwrap_or(text.len() - value_start);
            let span = header_credential_span(&name, &text[value_start..value_end]);
            if span == 0 {
                position = value_start;
                continue;
            }
            out.push_str(&text[copied..value_start]);
            out.push_str(REDACTED);
            copied = value_start + span;
            count += 1;
            position = copied;
            continue;
        }
        // Nothing between the name and its value: `-H authorization:abc123def`
        // is a shell word, and the literal grammar says where it ends.
        let (resume, replacement) = literal_replacement(text, start);
        position = resume;
        let Some(replacement) = replacement else {
            continue;
        };
        out.push_str(&text[copied..start]);
        out.push_str(&replacement);
        copied = resume;
        count += 1;
    }
    if count == 0 {
        return (text.to_string(), 0);
    }
    out.push_str(&text[copied..]);
    (out, count)
}

/// Scrub the credential-bearing parameters of every `https?://…` run.
///
/// The trailing punctuation a run swallowed is put back after the scrub, so a
/// markdown link keeps its bracket and a sentence its full stop.
fn redact_urls_in_text(text: &str) -> (String, usize) {
    let mut count = 0usize;
    let out = URL_IN_TEXT_RE
        .replace_all(text, |caps: &Captures| {
            let run = &caps[0];
            let trimmed = run.trim_end_matches(URL_TRAILING_PUNCTUATION);
            let (cleaned, found) = redact_url(trimmed);
            count += found;
            format!("{cleaned}{}", &run[trimmed.len()..])
        })
        .into_owned();
    (out, count)
}

/// Replace every known token shape, in the order the rule set lists them.
fn redact_token_shapes(text: &str) -> (String, usize) {
    let mut count = 0usize;
    let mut out = text.to_string();
    let present = TOKEN_SET.matches(&out);
    for (index, (shape, pattern)) in TOKEN_RULES.iter().enumerate() {
        if !present.matched(index) {
            continue;
        }
        // The bearer rule keeps the word "Bearer", so the graph still records
        // that the call was bearer-authenticated.
        let replacement = if *shape == "bearer" {
            format!("Bearer [redacted:{shape}]")
        } else {
            format!("[redacted:{shape}]")
        };
        // The boundary character belongs to the text around the secret, not
        // to the secret, so it goes back where it was. A match that fails the
        // shape's checks is left in place, character for character.
        let mut replaced = String::with_capacity(out.len());
        let mut copied = 0usize;
        for caps in pattern.captures_iter(&out) {
            let whole = caps.get(0).expect("the whole match");
            let boundary = caps.get(1).expect("the boundary group");
            let value = &out[boundary.end()..whole.end()];
            if !shape_match_passes(shape, value, &out[whole.end()..]) {
                continue;
            }
            replaced.push_str(&out[copied..boundary.end()]);
            replaced.push_str(&replacement);
            copied = whole.end();
            count += 1;
        }
        replaced.push_str(&out[copied..]);
        out = replaced;
    }
    (out, count)
}

fn scrub_optional(field: &mut Option<String>) -> usize {
    let Some(value) = field.as_deref() else {
        return 0;
    };
    if value.is_empty() {
        return 0;
    }
    let (cleaned, count) = redact_text(value);
    if count > 0 {
        *field = Some(cleaned);
    }
    count
}

fn scrub_required(field: &mut String) -> usize {
    if field.is_empty() {
        return 0;
    }
    let (cleaned, count) = redact_text(field);
    if count > 0 {
        *field = cleaned;
    }
    count
}

fn scrub_tool(tool: &mut Tool) -> usize {
    scrub_optional(&mut tool.name) + scrub_optional(&mut tool.arguments)
}

/// Scrub the content of a batch of events, in place. Returns the number of
/// replacements made across the whole batch.
///
/// `id`, `role` and `principal` are left alone. `id` is structure — edges
/// name events by id, so a scrubber that touched one would detach every
/// dependency it "protected" — and `principal` is stamped by the gRPC
/// boundary from the caller's credentials, not by the client. Everything
/// else is a free-form client string, `agent`, `entity` and the adapter's
/// `metadata` record included, and a free-form string is somewhere a secret
/// can land.
pub fn redact_events(events: &mut [Event]) -> usize {
    let mut count = 0usize;
    for event in events.iter_mut() {
        count += scrub_optional(&mut event.text);
        count += scrub_optional(&mut event.agent);
        count += scrub_optional(&mut event.entity);
        count += scrub_optional(&mut event.metadata);
        for tool in event.tools.iter_mut() {
            count += scrub_tool(tool);
        }
        if let Some(tool) = event.derived_from.as_mut() {
            count += scrub_tool(tool);
        }
    }
    count
}

/// Scrub the entity label on a batch of dependency edges, in place. Returns
/// the number of replacements made.
///
/// An edge's `source` and `destination` are ids, and its `message_index` and
/// `proximal` are structure — an edge that lost them would name nothing. Its
/// `principal` is stamped by the gRPC boundary. That leaves `entity`, "a
/// user-supplied actor in the caller's domain": a free-form client string
/// that is written into the store and mirrored to Neo4j like any other. The
/// identically-named field on an event is scrubbed, and this one is too, so
/// a client cannot put a credential in the graph by labelling an edge with
/// it.
pub fn redact_edges(edges: &mut [Edge]) -> usize {
    let mut count = 0usize;
    for edge in edges.iter_mut() {
        count += scrub_optional(&mut edge.entity);
    }
    count
}

/// Scrub the content of a batch of OTel spans, in place.
///
/// `status_message` is where an exception's text lands, and
/// `attributes_json` / `events_json` carry span attributes — both are routes
/// for the same URL-in-error-text leak. `name`, `service_name` and
/// `service_version` are free-form client strings too: naming an HTTP client
/// span after the request it made is a common convention, and that name
/// carries the query string with it. `entity` is the same user-supplied
/// label an event and a dependency edge carry — the gRPC boundary stamps
/// `principal` and leaves `entity` exactly as the client wrote it, so it is
/// scrubbed here like every other client string. Ids and timings are left
/// alone.
pub fn redact_computations(computations: &mut [Computation]) -> usize {
    let mut count = 0usize;
    for computation in computations.iter_mut() {
        count += scrub_required(&mut computation.name);
        count += scrub_optional(&mut computation.status_message);
        count += scrub_required(&mut computation.attributes_json);
        count += scrub_required(&mut computation.events_json);
        count += scrub_optional(&mut computation.service_name);
        count += scrub_optional(&mut computation.service_version);
        count += scrub_optional(&mut computation.entity);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every literal below is synthetic: a string shaped like a credential,
    // never a real one.
    const GOOGLE_KEY: &str = "AIzaSyDaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OPENAI_KEY: &str = "sk-proj-AAAAAAAAAAAAAAAAAAAA1234";
    const ANTHROPIC_KEY: &str = "sk-ant-api03-BBBBBBBBBBBBBBBBBBBBBBBB";
    const AWS_KEY_ID: &str = "AKIAQQQQQQQQQQQQQQQQ";
    const AWS_TEMP_KEY_ID: &str = "ASIAQQQQQQQQQQQQQQQQ";
    const GITHUB_CLASSIC: &str = "ghp_cccccccccccccccccccccccccccccccccccc";
    const GITHUB_OAUTH: &str = "gho_dddddddddddddddddddddddddddddddddddd";
    const GITHUB_PAT: &str =
        "github_pat_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const GITLAB_TOKEN: &str = "glpat-ffffffffffffffffffff";
    const SLACK_TOKEN: &str = "xoxb-111111111111";
    const NPM_TOKEN: &str = "npm_gggggggggggggggggggggggggggggggggggg";
    const PYPI_TOKEN: &str = "pypi-AgEIhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh";
    const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.AAAABBBBCCCC";
    const PEM_BLOCK: &str = concat!(
        "-----BEGIN RSA PRIVATE KEY-----\n",
        "MIIsyntheticsyntheticsynthetic\n",
        "-----END RSA PRIVATE KEY-----"
    );

    fn scrub(text: &str) -> String {
        redact_text(text).0
    }

    /// Requirement: one rule set, implemented three times.
    ///
    /// Nothing else compares the two. A name added to this crate and not to
    /// the rule file, or dropped from one of them, would leave the suite green
    /// while the two disagree about what a secret is. So the crate is checked
    /// against one checked-in file.
    ///
    /// A shape that carries a `linear_engine_pattern` is compared against
    /// that rather than against `pattern`, because this engine leaves the PEM
    /// body unbounded and cannot compile the lookaheads the reference pattern
    /// writes elsewhere. Every other shape is compared against
    /// `pattern`, and the lookaheads dropped here are compared against
    /// [`SHAPE_CHECKS`]. Every deviation is written down in the file, so none
    /// of them can happen silently.
    #[test]
    fn the_rule_set_is_the_one_the_other_two_implementations_carry() {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/redaction/rules.json"))
                .expect("the shared rule set parses");
        let strings = |key: &str| -> Vec<String> {
            doc[key]
                .as_array()
                .expect("a list")
                .iter()
                .map(|v| v.as_str().expect("a string").to_string())
                .collect()
        };

        let mut unambiguous: Vec<String> =
            UNAMBIGUOUS_PARAMS.iter().map(|s| s.to_string()).collect();
        unambiguous.sort();
        assert_eq!(unambiguous, strings("query_params_unambiguous"));

        let mut query_string_only: Vec<String> = QUERY_STRING_ONLY_PARAMS
            .iter()
            .map(|s| s.to_string())
            .collect();
        query_string_only.sort();
        assert_eq!(query_string_only, strings("query_params_query_string_only"));

        assert_eq!(
            PARAM_DELIMITER,
            doc["param_delimiter"].as_str().expect("a string")
        );
        assert_eq!(
            QUERY_STRING_DELIMITER,
            doc["query_string_delimiter"].as_str().expect("a string")
        );
        assert_eq!(PARAM_VALUE, doc["param_value"].as_str().expect("a string"));
        assert_eq!(
            SPACE_CHARS,
            doc["space_chars"].as_str().expect("a string"),
            "the space class is spelled as the rule file spells it"
        );
        assert_eq!(
            URL_IN_TEXT,
            doc["url_in_text"].as_str().expect("a string"),
            "the URL run stops where the rule file says it stops"
        );
        assert_eq!(
            URL_USERINFO,
            doc["url_userinfo"].as_str().expect("a string"),
            "the URL userinfo run stops where the rule file says it stops"
        );
        assert_eq!(
            QUERY_SEPARATOR,
            doc["query_separator"].as_str().expect("a string")
        );

        let mut headers: Vec<String> = SENSITIVE_HEADERS.iter().map(|s| s.to_string()).collect();
        headers.sort();
        assert_eq!(headers, strings("headers"));

        assert_eq!(
            HEADER_BOUNDARY,
            doc["header_boundary"].as_str().expect("a string")
        );
        assert_eq!(
            HEADER_VALUE_TERMINATORS,
            doc["header_value_terminators"].as_str().expect("a string")
        );
        assert_eq!(
            HEADER_NAME_SEPARATOR,
            doc["header_name_separator"].as_str().expect("a string"),
            "a header name is read behind an escaped quote"
        );
        let mut scheme_headers: Vec<String> =
            SCHEME_HEADERS.iter().map(|s| s.to_string()).collect();
        scheme_headers.sort();
        assert_eq!(scheme_headers, strings("scheme_headers"));
        let mut cookie_headers: Vec<String> =
            COOKIE_HEADERS.iter().map(|s| s.to_string()).collect();
        cookie_headers.sort();
        assert_eq!(cookie_headers, strings("cookie_headers"));

        assert_eq!(
            SHAPE_BOUNDARY,
            doc["shape_boundary"].as_str().expect("a string")
        );

        // The literal grammar: the class an unquoted value is made of, what
        // ends it, and what is a reference rather than a secret.
        assert_eq!(
            LITERAL_VALUE_CHARS,
            doc["literal_value_chars"].as_str().expect("a string")
        );
        assert_eq!(
            LITERAL_TERMINATORS,
            doc["literal_terminators"].as_str().expect("a string")
        );
        let mut placeholders: Vec<String> =
            PLACEHOLDER_VALUES.iter().map(|s| s.to_string()).collect();
        placeholders.sort();
        assert_eq!(placeholders, strings("placeholder_values"));
        let prefixes: Vec<String> = PLACEHOLDER_PREFIXES.iter().map(|s| s.to_string()).collect();
        assert_eq!(prefixes, strings("placeholder_prefixes"));

        // The name forms outside a query string.
        assert_eq!(
            NAME_VALUE_SEPARATOR,
            doc["name_value_separator"].as_str().expect("a string")
        );
        assert_eq!(
            NAME_EQUALS_SEPARATOR,
            doc["name_equals_separator"].as_str().expect("a string"),
            "the bare environment names read under `=` and nothing else"
        );
        assert_eq!(
            UNAMBIGUOUS_QUERY_DELIMITER,
            doc["unambiguous_query_delimiter"]
                .as_str()
                .expect("a string")
        );
        let suffixes: Vec<String> = ENV_NAME_SUFFIXES.iter().map(|s| s.to_string()).collect();
        assert_eq!(suffixes, strings("env_name_suffixes"));
        let bare: Vec<String> = ENV_BARE_NAMES.iter().map(|s| s.to_string()).collect();
        assert_eq!(bare, strings("env_bare_names"));
        assert_eq!(
            USER_PASSWORD_FLAG,
            doc["user_password_flag"].as_str().expect("a string")
        );

        // The URL readings.
        assert_eq!(URL_IN_TEXT, doc["url_in_text"].as_str().expect("a string"));
        assert_eq!(
            URL_USERINFO,
            doc["url_userinfo"].as_str().expect("a string")
        );
        assert_eq!(
            URL_TRAILING_PUNCTUATION.iter().collect::<String>(),
            doc["url_trailing_punctuation"].as_str().expect("a string")
        );

        // Which shapes deviate from the SDK pattern, and why, is written
        // down: the PEM body for the counted bound, the rest because their
        // SDK pattern carries a lookahead this engine cannot compile. A
        // deviation added to the file without a word said here fails right
        // there.
        let deviating: Vec<&str> = doc["shapes"]
            .as_array()
            .expect("a list")
            .iter()
            .filter(|shape| shape.get("linear_engine_pattern").is_some())
            .map(|shape| shape["name"].as_str().expect("a name"))
            .collect();
        assert_eq!(
            deviating,
            [
                "pem-private-key",
                "openai-key",
                "aws-access-key-id",
                "gitlab-token",
                "npm-token",
            ]
        );

        // Every lookahead the reference patterns write is a check this module
        // applies in code, and the rule set names them shape by shape.
        let expected_checks: Vec<(String, Vec<String>)> = doc["shapes"]
            .as_array()
            .expect("a list")
            .iter()
            .filter_map(|shape| {
                let checks = shape.get("checks")?;
                Some((
                    shape["name"].as_str().expect("a name").to_string(),
                    checks
                        .as_array()
                        .expect("a list")
                        .iter()
                        .map(|v| v.as_str().expect("a string").to_string())
                        .collect(),
                ))
            })
            .collect();
        let actual_checks: Vec<(String, Vec<String>)> = SHAPE_CHECKS
            .iter()
            .map(|(name, checks)| {
                (
                    name.to_string(),
                    checks.iter().map(|c| c.to_string()).collect(),
                )
            })
            .collect();
        assert_eq!(actual_checks, expected_checks);

        // Order is part of the rule set: PEM before JWT before bearer.
        let expected: Vec<(String, String)> = doc["shapes"]
            .as_array()
            .expect("a list")
            .iter()
            .map(|shape| {
                let pattern = shape
                    .get("linear_engine_pattern")
                    .unwrap_or_else(|| &shape["pattern"]);
                (
                    shape["name"].as_str().expect("a name").to_string(),
                    pattern.as_str().expect("a pattern").to_string(),
                )
            })
            .collect();
        let actual: Vec<(String, String)> = TOKEN_SHAPES
            .iter()
            .map(|(name, pattern)| (name.to_string(), pattern.to_string()))
            .collect();
        assert_eq!(actual, expected);
    }

    /// The literal class is pinned as a character class in the rule set and
    /// read here character by character, so the two readings are compared.
    /// Without this, [`is_literal_value_char`] could drift from
    /// [`LITERAL_VALUE_CHARS`] and the file would still say they agree.
    #[test]
    fn the_literal_value_class_is_the_one_the_rule_set_names() {
        let class = Regex::new(&format!("^[{LITERAL_VALUE_CHARS}]$")).expect("a valid class");
        for code in 0u32..0x2FF {
            let Some(character) = char::from_u32(code) else {
                continue;
            };
            assert_eq!(
                class.is_match(character.to_string().as_str()),
                is_literal_value_char(character),
                "{character:?}"
            );
        }
    }

    /// Requirement: byte-identical output, not merely the same rule names.
    ///
    /// `tests/fixtures/redaction/rules.json` pins the names and the patterns;
    /// `tests/fixtures/redaction/corpus.json` pins what they do to a string. Every case is
    /// checked here, so a change to this crate that moves any output fails.
    #[test]
    fn the_shared_corpus_comes_out_as_the_file_pins_it() {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/redaction/corpus.json"
        ))
        .expect("the shared corpus parses");
        let cases = doc["cases"].as_array().expect("a list");
        // The file's own row count, so a row deleted rather than fixed fails
        // here instead of shrinking the contract in silence.
        assert!(cases.len() >= 303, "the corpus is {} cases", cases.len());
        // Every row is run before anything is asserted: a first failing row
        // that stopped the test hid how many others were failing with it.
        let mut divergences: Vec<String> = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            let input = case["input"].as_str().expect("a string");
            let expected = case["expected"].as_str().expect("a string");
            let count = case["count"].as_u64().expect("a count") as usize;
            let actual = redact_text(input);
            if actual != (expected.to_string(), count) {
                divergences.push(format!(
                    "row {index}\n  input:    {input:?}\n  expected: {expected:?} ({count})\n  \
                     actual:   {:?} ({})",
                    actual.0, actual.1
                ));
            }
        }
        assert!(
            divergences.is_empty(),
            "{} of {} corpus rows differ:\n{}",
            divergences.len(),
            cases.len(),
            divergences.join("\n")
        );
    }

    /// No corpus input is pinned twice.
    ///
    /// The corpus is the contract an implementation is checked against, and a
    /// row added a second time with a different expected output would make it
    /// unsatisfiable: which copy wins depends on which was read last. A
    /// duplicate with the SAME output is dead weight.
    #[test]
    fn no_corpus_input_is_pinned_twice() {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/redaction/corpus.json"
        ))
        .expect("the shared corpus parses");
        let cases = doc["cases"].as_array().expect("a list");
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = Vec::new();
        for case in cases {
            let input = case["input"].as_str().expect("a string");
            if !seen.insert(input) {
                duplicates.push(input);
            }
        }
        assert!(duplicates.is_empty(), "pinned twice: {duplicates:?}");
    }

    /// The corpus rows the JSON invariant below does not hold for, named one
    /// by one so the shape is a stated limit rather than a silent gap. A JSON
    /// array element that ends on a credential name, standing next to an
    /// element that is not a string, puts `, 1, ` or `, true, ` between the
    /// string's own closing quote and the next one; that span carries a digit
    /// or letters, so the reader takes it for the value and writes the marker
    /// over it. An object is safe, because a quoted key always puts a bare
    /// `, ` there and punctuation alone is not a credential. This is the
    /// same-line splice the docs list as a residual, and closing it needs a
    /// JSON reader rather than a rule about characters.
    const JSON_INVARIANT_LIMITS: &[&str] = &[r#"{"lines": ["password: ", 1, "host: db"]}"#];

    /// Requirement: a row that arrives as JSON leaves as JSON.
    ///
    /// A tool call reaches the server as a JSON blob, and `@json_get_str` in
    /// the policy functors stops at the first unescaped quote: a marker that
    /// added a quote inside a JSON string left the blob unparseable and blind
    /// the policy that reads it. This holds the whole corpus to it rather
    /// than the handful of rows a single case would name.
    #[test]
    fn a_corpus_row_that_is_json_going_in_is_json_coming_out() {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/redaction/corpus.json"
        ))
        .expect("the shared corpus parses");
        let cases = doc["cases"].as_array().expect("a list");
        let mut checked = 0usize;
        let mut inner_checked = 0usize;
        for case in cases {
            let input = case["input"].as_str().expect("a string");
            if JSON_INVARIANT_LIMITS.contains(&input) {
                continue;
            }
            if serde_json::from_str::<serde_json::Value>(input).is_err() {
                continue;
            }
            checked += 1;
            let (scrubbed, _) = redact_text(input);
            let recleaned =
                serde_json::from_str::<serde_json::Value>(&scrubbed).unwrap_or_else(|error| {
                    panic!("{input:?} scrubbed to {scrubbed:?}, which no longer parses: {error}")
                });
            // One level down as well. A tool result quoting an HTTP body
            // back, and a span attribute holding a JSON blob that itself
            // holds JSON, reach this code double-encoded: the outer blob can
            // stay parseable while the marker has spliced the inner one
            // apart, and an invariant that stopped at the top level could not
            // see it.
            let parsed = serde_json::from_str::<serde_json::Value>(input).expect("a document");
            let mut paths = Vec::new();
            nested_json_blobs(&parsed, &mut Vec::new(), &mut paths);
            for path in &paths {
                inner_checked += 1;
                let written = at_path(&recleaned, path)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| panic!("{input:?}: no string at {path:?}"));
                serde_json::from_str::<serde_json::Value>(written).unwrap_or_else(|error| {
                    panic!(
                        "{input:?}: the blob at {path:?} no longer parses: \
                         {written:?}: {error}"
                    )
                });
            }
        }
        assert!(checked >= 20, "only {checked} corpus rows are JSON");
        assert!(
            inner_checked >= 4,
            "only {inner_checked} corpus rows carry a nested blob"
        );
    }

    /// One step of a path into a JSON document.
    #[derive(Debug)]
    enum Step {
        Key(String),
        Index(usize),
    }

    /// Where inside `value` a string is itself a JSON document.
    fn nested_json_blobs(
        value: &serde_json::Value,
        path: &mut Vec<Step>,
        found: &mut Vec<Vec<Step>>,
    ) {
        match value {
            serde_json::Value::String(text)
                if matches!(text.trim_start().as_bytes().first(), Some(b'{' | b'['))
                    && serde_json::from_str::<serde_json::Value>(text).is_ok() =>
            {
                found.push(
                    path.iter()
                        .map(|step| match step {
                            Step::Key(key) => Step::Key(key.clone()),
                            Step::Index(index) => Step::Index(*index),
                        })
                        .collect(),
                );
            }
            serde_json::Value::Object(members) => {
                for (key, item) in members {
                    path.push(Step::Key(key.clone()));
                    nested_json_blobs(item, path, found);
                    path.pop();
                }
            }
            serde_json::Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(Step::Index(index));
                    nested_json_blobs(item, path, found);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn at_path<'a>(value: &'a serde_json::Value, path: &[Step]) -> Option<&'a serde_json::Value> {
        let mut here = value;
        for step in path {
            here = match step {
                Step::Key(key) => here.get(key.as_str())?,
                Step::Index(index) => here.get(*index)?,
            };
        }
        Some(here)
    }

    #[test]
    fn a_token_of_each_known_shape_is_replaced_whole() {
        for (secret, shape) in [
            (OPENAI_KEY, "openai-key"),
            (ANTHROPIC_KEY, "openai-key"),
            (GOOGLE_KEY, "google-api-key"),
            (AWS_KEY_ID, "aws-access-key-id"),
            (AWS_TEMP_KEY_ID, "aws-access-key-id"),
            (GITHUB_CLASSIC, "github-token"),
            (GITHUB_OAUTH, "github-token"),
            (GITHUB_PAT, "github-pat"),
            (GITLAB_TOKEN, "gitlab-token"),
            (SLACK_TOKEN, "slack-token"),
            (NPM_TOKEN, "npm-token"),
            (PYPI_TOKEN, "pypi-token"),
            (JWT, "jwt"),
            (PEM_BLOCK, "pem-private-key"),
        ] {
            let cleaned = scrub(&format!("the call failed with {secret} attached"));
            assert!(!cleaned.contains(secret), "{shape}: secret survived");
            assert!(
                cleaned.contains(&format!("[redacted:{shape}]")),
                "{shape}: got {cleaned}"
            );
            assert!(
                !cleaned.contains(&secret[..12]),
                "{shape}: a prefix survived in {cleaned}"
            );
        }
    }

    #[test]
    fn a_bearer_token_is_replaced_but_the_word_bearer_stays() {
        let cleaned = scrub("sent Bearer zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz upstream");
        assert!(!cleaned.contains("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(cleaned.contains("Bearer [redacted:"), "got {cleaned}");
    }

    #[test]
    fn the_url_in_an_http_error_message_loses_its_key() {
        let message = format!(
            "Client error '400 Bad Request' for url \
             'https://generativelanguage.googleapis.com/v1beta/models/\
             gemini-2.0-flash:generateContent?key={GOOGLE_KEY}'"
        );
        let cleaned = scrub(&message);
        assert!(!cleaned.contains(GOOGLE_KEY));
        assert!(!cleaned.contains("AIza"));
        assert!(cleaned.contains("?key=[redacted]"), "got {cleaned}");
        assert!(cleaned.contains("generativelanguage.googleapis.com/v1beta/models"));
    }

    #[test]
    fn every_credential_bearing_query_parameter_loses_its_value() {
        for param in UNAMBIGUOUS_PARAMS.iter().chain(QUERY_STRING_ONLY_PARAMS) {
            let (cleaned, count) =
                redact_url(&format!("https://host/path?{param}=supersecret&page=2"));
            assert_eq!(
                cleaned,
                format!("https://host/path?{param}=[redacted]&page=2")
            );
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn a_query_parameter_name_is_matched_after_percent_decoding_and_case_folding() {
        let (cleaned, count) = redact_url("https://host/p?API%5FKEY=v&Api-Key=w&other=keep");
        assert_eq!(
            cleaned,
            "https://host/p?API%5FKEY=[redacted]&Api-Key=[redacted]&other=keep"
        );
        assert_eq!(count, 2);
    }

    #[test]
    fn a_url_keeps_its_host_path_and_fragment() {
        let (cleaned, _) = redact_url("https://h.example/a/b?token=abc&q=hi#frag");
        assert_eq!(cleaned, "https://h.example/a/b?token=[redacted]&q=hi#frag");
    }

    #[test]
    fn a_url_is_recognised_whatever_the_case_of_its_scheme() {
        let cleaned = scrub(&format!("redirected to HTTPS://host/v1?key={GOOGLE_KEY}"));
        assert!(!cleaned.contains(GOOGLE_KEY), "{cleaned}");
        assert!(cleaned.contains("?key=[redacted]"), "{cleaned}");
    }

    /// A span records `url.query` on its own, and an error message quotes the
    /// part of the request it failed on. Neither carries a scheme, so the
    /// query parameter has to be judged by its name where it stands.
    #[test]
    fn a_query_parameter_with_no_scheme_in_front_of_it_still_loses_its_value() {
        let cleaned = scrub("failed for generativelanguage.googleapis.com/v1?key=supersecretvalue");
        assert!(!cleaned.contains("supersecretvalue"), "{cleaned}");
        assert!(cleaned.contains("?key=[redacted]"), "{cleaned}");

        let (attributes, count) =
            redact_text(r#"{"url.query": "page=2&access_token=xyz&code=abc"}"#);
        assert!(!attributes.contains("xyz"), "{attributes}");
        assert!(!attributes.contains("abc"), "{attributes}");
        assert!(attributes.contains("page=2"), "{attributes}");
        assert_eq!(count, 2);
    }

    /// An unambiguous name needs no query string around it. OpenTelemetry
    /// defines `url.query` as the query component without its leading `?`, so
    /// the credential is often the very first thing in the field, with
    /// nothing but a quote in front of it — and a recorded command puts one
    /// straight after a quote too.
    ///
    /// The six ambiguous names are the other way round: they need a `?`, an
    /// `&`, an `&amp;` or a `;` in front of them, or ordinary content would
    /// be rewritten. See
    /// `ordinary_key_value_content_is_left_byte_identical`.
    #[test]
    fn an_unambiguous_parameter_loses_its_value_wherever_it_stands() {
        let (attributes, count) =
            redact_text(r#"{"url.path": "/v1/models", "url.query": "password=0paquesecret"}"#);
        assert!(!attributes.contains("0paquesecret"), "{attributes}");
        assert!(
            attributes.contains(r#""url.query": "password=[redacted]""#),
            "{attributes}"
        );
        assert!(
            attributes.contains(r#""url.path": "/v1/models""#),
            "{attributes}"
        );
        assert_eq!(count, 1);

        // The same parameter first rather than second in a recorded command.
        let (first, _) = redact_text(r#"curl -d "access_token=0paquesecret&user=x" https://h/t"#);
        assert!(!first.contains("0paquesecret"), "{first}");
        assert!(first.contains("access_token=[redacted]&user=x"), "{first}");
    }

    /// Content the agent read or wrote must come back as it went in.
    ///
    /// These are the strings a coding agent hands the recorder all day: a
    /// line of Python, a config snippet, a comparison. Each one contains a
    /// name from the ambiguous tier, and none of them is a credential —
    /// rewriting them corrupts the record, and a corrupted record is a policy
    /// reading a command that was never run.
    #[test]
    fn ordinary_key_value_content_is_left_byte_identical() {
        for kept in [
            "session=requests.Session()",
            "if code==200",
            "key=value\nname=thing\ntimeout=30",
            r#"token = "abc""#,
        ] {
            assert_eq!(scrub(kept), kept);
        }
    }

    /// A long CLI flag, a JVM property, an HTML-escaped separator, the legacy
    /// one. Each of these carries a real credential and none of them was
    /// reached by the old single delimiter class. What goes is the value and
    /// nothing else.
    #[test]
    fn a_credential_under_an_unambiguous_name_loses_exactly_its_value() {
        assert_eq!(scrub("--password=hunter2"), "--password=[redacted]");
        assert_eq!(scrub("-Dpassword=hunter2"), "-Dpassword=[redacted]");
        assert_eq!(scrub("&amp;key=SUPERSECRETVALUE1"), "&amp;key=[redacted]");
        assert_eq!(scrub("x;key=SUPERSECRETVALUE1"), "x;key=[redacted]");
    }

    /// Tool arguments arrive JSON-encoded, and a policy reads them back.
    ///
    /// A value that ran onto the backslash escaping the closing quote took
    /// the escape with it: the blob stopped parsing, and `@json_get_str` —
    /// which stops at the first unescaped quote — handed the policy a command
    /// with its destination gone. That is a fail-open, caused by the
    /// scrubber.
    #[test]
    fn scrubbing_a_json_encoded_command_leaves_valid_json_and_its_url() {
        let raw = serde_json::json!({"command": r#"curl -d "code=abc" https://h/"#}).to_string();
        let cleaned = scrub(&raw);
        let parsed: serde_json::Value =
            serde_json::from_str(&cleaned).expect("the scrubbed blob is still JSON");
        assert!(
            parsed["command"]
                .as_str()
                .expect("a string")
                .ends_with("https://h/"),
            "{cleaned}"
        );

        // The same command under an unambiguous name: the value goes, the
        // escape, the quote and the URL stay.
        let secret =
            serde_json::json!({"command": r#"curl -d "password=abc123" https://h/"#}).to_string();
        let scrubbed = scrub(&secret);
        let parsed: serde_json::Value =
            serde_json::from_str(&scrubbed).expect("the scrubbed blob is still JSON");
        assert_eq!(
            parsed["command"].as_str().expect("a string"),
            r#"curl -d "password=[redacted]" https://h/"#
        );
    }

    /// The name has to start where the run starts, not inside a word.
    #[test]
    fn a_word_ending_in_a_parameter_name_is_not_a_parameter() {
        for kept in ["mycode=42", "the_token=abc", "brisk-key=value"] {
            assert_eq!(scrub(kept), kept);
        }
    }

    /// A pre-signed URL carries its credential in a vendor-prefixed field,
    /// so the last hyphen-separated segment of the name is what is read.
    #[test]
    fn a_presigned_url_loses_its_signature_and_keeps_the_rest() {
        let (cleaned, count) = redact_text(
            "https://h/o?X-Amz-Signature=abc123deadbeef&X-Amz-Date=2026&X-Amz-Expires=60",
        );
        assert_eq!(
            cleaned,
            "https://h/o?X-Amz-Signature=[redacted]&X-Amz-Date=2026&X-Amz-Expires=60"
        );
        assert_eq!(count, 1);

        let (google, count) = redact_text("https://h/o?X-Goog-Signature=abc123deadbeef");
        assert_eq!(google, "https://h/o?X-Goog-Signature=[redacted]");
        assert_eq!(count, 1);

        // The segment has to be the whole of the last one: a field named for
        // something else that merely ends in those letters is content.
        let (kept, count) = redact_text("https://h/o?design=abc123&resig=abc123");
        assert_eq!(kept, "https://h/o?design=abc123&resig=abc123");
        assert_eq!(count, 0);
    }

    /// The bare-parameter pass runs after the URL pass, so a parameter both
    /// can see must be replaced once, not counted twice.
    #[test]
    fn a_parameter_inside_a_url_is_not_redacted_twice() {
        let (cleaned, count) = redact_text("failed for https://h/v1?key=supersecretvalue&page=2");
        assert_eq!(cleaned, "failed for https://h/v1?key=[redacted]&page=2");
        assert_eq!(count, 1);
        assert_eq!(redact_text(&cleaned).0, cleaned);
    }

    /// A credential name at the end of a quoted string reads no value: what
    /// stands between the quotes — `, `, `}, `, `], ` — carries no letter and
    /// no digit, so no credential is there. A closer holding letters, as in
    /// `echo "password: " >> notes.txt`, is still read as an opener; that
    /// residual is a stated limit.
    #[test]
    fn a_credential_name_at_the_end_of_a_quoted_string_reads_no_value() {
        for line in [
            r#"{"stdout": "Enter password: ", "exit_code": 1}"#,
            r#"{"content": "DB_PASSWORD=", "path": ".env"}"#,
            r#"{"lines": ["password: ", "host: db"]}"#,
        ] {
            assert_eq!(redact_text(line).0, line, "{line}");
        }
        // A string that goes on past the name still holds a value, and a
        // quote of the other kind is still an opener.
        for (line, expected) in [
            (
                r#"{"stdout": "Enter password: hunter2hunter2", "exit_code": 0}"#,
                r#"{"stdout": "Enter password: [redacted]", "exit_code": 0}"#,
            ),
            (
                r#"{"cmd": "mysql --password='hunter2hunter2'"}"#,
                r#"{"cmd": "mysql --password='[redacted]'"}"#,
            ),
        ] {
            assert_eq!(redact_text(line).0, expected, "{line}");
        }
    }

    /// Spring Boot's own spelling, and four more like it.
    ///
    /// [`PARAM_DELIMITER`] excludes the `.` from what may stand before a name,
    /// so every dotted spelling was read by no rule and stored as written. One
    /// shape closes it: the `=` glued to the name. The exclusion still holds
    /// for the spaced assignment, which is what keeps ordinary source unread,
    /// and the colon form of a dotted name goes with it; both are stated
    /// limits.
    #[test]
    fn a_dotted_credential_name_is_read_under_a_glued_equals() {
        for (taken, wanted) in [
            (
                "spring.datasource.password=s3cr3tpassword",
                "spring.datasource.password=[redacted]",
            ),
            (
                "helm install app --set postgres.password=hunter2hunter2",
                "helm install app --set postgres.password=[redacted]",
            ),
            (
                "-Dspring.datasource.password=hunter2hunter2",
                "-Dspring.datasource.password=[redacted]",
            ),
            (
                "mail.smtp.password=s3cr3tmail1",
                "mail.smtp.password=[redacted]",
            ),
            ("db.password=hunter2hunter2", "db.password=[redacted]"),
        ] {
            assert_eq!(redact_text(taken).0, wanted, "{taken}");
        }
        for kept in [
            "self.password = get_pw()",
            "config.api_key = settings.API_KEY",
            "self.password = other_password",
            "spring.datasource.password: hunter2hunter2",
        ] {
            assert_eq!(redact_text(kept).0, kept, "{kept}");
        }
    }

    /// A secret is written on one line; a value that spans two is not one.
    ///
    /// The reading holds at both encoding levels: a raw CR or LF ends the
    /// read, and in double-encoded text the two-character `\n` and `\r` end it
    /// too. A quote with no closer before the break opens no value, so a shell
    /// script that prompts on one line and reads the password on the next
    /// comes back byte-for-byte instead of losing the lines between.
    #[test]
    fn a_quoted_value_never_crosses_a_line_break() {
        for kept in [
            r#"{"content": "{\"stdout\": \"Enter password: \", \"exit_code\": 1}"}"#,
            concat!(
                r#"{"content": "echo -n \"Enter password: \"\nread -s PASSWORD"#,
                r#"\necho \"You entered $PASSWORD\"", "path": "a.sh"}"#
            ),
            "echo -n \"Enter password: \"\nread -s PASSWORD\necho \"You entered $PASSWORD\"",
            "password=\"unclosed\nhost: db",
        ] {
            assert_eq!(redact_text(kept).0, kept, "{kept}");
        }
        assert_eq!(read_value("\"abc\rrest", 0), (0, ValueKind::Nothing));
        assert_eq!(read_value(r#"\"one\nnext\""#, 0), (0, ValueKind::Nothing));
        // The raw-quote half of the same rule: a backslash escapes the next
        // character, but not a raw line break — the read ends at the break,
        // so the quote on the next line is not its closer.
        assert_eq!(read_value("\"abc\\\n\" x", 0), (0, ValueKind::Nothing));
    }

    /// A secret carries an ASCII letter or digit; `", "` does not.
    ///
    /// This is what stops the reader splicing over the `, ` between two JSON
    /// members when a credential name ends a string, and it is local to the
    /// read: no scan of what stands earlier on the line, so a line whose
    /// prefix holds an apostrophe or a stray quote still redacts.
    #[test]
    fn a_quoted_value_of_pure_punctuation_is_not_a_credential() {
        assert_eq!(read_value(r#"", "rest"#, 0), (0, ValueKind::Nothing));
        assert_eq!(read_value(r#""a"rest"#, 0), (3, ValueKind::Quoted));
        for (taken, wanted) in [
            (
                "# don't commit this: password='hunter2hunter2'",
                "# don't commit this: password='[redacted]'",
            ),
            (
                "app.py:12:# don't use in prod: api_key='abc123def456'",
                "app.py:12:# don't use in prod: api_key='[redacted]'",
            ),
            (
                r#"He said "hi, password="hunter2hunter2""#,
                r#"He said "hi, password="[redacted]""#,
            ),
            (
                "-- user's note: password='hunter2hunter2'",
                "-- user's note: password='[redacted]'",
            ),
            (
                r#"{"content": "{\"stdout\": \"Enter password: hunter2hunter2\"}"}"#,
                r#"{"content": "{\"stdout\": \"Enter password: [redacted]\"}"}"#,
            ),
        ] {
            assert_eq!(redact_text(taken).0, wanted, "{taken}");
        }
    }

    /// A lone word after a header name is a word, not a credential: no digit
    /// anywhere, and nothing outside letters, `-`, `/`, `.` and `_`.
    #[test]
    fn a_lone_word_after_a_header_name_is_kept() {
        for line in [
            "x-api-key: application/json",
            "x-api-key: not-set",
            "authorization: role-based",
            "x-goog-api-key: user-provided",
            "x-api-key: N/A",
        ] {
            assert_eq!(redact_text(line).0, line, "{line}");
        }
        for line in [
            "x-api-key: abc123def456",
            "x-api-key: sk_live_ex4mple0key0for0tests00",
            "x-api-key: 12345",
            "authorization: Basic dXNlcjpwYXNz",
        ] {
            assert!(
                redact_text(line).0.contains(REDACTED),
                "{line} keeps its value"
            );
        }
    }

    /// A cookie is a cookie string: it has to carry an `=` to be one. That is
    /// what leaves "We store a cookie: the session identifier" alone, so the
    /// two cookie headers are given a value of the shape a cookie has.
    ///
    /// The synthetic value carries a digit, because a digit-free value made
    /// of letters, `-`, `/`, `.` and `_` after a header name is a word by
    /// [`is_header_word`] and is kept by design.
    #[test]
    fn a_credential_header_line_in_free_text_loses_its_value() {
        for header in SENSITIVE_HEADERS {
            let value = if header.ends_with("cookie") {
                "sid=some-credential-material-7f3a"
            } else {
                "some-credential-material-7f3a"
            };
            let cleaned = scrub(&format!("request headers:\n{header}: {value}\naccept: */*"));
            assert!(
                !cleaned.contains("some-credential-material-7f3a"),
                "{header}"
            );
            assert!(
                cleaned.contains(&format!("{header}: [redacted]")),
                "{header}"
            );
            assert!(cleaned.contains("accept: */*"), "{header}");
        }
    }

    /// The PEM body is bounded, and the bound clears a real key: an
    /// 8192-bit RSA private key armours to roughly 6.5 KB of base64, and the
    /// pattern allows 8000 characters between the BEGIN and END lines.
    #[test]
    fn a_private_key_block_is_redacted_at_a_realistic_key_size() {
        let body = vec![format!("M{}", "I".repeat(63)); 100].join("\n"); // 6.4 KB
        let block =
            format!("-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----");
        let cleaned = scrub(&format!("the deploy key is\n{block}\nand that is all"));
        assert!(cleaned.contains("[redacted:pem-private-key]"), "{cleaned}");
        assert!(!cleaned.contains("MIIIII"));
    }

    /// A `curl -H` line is how a credential usually reaches this redactor
    /// from a coding agent. The header stands in the middle of the command,
    /// inside a quoted argument, so neither the line-anchored pass nor the
    /// quoted-pair pass can see it. The command around it — the URL included
    /// — is left intact: a command with its destination cut off is a broken
    /// record, and a policy that reads the destination out of it fails open.
    #[test]
    fn a_credential_header_inside_a_recorded_shell_command_loses_its_value() {
        for (header, secret) in [
            ("Authorization", "Basic dXNlcjpwYXNzd29yZA=="),
            ("x-api-key", "OPAQUESECRETVALUE123"),
        ] {
            let cleaned = scrub(&format!(
                r#"curl -H "{header}: {secret}" https://api.example/x"#
            ));
            assert!(!cleaned.contains(secret), "{cleaned}");
            assert_eq!(
                cleaned,
                format!(r#"curl -H "{header}: [redacted]" https://api.example/x"#)
            );
        }
    }

    /// Nothing says a header line starts at the start of the line: a verbose
    /// curl trace prefixes it with `> `, a log line with a timestamp.
    #[test]
    fn a_credential_header_on_a_prefixed_log_line_loses_its_value() {
        assert_eq!(
            scrub("> Authorization: Basic dXNlcjpwYXNzd29yZA=="),
            "> Authorization: [redacted]"
        );
        assert_eq!(
            scrub("2026-01-01 DEBUG authorization: Basic dXNlcjpwYXNzd29yZA=="),
            "2026-01-01 DEBUG authorization: [redacted]"
        );
    }

    /// Tool arguments arrive JSON-encoded and must stay parseable. The value
    /// ends at the backslash escaping the closing quote, so the escape, the
    /// quote and the URL after them all survive.
    #[test]
    fn a_header_inside_json_encoded_text_leaves_the_json_valid() {
        let raw = serde_json::json!({
            "command": r#"curl -H "Authorization: Basic ABCSECRET" https://h/"#
        })
        .to_string();
        let cleaned = scrub(&raw);
        assert!(!cleaned.contains("ABCSECRET"), "{cleaned}");
        let parsed: serde_json::Value =
            serde_json::from_str(&cleaned).expect("the scrubbed blob is still JSON");
        assert_eq!(
            parsed["command"].as_str().expect("a string"),
            r#"curl -H "Authorization: [redacted]" https://h/"#
        );
        assert_eq!(scrub(&cleaned), cleaned);
    }

    /// A header does not always arrive as a line. OpenTelemetry records
    /// `http.request.header.authorization` as a span attribute, and the whole
    /// attribute map reaches the store as one JSON string.
    #[test]
    fn a_credential_header_in_a_json_attribute_blob_loses_its_value() {
        let attributes =
            r#"{"http.request.header.authorization": "supersecretvalue", "http.method": "POST"}"#;
        let (cleaned, count) = redact_text(attributes);
        assert!(!cleaned.contains("supersecretvalue"), "{cleaned}");
        assert!(
            cleaned.contains(r#""http.request.header.authorization": "[redacted]""#),
            "{cleaned}"
        );
        assert!(cleaned.contains(r#""http.method": "POST""#), "{cleaned}");
        assert_eq!(count, 1);
        // A Python `repr` of a headers dictionary reads the same way.
        let (repr_cleaned, repr_count) = redact_text("{'x-api-key': 'supersecretvalue'}");
        assert_eq!(repr_cleaned, "{'x-api-key': '[redacted]'}");
        assert_eq!(repr_count, 1);
    }

    /// OpenTelemetry records `http.request.header.<key>` as a string array,
    /// so the serialisation that actually arrives is a one-element list, not
    /// a bare string.
    #[test]
    fn a_credential_header_recorded_as_an_array_loses_its_values() {
        let (attributes, count) =
            redact_text(r#"{"http.request.header.authorization": ["Bearer opaquesecret"]}"#);
        assert!(!attributes.contains("opaquesecret"), "{attributes}");
        assert_eq!(
            attributes,
            r#"{"http.request.header.authorization": ["[redacted]"]}"#
        );
        assert_eq!(count, 1);

        // Every element goes, not just the first.
        let (two, _) = redact_text(r#"{"http.request.header.cookie": ["a=1", "b=2"]}"#);
        assert!(!two.contains("a=1"), "{two}");
        assert!(!two.contains("b=2"), "{two}");
        assert_eq!(redact_text(&two).0, two);
    }

    /// A quoted cookie value (RFC 6265) is one value, not three. Reading the
    /// value as "everything up to the next quote" ends the match inside the
    /// credential: the tail survives in the store, and the blob it sits in
    /// stops being valid JSON.
    #[test]
    fn a_header_value_containing_a_quote_is_redacted_whole() {
        let (attributes, count) =
            redact_text(r#"{"http.request.header.set-cookie": "session=\"abc123secret\""}"#);
        assert!(!attributes.contains("abc123secret"), "{attributes}");
        assert_eq!(
            attributes,
            r#"{"http.request.header.set-cookie": "[redacted]"}"#
        );
        assert_eq!(count, 1);
    }

    /// The scrub runs on a request-handling thread, over text the client
    /// chose, and a batch may be as large as the gRPC decode limit. Hostile
    /// text — runs that open a shape and never complete it — must therefore
    /// cost a scan and not a search: four megabytes of it in under a second.
    #[test]
    fn megabytes_of_hostile_text_are_scrubbed_in_bounded_time() {
        let unclosed_pem = "-----BEGIN A PRIVATE KEY-----\n".repeat(60_000); // ~1.8 MB
        let repeated_jwt_prefix = format!("eyJ{}", "A".repeat(20)).repeat(80_000); // ~1.8 MB
        let start = std::time::Instant::now();
        assert_eq!(redact_text(&unclosed_pem).1, 0);
        assert_eq!(redact_text(&repeated_jwt_prefix).1, 0);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "3.6 MB of hostile text took {elapsed:?}"
        );
    }

    /// The name-based passes rewrite a value at a time, and a string that
    /// carries one credential per line carries tens of thousands of them.
    /// Copying the whole text once per replacement is quadratic, and this
    /// pass runs on a request-handling thread.
    ///
    /// The text is one line, and the quoted reading answers locally: it looks
    /// only at what stands between the quotes, so nothing here costs more than
    /// the text it reads.
    #[test]
    fn a_string_of_nothing_but_credentials_is_scrubbed_in_bounded_time() {
        let many = "password=hunter2 ".repeat(100_000); // ~1.7 MB, 100k values
        let start = std::time::Instant::now();
        let (cleaned, count) = redact_text(&many);
        let elapsed = start.elapsed();
        assert_eq!(count, 100_000);
        assert_eq!(cleaned, "password=[redacted] ".repeat(100_000));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "100k replacements took {elapsed:?}"
        );
    }

    /// A QUOTED header value, at the size a captured `curl -v` trace reaches.
    ///
    /// This is the branch that reads a value with the literal grammar, and a
    /// round that answered a question about the value by scanning the line in
    /// front of it made exactly this line quadratic — 19 s for a 200 KB trace
    /// in the SDK, on the thread it records spans on. The reading here answers
    /// locally, so the cost is the text's; a second is loose enough never to
    /// flake and tight enough to fail the moment a scan of the line comes
    /// back.
    #[test]
    fn a_quoted_header_line_repeated_over_a_captured_trace_costs_what_it_should() {
        let line = "authorization: \"abc123def456\"\n";
        let lines = 200_000 / line.len();
        let trace = line.repeat(lines);
        let start = std::time::Instant::now();
        let (cleaned, count) = redact_text(&trace);
        let elapsed = start.elapsed();
        assert_eq!(count, lines);
        assert_eq!(cleaned, "authorization: \"[redacted]\"\n".repeat(lines));
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "a 200 KB quoted-header trace took {elapsed:?}"
        );
    }

    #[test]
    fn scrubbing_already_scrubbed_text_changes_nothing() {
        let once = scrub(&format!(
            "{OPENAI_KEY} https://h/x?key={GOOGLE_KEY}\nAuthorization: {JWT}\n             {{\"cookie\": \"session=abc\"}}"
        ));
        assert_eq!(scrub(&once), once);
    }

    /// `-` and `_` keep a token run going, so without a left boundary any
    /// word ending in `sk` followed by sixteen more word characters reads as
    /// an OpenAI key — and the recorded path or command is rewritten before
    /// it is stored.
    #[test]
    fn a_hyphenated_word_is_not_mistaken_for_a_key() {
        for prose in [
            "risk-assessment-summary.md",
            "the risk-free-rate-calculation model",
            "run the task-force-recommendations report",
            "desk-side_support_ticket_00012 opened",
            "npm_install_helper_module_name",
            "glpat-is-not-a-token-here",
        ] {
            assert_eq!(scrub(prose), prose);
        }

        // The boundary is a boundary, not a required character.
        assert_eq!(scrub(OPENAI_KEY), "[redacted:openai-key]");
        assert_eq!(
            scrub(&format!("--model-key={OPENAI_KEY}")),
            "--model-key=[redacted:openai-key]"
        );
    }

    #[test]
    fn ordinary_prose_is_left_exactly_as_written() {
        let prose = "The user asked to book a flight from SFO to JFK. Cost: 412 USD.";
        let (cleaned, count) = redact_text(prose);
        assert_eq!(cleaned, prose);
        assert_eq!(count, 0);
    }

    #[test]
    fn event_content_is_scrubbed_and_event_identity_is_not() {
        let mut events = vec![Event {
            text: Some(format!("failed for url 'https://h/v1?key={GOOGLE_KEY}'")),
            agent: Some("assistant".into()),
            role: Some(2),
            id: Some("n1".into()),
            tools: vec![Tool {
                name: Some("fetch".into()),
                arguments: Some(format!("{{\"token\": \"{GITHUB_CLASSIC}\"}}")),
            }],
            derived_from: Some(Tool {
                name: Some("fetch".into()),
                arguments: Some(format!("{{\"k\": \"{OPENAI_KEY}\"}}")),
            }),
            principal: Some("p".into()),
            entity: Some("e".into()),
            metadata: Some(format!("{{\"key\": \"{GOOGLE_KEY}\"}}")),
        }];
        assert_eq!(redact_events(&mut events), 4);
        let event = &events[0];
        assert!(!event.text.as_deref().unwrap().contains(GOOGLE_KEY));
        assert!(!event.tools[0]
            .arguments
            .as_deref()
            .unwrap()
            .contains(GITHUB_CLASSIC));
        assert!(!event
            .derived_from
            .as_ref()
            .unwrap()
            .arguments
            .as_deref()
            .unwrap()
            .contains(OPENAI_KEY));
        assert!(!event.metadata.as_deref().unwrap().contains(GOOGLE_KEY));
        assert_eq!(event.id.as_deref(), Some("n1"));
        assert_eq!(event.agent.as_deref(), Some("assistant"));
        assert_eq!(event.role, Some(2));
        assert_eq!(event.entity.as_deref(), Some("e"));
        assert_eq!(event.principal.as_deref(), Some("p"));
    }

    /// An edge carries one free-form client string, and a free-form string
    /// is somewhere a secret can land. Its ids are structure and stay.
    #[test]
    fn an_edge_entity_is_scrubbed_and_the_edge_still_names_its_events() {
        let mut edges = vec![Edge {
            source: "n1".into(),
            destination: "n2".into(),
            message_index: Some(1),
            proximal: Some(true),
            principal: Some("p".into()),
            entity: Some(format!("tenant {OPENAI_KEY}")),
        }];
        assert_eq!(redact_edges(&mut edges), 1);
        assert!(!edges[0].entity.as_deref().unwrap().contains(OPENAI_KEY));
        assert_eq!(edges[0].source, "n1");
        assert_eq!(edges[0].destination, "n2");
        assert_eq!(edges[0].message_index, Some(1));
        assert_eq!(edges[0].principal.as_deref(), Some("p"));
    }

    /// A span carries the same free-form `entity` label an event and an edge
    /// carry, and the server does not overwrite it. Its ids are structure and
    /// stay.
    #[test]
    fn a_span_entity_is_scrubbed_and_the_span_still_names_its_trace() {
        let mut comps = vec![Computation {
            trace_id: "t1".into(),
            span_id: "s1".into(),
            name: "llm.call".into(),
            attributes_json: "{}".into(),
            events_json: "[]".into(),
            principal: Some("p".into()),
            entity: Some(format!("tenant {OPENAI_KEY}")),
            ..Default::default()
        }];
        assert_eq!(redact_computations(&mut comps), 1);
        let entity = comps[0].entity.as_deref().unwrap();
        assert!(!entity.contains(OPENAI_KEY), "{entity}");
        assert!(entity.contains("[redacted:openai-key]"), "{entity}");
        assert_eq!(comps[0].trace_id, "t1");
        assert_eq!(comps[0].span_id, "s1");
        assert_eq!(comps[0].name, "llm.call");
        assert_eq!(comps[0].principal.as_deref(), Some("p"));
    }

    #[test]
    fn a_span_status_message_is_scrubbed_before_it_is_stored() {
        let mut comps = vec![Computation {
            trace_id: "t1".into(),
            span_id: "s1".into(),
            name: "llm.call".into(),
            status_message: Some(format!(
                "HTTPStatusError for url 'https://h/v1?key={GOOGLE_KEY}'"
            )),
            attributes_json: format!("{{\"http.url\": \"https://h/v1?key={GOOGLE_KEY}\"}}"),
            events_json: "[]".into(),
            ..Default::default()
        }];
        assert_eq!(redact_computations(&mut comps), 2);
        assert!(!comps[0]
            .status_message
            .as_deref()
            .unwrap()
            .contains(GOOGLE_KEY));
        assert!(!comps[0].attributes_json.contains(GOOGLE_KEY));
        assert_eq!(comps[0].span_id, "s1");
        assert_eq!(comps[0].name, "llm.call");
    }

    /// A span's name and an event's agent are free-form client strings, and
    /// naming an HTTP client span after the request it made is a common
    /// convention — which puts the query string in the name.
    #[test]
    fn a_span_name_and_an_event_agent_are_scrubbed_like_any_other_client_string() {
        let mut comps = vec![Computation {
            name: format!("GET https://h/v1?key={GOOGLE_KEY}"),
            service_name: Some(format!("agent {OPENAI_KEY}")),
            attributes_json: "{}".into(),
            events_json: "[]".into(),
            ..Default::default()
        }];
        assert_eq!(redact_computations(&mut comps), 2);
        assert!(!comps[0].name.contains(GOOGLE_KEY), "{}", comps[0].name);
        assert!(
            comps[0].name.contains("?key=[redacted]"),
            "{}",
            comps[0].name
        );
        assert!(!comps[0]
            .service_name
            .as_deref()
            .unwrap()
            .contains(OPENAI_KEY));

        let mut events = vec![Event {
            agent: Some(format!("worker {GITHUB_CLASSIC}")),
            entity: Some(format!("tenant {AWS_KEY_ID}")),
            id: Some("n1".into()),
            ..Default::default()
        }];
        assert_eq!(redact_events(&mut events), 2);
        assert!(!events[0].agent.as_deref().unwrap().contains(GITHUB_CLASSIC));
        assert!(!events[0].entity.as_deref().unwrap().contains(AWS_KEY_ID));
        assert_eq!(events[0].id.as_deref(), Some("n1"));
    }

    /// Requirement: a name-based rule never leaves the text unparseable.
    ///
    /// Tool arguments reach the policy as JSON, and `@json_get_str` in the
    /// Soufflé functors stops at the first unescaped quote: a blob this pass
    /// broke blinds the policy on every field past the break. A bare number
    /// under a credential name is the one value whose marker has to be
    /// written as a JSON string for the blob to survive.
    #[test]
    fn a_numeric_json_value_is_replaced_by_a_marker_that_is_a_string() {
        let scrubbed = scrub(r#"{"password": 123456, "user": "app"}"#);
        assert_eq!(scrubbed, r#"{"password": "[redacted]", "user": "app"}"#);
        let parsed: serde_json::Value =
            serde_json::from_str(&scrubbed).expect("the scrubbed blob still parses");
        assert_eq!(parsed["password"], "[redacted]");
        assert_eq!(scrub(&scrubbed), scrubbed, "and the pass is idempotent");
    }

    /// Requirement: a separator never eats half of an operator.
    ///
    /// Every one of these is ordinary source. A pass that rewrote them would
    /// hand the policy a command or a comparison that was never written.
    #[test]
    fn a_two_character_operator_is_not_a_name_and_its_value() {
        for line in [
            "if password == \"admin\":",
            "if (password === userInput) {",
            "assert secret == expected_secret",
            "password := os.Getenv(\"DB_PASSWORD\")",
            "PASSWORD := hunter2",
        ] {
            assert_eq!(scrub(line), line);
        }
    }

    /// Requirement: a reference to a secret is not a secret.
    ///
    /// A dotted attribute path is a name for a credential, not one. The
    /// length clause took every one of these before the dot rule; the control
    /// carries no dot and still goes.
    #[test]
    fn an_attribute_path_is_a_reference_however_long_it_is() {
        for line in [
            "password = settings.DATABASE_PASSWORD",
            "password = config.database.password",
            "const password = process.env.DB_PASSWORD;",
            "password = self._resolve_password",
        ] {
            assert_eq!(scrub(line), line);
        }
        assert_eq!(
            scrub("password = correcthorsebatterystaple"),
            "password = [redacted]"
        );
    }

    /// Requirement: a header whose value is quoted loses the value and keeps
    /// the quotes — and the syntax around it.
    ///
    /// This is how a header is written in a YAML config, a JavaScript object
    /// literal and a TOML file. The bracketed line pins the closing bracket.
    #[test]
    fn a_quoted_header_value_goes_and_its_syntax_stays() {
        assert_eq!(
            scrub("  authorization: \"Basic ZGVwbG95Omh1bnRlcjI=\""),
            "  authorization: \"[redacted]\""
        );
        assert_eq!(
            scrub("axios.get(url, { headers: { authorization: \"Bearer OPAQUESECRET678\" } })"),
            "axios.get(url, { headers: { authorization: \"[redacted]\" } })"
        );
        assert_eq!(
            scrub("[authorization: Bearer abc123def456ghi789]"),
            "[authorization: [redacted]]"
        );
    }

    /// Requirement: a header value that is a single word has to be a
    /// credential by the literal grammar too.
    ///
    /// The single-token reading takes whatever word stands after the colon,
    /// so a type annotation, a reference and a placeholder all reached it.
    /// Only `<scheme> <token>` stays exempt: `Basic dXNlcjpwYXNz` is base64
    /// that may hold no digit at all.
    #[test]
    fn a_header_value_of_one_word_still_has_to_be_a_credential() {
        assert_eq!(
            scrub("def check(authorization: str):"),
            "def check(authorization: str):"
        );
        assert_eq!(
            scrub("const opts = { headers: { authorization: authHeader } };"),
            "const opts = { headers: { authorization: authHeader } };"
        );
        assert_eq!(scrub("Authorization: required"), "Authorization: required");
        assert_eq!(scrub("x-api-key: TODO"), "x-api-key: TODO");
        assert_eq!(scrub("  authorization: {}"), "  authorization: {}");
        assert_eq!(
            scrub("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc"),
            "Authorization: [redacted]"
        );
        assert_eq!(
            scrub("x-api-key: sk_live_ex4mple0key0for0tests00"),
            "x-api-key: [redacted]"
        );
    }

    /// Requirement: an environment name with no prefix loses its value too.
    ///
    /// Every suffix needed a character in front of it, so the three commonest
    /// bare spellings — a Django `settings.py`, a Terraform provider block, a
    /// service account's key — were read by no rule. A generic `_KEY` stays
    /// content, and a public key is public.
    #[test]
    fn an_environment_name_with_no_prefix_loses_its_value_too() {
        assert_eq!(
            scrub("SECRET_KEY=abc123def456ghi789"),
            "SECRET_KEY=[redacted]"
        );
        assert_eq!(
            scrub("ACCESS_KEY=AKIAIOSFODNN7EXAMPL"),
            "ACCESS_KEY=[redacted]"
        );
        assert_eq!(
            scrub("PRIVATE_KEY=MIIEpAIBAAKCAQEA1234567890"),
            "PRIVATE_KEY=[redacted]"
        );
        for line in [
            "CACHE_KEY=users:42",
            "SSH_KEY_PATH=~/.ssh/id_ed25519",
            "MAX_TOKENS=4096",
        ] {
            assert_eq!(scrub(line), line, "{line} is not a credential name");
        }
    }

    /// Requirement: an `=` with nothing after it does not take the next word.
    ///
    /// A logfmt field and a flag are both written empty, and reading on from
    /// there rewrote the tail of a recorded command — a policy would read an
    /// action that was never run. Only the asymmetric spacing says the value
    /// is empty; `a = b`, `a=b` and every colon form read as before.
    #[test]
    fn an_equals_with_nothing_after_it_does_not_take_the_next_word() {
        for line in [
            "level=info msg=connect password= host=db.internal port=5432",
            "mysql --password= --host=db -e 'show tables'",
        ] {
            assert_eq!(scrub(line), line, "{line} left its value empty");
        }
        assert_eq!(
            scrub("--password=hunter2hunter2 --host=db"),
            "--password=[redacted] --host=db"
        );
        assert_eq!(
            scrub("password = hunter2hunter2"),
            "password = [redacted]",
            "spacing on both sides is an ordinary assignment"
        );
    }

    /// Requirement: a bare name standing in a source assignment is a
    /// reference, whatever its length.
    ///
    /// A constant, a function and a variable are all spelled in letters and
    /// underscores, and none of them is a credential. The reading is confined
    /// to the `=` of source: under an environment-style name and in every
    /// colon form an unquoted word is the value itself, and a value that
    /// carries a digit or one case with no underscore is still a literal.
    #[test]
    fn a_name_under_a_source_assignment_is_a_reference_whatever_its_length() {
        for line in [
            "password = DEFAULT_ADMIN_PASSWORD",
            "const password = getDatabasePassword;",
            "api_key = SETTINGS_DEFAULT_API_KEY",
            "client_secret = CLIENT_SECRET_ENV_NAME",
        ] {
            assert_eq!(scrub(line), line, "{line} names a secret, it is not one");
        }
        assert_eq!(
            scrub("DB_PASSWORD=DefaultAdminPassword"),
            "DB_PASSWORD=[redacted]",
            "an environment-style name takes the word after it as the value"
        );
        assert_eq!(
            scrub("password: DEFAULT_ADMIN_PASSWORD"),
            "password: [redacted]",
            "a colon form is YAML, where the word is the value"
        );
        assert_eq!(
            scrub("password = correcthorsebatterystaple"),
            "password = [redacted]",
            "one case with no underscore is not an identifier reference"
        );
        assert_eq!(
            scrub("password = Tr0ub4dor_and_3"),
            "password = [redacted]",
            "a digit makes it a literal"
        );
    }

    /// Requirement: an environment-variable name reads in the forms a config
    /// writes it in, and only those.
    ///
    /// The colon form is a docker-compose `environment:` block and a
    /// GitHub Actions `env:` mapping; the spaced form is
    /// `~/.aws/credentials`, whose
    /// own spelling is lowercase. `MAX_TOKENS` ends in `TOKENS`, not
    /// `_TOKEN`, and is content.
    #[test]
    fn the_environment_variable_name_reads_its_three_forms() {
        assert_eq!(
            scrub("      POSTGRES_PASSWORD: s3cr3tpg"),
            "      POSTGRES_PASSWORD: [redacted]"
        );
        assert_eq!(
            scrub("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
            "aws_secret_access_key = [redacted]"
        );
        assert_eq!(
            scrub("-DB_PASSWORD=old1234pw"),
            "-DB_PASSWORD=[redacted]",
            "the removed line of a diff is where the old secret lives"
        );
        assert_eq!(scrub("MAX_TOKENS=4096"), "MAX_TOKENS=4096");
    }

    /// Requirement: a name behind an escaped quote is still a name.
    ///
    /// A response body quoted back inside a tool result, and a span attribute
    /// holding JSON that itself holds JSON, write the name as `\"name\"`.
    #[test]
    fn a_name_in_double_encoded_text_is_read_and_its_escapes_kept() {
        assert_eq!(
            scrub(r#"response body: {\"client_secret\": \"abc123secretvalue\"}"#),
            r#"response body: {\"client_secret\": \"[redacted]\"}"#
        );
    }
}
