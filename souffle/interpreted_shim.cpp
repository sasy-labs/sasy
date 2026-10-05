/**
 * Soufflé interpreted-mode evaluator adapter.
 *
 * Runs the policy file unchanged with the `souffle` CLI in interpreted mode.
 * Per query, writes RFC 4180 inputs in a private fact directory, invokes
 * `souffle`, and parses sectioned stdout. Rows use RFC 4180 where the policy
 * declares that output format.
 *
 * Build: g++ -std=c++17 -O2 -o souffle-interpreted interpreted_shim.cpp
 *        (the functors are built separately into a shared library from
 *        functors.cpp; see build-test-runtime.sh)
 * Run:   ./souffle-interpreted <policy.dl> [souffle-binary] [functor-lib] [functor-directory]
 *
 * Speaks the same IPC protocol as the compiled evaluator_shim.
 */

#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "json_string_codec.h"
#include "evaluator_protocol.h"

namespace json = sasy_json_codec;
namespace evaluator_protocol = sasy_evaluator_protocol;

extern char** environ;

// Graph state
struct GraphNode {
    std::string id, contents, agent, role, entity, tools_json, principal;
    // Session/conversation partition of the update that recorded this
    // node. The empty string is the global partition: where a node
    // announced without a session_id lands, and where an unsessioned
    // query is answered from.
    std::string session_id;
};
// An edge as this shim stores it: the session that announced it,
// then the source and destination message ids. Message ids are only
// unique within a session, so every per-edge value is keyed by all
// three — two sessions that reuse a pair of ids keep their own edge,
// their own principals and their own entities instead of overwriting
// each other, and a query is answered from the rows its own session
// recorded.
using EdgeKey = std::tuple<std::string,std::string,std::string>;

static EdgeKey edge_key(const std::string& session,
                        const std::string& src,
                        const std::string& dst) {
    return EdgeKey(session, src, dst);
}

struct GraphState {
    std::unordered_map<std::string, GraphNode> nodes;
    // Edges, indexed by (session, source, destination). An edge
    // re-announced under a second session belongs to both, so it
    // appears once per announcing session.
    std::set<EdgeKey> edges;
    // Who asserted each edge, indexed by (session, source,
    // destination). Both maps are sparse: an edge recorded without an
    // auth-derived principal, or without an entity, has no entry
    // and so contributes no fact row.
    std::map<EdgeKey, std::string> edge_principal;
    std::map<EdgeKey, std::string> edge_entity;
    // Tool-call provenance, keyed by node id. Scoped by the session of
    // the node it belongs to, which `nodes` holds.
    std::unordered_map<std::string, std::pair<std::string,std::string>> tool_results;
    // The adapter's canonical record of the source message
    // (`Event.metadata`), indexed by (recording session, node id).
    // Sparse: a node recorded without metadata has no entry and so
    // contributes no fact row. Each session keeps its own value for an
    // id, and a query writes the row held under the queried session
    // only — see evaluator_shim.cpp.
    std::map<std::pair<std::string, std::string>, std::string> node_metadata;
};
static GraphState g_state;
static std::vector<std::array<std::string, 3>> g_policy_metadata;
static std::string g_policy_path;
static std::string g_souffle_bin = "souffle";
static std::string g_functor_lib;
static std::string g_functor_directory;
static std::string g_fact_dir;

static const std::string FACT_DIR_PREFIX = "/tmp/souffle-interpreted-";

static bool is_private_fact_dir(const std::string& path) {
    return path.size() == FACT_DIR_PREFIX.size() + 6 &&
           path.compare(0, FACT_DIR_PREFIX.size(), FACT_DIR_PREFIX) == 0;
}

static std::string create_private_fact_dir() {
    std::string pattern = FACT_DIR_PREFIX + "XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* directory = ::mkdtemp(writable.data());
    return directory == nullptr ? "" : directory;
}

static void remove_private_fact_dir(const std::string& path) {
    if (!is_private_fact_dir(path)) return;
    std::error_code error;
    std::filesystem::remove_all(path, error);
}

class PrivateFactDirectory {
public:
    PrivateFactDirectory() : path_(create_private_fact_dir()) {}
    ~PrivateFactDirectory() { remove_private_fact_dir(path_); }

    const std::string& path() const { return path_; }
    bool valid() const { return !path_.empty(); }

private:
    std::string path_;
};

struct ChildProcessResult {
    bool launched = false;
    int status = -1;
    std::string output;
    std::string error;
};

static ChildProcessResult run_child_process(
    const std::vector<std::string>& arguments
) {
    ChildProcessResult result;
    if (arguments.empty()) {
        result.error = "child process has no executable";
        return result;
    }

    int output_pipe[2];
    if (::pipe(output_pipe) != 0) {
        result.error = "pipe failed: " + std::string(std::strerror(errno));
        return result;
    }

    posix_spawn_file_actions_t actions;
    int action_error = posix_spawn_file_actions_init(&actions);
    if (action_error != 0) {
        ::close(output_pipe[0]);
        ::close(output_pipe[1]);
        result.error = "spawn setup failed: " +
            std::string(std::strerror(action_error));
        return result;
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_addclose(
            &actions, output_pipe[0]
        );
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_adddup2(
            &actions, output_pipe[1], STDOUT_FILENO
        );
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_addclose(
            &actions, output_pipe[1]
        );
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_addopen(
            &actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0
        );
    }
    if (action_error != 0) {
        posix_spawn_file_actions_destroy(&actions);
        ::close(output_pipe[0]);
        ::close(output_pipe[1]);
        result.error = "spawn setup failed: " +
            std::string(std::strerror(action_error));
        return result;
    }

    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    pid_t child = -1;
    const int spawn_error = posix_spawnp(
        &child, arguments[0].c_str(), &actions, nullptr,
        argv.data(), environ
    );
    posix_spawn_file_actions_destroy(&actions);
    ::close(output_pipe[1]);
    if (spawn_error != 0) {
        ::close(output_pipe[0]);
        result.error = "could not launch child process: " +
            std::string(std::strerror(spawn_error));
        return result;
    }
    result.launched = true;

    char buffer[4096];
    while (true) {
        const ssize_t count = ::read(output_pipe[0], buffer, sizeof(buffer));
        if (count > 0) {
            result.output.append(buffer, static_cast<size_t>(count));
        } else if (count == 0) {
            break;
        } else if (errno != EINTR) {
            result.error = "could not read child output: " +
                std::string(std::strerror(errno));
            break;
        }
    }
    ::close(output_pipe[0]);

    while (::waitpid(child, &result.status, 0) == -1) {
        if (errno == EINTR) continue;
        result.status = -1;
        result.error = "could not wait for child process: " +
            std::string(std::strerror(errno));
        break;
    }
    return result;
}

// IPC. Frame layout: 4-byte big-endian payload length, 8-byte big-endian
// request id, then the JSON payload. The id belongs to the host and is echoed
// on the reply, so a host that gave up waiting recognises a late answer and
// discards it rather than reading it as the answer to its next query.
static bool read_message(std::string& out, uint64_t& id){
    uint32_t len;if(!std::cin.read(reinterpret_cast<char*>(&len),4))return false;
    len=__builtin_bswap32(len);
    uint64_t raw_id;if(!std::cin.read(reinterpret_cast<char*>(&raw_id),8))return false;
    id=__builtin_bswap64(raw_id);
    out.resize(len);
    return !!std::cin.read(out.data(),len);
}
static void write_message(const std::string& msg, uint64_t id){
    uint32_t len=__builtin_bswap32(static_cast<uint32_t>(msg.size()));
    uint64_t raw_id=__builtin_bswap64(id);
    std::cout.write(reinterpret_cast<const char*>(&len),4);
    std::cout.write(reinterpret_cast<const char*>(&raw_id),8);
    std::cout.write(msg.data(),msg.size());std::cout.flush();
}

// Drop every session's copy of one (source, destination) edge, and
// the attribution recorded for it. The store deletes an edge, not one
// session's view of it.
static void erase_edge_everywhere(const std::string& src, const std::string& dst) {
    for (auto it = g_state.edges.begin(); it != g_state.edges.end();) {
        if (std::get<1>(*it) == src && std::get<2>(*it) == dst) {
            it = g_state.edges.erase(it);
        } else {
            ++it;
        }
    }
    auto drop = [&](std::map<EdgeKey, std::string>& values) {
        for (auto it = values.begin(); it != values.end();) {
            if (std::get<1>(it->first) == src && std::get<2>(it->first) == dst) {
                it = values.erase(it);
            } else {
                ++it;
            }
        }
    };
    drop(g_state.edge_principal);
    drop(g_state.edge_entity);
}

static void apply_updates(const json::Value& updates) {
    for (size_t i = 0; i < updates.size(); ++i) {
        const auto& u = updates[i];
        if (!u["NodeCreated"].is_null()) {
            const auto& n = u["NodeCreated"];
            GraphNode gn{n["id"].str(),
                n["content"].is_null()?"":n["content"].str(),
                n["agent"].is_null()?"":n["agent"].str(),
                n["role"].is_null()?"assistant":n["role"].str(),
                n["entity"].is_null()?"":n["entity"].str(),
                !n["tools"].is_null() && n["tools"].type == json::Value::ARR
                    ? json::serialize(n["tools"])
                    : "[]",
                n["principal"].is_null()?"":n["principal"].str(),
                n["session_id"].is_null()?"":n["session_id"].str()};
            // Absent `derived_from` on an update means the update is
            // silent about provenance, not that the node lost it — see
            // the matching comment in evaluator_shim.cpp.
            if (!n["derived_from"].is_null())
                g_state.tool_results[gn.id] = {n["derived_from"]["name"].str(), n["derived_from"]["arguments"].str()};
            // Same rule for the adapter's record of the message.
            if (!n["metadata"].is_null())
                g_state.node_metadata[{gn.session_id, gn.id}] = n["metadata"].str();
            g_state.nodes[gn.id] = std::move(gn);
        } else if (!u["NodeDeleted"].is_null()) {
            const std::string deleted = u["NodeDeleted"].str();
            g_state.nodes.erase(deleted);
            g_state.tool_results.erase(deleted);
            // A deletion names an id and nothing else, and `nodes` holds
            // one node per id, so the node leaves every session: every
            // session's metadata for that id goes with it.
            for (auto mit = g_state.node_metadata.begin();
                 mit != g_state.node_metadata.end();) {
                if (mit->first.second == deleted) {
                    mit = g_state.node_metadata.erase(mit);
                } else {
                    ++mit;
                }
            }
        } else if (!u["EdgeCreated"].is_null()) {
            const auto& e = u["EdgeCreated"];
            auto src = e["source"].str();
            auto dst = e["destination"].str();
            std::string edge_sid;
            if (!e["session_id"].is_null()) edge_sid = e["session_id"].str();
            const EdgeKey key = edge_key(edge_sid, src, dst);
            // An edge already known to another session is new to this
            // one: it belongs to both, under each session's own key.
            g_state.edges.insert(key);
            // An update announces the edge as the store now holds it:
            // an absent field means the edge carries no such value any
            // more, so the stored one must go. `principal` is the field
            // that actually goes away — it is stamped from the
            // recording request and never merged with the stored edge's.
            if (!e["principal"].is_null())
                g_state.edge_principal[key] = e["principal"].str();
            else
                g_state.edge_principal.erase(key);
            if (!e["entity"].is_null())
                g_state.edge_entity[key] = e["entity"].str();
            else
                g_state.edge_entity.erase(key);
        } else if (!u["EdgeDeleted"].is_null()) {
            auto src = u["EdgeDeleted"]["source"].str();
            auto dst = u["EdgeDeleted"]["destination"].str();
            // The store deletes an edge, not one session's view of it,
            // so it leaves every session that announced it.
            erase_edge_everywhere(src, dst);
        }
    }
}

// Quote one outer RFC 4180 field. Souffle's RFC reader preserves embedded
// control bytes before its record/ADT reader parses the field contents.
static std::string rfc_escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    return out + "\"";
}

// Carry RFC 4180 quote state across appended fragments, so a multi-line record
// costs one pass over each byte instead of a rescan per continuation line.
// A doubled quote never straddles a fragment boundary: fragments are joined
// with a newline, so quotes split across the join are not adjacent.
static void rfc_advance_quote_state(const std::string& fragment, bool& quoted) {
    for (size_t index = 0; index < fragment.size(); ++index) {
        if (fragment[index] != '"') continue;
        if (quoted && index + 1 < fragment.size() &&
            fragment[index + 1] == '"') {
            ++index;
        } else {
            quoted = !quoted;
        }
    }
}

// Backstop against an unterminated quote consuming an unbounded output.
static const size_t kMaxRecordBytes = 16u * 1024u * 1024u;

static bool parse_rfc_record(
    const std::string& record,
    std::vector<std::string>& fields
) {
    fields.clear();
    std::string field;
    bool quoted = false;
    bool after_quote = false;
    bool at_field_start = true;
    for (size_t index = 0; index < record.size(); ++index) {
        const char byte = record[index];
        if (quoted) {
            if (byte == '"') {
                if (index + 1 < record.size() && record[index + 1] == '"') {
                    field += '"';
                    ++index;
                } else {
                    quoted = false;
                    after_quote = true;
                }
            } else {
                field += byte;
            }
        } else if (after_quote) {
            if (byte != ',') return false;
            fields.push_back(std::move(field));
            field.clear();
            after_quote = false;
            at_field_start = true;
        } else if (byte == ',') {
            fields.push_back(std::move(field));
            field.clear();
            at_field_start = true;
        } else if (byte == '"' && at_field_start) {
            quoted = true;
        } else if (byte == '"') {
            return false;
        } else {
            field += byte;
            at_field_start = false;
        }
    }
    if (quoted) return false;
    fields.push_back(std::move(field));
    return true;
}

static std::string decode_souffle_symbol(const std::string& value) {
    std::string decoded;
    for (size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '\\' && index + 1 < value.size() &&
            value[index + 1] == '"') {
            decoded += '"';
            ++index;
        } else {
            decoded += value[index];
        }
    }
    return decoded;
}

struct ParsedSouffleOutput {
    std::set<uint32_t> authorized;
    std::set<uint32_t> passthrough;
    std::map<uint32_t, json::Array> transforms;
    std::map<uint32_t, json::Array> denial_reasons;
    std::map<uint32_t, json::Array> allow_routes;
    std::set<uint32_t> ask_indices;
    bool has_auth = false;
    std::set<uint32_t> denied_indices;
    std::set<uint32_t> allowed_indices;
};

static bool parse_index(
    const std::string& value,
    size_t action_count,
    uint32_t& index
) {
    if (value.empty()) return false;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto result = std::from_chars(begin, end, index);
    return result.ec == std::errc() && result.ptr == end &&
           index < action_count;
}

static bool parse_souffle_output(
    const std::string& output,
    size_t action_count,
    ParsedSouffleOutput& parsed
) {
    static const std::map<std::string, std::string> required_headers = {
        {"AllowPassthrough", "idx"},
        {"ApplyTransform", "idx\ttransform_id"},
        {"Authorized", "idx"},
        {"DenialReason", "idx\tkind\treason\tsuggestion"},
        {"HasPrincipal", ""},
        {"IsAuthorized", "idx"},
        {"Unauthorized", "idx"},
    };
    static const std::string section_start = "---------------";
    static const std::string data_boundary = "===============";
    std::set<std::string> seen_required;
    bool diagnostic_valid = true, diagnostic_seen = false;
    size_t diagnostic_bytes = 0, diagnostic_rows = 0;
    std::istringstream stream(output);
    std::string line;

    while (std::getline(stream, line)) {
        if (line != section_start) return false;
        std::string section;
        std::string header;
        std::string boundary;
        if (!std::getline(stream, section) || section.empty() ||
            !std::getline(stream, header) ||
            !std::getline(stream, boundary) ||
            boundary != data_boundary) return false;

        const auto required = required_headers.find(section);
        const bool diagnostic = section == "SasyAllowRoute";
        if (diagnostic) {
            if (diagnostic_seen || header != "idx\trule_id\tstatus\tdetails\tsuggestion\tsource_location")
                diagnostic_valid = false;
            diagnostic_seen = true;
        }
        if (required != required_headers.end() &&
            (header != required->second ||
             !seen_required.insert(section).second)) return false;

        bool section_closed = false;
        while (std::getline(stream, line)) {
            if (line == data_boundary) {
                section_closed = true;
                break;
            }
            // Relations the shim does not consume are framed, not validated:
            // their rows are plain .output text and may hold a bare quote.
            if (required == required_headers.end() && (!diagnostic || !diagnostic_valid)) continue;
            std::string record = line;
            bool quoted = false;
            rfc_advance_quote_state(record, quoted);
            while (quoted) {
                if (record.size() > kMaxRecordBytes) return false;
                if (!std::getline(stream, line)) return false;
                record += "\n";
                record += line;
                rfc_advance_quote_state(line, quoted);
            }
            std::vector<std::string> fields;
            if (!parse_rfc_record(record, fields)) {
                if (diagnostic) { diagnostic_valid = false; continue; }
                return false;
            }
            uint32_t index = 0;
            if (diagnostic) {
                if (fields.size() != 6 || !parse_index(fields[0], action_count, index) ||
                    ++diagnostic_rows > 512) { diagnostic_valid = false; continue; }
                json::Object row;
                const char* names[] = {"rule_id", "status", "details", "suggestion", "source_location"};
                for (size_t column = 1; column < 6; ++column) {
                    const auto value = decode_souffle_symbol(fields[column]);
                    diagnostic_bytes += value.size();
                    if (diagnostic_bytes > 262144) { diagnostic_valid = false; break; }
                    row[names[column - 1]] = json::Value(value);
                }
                if (diagnostic_valid) parsed.allow_routes[index].push_back(json::Value(std::move(row)));
            } else if (section == "Authorized" ||
                section == "AllowPassthrough" ||
                section == "Unauthorized" ||
                section == "IsAuthorized") {
                if (fields.size() != 1 ||
                    !parse_index(fields[0], action_count, index)) return false;
                if (section == "Authorized") parsed.authorized.insert(index);
                else if (section == "AllowPassthrough")
                    parsed.passthrough.insert(index);
                else if (section == "Unauthorized")
                    parsed.denied_indices.insert(index);
                else parsed.allowed_indices.insert(index);
            } else if (section == "HasPrincipal") {
                if (fields.size() != 1 || fields[0] != "()" ||
                    parsed.has_auth) return false;
                parsed.has_auth = true;
            } else if (section == "DenialReason" ||
                       section == "ApplyTransform") {
                const size_t arity = section == "DenialReason" ? 4 : 2;
                if (fields.size() != arity ||
                    !parse_index(fields[0], action_count, index)) return false;
                if (section == "ApplyTransform") {
                    parsed.transforms[index].push_back(json::Value(
                        decode_souffle_symbol(fields[1])));
                } else {
                    const std::string kind =
                        decode_souffle_symbol(fields[1]);
                    if (kind != "ask" && kind != "block") return false;
                    if (kind == "ask") parsed.ask_indices.insert(index);
                    parsed.denial_reasons[index].push_back(
                        json::Value(json::Object{
                            {"kind", json::Value(kind)},
                            {"reason", json::Value(
                                decode_souffle_symbol(fields[2]))},
                            {"suggestion", json::Value(
                                decode_souffle_symbol(fields[3]))},
                        }));
                }
            }
        }
        if (!section_closed) return false;
    }
    if (!diagnostic_valid) parsed.allow_routes.clear();
    return seen_required.size() == required_headers.size();
}

// Soufflé ADT serialization for TSV: $BranchName(field1, field2, ...)
static std::string adt_agent_role(const std::string& role) {
    // Nullary ADT branches: no parentheses in Soufflé's TSV serialization
    if (role == "system") return "$SystemRole";
    if (role == "user") return "$UserRole";
    if (role == "agent") return "$AgentType";
    if (role == "assistant" || role == "llm") return "$Assistant";
    // Unknown role → Assistant: fail-safe untrusted default (see evaluator_shim).
    return "$Assistant";
}

// Quote a symbol value for Soufflé record fields.
// Soufflé's readSymbol checks for a leading '"' and uses readQuotedSymbol
// which handles escaped quotes (\") and stops at the closing '"'.
// This allows commas, brackets, etc. inside symbol values.
static std::string rec_escape(const std::string& s) {
    if (s.empty()) return "\"\"";
    // Check if quoting is needed (contains delimiters or whitespace)
    bool needs_quoting = false;
    for (char c : s) {
        if (c == ',' || c == '[' || c == ']' || c == '(' || c == ')'
            || c == '{' || c == '}' || static_cast<unsigned char>(c) < 0x20
            || c == ' ' || c == '"' || c == '\\') {
            needs_quoting = true;
            break;
        }
    }
    if (!needs_quoting) return s;
    // Quote with escaped internal quotes and backslashes.
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    out += '"';
    return out;
}

static std::string adt_message(const GraphNode& n) {
    // Record: [contents, tools_json, agent, agent_role, entity, principal]
    // tools_json: use quoting to preserve brackets/commas
    return "[" + rec_escape(n.contents) + ", " + rec_escape(n.tools_json) + ", " +
           rec_escape(n.agent) + ", " + adt_agent_role(n.role) + ", " +
           rec_escape(n.entity) + ", " + rec_escape(n.principal) + "]";
}

template <typename WriteFacts>
static bool write_fact_file(
    const std::string& fact_dir,
    const std::string& filename,
    WriteFacts write_facts
) {
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    try {
        file.open(fact_dir + "/" + filename, std::ios::out | std::ios::trunc);
        write_facts(file);
        file.flush();
        file.close();
        return true;
    } catch (const std::ios_base::failure&) {
        file.exceptions(std::ios::goodbit);
        if (file.is_open()) file.close();
        return false;
    }
}

static json::Value fact_write_error(const std::string& filename) {
    return json::Value(json::Object{{"Error", json::Value(json::Object{
        {"message", json::Value("failed to write fact file: " + filename)}
    })}});
}

static std::vector<std::string> souffle_query_arguments(const std::string& fact_dir) {
    std::vector<std::string> arguments = {
        g_souffle_bin,
        g_policy_path,
        "-F" + fact_dir,
        "-D-",
    };
    if (!g_functor_lib.empty()) {
        arguments.push_back("-l" + g_functor_lib);
        if (!g_functor_directory.empty()) {
            // The server selected this library explicitly. A same-named
            // library beside the adapter must never override that choice.
            arguments.push_back("-L" + g_functor_directory);
        } else {
            // Preserve standalone adapter compatibility when no directory
            // was supplied. Server-managed launches always supply one.
            arguments.push_back("-L/usr/local/lib");
            arguments.push_back("-L/opt/homebrew/lib");
            arguments.push_back("-L.");
        }
    }
    return arguments;
}

static json::Value run_query(const json::Value& req) {
    // Write RFC 4180 facts, then invoke souffle with -F pointing at the fact dir.
    // The policy keeps its .input directives so souffle reads from the files.
    const std::string& fact_dir = g_fact_dir;

    // Every per-session relation is answered from the querying
    // session's own rows, the way the compiled shim answers from that
    // session's program instance: Edge, EdgePrincipal, EdgeEntity,
    // SentMessage and ToolResult. A query with no session_id is
    // answered from the global partition, the empty-string session.
    // (EdgeData is per-session there too, but this path carries no
    // edge metadata at all — see the empty file written below.)
    std::string query_session;
    if (!req["session_id"].is_null()) query_session = req["session_id"].str();

    // Edge.facts
    if (!write_fact_file(fact_dir, "Edge.facts", [&](std::ostream& file) {
        for (const auto& key : g_state.edges) {
            if (std::get<0>(key) != query_session) continue;
            file << rfc_escape(std::get<1>(key)) << ","
                 << rfc_escape(std::get<2>(key)) << "\n";
        }
    })) return fact_write_error("Edge.facts");

    // EdgePrincipal.facts / EdgeEntity.facts — who asserted each edge.
    // Both are sparse: only edges that carry the value get a row. The
    // rows are the querying session's own: attribution is keyed by the
    // session that recorded it, so a second session that reuses a pair
    // of message ids never answers from the first session's principal.
    auto write_attribution = [&](const char* filename,
                                 const std::map<EdgeKey, std::string>& values) {
        return write_fact_file(fact_dir, filename, [&](std::ostream& file) {
            for (const auto& [key, value] : values) {
                if (std::get<0>(key) != query_session) continue;
                file << rfc_escape(std::get<1>(key)) << ","
                     << rfc_escape(std::get<2>(key)) << ","
                     << rfc_escape(value) << "\n";
            }
        });
    };
    if (!write_attribution("EdgePrincipal.facts", g_state.edge_principal)) {
        return fact_write_error("EdgePrincipal.facts");
    }
    if (!write_attribution("EdgeEntity.facts", g_state.edge_entity)) {
        return fact_write_error("EdgeEntity.facts");
    }

    // SentMessage.facts — second column is the ADT-serialized Message record
    if (!write_fact_file(
        fact_dir, "SentMessage.facts", [&](std::ostream& file) {
            for (auto& [id, node] : g_state.nodes) {
                if (node.session_id != query_session) continue;
                file << rfc_escape(id) << ","
                     << rfc_escape(adt_message(node)) << "\n";
            }
        }
    )) return fact_write_error("SentMessage.facts");

    // ToolResult.facts — a tool result belongs to the session of the
    // node that carries it, so it is scoped the same way SentMessage is.
    if (!write_fact_file(
        fact_dir, "ToolResult.facts", [&](std::ostream& file) {
            for (auto& [id, tool_result] : g_state.tool_results) {
                auto node = g_state.nodes.find(id);
                if (node == g_state.nodes.end()) continue;
                if (node->second.session_id != query_session) continue;
                file << rfc_escape(id) << ","
                     << rfc_escape(tool_result.first) << ","
                     << rfc_escape(tool_result.second) << "\n";
            }
        }
    )) return fact_write_error("ToolResult.facts");

    // MessageMetadata.facts — the adapter's record of the message,
    // scoped by its node's session like ToolResult, and taken from the
    // queried session's own entry, so a re-record of the same id under
    // another session neither carries the value over nor displaces it.
    if (!write_fact_file(
        fact_dir, "MessageMetadata.facts", [&](std::ostream& file) {
            for (auto& [key, metadata] : g_state.node_metadata) {
                const auto& [session, id] = key;
                if (session != query_session) continue;
                auto node = g_state.nodes.find(id);
                if (node == g_state.nodes.end()) continue;
                if (node->second.session_id != query_session) continue;
                file << rfc_escape(id) << ","
                     << rfc_escape(metadata) << "\n";
            }
        }
    )) return fact_write_error("MessageMetadata.facts");

    // Current.facts
    if (!write_fact_file(
        fact_dir, "Current.facts", [&req](std::ostream& file) {
            for (size_t i = 0; i < req["current_node_ids"].size(); ++i) {
                file << rfc_escape(req["current_node_ids"][i].str())
                     << "\n";
            }
        }
    )) return fact_write_error("Current.facts");

    // Actions.facts — second column is ADT-serialized Action
    if (!write_fact_file(
        fact_dir, "Actions.facts", [&req](std::ostream& file) {
            const auto& actions = req["actions"];
            for (size_t i = 0; i < actions.size(); ++i) {
                const auto& action_value = actions[i];
                std::ostringstream action;
                // Quote fields inside ADT/record constructors so their
                // punctuation is not parsed as constructor syntax.
                if (!action_value["ToolCall"].is_null()) {
                    action << "$CallTool("
                           << rec_escape(
                                  action_value["ToolCall"]["fn_name"].str())
                           << ", "
                           << rec_escape(
                                  action_value["ToolCall"]["args"].str())
                           << ")";
                } else if (!action_value["HttpRequest"].is_null()) {
                    action << "$HTTPRequest("
                           << rec_escape(
                                  action_value["HttpRequest"]["url"].str())
                           << ", "
                           << rec_escape(
                                  action_value["HttpRequest"]["body"].str())
                           << ", \"[]\")";
                } else if (!action_value["SendMessage"].is_null()) {
                    const auto role =
                        action_value["SendMessage"]["agent_role"].str();
                    const auto entity =
                        action_value["SendMessage"]["entity"].is_null()
                            ? ""
                            : action_value["SendMessage"]["entity"].str();
                    action << "$SendAttempt(["
                           << rec_escape(
                                  action_value["SendMessage"]["content"].str())
                           << ", \"[]\", "
                           << rec_escape(
                                  action_value["SendMessage"]["agent"].str())
                           << ", " << adt_agent_role(role) << ", "
                           << rec_escape(entity) << ", \"\"])";
                }
                file << i << "," << rfc_escape(action.str()) << "\n";
            }
        }
    )) return fact_write_error("Actions.facts");

    // Principal.facts — server-stamped, auth-derived principal. Mirrors the
    // compiled shim's `Principal` relation. Every `.input` relation needs its
    // fact file under the exact declared name, or souffle aborts on load and
    // every request is denied.
    if (!write_fact_file(
        fact_dir, "Principal.facts", [&req](std::ostream& file) {
            if (!req["principal"].is_null()) {
                file << rfc_escape(req["principal"].str()) << "\n";
            }
        }
    )) return fact_write_error("Principal.facts");

    // PrincipalRole.facts — keyed on the principal (not the user entity).
    if (!write_fact_file(
        fact_dir, "PrincipalRole.facts", [&req](std::ostream& file) {
            if (!req["principal"].is_null()) {
                for (size_t i = 0; i < req["roles"].size(); ++i) {
                    file << rfc_escape(req["principal"].str()) << ","
                         << rfc_escape(req["roles"][i].str()) << "\n";
                }
            }
        }
    )) return fact_write_error("PrincipalRole.facts");

    // Entity.facts — user-supplied actor (free-form, not auth-attested).
    if (!write_fact_file(
        fact_dir, "Entity.facts", [&req](std::ostream& file) {
            if (!req["entity"].is_null()) {
                file << rfc_escape(req["entity"].str()) << "\n";
            }
        }
    )) return fact_write_error("Entity.facts");

    // EdgeData.facts — sparse; the interpreted path carries no edge metadata,
    // but `.input EdgeData` still needs the file to exist or souffle aborts.
    if (!write_fact_file(
        fact_dir, "EdgeData.facts", [](std::ostream&) {}
    )) return fact_write_error("EdgeData.facts");

    // TenantId.facts
    if (!write_fact_file(
        fact_dir, "TenantId.facts", [&req](std::ostream& file) {
            if (!req["tenant_id"].is_null()) {
                file << rfc_escape(req["tenant_id"].str()) << "\n";
            }
        }
    )) return fact_write_error("TenantId.facts");

    // Static policy metadata persists across queries until SetMetadata replaces
    // it. The relation remains generic: the policy decides how to project each
    // (rel, a, b) tuple.
    if (!write_fact_file(
        fact_dir, "PolicyMetadata.facts", [](std::ostream& file) {
            for (const auto& fact : g_policy_metadata) {
                file << rfc_escape(fact[0]) << "," << rfc_escape(fact[1])
                     << "," << rfc_escape(fact[2]) << "\n";
            }
        }
    )) return fact_write_error("PolicyMetadata.facts");

    // Per-action metadata is query-scoped and indexed by the corresponding
    // action position.
    if (!write_fact_file(
        fact_dir, "ActionMetadata.facts", [&req](std::ostream& file) {
            const auto& entries = req["action_metadata"];
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& entry = entries[i];
                const auto& facts = entry["facts"];
                for (size_t j = 0; j < facts.size(); ++j) {
                    const auto& fact = facts[j];
                    file << entry["index"].num() << ","
                         << rfc_escape(fact["rel"].str()) << ","
                         << rfc_escape(fact["a"].str()) << ","
                         << rfc_escape(fact["b"].str()) << "\n";
                }
            }
        }
    )) return fact_write_error("ActionMetadata.facts");

    const auto arguments = souffle_query_arguments(fact_dir);
    const ChildProcessResult process = run_child_process(arguments);
    if (!process.launched || !process.error.empty()) {
        return json::Value(json::Object{{"Error", json::Value(json::Object{
            {"message", json::Value(process.error)}
        })}});
    }
    if (!WIFEXITED(process.status) || WEXITSTATUS(process.status) != 0) {
        std::string message = "souffle backend failed";
        if (WIFEXITED(process.status)) {
            message += " with exit code " +
                std::to_string(WEXITSTATUS(process.status));
        } else if (WIFSIGNALED(process.status)) {
            message += " with signal " +
                std::to_string(WTERMSIG(process.status));
        }
        return json::Value(json::Object{{"Error", json::Value(json::Object{
            {"message", json::Value(message)}
        })}});
    }

    // Parse sectioned stdout, using RFC 4180 rows where the policy declares it.
    ParsedSouffleOutput parsed;
    if (!parse_souffle_output(process.output, req["actions"].size(), parsed)) {
        return json::Value(json::Object{{"Error", json::Value(json::Object{
            {"message", json::Value("malformed souffle output")}
        })}});
    }

    return evaluator_protocol::build_query_result(
        req["actions"].size(), parsed.has_auth, parsed.authorized,
        parsed.denied_indices, parsed.allowed_indices, parsed.passthrough,
        parsed.ask_indices, parsed.denial_reasons, parsed.transforms, parsed.allow_routes
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: souffle-interpreted <policy.dl> [souffle-bin] [functor-lib] [functor-directory]\n";
        return 1;
    }
    g_policy_path = argv[1];
    if (argc > 2) g_souffle_bin = argv[2];
    if (argc > 3) g_functor_lib = argv[3];
    if (argc > 4) {
        g_functor_directory = argv[4];
        if (g_functor_directory.empty()) {
            std::cerr << "Explicit functor directory must not be empty\n";
            return 1;
        }
    }

    PrivateFactDirectory fact_directory;
    if (!fact_directory.valid()) {
        std::cerr << "Could not create a private fact directory\n";
        return 1;
    }
    g_fact_dir = fact_directory.path();

    std::cerr << "Soufflé interpreted evaluator ready (policy=" << g_policy_path << ")\n";

    std::string msg;
    uint64_t request_id = 0;
    while (read_message(msg, request_id)) {
        auto request = json::parse(msg);
        std::string response_json;
        bool handled = false;

        if (!request["Update"].is_null()) {
            apply_updates(request["Update"]["updates"]);
            response_json = "\"UpdateOk\""; handled = true;
        } else if (!request["SetMetadata"].is_null()) {
            g_policy_metadata.clear();
            const auto& facts = request["SetMetadata"]["facts"];
            for (size_t i = 0; i < facts.size(); ++i) {
                const auto& fact = facts[i];
                g_policy_metadata.push_back({
                    fact["rel"].str(), fact["a"].str(), fact["b"].str()
                });
            }
            response_json = "\"UpdateOk\""; handled = true;
        } else if (!request["Query"].is_null()) {
            auto result = run_query(request["Query"]);
            response_json = json::serialize(result); handled = true;
        } else if (!request["LlmResult"].is_null()) {
            // Interpreted mode cannot support LLM oracle callbacks —
            // functors run in the souffle CLI process, not this shim.
            // @llm_check_fn returns 0 (fail-safe deny) on cache miss.
            response_json = "{\"Error\":{\"message\":\"LlmResult not supported in interpreted mode\"}}";
            handled = true;
        }
        if (!handled && request.type == json::Value::STR) {
            if (request.str() == "Reset") {
                g_state.nodes.clear(); g_state.edges.clear(); g_state.tool_results.clear();
                g_state.node_metadata.clear();
                g_state.edge_principal.clear(); g_state.edge_entity.clear();
                response_json = "\"ResetOk\""; handled = true;
            } else if (request.str() == "Shutdown") break;
        }
        if (!handled) response_json = "{\"Error\":{\"message\":\"Unknown request type\"}}";
        write_message(response_json, request_id);
    }
    return 0;
}
