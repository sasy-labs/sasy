/**
 * Soufflé C++ API evaluator shim.
 *
 * Speaks the IPC protocol (length-prefixed JSON over stdin/stdout)
 * and uses Soufflé's SouffleProgram C++ API for evaluation.
 *
 * Build:
 *   1. souffle -g policy_program.cpp policy.dl
 *   2. g++ -std=c++17 -O2 -D__EMBEDDED_SOUFFLE__ \
 *        -I/usr/include/souffle \
 *        evaluator_shim.cpp policy_program.cpp functors_common.cpp \
 *        -o souffle-evaluator -lpthread
 *
 * The generated policy_program.cpp registers itself via REGISTER_PROGRAM.
 * This shim creates instances via ProgramFactory::newInstance().
 *
 * Each query gets a fresh SouffleProgram instance — no shared state,
 * concurrent queries are safe (handled by the policy engine).
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "json_string_codec.h"
#include "evaluator_protocol.h"

// Sandbox (Linux only)
#ifdef __linux__
#include <sys/stat.h>
#include <unistd.h>
#include <sched.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/syscall.h>
#endif

// Soufflé interface
#include <souffle/SouffleInterface.h>

// LLM cache (defined in functors_common.cpp)
extern "C" void llm_cache_clear();

namespace json = sasy_json_codec;
namespace evaluator_protocol = sasy_evaluator_protocol;

// ── LLM oracle callback ──────────────────────────────────────────
// Called by llm_check_fn (in functors_common.cpp) on cache miss.
// Writes an LlmQuery message on stdout, reads an LlmResult from stdin.
// Must be serialized — Soufflé parallel evaluation may call functors
// from multiple threads, but only one oracle IPC can be in flight.

static std::mutex llm_oracle_mutex;

// Forward declarations for IPC framing (defined below)
static bool read_message(std::string& out, uint64_t& id);
static void write_message(const std::string& msg, uint64_t id);

// The request id the main loop is currently answering. The oracle callback
// runs in the middle of that request, so its LlmQuery must carry the same id
// or the host would take it for the late answer to an abandoned request and
// discard it.
static uint64_t g_current_request_id = 0;

// One frame read out of turn and handed back to the main loop.
//
// The oracle blocks on stdin waiting for its LlmResult. If the host gave up on
// the query in the meantime — its query budget expired, so it dropped the call
// and moved on — the next thing on the pipe is a NEW request: another query, or
// the graph updates buffered while the evaluator was busy. Reading that as the
// LlmResult would consume it: the host never sees an answer to it, the check is
// denied although this process is healthy, and if it was an Update those graph
// records are gone from a graph nothing marks as incomplete. So the frame is
// parked here instead and the main loop takes it on its next turn, in order,
// exactly as if the oracle had never looked at it.
static bool g_pushback_valid = false;
static std::string g_pushback_msg;
static uint64_t g_pushback_id = 0;

bool llm_oracle_query(const std::string& prompt, const std::string& context) {
    std::lock_guard<std::mutex> lock(llm_oracle_mutex);

    // Build {"LlmQuery":{"prompt":"...","context":"..."}}
    json::Value query(json::Object{
        {"LlmQuery", json::Value(json::Object{
            {"prompt", json::Value(prompt)},
            {"context", json::Value(context)},
        })}
    });
    write_message(json::serialize(query), g_current_request_id);

    // Read LlmResult response: {"LlmResult":{"result":true/false}}
    std::string response;
    uint64_t response_id = 0;
    if (!read_message(response, response_id)) {
        std::cerr << "LLM oracle: failed to read response, returning false\n";
        return false;
    }

    if (response_id != g_current_request_id) {
        // Not our LlmResult: the host abandoned this query and this is the
        // request it sent next. Give it back to the main loop unconsumed and
        // abandon the oracle. The in-flight query still runs to an answer, and
        // the host discards that answer by id — which is the outcome it
        // already expects for a query it gave up on.
        std::cerr << "LLM oracle: frame carries request id " << response_id
                  << " while answering " << g_current_request_id
                  << " — the host abandoned this query; deferring that frame\n";
        g_pushback_msg = response;
        g_pushback_id = response_id;
        g_pushback_valid = true;
        return false;
    }

    auto parsed = json::parse(response);
    if (!parsed["LlmResult"].is_null()) {
        return parsed["LlmResult"]["result"].bool_val;
    }

    std::cerr << "LLM oracle: unexpected response, returning false\n";
    return false;
}

// ── Graph state ────────────────────────────────────────────────────
// Maintained in memory; populated into Soufflé on each query.

struct GraphNode {
    std::string id;
    std::string contents;
    std::string agent;
    std::string role;     // "user", "assistant", "system", "agent"
    // User-supplied actor in the caller's domain (free-form,
    // unvalidated). Populated from `Event.entity`.
    std::string entity;
    // Server-stamped auth-derived identity. Populated from
    // `Event.principal`. Distinct from `entity`: immutable to
    // clients, used for accountability rules.
    std::string principal;
    // Session/conversation partition. Empty string means the
    // per-tenant global partition: those tuples are visible to every
    // query regardless of the query's session_id, which is what a
    // query with no session_id gets.
    std::string session_id;
    // Tool result info (if this node is a tool result)
    std::string tool_fn_name;
    std::string tool_args;
    // JSON-encoded list of tool_call requests on this message
    // (for assistant turns containing tool_calls). Format matches
    // the Soufflé Message.tools_json schema:
    // [{"name":"...","arguments":"..."}, ...]
    // Empty/"[]" when the message has no tool_calls.
    std::string tools_json = "[]";
};

// Edge metadata from instrumentation. Either field may be unset
// independently; we use -1 sentinels in the Soufflé record to
// match the policy-language representation. `has_any()` tracks
// whether the edge carries any metadata at all so the caller
// can skip emitting an EdgeData tuple for plain edges.
struct EdgeMeta {
    int64_t message_index = -1;
    int64_t proximal = -1;

    bool has_any() const { return message_index >= 0 || proximal >= 0; }
};

// An edge as this shim stores it: the session that announced it,
// then the source and destination message ids. Message ids are only
// unique within a session, and each session gets its own program
// instance, so every per-edge map is keyed by all three — two
// sessions that reuse a pair of ids keep separate metadata and
// separate attribution instead of overwriting each other.
using EdgeKey = std::tuple<std::string, std::string, std::string>;

static EdgeKey edge_key(const std::string& session,
                        const std::string& src,
                        const std::string& dst) {
    return EdgeKey(session, src, dst);
}

struct GraphState {
    std::unordered_map<std::string, GraphNode> nodes;
    std::set<EdgeKey> edges;
    // Metadata for edges that were recorded with instrumentation
    // context. Indexed by (session, source, destination). Only edges
    // with metadata appear here — the EdgeData relation is sparse
    // relative to Edge.
    std::map<EdgeKey, EdgeMeta> edge_meta;
    // Who asserted each edge: the server-stamped principal and the
    // client-supplied entity, indexed by (session, source,
    // destination). Both are sparse — an edge recorded without an
    // auth-derived identity, or without an entity, is simply absent
    // from the corresponding map and emits no tuple.
    std::map<EdgeKey, std::string> edge_principal;
    std::map<EdgeKey, std::string> edge_entity;
    // The sessions that announced each (source, destination) pair.
    // An edge announced without a session_id belongs to the global
    // partition, which is the empty-string session.
    std::map<std::pair<std::string, std::string>, std::set<std::string>> edge_sessions;
    // Tool results tracked separately for ToolResult relation
    // (node_id -> (fn_name, args))
    std::unordered_map<std::string, std::pair<std::string, std::string>> tool_results;
    // The adapter's canonical record of the source message
    // (`Event.metadata`), for the MessageMetadata relation, indexed by
    // (recording session, node id). Sparse: a node recorded without
    // metadata has no entry and emits no tuple.
    //
    // The session is part of the KEY, so each session keeps its own
    // value for an id whatever another session records under the same
    // one: a second session that re-records the id without metadata of
    // its own inherits nothing, and recording under it does not disturb
    // the first session's value. A query emits the row held under the
    // queried session only.
    std::map<std::pair<std::string, std::string>, std::string> node_metadata;

    // ── Per-session reverse indices ────────────────────────
    // session_id → list of node ids whose GraphNode.session_id
    // matches. The empty-string key holds the per-tenant global
    // partition (session_id unset). Maintained alongside `nodes`
    // so a session-scoped rebuild iterates only the session's
    // tuples plus the global partition, instead of scanning all
    // of `nodes`.
    std::unordered_map<std::string, std::vector<std::string>> session_nodes;
    // session_id → list of (source, destination) edges. Edges
    // without an explicit session_id land in the empty-string
    // bucket so the global partition is visible to every query.
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> session_edges;
};

static GraphState g_state;
static std::string g_program_name;

// ── IPC framing ────────────────────────────────────────────────────

// Frame layout: 4-byte big-endian payload length, 8-byte big-endian request
// id, then the JSON payload. The id is the host's, echoed back on the reply:
// a host that gave up waiting (its query budget) recognises the late answer by
// that id and discards it instead of reading it as the answer to its next
// query. Both ends ship together — this shim is compiled per policy by the
// binary that talks to it — so the layout needs no version negotiation.
static bool read_message(std::string& out, uint64_t& id) {
    // A frame the oracle read out of turn comes first: it is older than
    // anything still on the pipe, and answering requests out of order would
    // strand whichever one is skipped.
    if (g_pushback_valid) {
        out = g_pushback_msg;
        id = g_pushback_id;
        g_pushback_valid = false;
        g_pushback_msg.clear();
        return true;
    }
    uint32_t len;
    if (!std::cin.read(reinterpret_cast<char*>(&len), 4)) return false;
    len = __builtin_bswap32(len); // big-endian
    uint64_t raw_id;
    if (!std::cin.read(reinterpret_cast<char*>(&raw_id), 8)) return false;
    id = __builtin_bswap64(raw_id);
    out.resize(len);
    if (!std::cin.read(out.data(), len)) return false;
    return true;
}

static void write_message(const std::string& msg, uint64_t id) {
    uint32_t len = __builtin_bswap32(static_cast<uint32_t>(msg.size()));
    uint64_t raw_id = __builtin_bswap64(id);
    std::cout.write(reinterpret_cast<const char*>(&len), 4);
    std::cout.write(reinterpret_cast<const char*>(&raw_id), 8);
    std::cout.write(msg.data(), msg.size());
    std::cout.flush();
}

// ── Persistent program instances ───────────────────────────────────
// One SouffleProgram instance per session_id (lazily created on
// first reference). Each instance owns its own symbol table,
// record table, and relation indices, so it *is* the materialised
// view for that session — no cross-session filtering needed and
// nothing to rebuild on session switch. ``g_prog`` is the
// currently-active instance; apply_updates and run_query swap it
// before invoking the existing insert_* / encode_* helpers so the
// helpers themselves don't need any plumbing.
//
// Empty session_id ("") is just another bucket — it covers queries
// with no session_id and the bootstrap "default" instance.

static std::unordered_map<std::string, souffle::SouffleProgram*> g_session_progs;
static souffle::SouffleProgram* g_prog = nullptr;
// Per-session "needs rebuild" flag, set when a delete happens in
// that session (Souffle has no delete API). Cleared on the next
// query for that session, after rebuild_input_relations runs.
static std::unordered_map<std::string, bool> g_session_needs_rebuild;

// Static policy configuration (SetPolicyRequest.policy_metadata), materialized
// as the `PolicyMetadata(rel, a, b)` EDB relation. Sent once via the SetMetadata
// IPC message after the evaluator spawns; constant for the evaluator's life,
// so it is seeded into every per-session program at creation. Decoupled from
// the compiled policy — a precompiled (restricted-build) evaluator accepts it
// without recompiling.
static std::vector<std::array<std::string, 3>> g_policy_metadata;

// Insert the policy-metadata facts into a program's PolicyMetadata relation.
static void seed_policy_metadata(souffle::SouffleProgram* p) {
    if (!p || g_policy_metadata.empty()) return;
    auto* rel = p->getRelation("PolicyMetadata");
    if (!rel) return;
    auto& sym = p->getSymbolTable();
    for (const auto& f : g_policy_metadata) {
        souffle::tuple t(rel);
        // PolicyMetadata(rel, a, b) is all symbols: assign the encoded ids via
        // operator[] (the `<<` RamDomain overload asserts a number/record/ADT
        // column and would abort on a symbol column).
        t[0] = sym.encode(f[0]);
        t[1] = sym.encode(f[1]);
        t[2] = sym.encode(f[2]);
        rel->insert(t);
    }
}

static souffle::SouffleProgram* prog_for_session(const std::string& sid) {
    auto it = g_session_progs.find(sid);
    if (it != g_session_progs.end()) return it->second;
    auto* p = souffle::ProgramFactory::newInstance(g_program_name);
    if (!p) return nullptr;
    g_session_progs[sid] = p;
    seed_policy_metadata(p);
    return p;
}

// ADT encoding helpers (depend on program's symbol/record tables)
static souffle::RamDomain encode_agent_role(souffle::SymbolTable& sym, souffle::RecordTable& rec,
                                             const std::string& role) {
    // AgentRole (enum, alphabetical): AgentType=0, Assistant=1, SystemRole=2, UserRole=3
    if (role == "agent") return 0;
    if (role == "system") return 2;
    if (role == "user") return 3;
    if (role == "assistant" || role == "llm") return 1;
    // Unknown role → Assistant: the fail-SAFE default (untrusted LLM content,
    // never the trusted USER principal). The known roles are listed explicitly
    // above so this is reached only on a producer/consumer role-name drift, not
    // silently for an expected role like "llm".
    return 1;
}

static souffle::RamDomain encode_message(souffle::SymbolTable& sym, souffle::RecordTable& rec,
                                          const std::string& contents, const std::string& agent,
                                          const std::string& role, const std::string& entity,
                                          const std::string& tools_json = "[]",
                                          const std::string& principal = "") {
    return rec.pack({
        sym.encode(contents), sym.encode(tools_json), sym.encode(agent),
        encode_agent_role(sym, rec, role), sym.encode(entity), sym.encode(principal),
    });
}

// Rebuild input relations on the active program (g_prog) from
// g_state's per-session reverse indices. Only invoked after a
// delete in that session — Souffle has no delete API, so the
// only way to remove a tuple is purge + re-insert without it.
// Inserts always live in their own session's program instance,
// so exactly the session's bucket is loaded.
static void rebuild_input_relations(const std::string& session_filter) {
    if (!g_prog) return;
    auto& sym = g_prog->getSymbolTable();
    auto& rec = g_prog->getRecordTable();

    auto encode_meta = [&](const EdgeMeta& meta) {
        return rec.pack({
            souffle::RamDomain(meta.proximal),
            souffle::RamDomain(meta.message_index),
        });
    };

    auto* edge_rel = g_prog->getRelation("Edge");
    auto* edge_data_rel = g_prog->getRelation("EdgeData");
    auto* edge_principal_rel = g_prog->getRelation("EdgePrincipal");
    auto* edge_entity_rel = g_prog->getRelation("EdgeEntity");
    auto* sent_rel = g_prog->getRelation("SentMessage");
    auto* tool_rel = g_prog->getRelation("ToolResult");
    auto* metadata_rel = g_prog->getRelation("MessageMetadata");

    if (edge_rel) edge_rel->purge();
    if (edge_data_rel) edge_data_rel->purge();
    if (edge_principal_rel) edge_principal_rel->purge();
    if (edge_entity_rel) edge_entity_rel->purge();
    if (sent_rel) sent_rel->purge();
    if (tool_rel) tool_rel->purge();
    if (metadata_rel) metadata_rel->purge();

    auto insert_edge_tuple = [&](const std::string& src, const std::string& dst) {
        if (edge_rel) {
            souffle::tuple t(edge_rel); t << src << dst; edge_rel->insert(t);
        }
        if (edge_data_rel) {
            auto mit = g_state.edge_meta.find(edge_key(session_filter, src, dst));
            if (mit != g_state.edge_meta.end()) {
                souffle::tuple t(edge_data_rel);
                t[0] = sym.encode(src);
                t[1] = sym.encode(dst);
                t[2] = encode_meta(mit->second);
                edge_data_rel->insert(t);
            }
        }
        auto insert_attr = [&](souffle::Relation* rel,
                               const std::map<EdgeKey, std::string>& values) {
            if (!rel) return;
            auto vit = values.find(edge_key(session_filter, src, dst));
            if (vit == values.end()) return;
            souffle::tuple t(rel); t << src << dst << vit->second; rel->insert(t);
        };
        insert_attr(edge_principal_rel, g_state.edge_principal);
        insert_attr(edge_entity_rel, g_state.edge_entity);
    };

    auto insert_node_tuple = [&](const std::string& id, const GraphNode& node) {
        if (sent_rel) {
            souffle::tuple t(sent_rel);
            t[0] = sym.encode(id);
            t[1] = encode_message(sym, rec, node.contents, node.agent, node.role, node.entity, node.tools_json, node.principal);
            sent_rel->insert(t);
        }
        if (tool_rel) {
            auto trit = g_state.tool_results.find(id);
            if (trit != g_state.tool_results.end()) {
                souffle::tuple t(tool_rel);
                t << id << trit->second.first << trit->second.second;
                tool_rel->insert(t);
            }
        }
        if (metadata_rel) {
            auto mdit = g_state.node_metadata.find({session_filter, id});
            if (mdit != g_state.node_metadata.end()) {
                souffle::tuple t(metadata_rel);
                t << id << mdit->second;
                metadata_rel->insert(t);
            }
        }
    };

    auto nit = g_state.session_nodes.find(session_filter);
    if (nit != g_state.session_nodes.end()) {
        for (const auto& id : nit->second) {
            auto vit = g_state.nodes.find(id);
            if (vit != g_state.nodes.end()) insert_node_tuple(id, vit->second);
        }
    }
    auto eit = g_state.session_edges.find(session_filter);
    if (eit != g_state.session_edges.end()) {
        for (const auto& [src, dst] : eit->second) insert_edge_tuple(src, dst);
    }
}

// Insert a single node into the persistent program's relations
static void insert_node(const GraphNode& node) {
    if (!g_prog) return;
    auto& sym = g_prog->getSymbolTable();
    auto& rec = g_prog->getRecordTable();

    if (auto* rel = g_prog->getRelation("SentMessage")) {
        souffle::tuple t(rel);
        t[0] = sym.encode(node.id);
        t[1] = encode_message(sym, rec, node.contents, node.agent, node.role, node.entity, node.tools_json, node.principal);
        rel->insert(t);
    }
}

static void insert_edge(const std::string& src, const std::string& dst) {
    if (!g_prog) return;
    if (auto* rel = g_prog->getRelation("Edge")) {
        souffle::tuple t(rel); t << src << dst; rel->insert(t);
    }
}

// Insert one attribution tuple (EdgePrincipal / EdgeEntity) for an
// edge. A no-op when the policy doesn't declare the relation, the
// same guard the EdgeData insertion uses.
static void insert_edge_attr(const char* relation, const std::string& src,
                             const std::string& dst, const std::string& value) {
    if (!g_prog) return;
    auto* rel = g_prog->getRelation(relation);
    if (!rel) return;
    souffle::tuple t(rel); t << src << dst << value; rel->insert(t);
}

static void insert_edge_data(const std::string& src, const std::string& dst,
                              const EdgeMeta& meta) {
    if (!g_prog) return;
    auto* rel = g_prog->getRelation("EdgeData");
    if (!rel) return;
    auto& sym = g_prog->getSymbolTable();
    auto& rec = g_prog->getRecordTable();
    auto record = rec.pack({
        souffle::RamDomain(meta.proximal),
        souffle::RamDomain(meta.message_index),
    });
    souffle::tuple t(rel);
    t[0] = sym.encode(src);
    t[1] = sym.encode(dst);
    t[2] = record;
    rel->insert(t);
}

static void insert_tool_result(const std::string& id, const std::string& fn, const std::string& args) {
    if (!g_prog) return;
    if (auto* rel = g_prog->getRelation("ToolResult")) {
        souffle::tuple t(rel); t << id << fn << args; rel->insert(t);
    }
}

static void insert_message_metadata(const std::string& id, const std::string& metadata) {
    if (!g_prog) return;
    if (auto* rel = g_prog->getRelation("MessageMetadata")) {
        souffle::tuple t(rel); t << id << metadata; rel->insert(t);
    }
}

// ── Apply graph updates (incremental) ─────────────────────────────

static void apply_updates(const json::Value& updates) {
    // Each update is routed to its session's SouffleProgram by
    // swapping g_prog before invoking the existing insert_*
    // helpers. The helpers read g_prog implicitly, so the only
    // plumbing needed at this layer is the swap.
    //
    // Updates from different sessions land in different program
    // instances; their relations don't share state, so a query
    // for session A is unaffected by inserts for session B (and
    // there's nothing to "rebuild" on session switch — the right
    // program is already up-to-date for its session).
    for (size_t i = 0; i < updates.size(); ++i) {
        const auto& u = updates[i];
        if (!u["NodeCreated"].is_null()) {
            const auto& node = u["NodeCreated"];
            GraphNode gn;
            gn.id = node["id"].str();
            gn.contents = node["content"].is_null() ? "" : node["content"].str();
            gn.agent = node["agent"].is_null() ? "" : node["agent"].str();
            gn.role = node["role"].is_null() ? "assistant" : node["role"].str();
            gn.entity = node["entity"].is_null() ? "" : node["entity"].str();
            gn.principal = node["principal"].is_null() ? "" : node["principal"].str();
            gn.session_id = node["session_id"].is_null() ? "" : node["session_id"].str();
            // tools: serialize the array of {name, arguments} objects
            // back to JSON so the policy can iterate via @json_array_*
            // functors. Empty array when absent or not an array.
            if (!node["tools"].is_null() && node["tools"].type == json::Value::ARR) {
                gn.tools_json = json::serialize(node["tools"]);
            } else {
                gn.tools_json = "[]";
            }
            g_prog = prog_for_session(gn.session_id);
            if (!g_prog) continue;
            auto previous = g_state.nodes.find(gn.id);
            const bool replacing = previous != g_state.nodes.end();
            const std::string previous_session = replacing
                ? previous->second.session_id
                : std::string();
            const std::string replacement_session = gn.session_id;
            // An update that carries no `derived_from` means "this
            // update says nothing about provenance", NOT "this node has
            // none". Erasing on absence would make every re-record of a
            // tool message drop its ToolResult fact and empty every policy
            // rule that joins on ToolResult. Only an explicit value
            // replaces the existing provenance.
            if (!node["derived_from"].is_null()) {
                auto fn = node["derived_from"]["name"].str();
                auto args = node["derived_from"]["arguments"].str();
                g_state.tool_results[gn.id] = {fn, args};
                if (!replacing) insert_tool_result(gn.id, fn, args);
            }
            // Same rule for the adapter's record of the message: an
            // update that carries no `metadata` says nothing about it,
            // so a re-record without one keeps the existing fact rather
            // than dropping it and emptying every rule that joins on
            // MessageMetadata. "Existing" means the value this session
            // recorded: the value is held under (session, id), so a
            // second session that re-records the same id inherits
            // nothing and overwrites nothing.
            //
            // Kept only when the compiled policy has the relation.
            // Soufflé drops an input relation no rule reads, so a policy
            // that never names MessageMetadata has none — and a whole
            // message as JSON is large enough that holding one per node
            // for a policy that cannot read it is worth skipping. The
            // same `getRelation` guard the EdgeData insertion uses.
            if (!node["metadata"].is_null() && g_prog->getRelation("MessageMetadata")) {
                auto metadata = node["metadata"].str();
                g_state.node_metadata[{gn.session_id, gn.id}] = metadata;
                if (!replacing) insert_message_metadata(gn.id, metadata);
            }
            if (!replacing) {
                insert_node(gn);
                g_state.session_nodes[gn.session_id].push_back(gn.id);
            } else if (previous_session != gn.session_id) {
                auto& previous_bucket = g_state.session_nodes[previous_session];
                previous_bucket.erase(
                    std::remove(previous_bucket.begin(), previous_bucket.end(), gn.id),
                    previous_bucket.end()
                );
                g_state.session_nodes[gn.session_id].push_back(gn.id);
            }
            g_state.nodes[gn.id] = std::move(gn);
            if (replacing) {
                // Souffle relations have no tuple-delete API. Rebuild from the
                // canonical maps so stale message and tool-result facts vanish.
                g_session_needs_rebuild[previous_session] = true;
                g_session_needs_rebuild[replacement_session] = true;
            }
        } else if (!u["NodeDeleted"].is_null()) {
            std::string id = u["NodeDeleted"].str();
            auto it = g_state.nodes.find(id);
            std::string sid;
            if (it != g_state.nodes.end()) {
                sid = it->second.session_id;
                auto& bucket = g_state.session_nodes[sid];
                bucket.erase(std::remove(bucket.begin(), bucket.end(), id), bucket.end());
            }
            g_state.nodes.erase(id);
            g_state.tool_results.erase(id);
            // A deletion names an id and nothing else, and `nodes` holds
            // one node per id, so it removes the node from every session
            // at once. The metadata entries follow it: every session's
            // value for the id goes, and every session that held one
            // rebuilds, since a program still holding the row would keep
            // answering from a node that no longer exists.
            for (auto mit = g_state.node_metadata.begin();
                 mit != g_state.node_metadata.end();) {
                if (mit->first.second == id) {
                    g_session_needs_rebuild[mit->first.first] = true;
                    mit = g_state.node_metadata.erase(mit);
                } else {
                    ++mit;
                }
            }
            // Souffle has no delete API — the node's own session
            // rebuilds on its next query.
            g_session_needs_rebuild[sid] = true;
        } else if (!u["EdgeCreated"].is_null()) {
            const auto& e = u["EdgeCreated"];
            auto src = e["source"].str();
            auto dst = e["destination"].str();
            std::string edge_sid;
            if (!e["session_id"].is_null()) {
                edge_sid = e["session_id"].str();
            }
            g_state.edge_sessions[{src, dst}].insert(edge_sid);
            const EdgeKey key = edge_key(edge_sid, src, dst);
            g_prog = prog_for_session(edge_sid);
            if (!g_prog) continue;
            // Only add to session bucket on first insertion so
            // re-applies (incremental retry, etc.) don't duplicate.
            // An edge already known to another session is new to this
            // one, so it lands in this session's bucket too.
            bool is_new_edge = g_state.edges.insert(key).second;
            insert_edge(src, dst);
            if (is_new_edge) {
                g_state.session_edges[edge_sid].push_back({src, dst});
            }
            // Optional instrumentation metadata. Each field is
            // independent; we emit an EdgeData tuple when any is
            // set. Unset fields are represented as -1 in the
            // Soufflé record.
            EdgeMeta meta;
            if (!e["message_index"].is_null()) {
                meta.message_index = e["message_index"].num();
            }
            if (!e["proximal"].is_null()) {
                meta.proximal = e["proximal"].boolean() ? 1 : 0;
            }
            // An update announces the edge as the store now holds it,
            // so a re-record can change a value; `principal` can also
            // go away, being stamped from the recording request at the
            // auth boundary and never merged with the stored edge's.
            // Souffle has no tuple delete, so a superseded value can
            // only leave the program through a purge and rebuild:
            // insert incrementally only when the value is new for this
            // edge, and mark the session for rebuild whenever a stored
            // value changed or went away.
            bool superseded = false;
            auto mit = g_state.edge_meta.find(key);
            if (meta.has_any()) {
                if (mit == g_state.edge_meta.end()) {
                    g_state.edge_meta.emplace(key, meta);
                    insert_edge_data(src, dst, meta);
                } else if (mit->second.message_index != meta.message_index ||
                           mit->second.proximal != meta.proximal) {
                    mit->second = meta;
                    superseded = true;
                }
            } else if (mit != g_state.edge_meta.end()) {
                g_state.edge_meta.erase(mit);
                superseded = true;
            }
            // Who asserted the edge. Both are optional and land in
            // their own relations, with the same lifecycle as EdgeData.
            auto apply_attr = [&](std::map<EdgeKey, std::string>& values,
                                  const json::Value& field,
                                  const char* relation) {
                auto vit = values.find(key);
                if (field.is_null()) {
                    if (vit == values.end()) return false;
                    values.erase(vit);
                    return true;
                }
                auto value = field.str();
                if (vit == values.end()) {
                    values.emplace(key, value);
                    insert_edge_attr(relation, src, dst, value);
                    return false;
                }
                if (vit->second == value) return false;
                vit->second = value;
                return true;
            };
            if (apply_attr(g_state.edge_principal, e["principal"], "EdgePrincipal")) {
                superseded = true;
            }
            if (apply_attr(g_state.edge_entity, e["entity"], "EdgeEntity")) {
                superseded = true;
            }
            if (superseded) {
                // Rebuild the program the edge belongs to. Every
                // per-edge value this update touched is keyed by the
                // announcing session, so only that session's program
                // holds a superseded tuple.
                g_session_needs_rebuild[edge_sid] = true;
            }
        } else if (!u["EdgeDeleted"].is_null()) {
            auto src = u["EdgeDeleted"]["source"].str();
            auto dst = u["EdgeDeleted"]["destination"].str();
            // The store deletes an edge, not one session's view of it,
            // so it leaves every session that announced the pair.
            auto sit = g_state.edge_sessions.find({src, dst});
            std::set<std::string> sids;
            if (sit != g_state.edge_sessions.end()) sids = sit->second;
            sids.insert(std::string());
            for (const auto& sid : sids) {
                auto& bucket = g_state.session_edges[sid];
                bucket.erase(
                    std::remove(bucket.begin(), bucket.end(), std::make_pair(src, dst)),
                    bucket.end()
                );
                const EdgeKey key = edge_key(sid, src, dst);
                g_state.edges.erase(key);
                g_state.edge_meta.erase(key);
                g_state.edge_principal.erase(key);
                g_state.edge_entity.erase(key);
                g_session_needs_rebuild[sid] = true;
            }
            g_state.edge_sessions.erase({src, dst});
        }
    }
}

// ── Run query ──────────────────────────────────────────────────────

static json::Value run_query(const json::Value& request) {
    // Route to this session's program instance. The instance's
    // input relations are already materialised for the session
    // (apply_updates writes incrementally into the right one), so
    // a session switch is just a map lookup — no purge, no
    // re-encode, no rebuild.
    std::string query_session;
    if (!request["session_id"].is_null()) {
        query_session = request["session_id"].str();
    }
    g_prog = prog_for_session(query_session);
    if (!g_prog) {
        return json::Value(json::Object{
            {"Error", json::Value(json::Object{
                {"message", json::Value("Program not initialized")}
            })}
        });
    }

    // Only this session's program needs a rebuild, and only after
    // a delete in this session since the last query. Souffle has
    // no delete API, so the only fix is purge + replay from
    // g_state's per-session reverse indices.
    auto rebuild_it = g_session_needs_rebuild.find(query_session);
    if (rebuild_it != g_session_needs_rebuild.end() && rebuild_it->second) {
        rebuild_input_relations(query_session);
        rebuild_it->second = false;
    }

    // Purge derived relations from previous query
    g_prog->purgeOutputRelations();
    g_prog->purgeInternalRelations();

    // Clear LLM local cache — each query gets fresh oracle calls.
    // The engine's optional Redis/Valkey cache handles cross-query caching.
    llm_cache_clear();

    // Purge instance input relations (per-query)
    auto purge_rel = [](souffle::SouffleProgram* p, const char* name) {
        if (auto* r = p->getRelation(name)) r->purge();
    };
    purge_rel(g_prog, "Current");
    purge_rel(g_prog, "Actions");
    purge_rel(g_prog, "ActionMetadata");
    purge_rel(g_prog, "Principal");
    purge_rel(g_prog, "Entity");
    purge_rel(g_prog, "PrincipalRole");
    purge_rel(g_prog, "TenantId");

    auto& symTab = g_prog->getSymbolTable();
    auto& recTab = g_prog->getRecordTable();

    // ── ADT encoding (alphabetical branch tags) ────────────────────
    // Action: CallTool=0, HTTPRequest=1, SendAttempt=2
    auto encode_call_tool = [&](const std::string& fn_name, const std::string& args) -> souffle::RamDomain {
        auto inner = recTab.pack({symTab.encode(fn_name), symTab.encode(args)});
        return recTab.pack({souffle::RamDomain(0), inner});
    };
    auto encode_http_request = [&](const std::string& url, const std::string& body) -> souffle::RamDomain {
        auto inner = recTab.pack({symTab.encode(url), symTab.encode(body), symTab.encode("[]")});
        return recTab.pack({souffle::RamDomain(1), inner});
    };
    auto encode_send_attempt = [&](const std::string& content, const std::string& agent,
                                   const std::string& role, const std::string& entity) -> souffle::RamDomain {
        auto inner = recTab.pack({encode_message(symTab, recTab, content, agent, role, entity)});
        return recTab.pack({souffle::RamDomain(2), inner});
    };

    // ── Insert instance facts only (persistent facts already loaded) ──

    if (auto* rel = g_prog->getRelation("Current")) {
        for (size_t i = 0; i < request["current_node_ids"].size(); ++i) {
            souffle::tuple t(rel);
            t << request["current_node_ids"][i].str();
            rel->insert(t);
        }
    }

    if (auto* rel = g_prog->getRelation("Actions")) {
        const auto& actions_arr = request["actions"];
        for (size_t i = 0; i < actions_arr.size(); ++i) {
            const auto& action = actions_arr[i];
            souffle::RamDomain action_rd;

            if (!action["ToolCall"].is_null()) {
                action_rd = encode_call_tool(
                    action["ToolCall"]["fn_name"].str(),
                    action["ToolCall"]["args"].str());
            } else if (!action["HttpRequest"].is_null()) {
                action_rd = encode_http_request(
                    action["HttpRequest"]["url"].str(),
                    action["HttpRequest"]["body"].str());
            } else if (!action["SendMessage"].is_null()) {
                action_rd = encode_send_attempt(
                    action["SendMessage"]["content"].str(),
                    action["SendMessage"]["agent"].str(),
                    action["SendMessage"]["agent_role"].str(),
                    action["SendMessage"]["entity"].is_null() ? "" : action["SendMessage"]["entity"].str());
            } else {
                continue;
            }

            souffle::tuple t(rel);
            t[0] = static_cast<souffle::RamDomain>(i);
            t[1] = action_rd;
            rel->insert(t);
        }
    }

    // Per-action metadata → ActionMetadata(idx, rel, a, b). Daemon-resolved
    // external context (e.g. supply-chain verdicts) for specific actions,
    // indexed by position in `actions`. Per-query instance relation, purged
    // above like Current/Actions.
    if (auto* rel = g_prog->getRelation("ActionMetadata")) {
        const auto& am = request["action_metadata"];
        for (size_t i = 0; i < am.size(); ++i) {
            const auto& entry = am[i];
            auto idx = static_cast<souffle::RamDomain>(entry["index"].num());
            const auto& facts = entry["facts"];
            for (size_t j = 0; j < facts.size(); ++j) {
                const auto& f = facts[j];
                souffle::tuple t(rel);
                t[0] = idx;
                t[1] = symTab.encode(f["rel"].str());
                t[2] = symTab.encode(f["a"].str());
                t[3] = symTab.encode(f["b"].str());
                rel->insert(t);
            }
        }
    }

    // Per-request principal — auth-derived, immutable. Populates
    // the `Principal` relation; `PrincipalRole` is keyed on the same
    // principal id.
    if (!request["principal"].is_null()) {
        auto principal_value = request["principal"].str();
        if (auto* rel = g_prog->getRelation("Principal")) {
            souffle::tuple t(rel);
            t << principal_value;
            rel->insert(t);
        }
        if (auto* rel = g_prog->getRelation("PrincipalRole")) {
            for (size_t i = 0; i < request["roles"].size(); ++i) {
                souffle::tuple t(rel);
                t << principal_value << request["roles"][i].str();
                rel->insert(t);
            }
        }
    }

    // Per-request user-supplied entity (free-form, not auth-attested).
    if (!request["entity"].is_null()) {
        if (auto* rel = g_prog->getRelation("Entity")) {
            souffle::tuple t(rel);
            t << request["entity"].str();
            rel->insert(t);
        }
    }

    if (!request["tenant_id"].is_null()) {
        if (auto* rel = g_prog->getRelation("TenantId")) {
            souffle::tuple t(rel);
            t << request["tenant_id"].str();
            rel->insert(t);
        }
    }

    // Run evaluation (semi-naive on current state)
    g_prog->run();

    // Extract results — use operator[] for direct RamDomain access
    // to avoid type assertion issues with unsigned/ADT columns
    const size_t action_count = request["actions"].size();
    const auto decode_result_index = [action_count](
        souffle::RamDomain raw_index,
        uint32_t& index
    ) {
        return evaluator_protocol::decode_query_result_index(
            static_cast<int64_t>(raw_index), action_count, index
        );
    };
    const auto invalid_result_index = []() {
        return evaluator_protocol::evaluator_error(
            "invalid evaluator result index"
        );
    };

    std::set<uint32_t> authorized_indices;
    if (auto* rel = g_prog->getRelation("Authorized")) {
        for (auto& row : *rel) {
            uint32_t index = 0;
            if (!decode_result_index(row[0], index)) {
                return invalid_result_index();
            }
            authorized_indices.insert(index);
        }
    }

    bool has_auth = false;
    if (auto* rel = g_prog->getRelation("HasPrincipal")) {
        has_auth = rel->size() > 0;
    }

    // Per-action denylist/allowlist, keyed by action index
    std::set<uint32_t> denied_indices, allowed_indices;
    if (auto* rel = g_prog->getRelation("Unauthorized")) {
        for (auto& row : *rel) {
            uint32_t index = 0;
            if (!decode_result_index(row[0], index)) {
                return invalid_result_index();
            }
            denied_indices.insert(index);
        }
    }
    if (auto* rel = g_prog->getRelation("IsAuthorized")) {
        for (auto& row : *rel) {
            uint32_t index = 0;
            if (!decode_result_index(row[0], index)) {
                return invalid_result_index();
            }
            allowed_indices.insert(index);
        }
    }

    // Extract denial reasons (correct by construction — only fired rules produce tuples).
    // DenialReason is 4-ary: (idx, kind, reason, suggestion); kind ∈ {block, ask}.
    // An "ask"-kind reason is a soft denial: the action stays authorized at the
    // Datalog level (only "block" derives Unauthorized) but requires user approval.
    std::map<uint32_t, json::Array> denial_reasons;
    std::set<uint32_t> ask_indices;
    if (auto* rel = g_prog->getRelation("DenialReason")) {
        for (auto& row : *rel) {
            uint32_t idx = 0;
            if (!decode_result_index(row[0], idx)) {
                return invalid_result_index();
            }
            std::string kind = symTab.decode(row[1]);
            if (kind == "ask") ask_indices.insert(idx);
            denial_reasons[idx].push_back(json::Value(json::Object{
                {"kind", json::Value(kind)},
                {"reason", json::Value(symTab.decode(row[2]))},
                {"suggestion", json::Value(symTab.decode(row[3]))},
            }));
        }
    }

    // Extract transforms
    std::map<uint32_t, json::Array> transforms;
    if (auto* rel = g_prog->getRelation("ApplyTransform")) {
        for (auto& row : *rel) {
            uint32_t idx = 0;
            if (!decode_result_index(row[0], idx)) {
                return invalid_result_index();
            }
            const std::string& tid = symTab.decode(row[1]);
            transforms[idx].push_back(json::Value(tid));
        }
    }

    // Extract deny/passthrough indices
    std::set<uint32_t> passthrough_indices;
    if (auto* rel = g_prog->getRelation("AllowPassthrough")) {
        for (auto& row : *rel) {
            uint32_t index = 0;
            if (!decode_result_index(row[0], index)) {
                return invalid_result_index();
            }
            passthrough_indices.insert(index);
        }
    }

    std::map<uint32_t, json::Array> allow_routes;
    if (auto* rel = g_prog->getRelation("SasyAllowRoute")) {
        bool valid_schema = rel->getArity() == 6;
        if (valid_schema) {
            valid_schema = *rel->getAttrType(0) == 'u';
            for (size_t column = 1; column < 6; ++column)
                valid_schema = valid_schema && *rel->getAttrType(column) == 's';
        }
        if (valid_schema) {
            size_t bytes = 0, rows = 0;
            for (auto& row : *rel) {
                uint32_t idx = 0;
                if (!decode_result_index(row[0], idx) || ++rows > 512) {
                    allow_routes.clear(); break;
                }
                json::Object diagnostic;
                const char* fields[] = {"rule_id", "status", "details", "suggestion", "source_location"};
                for (size_t column = 1; column < 6; ++column) {
                    const auto& value = symTab.decode(row[column]);
                    bytes += value.size();
                    if (bytes > 262144) break;
                    diagnostic[fields[column - 1]] = json::Value(value);
                }
                if (bytes > 262144) { allow_routes.clear(); break; }
                allow_routes[idx].push_back(json::Value(std::move(diagnostic)));
            }
        }
    }

    return evaluator_protocol::build_query_result(
        action_count, has_auth, authorized_indices, denied_indices,
        allowed_indices, passthrough_indices, ask_indices, denial_reasons,
        transforms, allow_routes
    );
}

// ── Filesystem sandbox ─────────────────────────────────────────────

/// Block dangerous evaluator syscalls such as ptrace, mount, bpf, and reboot.
/// Under bwrap, use this deny-list instead of the standalone allow-list: libc
/// initialization can require additional syscalls in the namespace environment.
/// Matches return EPERM. This supplements bwrap's namespace isolation.
static void apply_seccomp_denylist() {
#ifdef __linux__
#if defined(__x86_64__)
    constexpr uint32_t audit_arch_val = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
    constexpr uint32_t audit_arch_val = AUDIT_ARCH_AARCH64;
#else
    std::cerr << "Sandbox: seccomp deny-list unsupported arch; relying on bwrap\n";
    return;
#endif
    static const int denied[] = {
        SYS_ptrace,
        SYS_mount, SYS_pivot_root, SYS_setns, SYS_unshare,
        SYS_bpf, SYS_perf_event_open,
        SYS_keyctl, SYS_add_key, SYS_request_key,
        SYS_reboot, SYS_kexec_load,
#ifdef SYS_kexec_file_load
        SYS_kexec_file_load,
#endif
#ifdef SYS_umount2
        SYS_umount2,
#endif
#ifdef SYS_init_module
        SYS_init_module,
#endif
#ifdef SYS_finit_module
        SYS_finit_module,
#endif
#ifdef SYS_delete_module
        SYS_delete_module,
#endif
#ifdef SYS_process_vm_readv
        SYS_process_vm_readv,
#endif
#ifdef SYS_process_vm_writev
        SYS_process_vm_writev,
#endif
#ifdef SYS_open_by_handle_at
        SYS_open_by_handle_at,
#endif
#ifdef SYS_pidfd_getfd
        SYS_pidfd_getfd,
#endif
    };
    constexpr size_t M = sizeof(denied) / sizeof(denied[0]);

    std::vector<struct sock_filter> bpf;
    bpf.reserve(6 + M);
    // Reject foreign architectures outright (defends against arch bypass).
    bpf.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                            offsetof(struct seccomp_data, arch)));
    bpf.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, audit_arch_val, 1, 0));
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    // Load syscall number; each denied syscall jumps to the EPERM return.
    bpf.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                            offsetof(struct seccomp_data, nr)));
    for (size_t i = 0; i < M; ++i) {
        uint8_t jt = static_cast<uint8_t>(M - i);  // jump to ERRNO stmt
        bpf.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                                static_cast<uint32_t>(denied[i]), jt, 0));
    }
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K,
                            SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)));

    struct sock_fprog prog = {
        .len = static_cast<unsigned short>(bpf.size()),
        .filter = bpf.data(),
    };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        std::cerr << "Sandbox: PR_SET_NO_NEW_PRIVS failed (errno=" << errno << ")\n";
        return;
    }
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0) {
        std::cerr << "Sandbox: bwrap-path seccomp deny-list failed (errno="
                  << errno << ")\n";
        return;
    }
    std::cerr << "Sandbox: seccomp deny-list active under bwrap ("
              << M << " syscalls blocked)\n";
#endif
}

/// Apply OS isolation after program initialization.
/// Without bwrap, attempt a network namespace, chroot, and syscall allow-list.
/// When SASY_SKIP_IN_PROC_SANDBOX=1 marks host-provided bwrap isolation, apply
/// the compatible syscall deny-list instead of repeating namespace setup.
/// This in-process filter runs after static constructors; the host's pre-exec
/// filter is required to cover those constructors too.
static void sandbox_evaluator() {
#ifdef __linux__
    if (const char* skip = getenv("SASY_SKIP_IN_PROC_SANDBOX");
        skip && skip[0] == '1') {
        // bwrap already applied userns/mountns/netns isolation + chroot
        // equivalent. The full in-process allow-list is unsafe here (it
        // SIGKILLs under bwrap's glibc env), so layer a seccomp DENY-list
        // on top instead — see apply_seccomp_denylist().
        // The host sets
        // SASY_SECCOMP_PRE_EXEC=1 when it handed bwrap the same deny-list on a
        // descriptor, so bwrap installed it before this process ran its first
        // instruction — including the static constructors of any functor C++
        // linked in here. Without it, this copy is the first filter, and the
        // constructors have already run unfiltered.
        if (const char* pre = getenv("SASY_SECCOMP_PRE_EXEC");
            pre && pre[0] == '1') {
            std::cerr << "Sandbox: seccomp deny-list already installed before exec by bwrap; "
                         "installing the in-process copy as well\n";
        } else {
            std::cerr << "Sandbox: chroot/netns delegated to bwrap; applying seccomp deny-list "
                         "in-process, after this binary's static constructors\n";
        }
        apply_seccomp_denylist();
        return;
    }

    // ── 1. Network namespace isolation ──
    if (unshare(CLONE_NEWNET) == 0) {
        std::cerr << "Sandbox: network namespace isolated\n";
    } else {
        std::cerr << "Sandbox: unshare(CLONE_NEWNET) failed (errno="
                  << errno << "), network still accessible\n";
    }

    // ── 2. Chroot to empty directory ──
    const char* jail = "/tmp/.souffle-sandbox";
    mkdir(jail, 0700);  // ok if exists

    if (chroot(jail) == 0) {
        chdir("/");
        std::cerr << "Sandbox: chroot to " << jail << "\n";
    } else {
        std::cerr << "Sandbox: chroot failed (errno=" << errno
                  << "), filesystem still accessible\n";
    }

    // ── 2. Seccomp BPF filter ──
    // Architecture audit value for BPF
#if defined(__x86_64__)
    constexpr uint32_t audit_arch_val = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
    constexpr uint32_t audit_arch_val = AUDIT_ARCH_AARCH64;
#else
    std::cerr << "Sandbox: seccomp not supported on this architecture\n";
    return;
#endif

    // Syscall allow-list (everything else → SIGSYS kill)
    static const int allowed[] = {
        // stdio IPC
        SYS_read, SYS_write, SYS_close,
        // memory
        SYS_mmap, SYS_munmap, SYS_mprotect, SYS_brk, SYS_mremap, SYS_madvise,
        // threading (Soufflé parallel evaluation)
        SYS_clone, SYS_futex, SYS_set_robust_list, SYS_get_robust_list,
        SYS_rseq, SYS_sched_getaffinity, SYS_sched_yield,
        SYS_set_tid_address,
#ifdef SYS_clone3
        SYS_clone3,
#endif
        // signals
        SYS_rt_sigaction, SYS_rt_sigprocmask, SYS_rt_sigreturn, SYS_sigaltstack,
        // process lifecycle
        SYS_exit, SYS_exit_group, SYS_getpid, SYS_gettid, SYS_tgkill,
        // File metadata on pre-opened fds. newfstatat is needed by
        // Soufflé's runtime during evaluation. openat is NOT allowed,
        // so file contents cannot be read. stat() can probe existence
        // but chroot (when available) makes the filesystem empty.
        SYS_fstat, SYS_newfstatat,
        // misc glibc startup
        SYS_getrandom, SYS_prlimit64,
#ifdef SYS_arch_prctl
        SYS_arch_prctl,
#endif
        // chroot/chdir already happened, these are harmless now
        SYS_mkdirat, SYS_chroot, SYS_chdir,
    };
    constexpr size_t N = sizeof(allowed) / sizeof(allowed[0]);

    // Build BPF program:
    //   if (arch != AUDIT_ARCH) kill
    //   for each allowed syscall: if (nr == allowed[i]) allow
    //   kill (default)
    std::vector<struct sock_filter> bpf;
    bpf.reserve(3 + 2 * N + 2);

    // Load architecture
    bpf.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                            offsetof(struct seccomp_data, arch)));
    bpf.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, audit_arch_val, 1, 0));
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));

    // Load syscall number
    bpf.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                            offsetof(struct seccomp_data, nr)));

    // Check each allowed syscall
    for (size_t i = 0; i < N; ++i) {
        uint8_t remaining = static_cast<uint8_t>(N - i);
        bpf.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                                static_cast<uint32_t>(allowed[i]),
                                remaining, 0));
    }

    // Default: kill
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    // Allow
    bpf.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));

    struct sock_fprog prog = {
        .len = static_cast<unsigned short>(bpf.size()),
        .filter = bpf.data(),
    };

    // Enable seccomp strict mode with no-new-privs
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        std::cerr << "Sandbox: PR_SET_NO_NEW_PRIVS failed (errno=" << errno << ")\n";
        return;
    }
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0) {
        std::cerr << "Sandbox: seccomp filter failed (errno=" << errno << ")\n";
        return;
    }

    std::cerr << "Sandbox: seccomp filter active (" << N << " syscalls allowed)\n";
#endif
}

// ── Main loop ──────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // The program name matches what REGISTER_PROGRAM uses in the generated code.
    // It is derived from the .dl filename (e.g. "my_policy" from my_policy.dl).
    // The engine always passes it; the default matches the name it compiles to.
    if (argc > 1) {
        g_program_name = argv[1];
    } else {
        g_program_name = "policy_program";
    }

    // Bootstrap the default ("") session program. Additional
    // per-session instances are created lazily on first use via
    // prog_for_session.
    g_prog = prog_for_session(std::string());
    if (!g_prog) {
        std::cerr << "ERROR: Soufflé program '" << g_program_name << "' not registered.\n";
        std::cerr << "The generated C++ code must be linked into this binary.\n";
        return 1;
    }
    std::cerr << "Soufflé evaluator ready (program=" << g_program_name << ")\n";

    // Sandbox: chroot + seccomp after program is loaded.
    // All code/data is in memory; only stdin/stdout IPC needed.
    sandbox_evaluator();

    std::string msg;
    uint64_t request_id = 0;
    while (read_message(msg, request_id)) {
        g_current_request_id = request_id;
        auto request = json::parse(msg);
        std::string response_json;

        // Serde enum variants serialize as:
        //   unit variants: "Reset", "Shutdown"
        //   struct variants: {"Update": {...}}, {"Query": {...}}
        //   newtype variants: {"Query": {...}}
        bool handled = false;

        if (!request["Update"].is_null()) {
            apply_updates(request["Update"]["updates"]);
            response_json = "\"UpdateOk\"";
            handled = true;
        } else if (!request["SetMetadata"].is_null()) {
            // Static policy config → PolicyMetadata EDB. Reseed all live
            // session programs so it takes effect immediately.
            g_policy_metadata.clear();
            const auto& facts = request["SetMetadata"]["facts"];
            for (size_t i = 0; i < facts.size(); ++i) {
                const auto& f = facts[i];
                g_policy_metadata.push_back({f["rel"].str(), f["a"].str(), f["b"].str()});
            }
            for (auto& [sid, p] : g_session_progs) {
                if (auto* rel = p->getRelation("PolicyMetadata")) rel->purge();
                seed_policy_metadata(p);
            }
            response_json = "\"UpdateOk\"";
            handled = true;
        } else if (!request["Query"].is_null()) {
            auto result = run_query(request["Query"]);
            response_json = json::serialize(result);
            handled = true;
        } else if (!request["LlmResult"].is_null()) {
            // Unsolicited LlmResult outside a query — should not happen.
            response_json = "{\"Error\":{\"message\":\"Unexpected LlmResult outside query\"}}";
            handled = true;
        }

        // Handle bare string variants
        if (!handled && request.type == json::Value::STR) {
            const auto& s = request.str();
            if (s == "Reset") {
                g_state.nodes.clear();
                g_state.edges.clear();
                g_state.edge_meta.clear();
                g_state.edge_principal.clear();
                g_state.edge_entity.clear();
                g_state.edge_sessions.clear();
                g_state.tool_results.clear();
                g_state.node_metadata.clear();
                g_state.session_nodes.clear();
                g_state.session_edges.clear();
                // Drop every per-session program. Re-seed the
                // default ("") instance so queries with no
                // session_id still find a program.
                for (auto& [sid, p] : g_session_progs) {
                    delete p;
                }
                g_session_progs.clear();
                g_session_needs_rebuild.clear();
                g_prog = prog_for_session(std::string());
                response_json = "\"ResetOk\"";
                handled = true;
            } else if (s == "Shutdown") {
                break;
            }
        }

        if (!handled) {
            response_json = "{\"Error\":{\"message\":\"Unknown request type\"}}";
        }

        write_message(response_json, request_id);
    }

    delete g_prog;
    g_prog = nullptr;
    return 0;
}
