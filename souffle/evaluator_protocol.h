#ifndef SASY_SOUFFLE_EVALUATOR_PROTOCOL_H
#define SASY_SOUFFLE_EVALUATOR_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>

#include "json_string_codec.h"

namespace sasy_evaluator_protocol {

namespace json = sasy_json_codec;

inline json::Value evaluator_error(const std::string& message) {
    return json::Value(json::Object{{
        "Error", json::Value(json::Object{{
            "message", json::Value(message)
        }})
    }});
}

inline bool decode_query_result_index(
    int64_t raw_index,
    size_t action_count,
    uint32_t& index
) {
    if (raw_index < 0 ||
        action_count > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(raw_index) >= action_count) {
        return false;
    }
    index = static_cast<uint32_t>(raw_index);
    return true;
}

// Diagnostics are optional and cannot turn a valid decision into an error.
// Discard an incomplete/malformed set rather than present partial attribution.
inline json::Array checked_allow_routes(const json::Array& rows) {
    if (rows.size() > 512) return {};
    size_t remaining = 262144;
    for (const auto& row : rows) {
        for (const auto* field : {"rule_id", "status", "details", "suggestion", "source_location"}) {
            const auto& value = row[field];
            if (value.type != json::Value::STR || value.str().size() > remaining) return {};
            remaining -= value.str().size();
        }
        const auto status = row["status"].str();
        if (row["rule_id"].str().empty() ||
            (status != "blocked" && status != "possible" && status != "unknown")) return {};
    }
    return rows;
}

inline json::Value build_query_result(
    size_t action_count,
    bool has_auth,
    const std::set<uint32_t>& authorized,
    const std::set<uint32_t>& denied,
    const std::set<uint32_t>& allowed,
    const std::set<uint32_t>& passthrough,
    const std::set<uint32_t>& ask,
    const std::map<uint32_t, json::Array>& denial_reasons,
    const std::map<uint32_t, json::Array>& transforms,
    const std::map<uint32_t, json::Array>& allow_routes = {}
) {
    if (action_count > std::numeric_limits<uint32_t>::max()) {
        return evaluator_error("invalid evaluator action count");
    }
    const auto valid_index = [action_count](uint32_t index) {
        return static_cast<size_t>(index) < action_count;
    };
    for (const auto* indices : {
             &authorized, &denied, &allowed, &passthrough, &ask
         }) {
        for (uint32_t index : *indices) {
            if (!valid_index(index)) {
                return evaluator_error("invalid evaluator result index");
            }
        }
    }
    for (const auto* indexed_values : {&denial_reasons, &transforms}) {
        for (const auto& entry : *indexed_values) {
            if (!valid_index(entry.first)) {
                return evaluator_error("invalid evaluator result index");
            }
        }
    }
    for (const auto& entry : denial_reasons) {
        for (const auto& reason : entry.second) {
            const auto& kind = reason["kind"];
            if (kind.type != json::Value::STR ||
                (kind.str() != "ask" && kind.str() != "block")) {
                return evaluator_error(
                    "invalid evaluator denial reason kind"
                );
            }
        }
    }

    json::Array results;
    for (size_t index = 0; index < action_count; ++index) {
        const auto idx = static_cast<uint32_t>(index);
        const bool allow_passthrough = passthrough.count(idx) > 0;
        const bool requires_approval = ask.count(idx) > 0 && denied.count(idx) == 0;
        const auto reasons = denial_reasons.find(idx);
        const auto transform_ids = transforms.find(idx);
        const auto routes = allow_routes.find(idx);
        results.push_back(json::Value(json::Object{
            {"index", json::Value(static_cast<int64_t>(idx))},
            {"authorized", json::Value(authorized.count(idx) > 0 && !requires_approval)},
            {"is_authenticated", json::Value(has_auth)},
            {"is_denylisted", json::Value(denied.count(idx) > 0)},
            {"is_allowlisted", json::Value(allowed.count(idx) > 0)},
            {"requires_approval", json::Value(requires_approval)},
            {"transform_ids", json::Value(
                transform_ids == transforms.end() ? json::Array{} : transform_ids->second)},
            {"deny_if_unauthorized", json::Value(!allow_passthrough)},
            {"allow_passthrough", json::Value(allow_passthrough)},
            {"denial_reasons", json::Value(
                reasons == denial_reasons.end() ? json::Array{} : reasons->second)},
            {"allow_routes", json::Value(routes == allow_routes.end()
                ? json::Array{} : checked_allow_routes(routes->second))},
        }));
    }
    return json::Value(json::Object{{"QueryResult", json::Value(json::Object{
        {"results", json::Value(std::move(results))}
    })}});
}

}  // namespace sasy_evaluator_protocol

#endif  // SASY_SOUFFLE_EVALUATOR_PROTOCOL_H
