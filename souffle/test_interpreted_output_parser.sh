#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "${test_dir}"' EXIT

"${CXX:-g++}" -std=c++17 -I"${script_dir}" -x c++ \
    -o "${test_dir}/test" - <<'CPP'
#define main interpreted_shim_main
#include "interpreted_shim.cpp"
#undef main

#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

static const std::string VALID_OUTPUT =
    "---------------\n"
    "AllowPassthrough\n"
    "idx\n"
    "===============\n"
    "===============\n"
    "---------------\n"
    "ApplyTransform\n"
    "idx\ttransform_id\n"
    "===============\n"
    "0,\"transform\"\n"
    "===============\n"
    "---------------\n"
    "Authorized\n"
    "idx\n"
    "===============\n"
    "0\n"
    "===============\n"
    "---------------\n"
    "DenialReason\n"
    "idx\tkind\treason\tsuggestion\n"
    "===============\n"
    "0,\"ask\",\"reason\n===============\n---------------\",\"fix\"\n"
    "===============\n"
    "---------------\n"
    "HasPrincipal\n"
    "\n"
    "===============\n"
    "()\n"
    "===============\n"
    "---------------\n"
    "HasRole\n"
    "role\n"
    "===============\n"
    "\"===============\"\n"
    "===============\n"
    "---------------\n"
    "IsAuthorized\n"
    "idx\n"
    "===============\n"
    "0\n"
    "===============\n"
    "---------------\n"
    "Unauthorized\n"
    "idx\n"
    "===============\n"
    "===============\n";

static void expect_valid(const std::string& output) {
    ParsedSouffleOutput parsed;
    if (!parse_souffle_output(output, 1, parsed)) {
        std::cerr << "expected valid output\n";
        std::exit(1);
    }
    if (parsed.authorized.count(0) != 1 ||
        parsed.allowed_indices.count(0) != 1 ||
        parsed.ask_indices.count(0) != 1 ||
        !parsed.has_auth ||
        parsed.transforms.at(0).size() != 1 ||
        parsed.denial_reasons.at(0).size() != 1) {
        std::cerr << "valid output was parsed incorrectly\n";
        std::exit(1);
    }
}

static void expect_invalid(
    const std::string& name,
    std::string output
) {
    ParsedSouffleOutput parsed;
    if (parse_souffle_output(output, 1, parsed)) {
        std::cerr << "accepted invalid output: " << name << "\n";
        std::exit(1);
    }
}

static std::string replace_once(
    std::string value,
    const std::string& before,
    const std::string& after
) {
    const size_t position = value.find(before);
    if (position == std::string::npos) {
        std::cerr << "test fixture text not found\n";
        std::exit(1);
    }
    value.replace(position, before.size(), after);
    return value;
}

static json::Value build_boundary_result(
    const std::set<uint32_t>& authorized = {},
    const std::set<uint32_t>& denied = {},
    const std::set<uint32_t>& allowed = {},
    const std::set<uint32_t>& passthrough = {},
    const std::set<uint32_t>& ask = {},
    const std::map<uint32_t, json::Array>& denial_reasons = {},
    const std::map<uint32_t, json::Array>& transforms = {}
) {
    return evaluator_protocol::build_query_result(
        1, false, authorized, denied, allowed, passthrough, ask,
        denial_reasons, transforms
    );
}

static void expect_boundary_error(
    const std::string& name,
    const json::Value& result,
    const std::string& message
) {
    if (!result["QueryResult"].is_null() ||
        result["Error"]["message"].str() != message) {
        std::cerr << "shared boundary accepted invalid result: " << name
                  << "\n";
        std::exit(1);
    }
}

static json::Value denial_reason(const std::string& kind) {
    return json::Value(json::Object{
        {"kind", json::Value(kind)},
        {"reason", json::Value("reason")},
        {"suggestion", json::Value("suggestion")},
    });
}

static void expect_shared_boundary_rejections() {
    uint32_t decoded = 0;
    const int64_t too_large =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    if (evaluator_protocol::decode_query_result_index(-1, 1, decoded) ||
        evaluator_protocol::decode_query_result_index(
            too_large, 1, decoded
        ) ||
        !evaluator_protocol::decode_query_result_index(0, 1, decoded) ||
        decoded != 0) {
        std::cerr << "raw evaluator index validation failed\n";
        std::exit(1);
    }
    const uint32_t wrapped_negative = static_cast<uint32_t>(-1);
    const std::string index_error = "invalid evaluator result index";
    expect_boundary_error(
        "authorized index", build_boundary_result({wrapped_negative}),
        index_error
    );
    expect_boundary_error(
        "denied index", build_boundary_result({}, {wrapped_negative}),
        index_error
    );
    expect_boundary_error(
        "allowed index",
        build_boundary_result({}, {}, {wrapped_negative}), index_error
    );
    expect_boundary_error(
        "passthrough index",
        build_boundary_result({}, {}, {}, {wrapped_negative}), index_error
    );
    expect_boundary_error(
        "ask index",
        build_boundary_result({}, {}, {}, {}, {wrapped_negative}), index_error
    );
    expect_boundary_error(
        "denial reason index",
        build_boundary_result(
            {}, {}, {}, {}, {},
            {{wrapped_negative, {denial_reason("block")}}}
        ),
        index_error
    );
    expect_boundary_error(
        "transform index",
        build_boundary_result(
            {}, {}, {}, {}, {}, {},
            {{wrapped_negative, {json::Value("transform")}}}
        ),
        index_error
    );
    expect_boundary_error(
        "unknown denial kind",
        build_boundary_result(
            {}, {}, {}, {}, {}, {{0, {denial_reason("unknown")}}}
        ),
        "invalid evaluator denial reason kind"
    );
}

static void expect_fact_write_failure_stops_policy() {
    const std::string fact_directory = create_private_fact_dir();
    if (fact_directory.empty()) {
        std::cerr << "could not create fact directory for write failure test\n";
        std::exit(1);
    }

    const std::string prior_fact_dir = g_fact_dir;
    const std::string prior_souffle_bin = g_souffle_bin;
    const std::string prior_policy_path = g_policy_path;
    const std::filesystem::path marker =
        std::filesystem::path(fact_directory) / "policy-executed";
    g_fact_dir = fact_directory;
    g_souffle_bin = "touch " + marker.string() + ";";
    g_policy_path = "true";
    g_state.edges.insert({"", "source", "destination"});

    struct rlimit original_limit {};
    if (getrlimit(RLIMIT_FSIZE, &original_limit) != 0) {
        std::cerr << "could not read file-size limit\n";
        std::exit(1);
    }
    struct rlimit zero_limit = original_limit;
    zero_limit.rlim_cur = 0;
    const auto prior_signal = std::signal(SIGXFSZ, SIG_IGN);
    if (prior_signal == SIG_ERR ||
        setrlimit(RLIMIT_FSIZE, &zero_limit) != 0) {
        std::cerr << "could not impose fact write failure\n";
        std::exit(1);
    }

    const json::Value result = run_query(json::Value(json::Object{}));

    if (setrlimit(RLIMIT_FSIZE, &original_limit) != 0 ||
        std::signal(SIGXFSZ, prior_signal) == SIG_ERR) {
        std::cerr << "could not restore file-size limit\n";
        std::exit(1);
    }
    g_state.edges.clear();
    g_fact_dir = prior_fact_dir;
    g_souffle_bin = prior_souffle_bin;
    g_policy_path = prior_policy_path;

    if (!result["QueryResult"].is_null() ||
        result["Error"]["message"].str() !=
            "failed to write fact file: Edge.facts" ||
        std::filesystem::exists(marker)) {
        std::cerr << "fact write failure did not stop policy execution\n";
        std::exit(1);
    }
    remove_private_fact_dir(fact_directory);
}

static void expect_process_arguments_are_literal() {
    PrivateFactDirectory root;
    if (!root.valid()) {
        std::cerr << "could not create process test directory\n";
        std::exit(1);
    }
    const std::filesystem::path directory =
        std::filesystem::path(root.path()) / "prefix with spaces";
    const std::filesystem::path executable = directory / "echo tool";
    std::filesystem::create_directories(directory);
    std::filesystem::create_symlink("/bin/echo", executable);

    const ChildProcessResult result = run_child_process({
        executable.string(),
        "argument with spaces",
        "semi;colon",
        "$HOME",
    });
    if (!result.launched || result.status != 0 ||
        result.output != "argument with spaces semi;colon $HOME\n") {
        std::cerr << "process arguments were not passed literally\n";
        std::exit(1);
    }
}

int main() {
    g_souffle_bin = "/selected/souffle";
    g_policy_path = "/selected/policy.dl";
    g_functor_lib = "functors";
    g_functor_directory = "/selected/library with spaces";
    const std::vector<std::string> selected = {
        "/selected/souffle", "/selected/policy.dl", "-F/private/facts", "-D-",
        "-lfunctors", "-L/selected/library with spaces"
    };
    if (souffle_query_arguments("/private/facts") != selected) {
        std::cerr << "selected library must exclude competing cwd/legacy libraries\n";
        return 1;
    }
    g_functor_directory.clear();
    const auto legacy = souffle_query_arguments("/private/facts");
    if (legacy.size() != 8 || legacy[5] != "-L/usr/local/lib" ||
        legacy[6] != "-L/opt/homebrew/lib" || legacy[7] != "-L.") {
        std::cerr << "legacy standalone library search changed\n";
        return 1;
    }
    g_functor_lib.clear();
    const std::string first_directory = create_private_fact_dir();
    const std::string second_directory = create_private_fact_dir();
    if (first_directory.empty() || second_directory.empty() ||
        first_directory == second_directory ||
        !std::filesystem::is_directory(first_directory) ||
        !std::filesystem::is_directory(second_directory)) {
        std::cerr << "private fact directories are not isolated\n";
        std::exit(1);
    }
    remove_private_fact_dir(first_directory);
    remove_private_fact_dir(second_directory);
    if (std::filesystem::exists(first_directory) ||
        std::filesystem::exists(second_directory)) {
        std::cerr << "private fact directories were not cleaned up\n";
        std::exit(1);
    }

    expect_valid(VALID_OUTPUT);
    const std::string diagnostic_output = VALID_OUTPUT +
        "---------------\nSasyAllowRoute\n"
        "idx\trule_id\tstatus\tdetails\tsuggestion\tsource_location\n"
        "===============\n0,\"allow-1\",\"possible\",\"fixed conditions hold\",\"request approval\",\"policy:12\"\n===============\n";
    ParsedSouffleOutput diagnostic_parsed;
    if (!parse_souffle_output(diagnostic_output, 1, diagnostic_parsed) ||
        evaluator_protocol::checked_allow_routes(diagnostic_parsed.allow_routes[0]).size() != 1) {
        std::cerr << "optional attribution did not round-trip\n"; return 1;
    }
    auto bad_rows = diagnostic_parsed.allow_routes[0];
    bad_rows[0].obj_val["status"] = json::Value("invented");
    if (!evaluator_protocol::checked_allow_routes(bad_rows).empty() ||
        !evaluator_protocol::checked_allow_routes(json::Array(513, diagnostic_parsed.allow_routes[0][0])).empty()) {
        std::cerr << "invalid attribution was retained\n"; return 1;
    }
    const auto with_diagnostics = evaluator_protocol::build_query_result(
        1, diagnostic_parsed.has_auth, diagnostic_parsed.authorized,
        diagnostic_parsed.denied_indices, diagnostic_parsed.allowed_indices,
        diagnostic_parsed.passthrough, diagnostic_parsed.ask_indices,
        diagnostic_parsed.denial_reasons, diagnostic_parsed.transforms,
        diagnostic_parsed.allow_routes);
    if (!with_diagnostics["QueryResult"]["results"][0]["requires_approval"].boolean() ||
        with_diagnostics["QueryResult"]["results"][0]["authorized"].boolean()) {
        std::cerr << "attribution changed ask precedence\n"; return 1;
    }
    // A relation the shim does not consume may print plain .output rows,
    // including a bare quote; that must not fail the whole query.
    expect_valid(
        replace_once(VALID_OUTPUT, "HasRole\nrole\n===============\n",
                     "HasRole\nrole\n===============\nbare \" quote\n"));
    expect_shared_boundary_rejections();
    expect_fact_write_failure_stops_policy();
    expect_process_arguments_are_literal();
    expect_invalid(
        "numeric suffix",
        replace_once(VALID_OUTPUT, "Authorized\nidx\n===============\n0\n",
                     "Authorized\nidx\n===============\n0garbage\n"));
    expect_invalid(
        "numeric sign",
        replace_once(VALID_OUTPUT, "Authorized\nidx\n===============\n0\n",
                     "Authorized\nidx\n===============\n+0\n"));
    expect_invalid(
        "numeric overflow",
        replace_once(VALID_OUTPUT, "Authorized\nidx\n===============\n0\n",
                     "Authorized\nidx\n===============\n4294967296\n"));
    expect_invalid(
        "out-of-range index",
        replace_once(VALID_OUTPUT, "Authorized\nidx\n===============\n0\n",
                     "Authorized\nidx\n===============\n1\n"));
    expect_invalid(
        "wrong header",
        replace_once(VALID_OUTPUT, "Authorized\nidx\n",
                     "Authorized\nwrong\n"));
    expect_invalid(
        "missing section",
        replace_once(
            VALID_OUTPUT,
            "---------------\nUnauthorized\nidx\n"
            "===============\n===============\n",
            ""));
    expect_invalid(
        "truncated section",
        VALID_OUTPUT.substr(0, VALID_OUTPUT.size() - 17));
    expect_invalid(
        "bad RFC quote",
        replace_once(VALID_OUTPUT, "\"fix\"\n===============\n",
                     "\"fix\"garbage\n===============\n"));
    expect_invalid(
        "bad RFC symbol quote",
        replace_once(VALID_OUTPUT, "0,\"transform\"\n",
                     "0,\"transform\n"));
    expect_invalid(
        "unknown denial kind",
        replace_once(VALID_OUTPUT, "0,\"ask\",", "0,\"unknown\","));
}
CPP

"${test_dir}/test"
