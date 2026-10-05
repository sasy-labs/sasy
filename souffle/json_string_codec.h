#ifndef SASY_SOUFFLE_JSON_STRING_CODEC_H
#define SASY_SOUFFLE_JSON_STRING_CODEC_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sasy_json_codec {

inline int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

inline bool has_bytes(const char* value, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (value[index] == '\0') return false;
    }
    return true;
}

inline bool decode_hex_quad(const char* digits, uint32_t& value) {
    value = 0;
    for (int index = 0; index < 4; ++index) {
        const int digit = hex_digit(digits[index]);
        if (digit < 0) return false;
        value = (value << 4) | static_cast<uint32_t>(digit);
    }
    return true;
}

inline void append_utf8(uint32_t codepoint, std::string& out) {
    if (codepoint <= 0x7f) {
        out += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7ff) {
        out += static_cast<char>(0xc0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        out += static_cast<char>(0xe0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else {
        out += static_cast<char>(0xf0 | (codepoint >> 18));
        out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (codepoint & 0x3f));
    }
}

// `escape` points at the byte after a JSON backslash. On return it points at
// the last consumed byte so the caller's ordinary loop increment still works.
inline void append_decoded_escape(const char*& escape, std::string& out) {
    switch (*escape) {
        case '"': out += '"'; return;
        case '\\': out += '\\'; return;
        case '/': out += '/'; return;
        case 'b': out += '\b'; return;
        case 'f': out += '\f'; return;
        case 'n': out += '\n'; return;
        case 'r': out += '\r'; return;
        case 't': out += '\t'; return;
        case 'u': break;
        default: out += *escape; return;
    }

    uint32_t first;
    if (!has_bytes(escape + 1, 4) || !decode_hex_quad(escape + 1, first)) {
        out += 'u';
        return;
    }
    if (first >= 0xd800 && first <= 0xdbff) {
        uint32_t second;
        if (
            !has_bytes(escape + 5, 6) || escape[5] != '\\' ||
            escape[6] != 'u' ||
            !decode_hex_quad(escape + 7, second) ||
            second < 0xdc00 || second > 0xdfff
        ) {
            out += 'u';
            return;
        }
        append_utf8(
            0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00),
            out
        );
        escape += 10;
        return;
    }
    if (first >= 0xdc00 && first <= 0xdfff) {
        out += 'u';
        return;
    }
    append_utf8(first, out);
    escape += 4;
}

inline std::string escape(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (unsigned char byte : value) {
        switch (byte) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (byte < 0x20) {
                    out += "\\u00";
                    out += hex[byte >> 4];
                    out += hex[byte & 0x0f];
                } else {
                    out += static_cast<char>(byte);
                }
        }
    }
    return out;
}

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    enum Type { NUL, BOOL, INT, STR, ARR, OBJ } type = NUL;
    bool bool_val = false;
    int64_t int_val = 0;
    std::string str_val;
    Array arr_val;
    Object obj_val;

    Value() : type(NUL) {}
    Value(bool value) : type(BOOL), bool_val(value) {}
    Value(int64_t value) : type(INT), int_val(value) {}
    Value(const std::string& value) : type(STR), str_val(value) {}
    Value(const char* value) : type(STR), str_val(value) {}
    Value(Array value) : type(ARR), arr_val(std::move(value)) {}
    Value(Object value) : type(OBJ), obj_val(std::move(value)) {}

    const Value& operator[](const std::string& key) const {
        static Value null_value;
        if (type != OBJ) return null_value;
        auto iterator = obj_val.find(key);
        return iterator != obj_val.end() ? iterator->second : null_value;
    }

    const Value& operator[](size_t index) const {
        static Value null_value;
        if (type != ARR || index >= arr_val.size()) return null_value;
        return arr_val[index];
    }

    bool is_null() const { return type == NUL; }
    const std::string& str() const { return str_val; }
    int64_t num() const { return int_val; }
    bool boolean() const { return bool_val; }
    size_t size() const { return type == ARR ? arr_val.size() : 0; }
};

inline const char* skip_ws(const char* value) {
    while (
        *value == ' ' || *value == '\t' || *value == '\n' || *value == '\r'
    ) {
        ++value;
    }
    return value;
}

inline Value parse(const char*& value);

inline std::string parse_string(const char*& value) {
    if (*value != '"') return "";
    ++value;
    std::string result;
    while (*value && *value != '"') {
        if (*value == '\\') {
            ++value;
            if (!*value) break;
            append_decoded_escape(value, result);
        } else {
            result += *value;
        }
        ++value;
    }
    if (*value == '"') ++value;
    return result;
}

inline Value parse(const char*& value) {
    value = skip_ws(value);
    if (*value == '"') return Value(parse_string(value));
    if (*value == '{') {
        ++value;
        Object object;
        while (true) {
            value = skip_ws(value);
            if (*value == '}') {
                ++value;
                break;
            }
            if (*value == ',') {
                ++value;
                continue;
            }
            std::string key = parse_string(value);
            value = skip_ws(value);
            if (*value == ':') ++value;
            object[key] = parse(value);
        }
        return Value(std::move(object));
    }
    if (*value == '[') {
        ++value;
        Array array;
        while (true) {
            value = skip_ws(value);
            if (*value == ']') {
                ++value;
                break;
            }
            if (*value == ',') {
                ++value;
                continue;
            }
            array.push_back(parse(value));
        }
        return Value(std::move(array));
    }
    if (*value == 't' && strncmp(value, "true", 4) == 0) {
        value += 4;
        return Value(true);
    }
    if (*value == 'f' && strncmp(value, "false", 5) == 0) {
        value += 5;
        return Value(false);
    }
    if (*value == 'n' && strncmp(value, "null", 4) == 0) {
        value += 4;
        return Value();
    }
    char* end = nullptr;
    int64_t number = strtoll(value, &end, 10);
    if (end != value) {
        value = end;
        return Value(number);
    }
    return Value();
}

inline Value parse(const std::string& value) {
    const char* cursor = value.c_str();
    return parse(cursor);
}

inline std::string serialize(const Value& value) {
    switch (value.type) {
        case Value::NUL: return "null";
        case Value::BOOL: return value.bool_val ? "true" : "false";
        case Value::INT: return std::to_string(value.int_val);
        case Value::STR: return "\"" + escape(value.str_val) + "\"";
        case Value::ARR: {
            std::string out = "[";
            for (size_t index = 0; index < value.arr_val.size(); ++index) {
                if (index) out += ",";
                out += serialize(value.arr_val[index]);
            }
            return out + "]";
        }
        case Value::OBJ: {
            std::string out = "{";
            bool first = true;
            for (auto& [key, child] : value.obj_val) {
                if (!first) out += ",";
                first = false;
                out += "\"" + escape(key) + "\":" + serialize(child);
            }
            return out + "}";
        }
    }
    return "null";
}

}  // namespace sasy_json_codec

#endif  // SASY_SOUFFLE_JSON_STRING_CODEC_H
