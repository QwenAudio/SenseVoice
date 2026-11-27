#pragma once

#include <array>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace funasr_server {
namespace detail {

inline size_t skip_ws(const std::string &input, size_t pos) {
    while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos]))) ++pos;
    return pos;
}

inline int hex_value(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

inline void append_utf8(std::string &output, unsigned codepoint) {
    if (codepoint < 0x80) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint < 0x10000) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

inline bool parse_string(const std::string &input, size_t &pos, std::string &output) {
    if (pos >= input.size() || input[pos] != '"') return false;
    output.clear();
    ++pos;
    while (pos < input.size()) {
        unsigned char value = static_cast<unsigned char>(input[pos++]);
        if (value == '"') return true;
        if (value < 0x20) return false;
        if (value != '\\') {
            output.push_back(static_cast<char>(value));
            continue;
        }
        if (pos >= input.size()) return false;
        char escaped = input[pos++];
        switch (escaped) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > input.size()) return false;
                unsigned codepoint = 0;
                for (int i = 0; i < 4; ++i) {
                    int digit = hex_value(input[pos++]);
                    if (digit < 0) return false;
                    codepoint = (codepoint << 4) | static_cast<unsigned>(digit);
                }
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (pos + 6 > input.size() || input[pos] != '\\' || input[pos + 1] != 'u') return false;
                    pos += 2;
                    unsigned low = 0;
                    for (int i = 0; i < 4; ++i) {
                        int digit = hex_value(input[pos++]);
                        if (digit < 0) return false;
                        low = (low << 4) | static_cast<unsigned>(digit);
                    }
                    if (low < 0xdc00 || low > 0xdfff) return false;
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    return false;
                }
                append_utf8(output, codepoint);
                break;
            }
            default: return false;
        }
    }
    return false;
}

inline bool parse_compound(const std::string &input, size_t &pos, std::string &output) {
    const size_t start = pos;
    std::vector<char> closers;
    while (pos < input.size()) {
        if (input[pos] == '"') {
            std::string ignored;
            if (!parse_string(input, pos, ignored)) return false;
            continue;
        }
        if (input[pos] == '{') closers.push_back('}');
        else if (input[pos] == '[') closers.push_back(']');
        else if (input[pos] == '}' || input[pos] == ']') {
            if (closers.empty() || closers.back() != input[pos]) return false;
            closers.pop_back();
            ++pos;
            if (closers.empty()) {
                output = input.substr(start, pos - start);
                return true;
            }
            continue;
        }
        ++pos;
    }
    return false;
}

inline bool parse_value(const std::string &input, size_t &pos, std::string &output) {
    pos = skip_ws(input, pos);
    if (pos >= input.size()) return false;
    if (input[pos] == '"') return parse_string(input, pos, output);
    if (input[pos] == '{' || input[pos] == '[') return parse_compound(input, pos, output);
    const size_t start = pos;
    while (pos < input.size() && input[pos] != ',' && input[pos] != '}') ++pos;
    size_t end = pos;
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
    if (end == start) return false;
    output = input.substr(start, end - start);
    return true;
}

}  // namespace detail

inline std::string json_escape(const std::string &input) {
    std::string output;
    output.reserve(input.size() + 8);
    for (unsigned char value : input) {
        switch (value) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (value < 0x20) {
                    char encoded[8];
                    std::snprintf(encoded, sizeof encoded, "\\u%04x", value);
                    output += encoded;
                } else {
                    output.push_back(static_cast<char>(value));
                }
        }
    }
    return output;
}

inline std::string json_str(const std::string &input) { return "\"" + json_escape(input) + "\""; }

inline std::string format_transcription_delta(const std::string &item_id,
                                               const std::string &previous,
                                               const std::string &text) {
    if (text == previous) return {};
    const std::string delta = text.compare(0, previous.size(), previous) == 0
                                  ? text.substr(previous.size())
                                  : text;
    return "{\"type\":\"conversation.item.input_audio_transcription.delta\",\"item_id\":" + json_str(item_id) +
           ",\"delta\":" + json_str(delta) + ",\"transcript\":" + json_str(text) + "}";
}

inline std::string json_field(const std::string &input, const std::string &wanted_key) {
    size_t pos = detail::skip_ws(input, 0);
    if (pos >= input.size() || input[pos++] != '{') return "";
    bool found = false;
    std::string result;
    while (true) {
        pos = detail::skip_ws(input, pos);
        if (pos >= input.size()) return "";
        if (input[pos] == '}') {
            pos = detail::skip_ws(input, pos + 1);
            return pos == input.size() && found ? result : "";
        }
        std::string key;
        if (!detail::parse_string(input, pos, key)) return "";
        pos = detail::skip_ws(input, pos);
        if (pos >= input.size() || input[pos++] != ':') return "";
        std::string value;
        if (!detail::parse_value(input, pos, value)) return "";
        if (key == wanted_key) {
            found = true;
            result = value;
        }
        pos = detail::skip_ws(input, pos);
        if (pos >= input.size()) return "";
        if (input[pos] == '}') {
            pos = detail::skip_ws(input, pos + 1);
            return pos == input.size() && found ? result : "";
        }
        if (input[pos++] != ',') return "";
    }
}

inline bool base64_decode(const std::string &input, std::string &output) {
    static const std::array<signed char, 256> reverse = [] {
        std::array<signed char, 256> table{};
        table.fill(-1);
        const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < alphabet.size(); ++i) {
            table[static_cast<unsigned char>(alphabet[i])] = static_cast<signed char>(i);
        }
        return table;
    }();

    std::string clean;
    clean.reserve(input.size());
    for (unsigned char value : input) {
        if (!std::isspace(value)) clean.push_back(static_cast<char>(value));
    }
    output.clear();
    if (clean.empty()) return true;
    if (clean.size() % 4 != 0) return false;
    output.reserve(clean.size() / 4 * 3);
    for (size_t i = 0; i < clean.size(); i += 4) {
        const bool last = i + 4 == clean.size();
        int a = reverse[static_cast<unsigned char>(clean[i])];
        int b = reverse[static_cast<unsigned char>(clean[i + 1])];
        if (a < 0 || b < 0) return false;
        output.push_back(static_cast<char>((a << 2) | (b >> 4)));
        if (clean[i + 2] == '=') {
            if (!last || clean[i + 3] != '=' || (b & 0x0f) != 0) return false;
            continue;
        }
        int c = reverse[static_cast<unsigned char>(clean[i + 2])];
        if (c < 0) return false;
        output.push_back(static_cast<char>(((b & 0x0f) << 4) | (c >> 2)));
        if (clean[i + 3] == '=') {
            if (!last || (c & 0x03) != 0) return false;
            continue;
        }
        int d = reverse[static_cast<unsigned char>(clean[i + 3])];
        if (d < 0) return false;
        output.push_back(static_cast<char>(((c & 0x03) << 6) | d));
    }
    return true;
}

inline std::string format_timestamp(int milliseconds, bool webvtt) {
    int hours = milliseconds / 3600000;
    int minutes = (milliseconds / 60000) % 60;
    int seconds = (milliseconds / 1000) % 60;
    int millis = milliseconds % 1000;
    char output[64];
    std::snprintf(output, sizeof output, "%02d:%02d:%02d%c%03d", hours, minutes, seconds,
                  webvtt ? '.' : ',', millis);
    return output;
}

struct TranscriptSegment {
    int start_ms;
    int end_ms;
    std::string text;
};

inline std::string format_verbose_segments(const std::vector<TranscriptSegment> &segments) {
    std::string output;
    for (size_t index = 0; index < segments.size(); ++index) {
        if (!output.empty()) output.push_back(',');
        const auto &segment = segments[index];
        char prefix[192];
        std::snprintf(prefix, sizeof prefix,
                      "{\"id\":%zu,\"start\":%.2f,\"end\":%.2f,\"start_ms\":%d,\"end_ms\":%d,\"text\":",
                      index, segment.start_ms / 1000.0, segment.end_ms / 1000.0,
                      segment.start_ms, segment.end_ms);
        output += prefix;
        output += json_str(segment.text);
        output.push_back('}');
    }
    return output;
}

inline std::string format_subtitles(const std::vector<TranscriptSegment> &segments, bool webvtt) {
    std::string output = webvtt ? "WEBVTT\n\n" : "";
    size_t cue = 1;
    for (const auto &segment : segments) {
        if (segment.text.empty()) continue;
        if (!webvtt) output += std::to_string(cue++) + "\n";
        output += format_timestamp(segment.start_ms, webvtt) + " --> " +
                  format_timestamp(segment.end_ms, webvtt) + "\n" + segment.text + "\n\n";
    }
    return output;
}

}  // namespace funasr_server
