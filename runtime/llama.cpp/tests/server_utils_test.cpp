#include "server_utils.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
    using namespace funasr_server;

    int failures = 0;
#define CHECK(condition)                                                                                \
    do {                                                                                                \
        if (!(condition)) {                                                                             \
            std::cerr << "FAIL line " << __LINE__ << ": " #condition << "\n";                         \
            ++failures;                                                                                 \
        }                                                                                               \
    } while (false)

    CHECK(json_field("{\"type\":\"session.update\"}", "type") == "session.update");
    CHECK(json_field("{\"session\":{\"turn_detection\":{\"type\":\"server_vad\"}}}", "session") ==
          "{\"turn_detection\":{\"type\":\"server_vad\"}}");
    CHECK(json_field("{\"type\":", "type").empty());
    CHECK(json_field("{\"note\":\"\\\"type\\\": fake\"}", "type").empty());
    CHECK(json_field("{\"type\":\"bad\\u12xz\"}", "type").empty());
    CHECK(json_field("{\"session\":{\"items\":[1,2}}", "session").empty());
    CHECK(json_field("{\"type\":\"session.update\"} trailing", "type").empty());
    CHECK(json_field("{\"type\":\"\\ud83d\\ude00\"}", "type") == "\xf0\x9f\x98\x80");

    const auto appended = format_transcription_delta("it_0", "turn", "turn left");
    CHECK(json_field(appended, "type") == "conversation.item.input_audio_transcription.delta");
    CHECK(json_field(appended, "item_id") == "it_0");
    CHECK(json_field(appended, "delta") == " left");
    CHECK(json_field(appended, "transcript") == "turn left");

    const auto revised = format_transcription_delta("it_0", "turn left", "turn right");
    CHECK(json_field(revised, "delta") == "turn right");
    CHECK(json_field(revised, "transcript") == "turn right");

    const auto shortened = format_transcription_delta("it_0", "turn right", "turn");
    CHECK(json_field(shortened, "delta") == "turn");
    CHECK(json_field(shortened, "transcript") == "turn");
    const auto cleared = format_transcription_delta("it_0", "turn right", "");
    CHECK(cleared.find("\"transcript\":\"\"") != std::string::npos);
    CHECK(format_transcription_delta("it_0", "turn right", "turn right").empty());

    std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) {
        workers.emplace_back([] {
            for (int j = 0; j < 100; ++j) {
                std::string decoded;
                if (!base64_decode("AQID", decoded) || decoded != std::string("\x01\x02\x03", 3)) {
                    std::terminate();
                }
            }
        });
    }
    for (auto &worker : workers) worker.join();

    std::string invalid_output;
    CHECK(!base64_decode("A===", invalid_output));
    CHECK(!base64_decode("AQI", invalid_output));
    CHECK(format_timestamp(3723004, false) == "01:02:03,004");
    CHECK(format_timestamp(3723004, true) == "01:02:03.004");

    std::vector<TranscriptSegment> segments = {{770, 5980, "hello"}};
    CHECK(format_subtitles(segments, false) ==
          "1\n00:00:00,770 --> 00:00:05,980\nhello\n\n");
    CHECK(format_subtitles(segments, true) ==
          "WEBVTT\n\n00:00:00.770 --> 00:00:05.980\nhello\n\n");
    CHECK(format_verbose_segments(segments) ==
          "{\"id\":0,\"start\":0.77,\"end\":5.98,\"start_ms\":770,\"end_ms\":5980,\"text\":\"hello\"}");
    return failures == 0 ? 0 : 1;
}
