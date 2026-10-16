# opus1 documentation

These documents explain the Opus audio codec and `opus1`. Opus is defined in RFC 6716 (Request for Comments 6716, the Internet standard for Opus). `opus1` is the pure-Rust implementation of Opus in this repository, and it follows libopus 1.6. This page lists the documents with one line on each. It also suggests a reading order for people who use the library and for people who maintain it. The documents assume no knowledge of audio coding beyond "sample rate" and "bitrate". Each one defines its terms when it first uses them, and the [glossary](glossary.md) collects them in one place. The repository's [README](../README.md) has the short overview, the release notes and the latest timings.

## The documents

| Document | What it covers |
|---|---|
| [01 What Opus is](01-what-is-opus.md) | The two coding layers (SILK, a speech coder, and CELT, the Constrained Energy Lapped Transform coder), the three modes, bandwidths, sampling rates, frame durations, delay, packet loss tools, Ogg Opus, and how `opus1` relates to libopus. |
| [02 Opus packets](02-packets.md) | The bytes of a packet: the table-of-contents (TOC) byte, the 32 configurations, the four frame count codes, limits, self-delimited framing, and the `opus1::packet` module. |
| [03 The encoder](03-encoder.md) | One call of `OpusEncoder::encode`: how the mode, bandwidth and channel count are chosen, the stages of the SILK and CELT encoders, and every encoder setting. |
| [04 The decoder](04-decoder.md) | One call of `OpusDecoder::decode`: packet parsing, SILK and CELT decoding, mode switches, packet loss concealment, in-band forward error correction (FEC), 16-bit output and the final range. |
| [05 Multichannel audio](05-multichannel.md) | Multistream packets, channel layouts and mapping families, how the encoder shares the bitrate between streams, surround mode, and the application programming interface (API) of `opus1::multistream`. |
| [06 Using the library](06-using-the-library.md) | Recipes for calls, voice agents and music; choosing frame duration, bitrate and complexity; packet loss; the `#![no_std]` build; errors, threads and memory. |
| [07 Architecture](07-architecture.md) | Every source file and the libopus file it ports, the main types, floating point and fixed point, feature flags, processor-specific code, `unsafe` code and the layout of the tests. |
| [08 Performance](08-performance.md) | How speed is measured against libopus, the latest results, the changes that made the code fast, and how to reproduce the numbers. |
| [09 Testing](09-testing.md) | How libopus is linked for the tests, what the tests compare, the groups of test files, the documentation examples, fuzzing, and the command for each part. |
| [10 Differences from libopus](10-differences-from-libopus.md) | What `opus1` does not implement, where it differs on purpose, and how far its output differs from libopus, with the measured figures. |
| [Glossary](glossary.md) | The terms used in these documents, in alphabetical order. |

## Reading order for users

1. [01 What Opus is](01-what-is-opus.md), for the ideas and the terms.
2. [06 Using the library](06-using-the-library.md), to write code that encodes and decodes.
3. [05 Multichannel audio](05-multichannel.md), when you have more than two channels.
4. [10 Differences from libopus](10-differences-from-libopus.md), before you depend on a libopus control or on bit-exact output.
5. [02 Opus packets](02-packets.md), when you store, send or inspect packets yourself.

## Reading order for maintainers

1. [01 What Opus is](01-what-is-opus.md) and [02 Opus packets](02-packets.md), for the format.
2. [07 Architecture](07-architecture.md), to find the code behind each part.
3. [03 The encoder](03-encoder.md) and [04 The decoder](04-decoder.md), for the code paths in the order they run.
4. [05 Multichannel audio](05-multichannel.md), for the multistream layer.
5. [09 Testing](09-testing.md), before you change code.
6. [08 Performance](08-performance.md), before you change code for speed.
7. [10 Differences from libopus](10-differences-from-libopus.md), for the known gaps and their sizes.
