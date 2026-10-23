# Opus 编解码图解

An interactive, single-page explainer (in Chinese and English) of the Opus codec, end to end: from a sound wave to PCM, frames,
SILK / CELT / Hybrid, range coding, the packet, decoding, packet loss concealment, and back to sound. It is written
for readers who know basic acoustics and some programming, but nothing about audio coding.

## Open it

Open `index.html` in a browser. It works from the local file system; there is no build step. The fonts come from
Google Fonts when the browser is online and fall back to system fonts otherwise.

Every figure is in `js/figs/<name>.js` and is registered with `OX.fig(name, init)`; the shared runtime is
`js/core.js`. Two URL parameters help when working on a figure:

- `index.html?solo=<name>` shows one figure only, for example `?solo=pvq`;
- `index.html?eager` builds every figure at load time (by default figures are built as they scroll into view).

## Two languages

The button in the top bar switches between Chinese and English without reloading; the choice is saved in
`localStorage`. `?lang=zh` or `?lang=en` picks the language on load; without either, the saved choice or the browser
language decides. Every piece of text exists in both languages, so a change to one needs the same change to the other:

- In `index.html`, text comes in pairs inside the element that holds it:
  `<p><span lang="zh-CN">中文</span><span lang="en">English</span></p>`. CSS shows one span of each pair. Attributes
  keep the Chinese value and put the English in `data-en-<attr>` (`title`, `aria-label`, `alt`, `content`); sections
  use `data-title` / `data-title-en`, and sub-topic headings `data-sub` / `data-sub-en`.
- In `js/figs/*.js`, every visible string is `OX.t('中文', 'English')`, called while the figure is built or drawn
  (not when the file loads). On a switch the runtime rebuilds each figure that has been built. Canvases, `OX.loop`
  loops and `OX.onTheme` / `OX.onLang` callbacks are cleaned up automatically. A figure that sets up anything else
  (its own `ResizeObserver`, a timer, a window listener) returns a clean-up function from its init.

## The data

Figures tagged 真实数据 read `data/opus-demo-data.js`, which is produced by this crate:

```sh
cargo run --release --example opus_demo_data
```

The generator encodes a 3 s excerpt of `fixtures/answer_16k.wav` and a synthetic 48 kHz music clip with opus1 at
several bitrates, decodes them, and replays the SILK and CELT bitstreams with the crate's public functions to record
the per-frame parameters (signal type, pitch lags, LPC coefficients, band energies, bit allocation, ...). Each replay
is checked against the real decoder; if a check fails, the data file is left unchanged. It also records, as a
reference, what libopus does at the same settings, through the `opus` dev-dependency. That dependency links the C
libopus: build it with cmake, or set `LIBOPUS_LIB_DIR` to an installed libopus and `LIBOPUS_NO_PKG=1`.

The output is deterministic. `meta.git_commit` names the commit the data was built from, with `-dirty` when the
generator's inputs (`src/`, `fixtures/`, `Cargo.toml`, the generator itself) had uncommitted changes.
