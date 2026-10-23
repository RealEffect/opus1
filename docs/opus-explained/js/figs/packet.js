/* Figure 12: one real Opus packet, taken apart.
 * Pick a packet from an opus1 stream (speech: SILK, music: CELT, plus the other target bitrates). Its first
 * byte, the TOC, goes into an editor: 8 bit buttons grouped config(5) | s(1) | c(2), and a 32-cell table of
 * configurations. The editor decodes the byte and says how a decoder would split the rest of this packet with it
 * (RFC 6716 §3.2, rules R1–R7). The packet itself is shown as a hex dump with its size, duration and bitrate.
 * Keyboard: the packet slider, the bit buttons and the 32 configuration buttons are all ordinary controls.
 * The TOC table is meta.toc_configs (RFC 6716 §3.1, Table 2); the RFC rule is the fallback.
 */
OX.fig('packet', function (root, body) {
  const { el, segmented, slider, button, readout, legend } = OX.ui;
  const D = OX.data.raw;
  if (!D || !D.speech || !D.music || !D.speech.encodes || !D.music.encodes) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js，这张图无法显示。', 'The data file data/opus-demo-data.js is missing, so this figure cannot be shown.') }));
    return;
  }

  /* ---------- tables ---------- */
  // RFC 6716 Table 2. 0-11 SILK (NB, MB, WB × 10/20/40/60 ms), 12-15 Hybrid (SWB, FB × 10/20 ms),
  // 16-31 CELT (NB, WB, SWB, FB × 2.5/5/10/20 ms). Note: CELT 16-19 is NB; MB exists only for SILK.
  const rfcCfg = (c) => (c < 12 ? { config: c, mode: 'SILK', bandwidth: ['NB', 'MB', 'WB'][c >> 2], frame_ms: [10, 20, 40, 60][c & 3] }
    : c < 16 ? { config: c, mode: 'Hybrid', bandwidth: c < 14 ? 'SWB' : 'FB', frame_ms: [10, 20][c & 1] }
      : { config: c, mode: 'CELT', bandwidth: ['NB', 'WB', 'SWB', 'FB'][(c - 16) >> 2], frame_ms: [2.5, 5, 10, 20][c & 3] });
  const metaCfg = D.meta && D.meta.toc_configs;
  const CFG = Array.from({ length: 32 }, (_, c) => (metaCfg && metaCfg[c] && metaCfg[c].config === c ? metaCfg[c] : rfcCfg(c)));
  const BWS = (D.meta && D.meta.bandwidths) || {
    NB: { audio_hz: 4000, sample_rate: 8000 }, MB: { audio_hz: 6000, sample_rate: 12000 }, WB: { audio_hz: 8000, sample_rate: 16000 },
    SWB: { audio_hz: 12000, sample_rate: 24000 }, FB: { audio_hz: 20000, sample_rate: 48000 },
  };
  const BW_NAME = { NB: OX.t('窄带', 'narrowband'), MB: OX.t('中带', 'mediumband'), WB: OX.t('宽带', 'wideband'), SWB: OX.t('超宽带', 'super-wideband'), FB: OX.t('全带', 'fullband') };
  const MODE = {
    SILK: { cls: 'silk', tag: 'SILK-only', zh: OX.t('只用 SILK', 'no CELT layer') },
    Hybrid: { cls: 'hyb', tag: OX.t('混合模式（Hybrid）', 'Hybrid'), zh: OX.t('SILK 和 CELT 一起用', 'SILK and CELT together') },
    CELT: { cls: 'celt', tag: 'CELT-only', zh: OX.t('只用 CELT', 'no SILK layer') },
  };
  const C_MEANING = [OX.t('1 帧', '1 frame'), OX.t('2 帧，字节数相等', '2 frames of equal size'), OX.t('2 帧，字节数不同（第一帧的字节数写在 TOC 后面）', '2 frames of different sizes (the size of the first frame follows the TOC)'), OX.t('任意帧数（TOC 后面跟一个帧数字节）', 'any number of frames (a frame count byte follows the TOC)')];
  const TYPE_NAME = [OX.t('无语音', 'Inactive'), OX.t('清音', 'Unvoiced'), OX.t('浊音', 'Voiced')];
  const nBytes = (k) => `${k} byte${k === 1 ? '' : 's'}`; // English only
  const nFrames = (k) => `${k} frame${k === 1 ? '' : 's'}`; // English only
  const hex2 = (b) => b.toString(16).toUpperCase().padStart(2, '0');
  const bin = (v, n) => v.toString(2).padStart(n, '0');
  const khz = (hz) => (hz / 1000) + ' kHz';

  /* ---------- streams ---------- */
  const STREAMS = {
    speech: { src: D.speech, def: 16000, idx: 127 }, // 127: the voiced frame used across the page
    music: { src: D.music, def: 64000, idx: ((D.music.celt_detail || []).find((d) => d.role === 'tonal') || { index: 23 }).index }, // the tonal frame picked by the data generator
  };
  const st = { stream: 'speech', rate: 16000, idx: 127, toc: 0 };
  const encOf = (s, r) => STREAMS[s].src.encodes.find((e) => e.bitrate === r) || STREAMS[s].src.encodes[0];
  function cur() {
    const S = STREAMS[st.stream].src, enc = encOf(st.stream, st.rate);
    const bytes = OX.dsp.b64ToBytes(enc.packets[st.idx]);
    return { S, enc, bytes, toc0: bytes[0] };
  }
  {
    const e = encOf('speech', 16000);
    st.rate = e.bitrate;
    st.idx = Math.min(STREAMS.speech.idx, e.n_frames - 1);
    st.toc = OX.dsp.b64ToBytes(e.packets[st.idx])[0];
  }

  /* How a decoder splits the N bytes of a packet whose first byte is `toc` (RFC 6716 §3.2.1-3.2.5, §3.4). */
  function parseFrames(toc, bytes) {
    const N = bytes.length, c = toc & 3, ms = CFG[toc >> 3].frame_ms;
    const bad = (text) => ({ ok: false, text });
    const MAXF = 1275;
    const readLen = (pos) => { // 0..251: the length itself; 252..255: length = 4 × next byte + this byte
      if (pos >= N) return null;
      const b = bytes[pos];
      if (b < 252) return { len: b, used: 1, txt: '0x' + hex2(b) };
      if (pos + 1 >= N) return null;
      return { len: 4 * bytes[pos + 1] + b, used: 2, txt: `0x${hex2(b)} 0x${hex2(bytes[pos + 1])}` };
    };
    const L = N - 1;
    if (c === 0) {
      if (L > MAXF) return bad(OX.t(`1 帧，但 ${L} 字节超过了一帧最多 1275 字节的上限：无效。`, `1 frame, but ${L} bytes exceeds the limit of 1275 bytes per frame: invalid.`));
      return { ok: true, frames: 1, dur: ms, text: OX.t(`1 帧，占 TOC 后面的全部 ${L} 个字节，时长 ${ms} 毫秒。`, `1 frame. It takes all ${nBytes(L)} after the TOC and lasts ${ms} ms.`) };
    }
    if (c === 1) {
      if (L % 2) return bad(OX.t(`2 帧，字节数相等。但 TOC 后面有 ${L} 个字节，是奇数，不能平分：解码器会把这个包判为无效。`, `2 frames of equal size. But ${nBytes(L)} follow the TOC, an odd number that cannot be split evenly: a decoder rejects this packet as invalid.`));
      if (L / 2 > MAXF) return bad(OX.t(`2 帧，每帧 ${L / 2} 字节，超过 1275 字节的上限：无效。`, `2 frames of ${L / 2} bytes each, over the limit of 1275 bytes: invalid.`));
      return { ok: true, frames: 2, dur: 2 * ms, text: OX.t(`2 帧，各 ${L / 2} 字节，共 ${2 * ms} 毫秒。`, `2 frames of ${nBytes(L / 2)} each, ${2 * ms} ms in total.`) };
    }
    if (c === 2) {
      const r = readLen(1);
      if (!r) return bad(OX.t('2 帧，字节数不同。但包里放不下第一帧的字节数：无效。', '2 frames of different sizes. But the packet is too short to hold the size of the first frame: invalid.'));
      const rest = N - 1 - r.used;
      if (r.len > rest) return bad(OX.t(`2 帧，字节数不同。TOC 后面的 ${r.txt} 表示第一帧有 ${r.len} 字节，可后面只剩 ${rest} 个字节：解码器会把这个包判为无效。`, `2 frames of different sizes. The ${r.txt} after the TOC says that the first frame has ${nBytes(r.len)}, but only ${nBytes(rest)} are left: a decoder rejects this packet as invalid.`));
      if (rest - r.len > MAXF) return bad(OX.t(`2 帧，第二帧 ${rest - r.len} 字节，超过 1275 字节的上限：无效。`, `2 frames, and the second one has ${rest - r.len} bytes, over the limit of 1275 bytes: invalid.`));
      return { ok: true, frames: 2, dur: 2 * ms, text: OX.t(`2 帧，字节数不同。TOC 后面的 ${r.txt} 表示第一帧有 ${r.len} 字节，第二帧占剩下的 ${rest - r.len} 字节，共 ${2 * ms} 毫秒。`, `2 frames of different sizes. The ${r.txt} after the TOC says that the first frame has ${nBytes(r.len)}, and the second frame takes the remaining ${nBytes(rest - r.len)}. ${2 * ms} ms in total.`) };
    }
    if (N < 2) return bad(OX.t('任意帧数，但包里没有帧数字节：无效。', 'Any number of frames, but the packet has no frame count byte: invalid.'));
    const fc = bytes[1], v = fc >> 7, p = (fc >> 6) & 1, M = fc & 63;
    const head = OX.t(`TOC 后面的 0x${hex2(fc)}（${bin(fc, 8)}）被当成帧数字节：v = ${v}（${v ? '各帧字节数不同' : '各帧字节数相同'}），p = ${p}（${p ? '有填充' : '没有填充'}），M = ${M} 帧`, `The 0x${hex2(fc)} (${bin(fc, 8)}) after the TOC is read as the frame count byte: v = ${v} (${v ? 'frames of different sizes' : 'all frames the same size'}), p = ${p} (${p ? 'padding' : 'no padding'}), M = ${nFrames(M)}`);
    if (M === 0) return bad(OX.t(`${head}。帧数为 0：无效。`, `${head}. A frame count of 0: invalid.`));
    const dur = M * ms;
    if (dur > 120) return bad(OX.t(`${head}，${M} × ${ms} 毫秒 = ${dur} 毫秒，超过一个包最多 120 毫秒的上限：解码器会把这个包判为无效。`, `${head}. ${M} × ${ms} ms = ${dur} ms, over the limit of 120 ms per packet: a decoder rejects this packet as invalid.`));
    let pos = 2, pad = 0;
    if (p) {
      for (;;) { // each 255 means 254 bytes of padding and one more length byte
        if (pos >= N) return bad(OX.t(`${head}。填充长度一直写到了包尾：无效。`, `${head}. The padding length runs to the end of the packet: invalid.`));
        const b = bytes[pos++];
        pad += b === 255 ? 254 : b;
        if (b !== 255) break;
      }
    }
    if (!v) {
      const R = N - pos - pad;
      if (R < 0) return bad(OX.t(`${head}。填充有 ${pad} 字节，比剩下的包还长：无效。`, `${head}. The padding is ${nBytes(pad)}, longer than the rest of the packet: invalid.`));
      if (R % M) return bad(OX.t(`${head}。剩下的 ${R} 个字节不能平分成 ${M} 帧：解码器会把这个包判为无效。`, `${head}. The remaining ${nBytes(R)} cannot be split evenly into ${M} frames: a decoder rejects this packet as invalid.`));
      if (R / M > MAXF) return bad(OX.t(`${head}。每帧 ${R / M} 字节，超过 1275 字节的上限：无效。`, `${head}. Each frame would have ${R / M} bytes, over the limit of 1275 bytes: invalid.`));
      return { ok: true, frames: M, dur, text: OX.t(`${head}${p ? `，包尾有 ${pad} 字节填充` : ''}。每帧 ${R / M} 字节，共 ${dur} 毫秒。`, `${head}${p ? `, with ${nBytes(pad)} of padding at the end of the packet` : ''}. Each frame has ${nBytes(R / M)}. ${dur} ms in total.`) };
    }
    let sum = 0;
    for (let i = 0; i < M - 1; i++) {
      const r = readLen(pos);
      if (!r) return bad(OX.t(`${head}。包里放不下 ${M - 1} 个帧的字节数：无效。`, `${head}. The packet is too short to hold the sizes of ${nFrames(M - 1)}: invalid.`));
      if (r.len > MAXF) return bad(OX.t(`${head}。第 ${i + 1} 帧的字节数 ${r.len} 超过 1275 字节的上限：无效。`, `${head}. Frame ${i + 1} has a size of ${r.len} bytes, over the limit of 1275 bytes: invalid.`));
      pos += r.used; sum += r.len;
    }
    const last = N - pos - pad - sum;
    if (last < 0) return bad(OX.t(`${head}。接下来 ${M - 1} 个帧字节数加起来是 ${sum} 字节，超过了包里剩下的字节：解码器会把这个包判为无效。`, `${head}. The ${M - 1} frame size${M - 1 === 1 ? '' : 's'} that follow add up to ${nBytes(sum)}, more than the packet has left: a decoder rejects this packet as invalid.`));
    if (last > MAXF) return bad(OX.t(`${head}。最后一帧 ${last} 字节，超过 1275 字节的上限：无效。`, `${head}. The last frame has ${last} bytes, over the limit of 1275 bytes: invalid.`));
    return { ok: true, frames: M, dur, text: OX.t(`${head}${p ? `，包尾有 ${pad} 字节填充` : ''}。接着是 ${M - 1} 个帧字节数，最后一帧占剩下的 ${last} 字节，共 ${dur} 毫秒。`, `${head}${p ? `, with ${nBytes(pad)} of padding at the end of the packet` : ''}. Next come ${M - 1} frame size${M - 1 === 1 ? '' : 's'}, and the last frame takes the remaining ${nBytes(last)}. ${dur} ms in total.`) };
  }

  /* ---------- styles ---------- */
  OX.ui.css('packet', `
    .pk-ctl-note { font-size: var(--tiny); color: var(--ink-3); margin: 0; }
    .pk-strip { margin: 6px 0 0; }
    .pk-strip .ox-canvas { cursor: pointer; }
    .pk-main { display: grid; grid-template-columns: minmax(0, 1fr); gap: 22px 32px; margin-top: 16px; }
    @media (min-width: 900px) { .pk-main { grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); } }
    .pk-main > div { min-width: 0; }
    .pk-h { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 10px; margin: 0 0 10px; font: 600 var(--small)/1.4 var(--font-body); color: var(--ink); }
    .pk-h .sub { font-weight: 400; font-size: var(--tiny); color: var(--ink-3); }
    .pk-tochead { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-bottom: 10px; min-height: 34px; }
    .pk-hex { font: 600 1.45rem/1 var(--font-mono); color: var(--accent); font-variant-numeric: tabular-nums; }
    .pk-formula { font: 400 var(--tiny)/1.4 var(--font-mono); color: var(--ink-2); }
    .pk-edited { font: 500 0.72rem/1 var(--font-body); color: var(--accent); background: var(--accent-soft); padding: 4px 8px; border-radius: 999px; white-space: nowrap; }
    .pk-tochead .ox-btn { padding: 6px 10px; }
    .pk-bits { display: flex; gap: 8px; }
    .pk-grp { display: flex; flex-direction: column; align-items: center; gap: 5px; padding: 6px 6px 7px; border-radius: 8px; background: var(--sunken); min-width: 0; }
    .pk-glabel { font: 500 0.72rem/1.2 var(--font-body); color: var(--ink-2); white-space: nowrap; }
    .pk-glabel code { background: none; padding: 0; font-size: 1em; color: var(--ink); }
    .pk-btns { display: flex; gap: 3px; }
    .pk-bit { width: 34px; height: 40px; padding: 0; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--ink);
      font: 600 1rem/1 var(--font-mono); cursor: pointer; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; }
    .pk-bit small { font: 400 0.58rem/1 var(--font-mono); color: var(--ink-3); }
    .pk-bit:hover { border-color: var(--ink-3); }
    .pk-bit.on { background: var(--accent); border-color: var(--accent); color: #fff; }
    .pk-bit.on small { color: rgba(255, 255, 255, 0.78); }
    :root[data-theme="dark"] .pk-bit.on, :root[data-theme="dark"] .pk-bit.on small { color: #0d1117; }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .pk-bit.on, :root:not([data-theme="light"]) .pk-bit.on small { color: #0d1117; } }
    .pk-gval { font: 500 var(--tiny)/1.2 var(--font-mono); color: var(--ink); white-space: nowrap; }
    @media (max-width: 440px) {
      .pk-bit { width: 28px; height: 36px; font-size: 0.92rem; }
      .pk-bits { gap: 5px; }
      .pk-grp { padding: 5px 4px 6px; }
      .pk-btns { gap: 2px; }
    }
    .pk-dl { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 14px; margin: 14px 0 0; font-size: var(--small); line-height: 1.5; }
    .pk-dl dt { color: var(--ink-3); white-space: nowrap; }
    .pk-dl dd { margin: 0; color: var(--ink); min-width: 0; }
    .pk-dl dd .sub { display: block; color: var(--ink-2); font-size: var(--tiny); }
    .pk-dl .mono { font-variant-numeric: tabular-nums; }
    .pk-split { margin: 14px 0 0; padding: 8px 12px; font-size: var(--small); line-height: 1.65; color: var(--ink-2);
      border-left: 2px solid var(--accent); background: color-mix(in srgb, var(--accent-soft) 45%, transparent); border-radius: 0 6px 6px 0; overflow-wrap: anywhere; }
    .pk-split.bad { border-left-color: var(--bad); background: color-mix(in srgb, var(--bad) 7%, transparent); }
    .pk-split b { color: var(--ink); font-weight: 600; }
    .pk-split.bad b { color: var(--bad); }
    .pk-grid { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 4px; }
    @media (max-width: 520px) { .pk-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
    .pk-cell { min-width: 0; border: 1px solid transparent; border-radius: 6px; padding: 5px 1px 6px; cursor: pointer;
      display: flex; flex-direction: column; align-items: center; gap: 3px; font-family: var(--font-mono); }
    .pk-cell .n { font-size: 0.6rem; line-height: 1; opacity: 0.8; }
    .pk-cell .bw { font: 600 0.8rem/1 var(--font-mono); }
    .pk-cell .ms { font-size: 0.64rem; line-height: 1; white-space: nowrap; }
    .pk-cell.silk { background: var(--silk-soft); color: var(--silk); }
    .pk-cell.celt { background: var(--celt-soft); color: var(--celt); }
    .pk-cell.hyb { background: linear-gradient(90deg, var(--silk-soft) 50%, var(--celt-soft) 50%); color: var(--ink); }
    .pk-cell:hover { border-color: var(--ink-3); }
    .pk-cell.cur { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    .pk-cell.cur .n { opacity: 1; font-weight: 600; }
    .pk-grid-note { font-size: var(--tiny); color: var(--ink-3); margin: 8px 0 0; }
    .pk-pkt { margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--rule); }
    .pk-pkt .ox-readouts { margin: 0 0 12px; }
    .pk-dump { font: 400 0.8rem/1.8 var(--font-mono); background: var(--sunken); border-radius: 8px; padding: 8px 12px; overflow: auto; max-height: 23em; font-variant-numeric: tabular-nums; }
    .pk-row { display: flex; white-space: nowrap; }
    .pk-off { flex: none; width: 3.4em; color: var(--ink-3); }
    .pk-b { flex: none; width: 3ch; text-align: center; color: var(--ink); border-radius: 3px; }
    .pk-b.mid { margin-left: 1ch; }
    .pk-b.toc { background: var(--accent); color: #fff; font-weight: 600; }
    :root[data-theme="dark"] .pk-b.toc { color: #0d1117; }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .pk-b.toc { color: #0d1117; } }
    .pk-lay { display: flex; align-items: center; gap: 0; margin-top: 8px; height: 26px; border-radius: 6px; overflow: hidden;
      background: var(--sunken); font: 500 0.72rem/1 var(--font-body); color: var(--ink-2); }
    .pk-lay span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 0 8px; }
    .pk-lay .toc { flex: none; align-self: stretch; display: flex; align-items: center; background: var(--accent); color: #fff; font-family: var(--font-mono); font-weight: 600; }
    :root[data-theme="dark"] .pk-lay .toc { color: #0d1117; }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .pk-lay .toc { color: #0d1117; } }
    .pk-lay .fwd { flex: 1 1 auto; min-width: 0; }
    .pk-lay .mid { flex: 0 1 auto; color: var(--ink-3); min-width: 0; }
    .pk-lay .back { flex: 1 1 auto; min-width: 0; text-align: right; }
    .pk-lay .silk-c { color: var(--silk); } .pk-lay .celt-c { color: var(--celt); }
  `);

  /* ---------- controls ---------- */
  const streamSeg = segmented({
    id: 'pk-stream', label: OX.t('码流', 'Stream'), value: st.stream,
    options: [{ value: 'speech', label: OX.t('语音（16 kHz）', 'Speech (16 kHz)') }, { value: 'music', label: OX.t('音乐（48 kHz）', 'Music (48 kHz)') }],
    onChange: (v) => setStream(v),
  });
  const rateHolder = el('div', { style: { display: 'contents' } });
  const sliderOpts = { id: 'pk-idx', label: OX.t('包序号', 'Packet number'), min: 0, max: cur().enc.n_frames - 1, step: 1, value: st.idx, format: (v) => '#' + v, onInput: (v) => setIdx(v) };
  const idxSlider = slider(sliderOpts);
  const ctlNote = el('p', { class: 'pk-ctl-note' });
  const stripHost = el('div', { class: 'pk-strip' });

  function buildRateSeg() {
    const encs = STREAMS[st.stream].src.encodes;
    const seg = segmented({
      id: 'pk-rate-' + st.stream, label: OX.t('目标码率（kb/s）', 'Target bitrate (kb/s)'), value: String(st.rate),
      options: encs.map((e) => ({ value: String(e.bitrate), label: e.bitrate / 1000 + (e.rate_note ? ' *' : '') })),
      onChange: (v) => setRate(+v),
    });
    rateHolder.innerHTML = '';
    rateHolder.appendChild(seg);
    const odd = encs.find((e) => e.rate_note);
    ctlNote.textContent = odd
      ? OX.t(`* 这一路的目标是 ${odd.bitrate / 1000} kb/s，实际平均 ${odd.avg_kbps.toFixed(1)} kb/s，相差超过 25%。`, `* This stream targets ${odd.bitrate / 1000} kb/s, but its actual average is ${odd.avg_kbps.toFixed(1)} kb/s, more than 25% off the target.`)
      : '';
    ctlNote.style.display = odd ? '' : 'none';
  }

  /* ---------- TOC editor ---------- */
  const hexEl = el('span', { class: 'pk-hex' });
  const formulaEl = el('span', { class: 'pk-formula' });
  const editedEl = el('span', { class: 'pk-edited' });
  const restoreBtn = button({ icon: 'reset', label: OX.t('还原为这个包的 TOC', 'Restore the original TOC'), kind: 'ghost', onClick: () => { st.toc = cur().toc0; renderToc(); } });
  const bitBtns = {};
  const mkBit = (b, role) => {
    const btn = el('button', { type: 'button', class: 'pk-bit' });
    btn.addEventListener('click', () => { st.toc ^= 1 << b; renderToc(); });
    btn.dataset.role = role;
    bitBtns[b] = btn;
    return btn;
  };
  const gval = { cfg: el('span', { class: 'pk-gval' }), s: el('span', { class: 'pk-gval' }), c: el('span', { class: 'pk-gval' }) };
  const group = (labelHtml, bits, role, val) => el('div', { class: 'pk-grp' }, [
    el('span', { class: 'pk-glabel', html: labelHtml }),
    el('div', { class: 'pk-btns' }, bits.map((b) => mkBit(b, role))),
    val,
  ]);
  const bitsRow = el('div', { class: 'pk-bits', role: 'group', 'aria-label': OX.t('TOC 字节的 8 个比特，从最高位到最低位', 'The 8 bits of the TOC byte, from most to least significant') }, [
    group(OX.t('配置号 <code>config</code>', 'Configuration <code>config</code>'), [7, 6, 5, 4, 3], OX.t('配置号', 'configuration'), gval.cfg),
    group('<code>s</code>', [2], OX.t('立体声标志', 'stereo flag'), gval.s),
    group('<code>c</code>', [1, 0], OX.t('帧数代码', 'frame count code'), gval.c),
  ]);
  const dl = el('dl', { class: 'pk-dl' });
  const splitEl = el('p', { class: 'pk-split', 'aria-live': 'polite' });
  const left = el('div', {}, [
    el('div', { class: 'pk-h' }, [el('span', { text: OX.t('TOC 字节', 'TOC byte') }), el('span', { class: 'sub', text: OX.t('点任意一位可以翻转它', 'Click any bit to flip it') })]),
    el('div', { class: 'pk-tochead' }, [hexEl, formulaEl, editedEl, restoreBtn]),
    bitsRow, dl, splitEl,
  ]);

  /* ---------- config grid ---------- */
  const cells = CFG.map((c) => {
    const m = MODE[c.mode];
    const b = el('button', {
      type: 'button', class: 'pk-cell ' + m.cls,
      'aria-label': OX.t(`配置 ${c.config}：${m.tag}，${BW_NAME[c.bandwidth]}，${c.frame_ms} 毫秒`, `Configuration ${c.config}: ${m.tag}, ${BW_NAME[c.bandwidth]}, ${c.frame_ms} ms`),
    }, [el('span', { class: 'n', text: String(c.config) }), el('span', { class: 'bw', text: c.bandwidth }), el('span', { class: 'ms', text: c.frame_ms + ' ms' })]);
    b.addEventListener('click', () => { st.toc = (c.config << 3) | (st.toc & 7); renderToc(); });
    return b;
  });
  const right = el('div', {}, [
    el('div', { class: 'pk-h' }, [el('span', { text: OX.t('32 种配置', '32 configurations') }), el('span', { class: 'sub', text: OX.t('配置号 = TOC 的高 5 位；点一格就改成那一种', 'Configuration number = top 5 bits of the TOC. Click a cell to switch to it.') })]),
    el('div', { class: 'pk-grid' }, cells),
    legend([
      { label: 'SILK-only', color: 'var(--silk)', kind: 'box' },
      { label: OX.t('混合模式（Hybrid）', 'Hybrid'), color: 'linear-gradient(90deg, var(--silk) 50%, var(--celt) 50%)', kind: 'box' },
      { label: 'CELT-only', color: 'var(--celt)', kind: 'box' },
    ]),
    el('p', { class: 'pk-grid-note', text: OX.t('每格写着配置号、音频带宽和帧长。中带（MB）只有 SILK 有。CELT 的 16–19 号是窄带（NB）。混合模式只有超宽带和全带，帧长只有 10 或 20 毫秒。', 'Each cell shows the configuration number, the audio bandwidth, and the frame duration. Only SILK has mediumband (MB). CELT configurations 16–19 are narrowband (NB). Hybrid mode exists only for super-wideband and fullband, with frames of 10 or 20 ms only.') }),
  ]);

  /* ---------- the packet ---------- */
  const roSize = readout(OX.t('大小', 'Size'), '');
  const roDur = readout(OX.t('时长', 'Duration'), '');
  const roRate = readout(OX.t('这个包的码率', 'Bitrate of this packet'), '');
  const roPos = readout(OX.t('对应原始信号', 'Time in the original signal'), '');
  const roExtra = readout('', '');
  const dump = el('div', { class: 'pk-dump', role: 'region', 'aria-label': OX.t('数据包的全部字节，十六进制', 'All bytes of the packet, in hex'), tabindex: '0' });
  const dumpHint = el('p', { class: 'ox-note', style: { margin: '4px 0 0' } });
  const lay = el('div', { class: 'pk-lay', 'aria-hidden': 'true' });
  const payNote = el('p', { class: 'ox-note' });
  const pktSec = el('div', { class: 'pk-pkt' }, [
    el('div', { class: 'pk-h' }, [el('span', { text: OX.t('这个包的全部字节', 'All bytes of this packet') }), el('span', { class: 'sub', text: OX.t('十六进制；左边是字节偏移（十进制），蓝底的第一个字节是 TOC', 'In hex. The byte offset (in decimal) is on the left. The first byte, on blue, is the TOC.') })]),
    el('div', { class: 'ox-readouts' }, [roSize, roDur, roRate, roPos, roExtra]),
    dump, dumpHint, lay, payNote,
  ]);

  body.append(
    el('div', { class: 'ox-controls' }, [streamSeg, rateHolder, idxSlider]),
    ctlNote, stripHost,
    el('div', { class: 'pk-main' }, [left, right]),
    pktSec);
  buildRateSeg();

  /* ---------- strip of packet sizes (declared before the canvas: draw runs at once) ---------- */
  let stripDrag = false;
  function drawStrip(ctx, w, h) {
    const T = OX.theme, { enc } = cur(), n = enc.n_frames;
    const sizes = enc.sizes.slice(0, n), max = Math.max(...sizes);
    const top = 18, bot = h - 1, bw = w / n;
    ctx.font = OX.font(10.5, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(w < 520 ? OX.t(`${n} 个包，柱高 = 字节数`, `${n} packets, bar height = bytes`) : OX.t(`这一路的 ${n} 个包，柱高 = 字节数（最大 ${max} 字节）。点一下或拖动，可以换包`, `The ${n} packets of this stream; bar height = bytes (max ${max}). Click or drag to pick a packet`), 0, 11);
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.22 : 0.14);
    ctx.fillRect(st.idx * bw, top, Math.max(2, bw), bot - top);
    const g = bw > 3 ? 0.5 : 0;
    for (let k = 0; k < n; k++) {
      const hh = Math.max(1, ((bot - top) * sizes[k]) / max);
      ctx.fillStyle = k === st.idx ? T.accent : OX.alpha(T.ink3, T.dark ? 0.6 : 0.45);
      ctx.fillRect(k * bw + g, bot - hh, Math.max(0.8, bw - 2 * g), hh);
    }
  }
  const strip = OX.view.canvas(stripHost, { height: 48, label: OX.t('这一路每个包的字节数，选中的包用蓝色标出', 'Size in bytes of each packet in this stream, with the selected packet in blue') }, drawStrip);
  const pickStrip = (e) => {
    const r = strip.canvas.getBoundingClientRect(), n = cur().enc.n_frames;
    setIdx(Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))));
  };
  strip.canvas.addEventListener('pointerdown', (e) => { stripDrag = true; try { strip.canvas.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } pickStrip(e); });
  strip.canvas.addEventListener('pointermove', (e) => { if (stripDrag) pickStrip(e); });
  strip.canvas.addEventListener('pointerup', () => { stripDrag = false; });
  strip.canvas.addEventListener('pointercancel', () => { stripDrag = false; });

  /* ---------- rendering ---------- */
  function renderToc() {
    const t = st.toc, cfgN = t >> 3, s = (t >> 2) & 1, c = t & 3;
    const cfg = CFG[cfgN], m = MODE[cfg.mode], bw = BWS[cfg.bandwidth];
    const { bytes, toc0 } = cur();
    const edited = t !== toc0;
    hexEl.textContent = '0x' + hex2(t);
    formulaEl.textContent = `= ${bin(cfgN, 5)} ${s} ${bin(c, 2)} = ${cfgN} × 8 + ${s} × 4 + ${c}`;
    editedEl.textContent = OX.t(`已修改，原为 0x${hex2(toc0)}`, `Edited, was 0x${hex2(toc0)}`);
    editedEl.style.display = edited ? '' : 'none';
    restoreBtn.style.display = edited ? '' : 'none';
    for (let b = 0; b < 8; b++) {
      const v = (t >> b) & 1, btn = bitBtns[b];
      btn.innerHTML = `${v}<small>${b}</small>`;
      btn.classList.toggle('on', !!v);
      btn.setAttribute('aria-pressed', v ? 'true' : 'false');
      btn.setAttribute('aria-label', OX.t(`第 ${b} 位（${btn.dataset.role}），现在是 ${v}`, `Bit ${b} (${btn.dataset.role}), now ${v}`));
    }
    gval.cfg.textContent = '= ' + cfgN;
    gval.s.textContent = s ? OX.t('立体声', 'stereo') : OX.t('单声道', 'mono');
    gval.c.textContent = '= ' + c;

    let bwSub;
    if (cfg.mode === 'SILK') bwSub = OX.t(`SILK 以 ${khz(bw.sample_rate)} 采样率编码`, `SILK codes at a ${khz(bw.sample_rate)} sample rate`);
    else if (cfg.mode === 'Hybrid') bwSub = OX.t('8 kHz 以下由 SILK 编码（16 kHz 采样），8 kHz 以上由 CELT 编码', 'SILK codes below 8 kHz (at a 16 kHz sample rate), and CELT codes above 8 kHz');
    else bwSub = OX.t(`CELT 内部按 48 kHz 工作，只编码到 ${khz(bw.audio_hz)}`, `CELT runs at 48 kHz internally and codes only up to ${khz(bw.audio_hz)}`);
    let durSub = '';
    if (cfg.mode === 'SILK' && cfg.frame_ms > 20) durSub = OX.t(`SILK 在这一帧里连续编 ${cfg.frame_ms / 20} 个 20 毫秒的 SILK 帧`, `Within this frame, SILK codes ${cfg.frame_ms / 20} SILK frames of 20 ms in a row`);
    else if (cfg.mode === 'SILK') durSub = cfg.frame_ms === 10 ? OX.t('10 毫秒的帧只有 2 个子帧', 'A 10 ms frame has only 2 subframes') : OX.t('4 个 5 毫秒的子帧', '4 subframes of 5 ms');
    const row = (k, v, sub) => [el('dt', { text: k }), el('dd', {}, [v, sub ? el('span', { class: 'sub', text: sub }) : null])];
    dl.innerHTML = '';
    [
      row(OX.t('配置号', 'Configuration'), el('span', { class: 'mono', text: OX.t(`${cfgN}（共 0–31）`, `${cfgN} (of 0–31)`) })),
      row(OX.t('模式', 'Mode'), el('span', {}, [el('span', { class: 'pill ' + m.cls, text: m.tag }), ' ' + m.zh])),
      row(OX.t('音频带宽', 'Audio bandwidth'), OX.t(`${BW_NAME[cfg.bandwidth]}（${cfg.bandwidth}），最高 ${khz(bw.audio_hz)}`, `${cfg.bandwidth} (${BW_NAME[cfg.bandwidth]}), up to ${khz(bw.audio_hz)}`), bwSub),
      row(OX.t('帧长', 'Frame duration'), el('span', { class: 'mono', text: cfg.frame_ms + ' ms' }), durSub),
      row(OX.t('声道', 'Channels'), s ? OX.t('立体声（s = 1）', 'Stereo (s = 1)') : OX.t('单声道（s = 0）', 'Mono (s = 0)')),
      row(OX.t('帧数', 'Frame count'), OX.t(`c = ${c}：${C_MEANING[c]}`, `c = ${c}: ${C_MEANING[c]}`)),
    ].forEach((pair) => dl.append(...pair));

    const r = parseFrames(t, bytes);
    splitEl.className = 'pk-split' + (r.ok ? '' : ' bad');
    splitEl.innerHTML = '';
    splitEl.append(el('b', { text: edited ? OX.t('用改过的 TOC 读这个包：', 'Read with the edited TOC: ') : OX.t('解码器这样拆这个包：', 'How a decoder splits this packet: ') }), r.text);
    cells.forEach((b, i) => { b.classList.toggle('cur', i === cfgN); b.setAttribute('aria-pressed', i === cfgN ? 'true' : 'false'); });
  }

  let bpr = 16;
  function renderDump() {
    const { bytes } = cur();
    bpr = dump.clientWidth >= 430 || !dump.clientWidth ? 16 : 8;
    const frag = document.createDocumentFragment();
    for (let off = 0; off < bytes.length; off += bpr) {
      const rowEl = el('div', { class: 'pk-row' }, [el('span', { class: 'pk-off', text: String(off).padStart(3, ' ') })]);
      for (let i = off; i < Math.min(bytes.length, off + bpr); i++) {
        const cls = 'pk-b' + (i === 0 ? ' toc' : '') + (bpr === 16 && i % 16 === 8 ? ' mid' : '');
        rowEl.appendChild(el('span', { class: cls, text: hex2(bytes[i]), title: i === 0 ? OX.t('TOC 字节', 'TOC byte') : null }));
      }
      frag.appendChild(rowEl);
    }
    dump.innerHTML = '';
    dump.appendChild(frag);
    dump.scrollTop = 0;
    const rows = Math.ceil(bytes.length / bpr);
    const scrolls = dump.scrollHeight > dump.clientHeight + 2;
    dumpHint.textContent = scrolls ? OX.t(`共 ${rows} 行，框里可以上下滚动。`, `${rows} rows in total. Scroll inside the box to see them all.`) : '';
    dumpHint.style.display = scrolls ? '' : 'none';
  }
  const dumpRO = window.ResizeObserver ? new ResizeObserver(() => { const want = dump.clientWidth >= 430 ? 16 : 8; if (want !== bpr) renderDump(); }) : null;
  if (dumpRO) dumpRO.observe(dump);

  function renderPacket() {
    const { S, enc, bytes, toc0 } = cur();
    const N = bytes.length, cfg = CFG[toc0 >> 3];
    const r = parseFrames(toc0, bytes);
    const dur = r.ok ? r.dur : cfg.frame_ms;
    roSize.set(OX.t(`${N} 字节（TOC 1 + 数据 ${N - 1}）`, `${N} bytes (TOC 1 + data ${N - 1})`));
    roDur.set(`${dur} ms`);
    roRate.set(`${((N * 8) / dur).toFixed(1)} kb/s`);
    const F = S.frame_samples, fs = S.sample_rate, a = st.idx * F - enc.delay_samples;
    const sec = (x) => (x < 0 ? '−' : '') + (Math.abs(x) / fs).toFixed(3);
    roPos.set(`${sec(a)}–${sec(a + F)} s`);
    // per-frame facts exist only for the stream they were traced from
    let extra = null;
    const sf = S.silk_frames, cf = S.celt_frames;
    if (sf && sf.bitrate === enc.bitrate && sf.frames[st.idx]) extra = [OX.t('SILK 信号类型', 'SILK signal type'), TYPE_NAME[sf.frames[st.idx].signal_type]];
    else if (cf && cf.bitrate === enc.bitrate && cf.frames[st.idx]) extra = [OX.t('CELT 块长', 'CELT block size'), cf.frames[st.idx].transient ? OX.t('瞬态帧：8 个短块', 'Transient: 8 short blocks') : OX.t('1 个长块', '1 long block')];
    roExtra.style.display = extra ? '' : 'none';
    if (extra) { roExtra.querySelector('.ox-ro-label').textContent = extra[0]; roExtra.set(extra[1]); }
    renderDump();

    // what follows the TOC (RFC 6716 §4.1: range-coded data from the front, raw bits from the back)
    lay.innerHTML = '';
    const sp = (cls, text) => el('span', { class: cls, text });
    if (cfg.mode === 'SILK') {
      lay.append(sp('toc', 'TOC'), sp('fwd silk-c', OX.t('区间编码器的输出 →', 'Range coder output →')), sp('back', OX.t('SILK 不用原始比特', 'SILK uses no raw bits')));
      payNote.textContent = OX.t(`TOC 后面的 ${N - 1} 个字节全部来自区间编码器。里面大致依次是语音活动标志、信号类型、增益、LPC 系数（以 NLSF 形式）、浊音帧的基音和 LTP 参数，最后是激励脉冲。`, `All ${nBytes(N - 1)} after the TOC come from the range coder. Roughly in order, they hold the voice activity flag, the signal type, the gains, the LPC coefficients (as NLSFs), the pitch and LTP parameters of voiced frames, and finally the excitation pulses. `)
        + OX.t('SILK 不用原始比特（不用概率表、直接写进包里的比特）。', 'SILK uses no raw bits (bits written into the packet directly, without a probability table). ')
        + OX.t('字节之间没有分界，只能从头一个符号一个符号地解。编码器还会去掉末尾多余的零字节，解码器读到包尾之后一律当作 0。', 'Symbols do not line up with byte boundaries, so the data can only be decoded one symbol at a time, from the start. The encoder also drops unneeded zero bytes at the end, and the decoder reads everything past the end of the packet as 0.');
    } else {
      const fwd = cfg.mode === 'Hybrid' ? OX.t('SILK 层 → CELT 层的区间编码符号 →', 'SILK → CELT, range coded →') : OX.t('区间编码的符号 →', 'Range coded →');
      lay.append(sp('toc', 'TOC'), sp('fwd ' + (cfg.mode === 'Hybrid' ? 'silk-c' : 'celt-c'), fwd), sp('mid', OX.t('分界不固定', 'Boundary varies')), sp('back celt-c', OX.t('← 原始比特', '← Raw bits')));
      // Hybrid: the CELT layer starts at band 17 and never codes post-filter fields (they need start_band == 0:
      // src/celt.rs:4848 encoder, :8066 decoder; libopus celt_decoder.c `if (start==0 && ...)`).
      // CELT-only: of the post-filter fields only the low bits of the period and the 3-bit gain are raw bits.
      payNote.textContent = (cfg.mode === 'Hybrid'
        ? OX.t('TOC 后面先是 SILK 层（8 kHz 以下），CELT 层接着用同一个区间编码器往后写。', 'The SILK layer (below 8 kHz) comes first after the TOC. The CELT layer follows it, written by the same range coder. ')
          + OX.t('CELT 层的精细能量、PVQ 序号的低位等是“原始比特”（不用概率表、直接写进包里的比特），从包的最后一个字节往前写。', 'Some CELT-layer values, such as the fine energy and the low bits of the PVQ indices, are “raw bits” (bits written into the packet directly, without a probability table). They are written backward, starting from the last byte of the packet. ')
          + OX.t('混合模式下 CELT 层不开后滤波器，包里没有它的参数。', 'In hybrid mode the CELT layer does not use the post-filter, so the packet carries no post-filter parameters. ')
        : OX.t('TOC 后面有两种数据。能量、比特分配参数、PVQ 序号的高位等用区间编码，从前往后写。', 'Two kinds of data follow the TOC. The energies, the bit allocation parameters, the high bits of the PVQ indices, and so on are range coded and written from front to back. ')
          + OX.t('精细能量、PVQ 序号的低位、后滤波器周期的低位和增益等是“原始比特”（不用概率表、直接写进包里的比特），从包的最后一个字节往前写。', 'The fine energy, the low bits of the PVQ indices, the low bits of the post-filter period, the post-filter gain, and so on are “raw bits” (bits written into the packet directly, without a probability table). They are written backward, starting from the last byte of the packet. '))
        + OX.t('两段在中间相遇，包里没有记录分界在哪，这里也不标出。', 'The two parts meet in the middle. The packet does not record where the boundary is, so it is not marked here either.');
    }
  }

  function render() {
    renderToc();
    renderPacket();
    strip.redraw();
  }
  function loadPacket() { st.toc = cur().toc0; render(); }
  function setIdx(v) {
    if (v === st.idx) return;
    st.idx = v; STREAMS[st.stream].idx = v;
    idxSlider.set(v);
    loadPacket();
  }
  function setRate(r) {
    st.rate = r;
    st.idx = Math.min(st.idx, encOf(st.stream, r).n_frames - 1);
    loadPacket();
  }
  function setStream(s) {
    st.stream = s;
    const e = encOf(s, STREAMS[s].def);
    st.rate = e.bitrate;
    st.idx = Math.min(STREAMS[s].idx, e.n_frames - 1);
    sliderOpts.max = e.n_frames - 1;
    idxSlider.input.max = String(e.n_frames - 1);
    idxSlider.set(st.idx);
    buildRateSeg();
    loadPacket();
  }

  loadPacket();
  return () => { if (dumpRO) dumpRO.disconnect(); };
});
