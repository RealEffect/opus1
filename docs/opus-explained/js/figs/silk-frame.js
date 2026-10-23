/* Figure 6: one real frame of SILK speech.
 * Top: the 3 s speech excerpt with one color band per 20 ms SILK frame (inactive, unvoiced, voiced). The types and
 * every parameter below are read back from the real 16 kb/s opus1 bitstream (speech.silk_frames).
 * Click a frame to see
 *   (a) its waveform, original vs decoded, with the pitch lag drawn as arrows;
 *   (b) the spectrum of the original frame with the LPC envelope 1/|A(e^jw)| from the decoded coefficients and the
 *       16 NLSFs (normalized line spectral frequencies) as ticks;
 *   (c) the coded excitation pulses (one small integer per sample) with the subframe gains, and below them the LPC
 *       input worked back from the decoded output (the restored excitation, plus the LTP part in voiced frames).
 */
OX.fig('silk-frame', function (root, body) {
  const { el, segmented, button, readout } = OX.ui;
  const D = OX.data.raw;
  const sp = D && D.speech;
  const SF = sp && sp.silk_frames;
  if (!SF || !SF.frames || !sp.pcm) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js（或其中的 speech.silk_frames），这张图无法显示。', 'This figure cannot be shown: the data file data/opus-demo-data.js (or its speech.silk_frames) is missing.') }));
    return;
  }

  const FS = sp.sample_rate; // 16000
  const N = sp.frame_samples; // 320 samples = 20 ms
  const SUB = N / 4; // 80 samples = 5 ms per subframe
  const NS = sp.n_samples;
  const enc = (sp.encodes || []).find((e) => e.bitrate === SF.bitrate);
  const pcm = OX.data.audio('speech.pcm', sp.pcm);
  const dec = enc ? OX.data.audio('speech.encodes.' + enc.bitrate + '.decoded', enc.decoded) : null;
  // The coded frames; the last ones only flush the encoder delay with appended silence.
  const nFrames = Math.min(SF.frames.length, (enc && enc.n_frames) || SF.frames.length);
  const frames = SF.frames.slice(0, nFrames);
  const start0 = frames[0].start_sample;
  const order = SF.lpc_order; // 16 for wideband
  const detail = {};
  (sp.silk_detail || []).forEach((d) => { detail[d.role] = d.index; });
  const TYPE_NAME = [OX.t('无语音', 'Inactive'), OX.t('清音', 'Unvoiced'), OX.t('浊音', 'Voiced')];
  const fmtAmp = (v) => {
    if (v === 0) return '0';
    const s = Math.abs(v) >= 0.1 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toFixed(3);
    return s.replace('-', '−');
  };
  const fmtGain = (g) => (g >= 100 ? String(Math.round(g)) : g >= 10 ? g.toFixed(1).replace(/\.0$/, '') : g.toFixed(2).replace(/0+$/, '').replace(/\.$/, ''));
  const fmtBits = (b) => b.toFixed(1).replace(/\.0$/, '');
  /**
   * "payload = 开头 + 参数 + 激励 + 末尾未用", in bitstream order. Each part is rounded to 0.1 bit (up or down, by
   * largest remainder) so that the shown parts still add up exactly to the payload, and a remainder of exactly 0 stays
   * 0. The remainder is negative when the encoder dropped trailing zero bytes (the decoder reads zeros past the end).
   */
  function bitsText(b) {
    const parts = [b.header, b.side_info, b.excitation, b.payload - b.header - b.side_info - b.excitation];
    const tenths = parts.map((v) => Math.round(v * 1000) / 100);
    const fl = tenths.map(Math.floor);
    const left = Math.round(b.payload * 10 - fl.reduce((a, c) => a + c, 0));
    const byRem = tenths.map((v, i) => [v - fl[i], i]).sort((p, q) => q[0] - p[0]);
    for (let j = 0; j < left; j++) fl[byRem[j][1]] += 1;
    const [head, side, exc, rest] = fl.map((v) => v / 10);
    return OX.t(`${b.payload} = 开头 ${fmtBits(head)} + 参数 ${fmtBits(side)} + 激励 ${fmtBits(exc)}`, `${b.payload} = header ${fmtBits(head)} + parameters ${fmtBits(side)} + excitation ${fmtBits(exc)}`)
      + (rest >= 0 ? OX.t(` + 末尾未用 ${fmtBits(rest)}`, ` + unused at end ${fmtBits(rest)}`) : OX.t(` − ${fmtBits(-rest)}（末尾全为 0 的字节没有发送）`, ` − ${fmtBits(-rest)} (trailing all-zero bytes are not sent)`));
  }
  const JUMPS = [
    { value: 'voiced', label: OX.t('浊音帧', 'Voiced frame') },
    { value: 'unvoiced', label: OX.t('清音帧', 'Unvoiced frame') },
    { value: 'inactive', label: OX.t('无语音帧', 'Inactive frame') },
  ].filter((j) => detail[j.value] != null && detail[j.value] < nFrames);
  let sel = detail.voiced != null && detail.voiced < nFrames ? detail.voiced : Math.floor(nFrames / 2);

  const peak = OX.data.memo('sf:peak', () => { let m = 1e-6; for (let i = 0; i < pcm.length; i++) m = Math.max(m, Math.abs(pcm[i])); return m; });

  /** Decoded excitation pulses of frame k (int8), or null when the data file has none for it. */
  function pulsesOf(k) {
    return OX.data.memo('sf:pulses:' + k, () => {
      const f = SF.frames[k];
      if (f.pulses_b64) { const b = OX.dsp.b64ToBytes(f.pulses_b64); return new Int8Array(b.buffer, b.byteOffset, b.length); }
      const d = (sp.silk_detail || []).find((x) => x.index === k);
      return d ? Int8Array.from(d.pulses) : null;
    });
  }

  /**
   * Pitch marks for a voiced frame: positions t_0 < t_1 < ... inside the frame, one pitch lag apart (the lag of the
   * subframe the interval ends in). The phase is the one that puts the marks on the bursts of `x` (local energy over
   * 5 samples), so with the LPC input they land on the pitch pulses.
   */
  function pitchMarks(x, lags) {
    const e = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) { const v = x[i + d]; if (v === v && v != null) s += v * v; }
      e[i] = s;
    }
    const lagAt = (p) => lags[Math.min(3, Math.max(0, Math.floor(p / SUB)))];
    let best = -Infinity, bestT = null;
    for (let phi = 0; phi < Math.min(N, lags[0]); phi++) {
      const t = [phi];
      for (;;) {
        const p = t[t.length - 1];
        const q = p + lagAt(p + lagAt(p));
        if (q >= N) break;
        t.push(q);
      }
      let sc = 0;
      for (const q of t) sc += e[q];
      sc /= t.length;
      if (sc > best) { best = sc; bestT = t; }
    }
    return bestT;
  }

  /** Everything panels (a)-(c) need for frame k, computed once. */
  function model(k) {
    return OX.data.memo('sf:frame:' + k, () => {
      const f = SF.frames[k];
      const s0 = f.start_sample;
      const orig = new Float32Array(N), decd = new Float32Array(N), res = new Float32Array(N);
      for (let j = 0; j < N; j++) {
        const i = s0 + j;
        orig[j] = i >= 0 && i < pcm.length ? pcm[i] : NaN;
        decd[j] = dec && i >= 0 && i < dec.length ? dec[i] : NaN;
        // LPC input, worked back from the decoded output: SILK computes y = g * s with s[n] = r[n] + sum a_k s[n-k],
        // and rescales the filter state when the gain changes, so r[n] = (y[n] - sum a_k y[n-k]) / g exactly
        // (up to the int16 rounding of y). Subframes 0-1 use the interpolated coefficients when there are any.
        res[j] = NaN;
        if (dec && i - order >= 0 && i < dec.length) {
          const sub = Math.floor(j / SUB);
          const a = sub < 2 && f.lpc_a_first_half ? f.lpc_a_first_half : f.lpc_a;
          let pred = 0;
          for (let q = 0; q < order; q++) pred += a[q] * dec[i - 1 - q];
          res[j] = ((dec[i] - pred) * 32768) / f.gains[sub];
        }
      }
      // spectrum of the original frame (samples outside the excerpt count as 0)
      const NFFT = 1024;
      const x0 = orig.map((v) => (v === v ? v : 0));
      const spec = OX.dsp.spectrumDb(x0, NFFT, OX.dsp.hann(N), -140);
      // LPC envelope 1/|A(e^jw)| from the decoded coefficients (subframes 2-3; the bitstream's NLSFs describe them)
      const env = OX.dsp.lpcEnvelopeDb(f.lpc_a, 1, NFFT / 2 + 1);
      // vertical level: give the envelope the same total power as the frame's spectrum
      let ps = 0, pe = 0;
      for (let b = 0; b < spec.length; b++) { ps += Math.pow(10, spec[b] / 10); pe += Math.pow(10, env[b] / 10); }
      const off = 10 * Math.log10((ps + 1e-30) / (pe + 1e-30));
      for (let b = 0; b < env.length; b++) env[b] += off;
      let amp = 1e-4;
      for (let j = 0; j < N; j++) { if (orig[j] === orig[j]) amp = Math.max(amp, Math.abs(orig[j])); if (decd[j] === decd[j]) amp = Math.max(amp, Math.abs(decd[j])); }
      const hasRes = res.some((v) => v === v);
      const marks = f.signal_type === 2 && f.pitch_lags ? pitchMarks(hasRes ? res : orig, f.pitch_lags) : null;
      return { f, orig, decd, res: hasRes ? res : null, spec, env, amp, marks, pulses: pulsesOf(k) };
    });
  }

  /* ---------------- DOM ---------------- */
  OX.ui.css('silk-frame', `
    .sf-tl { margin-top: 4px; }
    .sf-tl canvas { cursor: pointer; touch-action: pan-y; }
    .sf-tl canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .sf-main { margin-top: 14px; }
    .sf-ro .ox-ro-val { font-size: 0.9rem; }
    .sf-notes { margin-top: 12px; }
    [data-fig="silk-frame"] { --sf-unv: color-mix(in srgb, var(--silk) 38%, transparent); --sf-off: color-mix(in srgb, var(--ink-3) 55%, transparent); }
    :root[data-theme="dark"] [data-fig="silk-frame"] { --sf-unv: color-mix(in srgb, var(--silk) 52%, transparent); --sf-off: color-mix(in srgb, var(--ink-3) 45%, transparent); }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) [data-fig="silk-frame"] { --sf-unv: color-mix(in srgb, var(--silk) 52%, transparent); --sf-off: color-mix(in srgb, var(--ink-3) 45%, transparent); } }
  `);
  const jump = segmented({
    id: 'sf-jump', label: OX.t('跳到', 'Jump to'), value: JUMPS.find((j) => detail[j.value] === sel) ? JUMPS.find((j) => detail[j.value] === sel).value : '',
    options: JUMPS, onChange: (v) => select(detail[v]),
  });
  const prevBtn = button({ label: OX.t('上一帧', 'Previous frame'), onClick: () => select(sel - 1) });
  const nextBtn = button({ label: OX.t('下一帧', 'Next frame'), onClick: () => select(sel + 1) });
  const tlHost = el('div', { class: 'sf-tl' });
  const mainHost = el('div', { class: 'sf-main' });
  const ro = {
    frame: readout(OX.t('帧号与时间', 'Frame and time'), ''),
    type: readout(OX.t('信号类型', 'Signal type'), ''),
    pitch: readout(OX.t('LTP 延迟（基音周期或它的整数倍）', 'LTP lag (the pitch period or a multiple of it)'), ''),
    gain: readout(OX.t('子帧增益', 'Subframe gains'), ''),
    bits: readout(OX.t('本帧 SILK 数据的比特数', 'Bits of SILK data in this frame'), ''),
    order: readout(OX.t('LPC 阶数', 'LPC order'), String(SF.lpc_order)),
  };
  const roWrap = el('div', { class: 'ox-readouts sf-ro', 'aria-live': 'polite' }, Object.values(ro));
  body.append(
    el('div', { class: 'ox-controls' }, [jump, el('div', { style: { display: 'flex', gap: '8px' } }, [prevBtn, nextBtn])]),
    tlHost,
    OX.ui.legend([
      { label: OX.t('浊音', 'Voiced'), color: 'var(--silk)', kind: 'box' },
      { label: OX.t('清音', 'Unvoiced'), color: 'var(--sf-unv)', kind: 'box' },
      { label: OX.t('无语音', 'Inactive (no voice activity)'), color: 'var(--sf-off)', kind: 'box' },
      { label: OX.t('选中的帧', 'Selected frame'), color: 'var(--accent)', kind: 'box' },
    ]),
    mainHost,
    roWrap,
    el('div', { class: 'sf-notes' }, [
      el('p', { class: 'ox-note', text: OX.t('包络：用码流里的 LPC 系数算出合成滤波器 1/A(z) 的频率响应（dB），高度按这一帧频谱的总能量对齐。这段录音是电话频带，约 3.8 kHz 以上几乎没有能量。', 'Envelope: the frequency response (dB) of the synthesis filter 1/A(z), computed from the LPC coefficients in the bitstream. It is shifted vertically so that its total energy matches that of the frame’s spectrum. This recording has telephone bandwidth, with almost no energy above about 3.8 kHz.') }),
      el('p', { class: 'ox-note', text: OX.t('激励：码流里只有整数脉冲。解码器分三步把脉冲还原成激励。第一步，非零脉冲的绝对值减去 0.078。第二步，加上一个量化偏移；它由信号类型和码流里的偏移类型决定，取 0.031、0.098 或 0.234。第三步，按一个伪随机序列翻转正负号；编码器用的是同一个序列，量化时已经考虑了这一步。所以最下一行里约一半尖峰的方向和上一行的脉冲相反，脉冲为 0 的位置也有一个小的 ± 偏移。', 'Excitation: the bitstream contains only integer pulses. The decoder turns the pulses back into the excitation in three steps. First, it subtracts 0.078 from the magnitude of each nonzero pulse. Second, it adds a quantization offset, which depends on the signal type and on the offset type in the bitstream: 0.031, 0.098, or 0.234. Third, it flips signs according to a pseudo-random sequence. The encoder uses the same sequence and has already taken this step into account when quantizing. That is why about half of the spikes in the bottom row point the opposite way from the pulses in the row above, and why positions with a zero pulse also get a small ± offset.') }),
      el('p', { class: 'ox-note', text: OX.t('最下一行是把解码输出 y 反过来送进 A(z)，再除以增益 g 得到的：r[n] = (y[n] − Σ a_k·y[n−k]) / g。它就是 LPC 合成滤波器的输入。两行的单位相同，1 是一个量化台阶。增益以 16 位样本值为单位，只是激励的放大倍数，不是响度。', 'The bottom row is obtained by passing the decoded output y back through A(z) and dividing by the gain g: r[n] = (y[n] − Σ a_k·y[n−k]) / g. This is the input of the LPC synthesis filter. The two rows use the same unit: 1 is one quantization step. The gains are in units of 16-bit sample values. A gain only scales the excitation; it is not the loudness.') }),
      el('p', { class: 'ox-note', text: OX.t('LTP 延迟不一定等于基音周期，也可能是它的 2 倍、3 倍甚至更多。默认显示的第 127 帧里，一个延迟（48 个样本）内有两个波形周期。第 128 帧的波形同样大约每 24 个样本重复一次（约 670 Hz），LTP 延迟却是 72 个样本，是它的 3 倍，所以读数里的 222 Hz 只有音高的 1/3。SILK 的延迟最短 2 毫秒（16 kHz 下是 32 个样本），所以音高超过 500 Hz 时只能取倍数。帧号从 0 数起。', 'The LTP lag is not always the pitch period. It can also be 2, 3, or more times the period. In frame 127, shown by default, one lag (48 samples) spans two periods of the waveform. In frame 128 the waveform again repeats about every 24 samples (about 670 Hz), but the LTP lag is 72 samples, 3 times as long, so the 222 Hz in the readout is only 1/3 of the pitch. The shortest SILK lag is 2 ms (32 samples at 16 kHz), so for a pitch above 500 Hz the lag can only be a multiple of the period. Frames are numbered from 0.') }),
      el('p', { class: 'ox-note', text: OX.t('比特数里的“开头 3”：语音活动标志 1 比特；LBRR 标志 1 比特，表示这个包有没有附带上一帧的一份低码率副本（LBRR，low bitrate redundancy，丢包时用来恢复，见第 9 节“丢包了怎么办”）；再加上区间编码器固定计入的 1 比特（为收尾预留）。', 'The “header 3” in the bit count: 1 bit for the voice activity flag; 1 bit for the LBRR flag, which says whether the packet also carries a low-bitrate copy of the previous frame (low bitrate redundancy, LBRR, used for recovery after packet loss; see “When packets are lost” in Section 9); and 1 bit that the range coder always counts (reserved for finishing the stream).') }),
    ]),
  );

  /* ---------------- timeline ---------------- */
  const tl = OX.view.canvas(tlHost, { height: (w) => (w < 560 ? 96 : 108) }, drawTimeline);
  tl.canvas.setAttribute('role', 'slider');
  tl.canvas.setAttribute('tabindex', '0');
  tl.canvas.setAttribute('aria-label', OX.t('3 秒语音的波形和每帧的信号类型。点一下或用左右方向键选择一帧', 'Waveform of 3 s of speech and the signal type of each frame. Click or tap a frame, or use the left and right arrow keys, to select it.'));
  tl.canvas.setAttribute('aria-valuemin', '0');
  tl.canvas.setAttribute('aria-valuemax', String(nFrames - 1));

  function frameAtX(x) {
    const w = tl.w;
    const s = (x / w) * NS;
    return Math.max(0, Math.min(nFrames - 1, Math.floor((s - start0) / N)));
  }
  let dragging = false;
  tl.canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    try { tl.canvas.setPointerCapture(e.pointerId); } catch (err) { /* not capturable */ }
    const r = tl.canvas.getBoundingClientRect();
    select(frameAtX(e.clientX - r.left));
  });
  tl.canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const r = tl.canvas.getBoundingClientRect();
    select(frameAtX(e.clientX - r.left));
  });
  const endDrag = () => { dragging = false; };
  tl.canvas.addEventListener('pointerup', endDrag);
  tl.canvas.addEventListener('pointercancel', endDrag);
  tl.canvas.addEventListener('keydown', (e) => {
    const step = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 }[e.key];
    if (step) { select(sel + step); e.preventDefault(); }
    else if (e.key === 'Home') { select(0); e.preventDefault(); }
    else if (e.key === 'End') { select(nFrames - 1); e.preventDefault(); }
  });

  function typeColor(T, t) {
    return t === 2 ? T.silk : t === 1 ? OX.alpha(T.silk, T.dark ? 0.52 : 0.38) : OX.alpha(T.ink3, T.dark ? 0.45 : 0.55);
  }

  function drawTimeline(ctx, w, h) {
    const T = OX.theme;
    const xs = OX.view.linear(0, NS, 0, w);
    const wy0 = 4, wy1 = h - 44, by0 = h - 40, by1 = h - 31;
    const mid = (wy0 + wy1) / 2, half = (wy1 - wy0) / 2;
    const ys = (v) => mid - (v / peak) * half;
    // selected frame
    const f = frames[sel];
    const sx0 = xs(Math.max(0, f.start_sample)), sx1 = xs(Math.min(NS, f.start_sample + N));
    const cx = (sx0 + sx1) / 2, sw = Math.max(3, sx1 - sx0);
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.22 : 0.14);
    ctx.fillRect(cx - sw / 2, wy0 - 2, sw, by1 - wy0 + 2);
    // waveform
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, Math.round(mid) + 0.5); ctx.lineTo(w, Math.round(mid) + 0.5); ctx.stroke();
    OX.view.wave(ctx, pcm, 0, NS, xs, ys, OX.alpha(T.ink, T.dark ? 0.62 : 0.55), 1);
    // one band per frame
    const pxPer = (N / NS) * w;
    const gap = pxPer >= 5 ? 1 : 0;
    for (const fr of frames) {
      const a = xs(Math.max(0, fr.start_sample)), b = xs(Math.min(NS, fr.start_sample + N));
      ctx.fillStyle = typeColor(T, fr.signal_type);
      ctx.fillRect(a, by0, Math.max(0.5, b - a - gap), by1 - by0);
    }
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5;
    ctx.strokeRect(Math.round(cx - sw / 2) + 0.5, wy0 - 2 + 0.5, Math.round(sw), Math.round(by1 - wy0 + 1));
    // marker under the band
    ctx.fillStyle = T.accent;
    ctx.beginPath(); ctx.moveTo(cx, by1 + 2); ctx.lineTo(cx - 5, by1 + 8); ctx.lineTo(cx + 5, by1 + 8); ctx.closePath(); ctx.fill();
    // time axis
    ctx.font = OX.font(10.5, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
    const step = w < 560 ? 1 : 0.5;
    for (let t = 0; t <= NS / FS + 1e-9; t += step) {
      const x = xs(t * FS);
      ctx.textAlign = t === 0 ? 'left' : t >= NS / FS - 1e-9 ? 'right' : 'center';
      ctx.fillText(t === 0 ? '0' : t.toFixed(step < 1 ? 1 : 0) + (t >= NS / FS - 1e-9 ? ' s' : ''), x, h - 20);
    }
  }

  /* ---------------- panels (a) (b) (c) ---------------- */
  const main = OX.view.canvas(mainHost, {
    height: (w) => (w >= 720 ? 452 : w < 480 ? 770 : 810),
    label: OX.t('选中帧的波形、频谱与 LPC 包络、激励脉冲，以及送进 LPC 滤波器的信号', 'The selected frame: its waveform, its spectrum with the LPC envelope, the excitation pulses, and the signal fed into the LPC filter'),
  }, drawMain);

  /** Panel boxes: A waveform, C pulses, R LPC input (these three share the time axis), B spectrum. */
  function layout(w, h) {
    const TT = 22, gapV = 14, axis = 30;
    if (w >= 720) {
      const gap = 34, lw = Math.round((w - gap) * 0.56), rw = w - gap - lw;
      const usable = h - 3 * TT - 2 * gapV - axis;
      const aH = Math.round(usable * 0.42), cH = Math.round(usable * 0.27), rH = usable - aH - cH;
      const cY = TT + aH + gapV + TT, rY = cY + cH + gapV + TT;
      return {
        A: { x: 0, y: TT, w: lw, h: aH }, C: { x: 0, y: cY, w: lw, h: cH }, R: { x: 0, y: rY, w: lw, h: rH },
        B: { x: lw + gap, y: TT, w: rw, h: h - TT - axis },
      };
    }
    const gapB = 26, KEY = 52;
    const usable = h - 4 * TT - 2 * gapV - 2 * axis - gapB - KEY;
    const aH = Math.round(usable * 0.3), cH = Math.round(usable * 0.18), rH = Math.round(usable * 0.2), bH = usable - aH - cH - rH;
    const cY = TT + aH + gapV + TT, rY = cY + cH + gapV + TT, bY = rY + rH + axis + gapB + TT + KEY;
    return { A: { x: 0, y: TT, w, h: aH }, C: { x: 0, y: cY, w, h: cH }, R: { x: 0, y: rY, w, h: rH }, B: { x: 0, y: bY, w, h: bH, keyTop: bY - KEY } };
  }

  /** Nice tick step for a range of about `span`, aiming for `n` ticks. */
  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }

  function drawMain(ctx, w, h) {
    const T = OX.theme;
    const narrow = w < 560;
    const L = layout(w, h);
    const m = model(sel);
    const f = m.f;
    const ML = 40; // left margin for y labels
    const xsT = (P) => OX.view.linear(0, N, P.x + ML, P.x + P.w);

    const title = (P, text) => {
      ctx.font = OX.font(narrow ? 11.5 : 12, 'body', 600); ctx.fillStyle = T.ink;
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(text, P.x, (P.keyTop != null ? P.keyTop : P.y) - 8);
    };
    /** Inline key at the right end of a title row: items [{label, color, kind: 'line'|'dash'|'stem'}]. */
    const key = (P, items) => {
      ctx.font = OX.font(10.5, 'body'); ctx.textBaseline = 'alphabetic';
      let x = P.x + P.w;
      const y = P.y - 8;
      for (let i = items.length - 1; i >= 0; i--) {
        const it = items[i];
        const tw = ctx.measureText(it.label).width;
        ctx.fillStyle = T.ink2; ctx.textAlign = 'right'; ctx.fillText(it.label, x, y);
        x -= tw + 5;
        ctx.strokeStyle = it.color; ctx.fillStyle = it.color; ctx.lineWidth = it.kind === 'line' ? 1.8 : 1.4;
        ctx.setLineDash(it.kind === 'dash' ? [3, 3] : []);
        ctx.beginPath();
        if (it.kind === 'stem') { ctx.moveTo(x - 4, y + 1); ctx.lineTo(x - 4, y - 9); }
        else { ctx.moveTo(x - 16, y - 4); ctx.lineTo(x - 2, y - 4); }
        ctx.stroke();
        ctx.setLineDash([]);
        x -= (it.kind === 'stem' ? 8 : 18) + 12;
      }
    };
    const frameBox = (P) => {
      ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(P.x + ML) + 0.5, Math.round(P.y) + 0.5, Math.round(P.w - ML) - 1, Math.round(P.h));
    };
    const subGrid = (P, xs) => {
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
      for (let s = 1; s < 4; s++) { const x = Math.round(xs(s * SUB)) + 0.5; ctx.beginPath(); ctx.moveTo(x, P.y + 1); ctx.lineTo(x, P.y + P.h); ctx.stroke(); }
    };
    const pitchLines = (P, xs) => {
      if (!m.marks) return;
      ctx.strokeStyle = OX.alpha(T.silk, 0.55); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      for (const t of m.marks) { const x = Math.round(xs(t)) + 0.5; ctx.beginPath(); ctx.moveTo(x, P.y + 1); ctx.lineTo(x, P.y + P.h); ctx.stroke(); }
      ctx.setLineDash([]);
    };
    const yLabel = (y, text) => {
      ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(text, ML - 6 + (y.x || 0), y.y);
    };
    const line = (arr, xs, ys, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let j = 0; j < arr.length; j++) {
        const v = arr[j];
        if (!(v === v)) { pen = false; continue; }
        const X = xs(j), Y = ys(v);
        if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
      }
      ctx.stroke();
    };

    /* (a) waveform */
    {
      const P = L.A, xs = xsT(P);
      title(P, OX.t('波形', 'Waveform'));
      key(P, [{ label: OX.t('原始', 'Original'), color: T.orig, kind: 'line' }, { label: OX.t('解码（16 kb/s）', 'Decoded (16 kb/s)'), color: T.dec, kind: 'line' }]);
      frameBox(P); subGrid(P, xs);
      const top = P.y + (m.marks ? 22 : 8), bot = P.y + P.h - 6;
      const A = m.amp * 1.08;
      const ys = OX.view.linear(-A, A, bot, top);
      const st = niceStep(m.amp, 2);
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
      for (const v of [-st, 0, st]) {
        if (Math.abs(v) > A) continue;
        const y = Math.round(ys(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(P.x + ML, y); ctx.lineTo(P.x + P.w, y); ctx.stroke();
        yLabel({ x: P.x, y }, fmtAmp(v));
      }
      pitchLines(P, xs);
      ctx.save(); ctx.beginPath(); ctx.rect(P.x + ML, P.y, P.w - ML, P.h); ctx.clip();
      line(m.orig, xs, ys, OX.alpha(T.orig, T.dark ? 0.75 : 0.7), 1.25);
      line(m.decd, xs, ys, T.dec, 1.25);
      ctx.restore();
      // pitch-lag arrows between consecutive marks
      if (m.marks) {
        const ay = P.y + 11;
        ctx.strokeStyle = T.silk; ctx.fillStyle = T.silk; ctx.lineWidth = 1.2;
        ctx.font = OX.font(10, 'mono', 500); ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
        for (let i = 1; i < m.marks.length; i++) {
          const a = xs(m.marks[i - 1]), b = xs(m.marks[i]);
          const lag = m.marks[i] - m.marks[i - 1];
          const label = String(lag);
          const tw = ctx.measureText(label).width + 8;
          const c = (a + b) / 2;
          const gapL = b - a > tw + 16 ? tw / 2 : 0;
          ctx.beginPath(); ctx.moveTo(a + 2, ay); ctx.lineTo(c - gapL, ay); ctx.moveTo(c + gapL, ay); ctx.lineTo(b - 2, ay); ctx.stroke();
          arrowHead(ctx, a + 2, ay, -1); arrowHead(ctx, b - 2, ay, 1);
          if (gapL) ctx.fillText(label, c, ay + 0.5);
        }
      }
      if (!m.marks) {
        ctx.font = OX.font(10.5, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        ctx.fillText(f.signal_type === 2 ? '' : OX.t('非浊音帧，没有长时预测', 'Not a voiced frame: no long-term prediction'), P.x + P.w - 6, P.y + 5);
      }
    }

    /* (c) excitation pulses, then the LPC input worked back from the decoded output; both in units of one
       quantization step, so the y ticks are comparable */
    const intTicks = (P, ys, M) => {
      const stepY = M <= 3 ? 1 : M <= 6 ? 2 : niceStep(M, 2);
      for (let v = -Math.floor(M / stepY) * stepY; v <= M + 1e-9; v += stepY) {
        const y = Math.round(ys(v)) + 0.5;
        ctx.strokeStyle = v === 0 ? T.rule : T.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(P.x + ML, y); ctx.lineTo(P.x + P.w, y); ctx.stroke();
        yLabel({ x: P.x, y }, v === 0 ? '0' : (v > 0 ? '+' : '−') + Math.abs(v));
      }
    };
    const centerMsg = (P, text) => {
      ctx.font = OX.font(11, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(text, P.x + ML + (P.w - ML) / 2, P.y + P.h / 2);
    };
    {
      const P = L.C, xs = xsT(P);
      // the long English title needs about 130 px of room on its right for the key; otherwise a shorter one is used
      const cLong = OX.t('激励脉冲（码流里的整数）', 'Excitation pulses (integers in the bitstream)');
      ctx.font = OX.font(narrow ? 11.5 : 12, 'body', 600);
      title(P, ctx.measureText(cLong).width + 130 > P.w ? OX.t('激励脉冲（码流里的整数）', 'Excitation pulses (integers)') : cLong);
      key(P, m.marks ? [{ label: OX.t('脉冲', 'Pulses'), color: T.silk, kind: 'stem' }, { label: OX.t('LTP 延迟', 'LTP lag'), color: OX.alpha(T.silk, 0.8), kind: 'dash' }] : [{ label: OX.t('脉冲', 'Pulses'), color: T.silk, kind: 'stem' }]);
      frameBox(P); subGrid(P, xs);
      const p = m.pulses;
      if (!p) centerMsg(P, OX.t('数据文件里没有这一帧的脉冲', 'The data file has no pulses for this frame'));
      else {
        let M = 1; for (let j = 0; j < p.length; j++) M = Math.max(M, Math.abs(p[j]));
        const ys = OX.view.linear(-M - 0.4, M + 0.4, P.y + P.h - 3, P.y + 15);
        intTicks(P, ys, M);
        pitchLines(P, xs);
        const y0 = ys(0);
        ctx.strokeStyle = T.silk; ctx.fillStyle = T.silk; ctx.lineWidth = 1.3;
        for (let j = 0; j < p.length; j++) {
          if (!p[j]) continue;
          const X = Math.round(xs(j + 0.5)) + 0.5, Y = ys(p[j]);
          ctx.beginPath(); ctx.moveTo(X, y0); ctx.lineTo(X, Y); ctx.stroke();
          ctx.beginPath(); ctx.arc(X, Y, 1.7, 0, Math.PI * 2); ctx.fill();
        }
        // subframe gains at the top of each subframe
        ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        for (let s = 0; s < 4; s++) ctx.fillText((narrow ? 'g ' : OX.t('增益 ', 'gain ')) + fmtGain(f.gains[s]), xs((s + 0.5) * SUB), P.y + 2);
      }
    }
    {
      const P = L.R, xs = xsT(P);
      const voiced = f.signal_type === 2;
      const tText = narrow ? OX.t('LPC 滤波器的输入（由输出反推）', 'LPC filter input (from the output)') : OX.t('送进 LPC 滤波器的信号（由解码输出反推）', 'LPC filter input (worked back from the output)');
      title(P, tText);
      ctx.font = OX.font(narrow ? 11.5 : 12, 'body', 600);
      const tW = ctx.measureText(tText).width;
      let hint = voiced ? (narrow ? OX.t('= 还原的激励 + LTP', '= restored excitation + LTP') : OX.t('= 还原的激励 + 长时预测（LTP）', '= restored excitation + long-term prediction (LTP)')) : (narrow ? OX.t('= 还原的激励', '= restored excitation') : OX.t('没有长时预测：就是还原的激励', 'No long-term prediction: the restored excitation'));
      ctx.font = OX.font(10.5, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
      // the long English hints often do not fit next to the title; then the short (narrow-screen) wording is used
      if (!narrow && tW + ctx.measureText(hint).width + 12 > P.w) hint = voiced ? OX.t('= 还原的激励 + 长时预测（LTP）', '= restored excitation + LTP') : OX.t('没有长时预测：就是还原的激励', '= restored excitation');
      if (tW + ctx.measureText(hint).width + 12 <= P.w) ctx.fillText(hint, P.x + P.w, P.y - 8);
      frameBox(P); subGrid(P, xs);
      const r = m.res;
      if (!r) centerMsg(P, OX.t('缺少解码数据', 'Decoded data is missing'));
      else {
        let M = 1;
        for (let j = 0; j < N; j++) if (r[j] === r[j]) M = Math.max(M, Math.abs(r[j]));
        const ys = OX.view.linear(-M * 1.06, M * 1.06, P.y + P.h - 3, P.y + 3);
        intTicks(P, ys, M);
        pitchLines(P, xs);
        const y0 = ys(0);
        ctx.strokeStyle = OX.alpha(T.dec, 0.85); ctx.lineWidth = 1;
        ctx.beginPath();
        for (let j = 0; j < N; j++) {
          if (!(r[j] === r[j])) continue;
          const X = Math.round(xs(j + 0.5)) + 0.5;
          ctx.moveTo(X, y0); ctx.lineTo(X, ys(r[j]));
        }
        ctx.stroke();
      }
      // time axis, shared by the three panels on the left
      const ax = P.y + P.h;
      OX.view.axis(ctx, xs, { side: 'bottom', at: ax, ticks: [0, 1, 2, 3, 4].map((s) => ({ v: s * SUB, label: s === 4 ? '20 ms' : String(s * 5), align: s === 4 ? 'right' : s === 0 ? 'left' : 'center' })), fontSize: 10 });
      if (!narrow) {
        ctx.font = OX.font(10, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        // "Subframe 4" would run into the "20 ms" tick label, so in English these labels get their own line
        const sfY = OX.lang === 'en' ? ax + 18 : ax + 6;
        for (let s = 0; s < 4; s++) ctx.fillText(OX.t('子帧 ', 'Subframe ') + (s + 1), xs((s + 0.5) * SUB), sfY);
      }
    }

    /* (b) spectrum and envelope */
    {
      const P = L.B;
      const MLB = 44;
      title(P, OX.t('频谱与 LPC 包络', 'Spectrum and LPC envelope'));
      const x0 = P.x + MLB, x1 = P.x + P.w;
      const xs = OX.view.linear(0, FS / 2, x0, x1);
      let mx = -200;
      for (let b = 0; b < m.spec.length; b++) mx = Math.max(mx, m.spec[b], m.env[b]);
      const hi = Math.ceil((mx + 4) / 10) * 10, lo = hi - (P.h < 150 ? 70 : 80);
      const ys = OX.view.linear(lo, hi, P.y + P.h, P.y);
      ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(x0) + 0.5, Math.round(P.y) + 0.5, Math.round(x1 - x0) - 1, Math.round(P.h));
      // grid
      ctx.strokeStyle = T.grid;
      for (let v = lo + 20 - (lo % 20 === 0 ? 0 : (lo % 20 + 20) % 20); v < hi; v += 20) {
        const y = Math.round(ys(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText((v < 0 ? '−' + -v : String(v)) + ' dB', x0 - 5, y);
      }
      const fstep = (x1 - x0) < 300 ? 2000 : 1000;
      for (let fz = fstep; fz < FS / 2; fz += fstep) { const x = Math.round(xs(fz)) + 0.5; ctx.beginPath(); ctx.moveTo(x, P.y + 1); ctx.lineTo(x, P.y + P.h); ctx.stroke(); }
      const binHz = (FS / 2) / (m.spec.length - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(x0, P.y, x1 - x0, P.h); ctx.clip();
      ctx.strokeStyle = OX.alpha(T.ink, T.dark ? 0.45 : 0.38); ctx.lineWidth = 1; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let b = 0; b < m.spec.length; b++) { const X = xs(b * binHz), Y = ys(Math.max(lo - 5, m.spec[b])); if (b) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
      ctx.stroke();
      ctx.strokeStyle = T.silk; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let b = 0; b < m.env.length; b++) { const X = xs(b * binHz), Y = ys(Math.max(lo - 5, m.env[b])); if (b) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
      ctx.stroke();
      ctx.restore();
      // NLSF ticks on the frequency axis
      ctx.strokeStyle = T.silk; ctx.lineWidth = 1.5;
      const yb = P.y + P.h;
      for (const v of f.nlsf) { const x = Math.round(xs(v * FS / 2)) + 0.5; ctx.beginPath(); ctx.moveTo(x, yb - 1); ctx.lineTo(x, yb - 9); ctx.stroke(); }
      OX.view.axis(ctx, xs, {
        side: 'bottom', at: yb, fontSize: 10,
        ticks: Array.from({ length: FS / 2 / fstep + 1 }, (_, i) => i * fstep).map((v) => ({ v, label: v === FS / 2 ? (v / 1000) + ' kHz' : String(v / 1000), align: v === FS / 2 ? 'right' : v === 0 ? 'left' : 'center' })),
      });
      // key and the NLSF explanation: in the (empty) top-right corner, or above the plot on narrow screens
      const above = P.keyTop != null;
      const kx = above ? P.x : x1 - 8;
      ctx.font = OX.font(10.5, 'body'); ctx.textBaseline = 'alphabetic'; ctx.textAlign = above ? 'left' : 'right';
      // The English labels are longer. In the corner, a label that would reach left into the low frequencies (below
      // about 1.5 kHz, where the spectrum peaks) is replaced by a shorter one. The Chinese labels are the same in both.
      const roomW = 0.81 * (x1 - x0) - 30;
      const fit = (long, short) => (above || ctx.measureText(long).width <= roomW ? long : short);
      const rows = [
        { label: fit(OX.t('原始语音的频谱（汉宁窗，1024 点）', 'Original spectrum (Hann window, 1024 points)'), OX.t('原始语音的频谱（汉宁窗，1024 点）', 'Original spectrum (Hann, 1024 points)')), color: OX.alpha(T.ink, T.dark ? 0.55 : 0.45), kind: 'line' },
        { label: fit(OX.t('码流里的 LPC 系数算出的包络', 'Envelope from the LPC coefficients in the bitstream'), OX.t('码流里的 LPC 系数算出的包络', 'LPC envelope from the bitstream')), color: T.silk, kind: 'line', w: 2 },
        { label: fit(OX.t('线谱频率（NLSF）：成对出现在共振峰附近', 'Line spectral frequencies (NLSF): in pairs near formants'), OX.t('线谱频率（NLSF）：成对出现在共振峰附近', 'NLSFs: in pairs near the formants')), color: T.silk, kind: 'tick' },
      ];
      let ky = above ? P.keyTop + 10 : P.y + 16;
      for (const r of rows) {
        // swatch first when left-aligned, last when right-aligned
        const sx = above ? kx : kx - 16;
        ctx.fillStyle = T.ink2; ctx.fillText(r.label, above ? kx + 22 : kx - 22, ky);
        ctx.strokeStyle = r.color; ctx.lineWidth = r.w || 1.4;
        ctx.beginPath();
        if (r.kind === 'tick') { ctx.moveTo(sx + 8, ky + 1); ctx.lineTo(sx + 8, ky - 9); }
        else { ctx.moveTo(sx, ky - 4); ctx.lineTo(sx + 16, ky - 4); }
        ctx.stroke();
        ky += 15;
      }
    }
  }

  function arrowHead(ctx, x, y, dir) {
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - dir * 5, y - 3); ctx.lineTo(x - dir * 5, y + 3); ctx.closePath(); ctx.fill();
  }

  /* ---------------- state ---------------- */
  function updateReadouts() {
    const f = frames[sel];
    const t0 = f.start_sample / FS, t1 = (f.start_sample + N) / FS;
    ro.frame.set(OX.t(`第 ${sel} 帧 · ${Math.max(0, t0).toFixed(3)}–${Math.min(NS / FS, t1).toFixed(3)} s`, `Frame ${sel} · ${Math.max(0, t0).toFixed(3)}–${Math.min(NS / FS, t1).toFixed(3)} s`));
    ro.type.set(TYPE_NAME[f.signal_type] || '?');
    if (f.pitch_lags) {
      const lags = f.pitch_lags;
      const mean = lags.reduce((a, b) => a + b, 0) / lags.length;
      const same = lags.every((v) => v === lags[0]);
      ro.pitch.set(OX.t(`${same ? lags[0] : lags.join(' / ')} 个样本 · ${(mean / FS * 1000).toFixed(1)} ms · ${Math.round(FS / mean)} Hz`, `${same ? lags[0] : lags.join(' / ')} samples · ${(mean / FS * 1000).toFixed(1)} ms · ${Math.round(FS / mean)} Hz`));
    } else ro.pitch.set(OX.t('—（只有浊音帧有）', '— (voiced frames only)'));
    ro.gain.set(f.gains.map(fmtGain).join(' / '));
    ro.bits.set(bitsText(f.bits));
    tl.canvas.setAttribute('aria-valuenow', String(sel));
    tl.canvas.setAttribute('aria-valuetext', OX.t(`第 ${sel} 帧，${TYPE_NAME[f.signal_type]}`, `Frame ${sel}, ${TYPE_NAME[f.signal_type]}`));
    const j = JUMPS.find((x) => detail[x.value] === sel);
    jump.set(j ? j.value : '');
    // move focus off a step button before it becomes disabled, or it would fall back to <body>
    const atStart = sel <= 0, atEnd = sel >= nFrames - 1;
    if (atEnd && document.activeElement === nextBtn) prevBtn.focus();
    if (atStart && document.activeElement === prevBtn) nextBtn.focus();
    prevBtn.disabled = atStart; nextBtn.disabled = atEnd;
  }
  function select(k) {
    k = Math.max(0, Math.min(nFrames - 1, k | 0));
    if (k === sel) return;
    sel = k;
    updateReadouts();
    tl.redraw(); main.redraw();
  }
  updateReadouts();
});
