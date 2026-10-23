/* Figure 10: why CELT switches to short blocks on a transient (pre-echo).
 * A synthetic 48 kHz signal: a very soft low tone, then a castanet-like click 70 % into the middle 20 ms frame.
 * Four frames are coded the same simplified CELT way (band energy kept exact, band shape quantised with K PVQ
 * pulses). The middle frame uses either one long MDCT (960 coefficients) or 8 short MDCTs (120 each, interleaved
 * as in CELT, so the 21 bands, their sizes and their pulses are identical). The figure shows where the
 * quantisation error lands in time: across the whole long window, or only in the short blocks around the click.
 */
(function () {
  'use strict';
  if (typeof OX === 'undefined' || !OX.fig) return;

  /* ------------------------------------------------------------------ */
  /* Model (pure functions)                                               */
  /* ------------------------------------------------------------------ */
  const FS = 48000;
  const N = 960; // long block: 960 MDCT coefficients (20 ms)
  const OV = 120; // CELT overlap: 120 samples (2.5 ms) for every block size
  const B = 8; // short blocks per 20 ms frame
  const NS = N / B; // 120 coefficients per short block
  // CELT band edges in units of 200 Hz (EBAND_5MS, src/modes.rs:90); at 20 ms a band holds (eb[i+1]-eb[i]) << 3 coefficients.
  const EB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];
  const NBANDS = 21;
  const NF = 4; // frames coded: the frame before, the transient frame, the frame after, and one more for the overlap
  const PAD = OV; // samples before the shown excerpt (the rising overlap of the first frame)
  const LEN = NF * N + OV;
  const SHOW = 3 * N; // shown excerpt: 60 ms
  const TF = 1; // index of the transient frame
  const ATT = PAD + TF * N + 672; // the click: 672 / 960 = 70 % into the transient frame (34 ms)
  const RHO = [1, 1 / 2, 1 / 4, 1 / 8]; // pulses per coefficient, fine -> coarse
  const KMIN = 8; // every band gets at least 8 pulses (CELT also gives low bands more bits per coefficient)

  const ms = (samples) => (samples / FS) * 1000;

  /** Frame f covers array samples [f*N, f*N + N + OV): its rising overlap, then its own 960 samples. */
  const frameStart = (f) => f * N;

  // Fast exact MDCT with the same definition and scaling as OX.dsp.mdct / OX.dsp.imdct:
  // X[k] = sum x[n] cos(pi/n (i + 1/2 + n/2)(k + 1/2)); the angle is pi*(2i+1+n)(2k+1)/(4n), so one table of 8n cosines serves.
  const cosTabs = {};
  function cosTab(n) {
    if (!cosTabs[n]) {
      const t = new Float64Array(8 * n);
      for (let m = 0; m < 8 * n; m++) t[m] = Math.cos((Math.PI * m) / (4 * n));
      cosTabs[n] = t;
    }
    return cosTabs[n];
  }
  function mdct(x, n) {
    const t = cosTab(n), M = 8 * n, X = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const step = 2 * (2 * k + 1);
      let idx = ((1 + n) * (2 * k + 1)) % M, acc = 0;
      for (let i = 0; i < 2 * n; i++) { acc += x[i] * t[idx]; idx += step; if (idx >= M) idx -= M; }
      X[k] = acc;
    }
    return X;
  }
  function imdct(X, n) {
    const t = cosTab(n), M = 8 * n, y = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) {
      const a = 2 * i + 1 + n, step = (2 * a) % M;
      let idx = a % M, acc = 0;
      for (let k = 0; k < n; k++) { acc += X[k] * t[idx]; idx += step; if (idx >= M) idx -= M; }
      y[i] = (2 / n) * acc;
    }
    return y;
  }

  /** CELT's low-overlap window over 2n samples: zeros, the 120-sample power-complementary taper, a flat top, the mirrored taper, zeros. */
  const taper = OX.dsp.celtWindow(OV);
  /** Window value at position i of a block's support (length n + OV). */
  const winAt = (i, n) => (i < 0 || i >= n + OV ? 0 : i < OV ? taper[i] : i < n ? 1 : taper[n + OV - 1 - i]);
  const windows = {};
  function win2n(n) {
    if (!windows[n]) {
      const w = new Float64Array(2 * n), z = (n - OV) / 2;
      for (let j = 0; j < 2 * n; j++) w[j] = winAt(j - z, n);
      windows[n] = w;
    }
    return windows[n];
  }
  /** MDCT of one block whose support (n + OV samples) starts at array index `start`. */
  function analyseBlock(x, start, n) {
    const w = win2n(n), z = (n - OV) / 2, seg = new Float64Array(2 * n);
    for (let j = 0; j < 2 * n; j++) { const p = start - z + j; seg[j] = w[j] && p >= 0 && p < x.length ? x[p] * w[j] : 0; }
    return mdct(seg, n);
  }
  function synthBlock(X, start, n, out) {
    const w = win2n(n), z = (n - OV) / 2, y = imdct(X, n);
    for (let j = 0; j < 2 * n; j++) { const p = start - z + j; if (w[j] && p >= 0 && p < out.length) out[p] += y[j] * w[j]; }
  }
  /** 960 coefficients of frame f. Short blocks are interleaved as in CELT: coefficient k of block b sits at k*B + b. */
  function analyseFrame(x, f, short) {
    const s0 = frameStart(f);
    if (!short) return analyseBlock(x, s0, N);
    const C = new Float64Array(N);
    for (let b = 0; b < B; b++) { const Xb = analyseBlock(x, s0 + b * NS, NS); for (let k = 0; k < NS; k++) C[k * B + b] = Xb[k]; }
    return C;
  }
  function synthFrame(C, f, short, out) {
    const s0 = frameStart(f);
    if (!short) { synthBlock(C, s0, N, out); return; }
    const Xb = new Float64Array(NS);
    for (let b = 0; b < B; b++) { for (let k = 0; k < NS; k++) Xb[k] = C[k * B + b]; synthBlock(Xb, s0 + b * NS, NS, out); }
  }

  /** CELT-style PVQ search (src/pvq.rs pvq_search_scalar): projection when K > n/2, then greedy pulses. */
  function pvq(x, K) {
    const n = x.length, ax = new Float64Array(n), y = new Int32Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) { ax[i] = Math.abs(x[i]); sum += ax[i]; }
    let xy = 0, yy = 0, left = K;
    if (K > n / 2 && sum > 0) {
      const r = (K + 0.8) / sum;
      for (let i = 0; i < n; i++) { y[i] = Math.floor(r * ax[i]); left -= y[i]; xy += ax[i] * y[i]; yy += y[i] * y[i]; }
    }
    while (left > 0) {
      let best = 0, bn = -1, bd = 1;
      for (let i = 0; i < n; i++) {
        const num = (xy + ax[i]) * (xy + ax[i]), den = yy + 2 * y[i] + 1;
        if (num * bd > bn * den) { best = i; bn = num; bd = den; }
      }
      xy += ax[best]; yy += 2 * y[best] + 1; y[best]++; left--;
    }
    for (let i = 0; i < n; i++) if (x[i] < 0) y[i] = -y[i];
    return y;
  }
  const bandSize = (i) => (EB[i + 1] - EB[i]) << 3;
  const pulsesFor = (n, rho) => Math.max(KMIN, Math.round(rho * n));

  /** Per band: keep the energy (L2 norm) exact, replace the shape by its K-pulse PVQ codeword. Above 20 kHz: nothing coded. */
  function quantise(C, rho) {
    const Q = new Float64Array(N);
    for (let i = 0; i < NBANDS; i++) {
      const lo = EB[i] << 3, hi = EB[i + 1] << 3, v = C.subarray(lo, hi);
      let E = 0;
      for (let j = 0; j < v.length; j++) E += v[j] * v[j];
      E = Math.sqrt(E);
      if (E < 1e-12) continue;
      const y = pvq(v, pulsesFor(hi - lo, rho));
      let yn = 0;
      for (let j = 0; j < y.length; j++) yn += y[j] * y[j];
      yn = Math.sqrt(yn);
      for (let j = 0; j < y.length; j++) Q[lo + j] = (E * y[j]) / yn;
    }
    return Q;
  }

  /** log2 of the PVQ codebook size V(n, K), by V(n,K) = V(n-1,K) + V(n,K-1) + V(n-1,K-1). */
  const log2VCache = {};
  function log2V(n, K) {
    const key = n + ',' + K;
    if (log2VCache[key] == null) {
      let prev = new Float64Array(K + 1); prev[0] = 1;
      for (let m = 1; m <= n; m++) {
        const cur = new Float64Array(K + 1); cur[0] = 1;
        for (let k = 1; k <= K; k++) cur[k] = prev[k] + cur[k - 1] + prev[k - 1];
        prev = cur;
      }
      log2VCache[key] = Math.log2(prev[K]);
    }
    return log2VCache[key];
  }
  /** Bits for all band shapes of one frame: sum of log2 V(n_i, K_i). The same for long and short blocks. */
  function shapeBits(rho) {
    let s = 0;
    for (let i = 0; i < NBANDS; i++) { const n = bandSize(i); s += log2V(n, pulsesFor(n, rho)); }
    return s;
  }

  /** The test signal: a very soft 300 Hz tone (with a weak 600 Hz partial) and a band-passed noise click. */
  let sigCache = null;
  function signal() {
    if (sigCache) return sigCache;
    const x = new Float64Array(LEN), burst = new Float64Array(LEN);
    let seed = 20261016;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed / 4294967296) * 2 - 1; };
    // RBJ band-pass biquad around 3.2 kHz (Q 0.8): a castanet-like "tick".
    const w0 = (2 * Math.PI * 3200) / FS, al = Math.sin(w0) / (2 * 0.8);
    const b0 = al, b2 = -al, a0 = 1 + al, a1 = -2 * Math.cos(w0), a2 = 1 - al;
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0, pk = 0;
    const tau = 0.003 * FS; // 3 ms exponential decay
    for (let i = 0; i < LEN; i++) {
      const k = i - ATT;
      const env = k < 0 ? 0 : (1 - Math.exp(-k / 4)) * Math.exp(-k / tau);
      const u = rnd() * env;
      const y = (b0 * u + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      x2 = x1; x1 = u; y2 = y1; y1 = y;
      burst[i] = y;
      pk = Math.max(pk, Math.abs(y));
    }
    for (let i = 0; i < LEN; i++) {
      const t = (i - PAD) / FS;
      x[i] = 0.03 * (Math.sin(2 * Math.PI * 300 * t) + 0.35 * Math.sin(2 * Math.PI * 600 * t + 0.6)) + (0.85 * burst[i]) / pk;
    }
    sigCache = x;
    return x;
  }

  /** Code all frames; the transient frame uses short blocks when `short`. rho = null codes without quantisation. */
  function code(rho, short) {
    const x = signal(), out = new Float64Array(LEN);
    for (let f = 0; f < NF; f++) {
      const sh = short && f === TF;
      const C = analyseFrame(x, f, sh);
      synthFrame(rho == null ? C : quantise(C, rho), f, sh, out);
    }
    return out;
  }

  const sumSq = (a, lo, hi) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i] * a[i]; return s; };
  /** The long block's error can reach back to its window start; a short block's only to the first short window holding the click. */
  const REACH_LONG = ATT - frameStart(TF);
  const firstShort = Math.floor((ATT - frameStart(TF)) / NS - 2) + 1;
  const REACH_SHORT = ATT - (frameStart(TF) + firstShort * NS);

  function stats(rec) {
    const x = signal(), e = new Float64Array(LEN);
    for (let i = 0; i < LEN; i++) e[i] = rec[i] - x[i];
    const p0 = ATT - REACH_LONG, p1 = ATT;
    return {
      err: e,
      preDb: 10 * Math.log10(sumSq(e, p0, p1) / sumSq(x, p0, p1)),
      // the same, but stopping where the short blocks' error could start
      farDb: 10 * Math.log10(sumSq(e, p0, ATT - REACH_SHORT) / sumSq(x, p0, ATT - REACH_SHORT)),
      snrDb: 10 * Math.log10(sumSq(x, PAD, PAD + SHOW) / sumSq(e, PAD, PAD + SHOW)),
    };
  }

  /* ------------------------------------------------------------------ */
  /* Figure                                                               */
  /* ------------------------------------------------------------------ */
  OX.fig('transient', function (root, body) {
    const { el, slider, segmented, button } = OX.ui;
    const st = { view: 'cmp', q: 2, playing: null, playKind: null };
    const REPS = 4; // the 60 ms excerpt is played this many times, with a short gap
    const results = {};
    function result(q) {
      if (!results[q]) {
        const rho = RHO[q];
        const L = stats(code(rho, false)), S = stats(code(rho, true));
        const recL = new Float64Array(LEN), recS = new Float64Array(LEN), x = signal();
        for (let i = 0; i < LEN; i++) { recL[i] = x[i] + L.err[i]; recS[i] = x[i] + S.err[i]; }
        results[q] = { L, S, recL, recS, bits: shapeBits(rho) };
      }
      return results[q];
    }
    const x = signal();
    let R = result(st.q);

    OX.ui.css('transient', `
      .tr-below { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 14px 28px; margin-top: 12px; align-items: start; }
      @media (max-width: 760px) { .tr-below { grid-template-columns: minmax(0, 1fr); } }
      .tr-tbl { width: 100%; border-collapse: collapse; font-size: var(--small); }
      .tr-tbl th, .tr-tbl td { padding: 6px 8px 6px 0; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: baseline; }
      .tr-tbl thead th { font: 600 var(--tiny)/1.3 var(--font-body); color: var(--ink-2); }
      .tr-tbl tbody th { font-weight: 400; color: var(--ink-2); line-height: 1.45; }
      .tr-tbl tbody tr.sub th { padding-left: 1.2em; color: var(--ink-3); }
      .tr-tbl td { font: 500 0.92rem/1.3 var(--font-mono); color: var(--ink); font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; padding-right: 2px; padding-left: 10px; }
      .tr-tbl thead th.num { text-align: right; padding-left: 10px; padding-right: 2px; white-space: nowrap; }
      .tr-tbl td.hot { color: var(--bad); }
      .tr-play { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center; }
      .tr-play .ox-ctl-label { font-size: var(--small); color: var(--ink-2); margin-right: 2px; }
      .tr-tile-note { margin-top: 4px; }
      @media (max-width: 560px) { html[data-lang="en"] .tr-tbl thead th.num { white-space: normal; } }
    `);

    // --- controls -------------------------------------------------------
    const viewCtl = segmented({
      id: 'tr-view', label: OX.t('显示', 'Show'), value: st.view,
      options: [{ value: 'cmp', label: OX.t('对比两种误差', 'Compare errors') }, { value: 'L', label: OX.t('长块结果', 'Long block') }, { value: 'S', label: OX.t('短块结果', 'Short blocks') }],
      onChange: (v) => { st.view = v; syncLegend(); cv.redraw(); },
    });
    let legendEl = null;
    // the decoded signal is drawn only in the two single-variant views
    const syncLegend = () => { if (legendEl) legendEl.children[1].style.display = st.view === 'cmp' ? 'none' : ''; };
    const qCtl = slider({
      id: 'tr-q', label: OX.t('量化粗细', 'Quantization coarseness'), min: 0, max: RHO.length - 1, step: 1, value: st.q,
      format: (v) => ['K = n', 'K = n/2', 'K = n/4', 'K = n/8'][v],
      onInput: (v) => { st.q = v; R = result(v); update(); },
    });
    const host = el('div');
    const playBtns = {
      orig: button({ icon: 'sound', label: OX.t('原始', 'Original'), onClick: () => play('orig') }),
      L: button({ icon: 'sound', label: OX.t('长块结果', 'Long block'), onClick: () => play('L') }),
      S: button({ icon: 'sound', label: OX.t('短块结果', 'Short blocks'), onClick: () => play('S') }),
    };
    const playRow = el('div', { class: 'tr-play' }, [el('span', { class: 'ox-ctl-label', text: OX.t(`听这 ${ms(SHOW)} 毫秒（循环 ${REPS} 遍）`, `Listen to these ${ms(SHOW)} ms (looped ${REPS} times)`) }), playBtns.orig, playBtns.L, playBtns.S]);

    // --- table of numbers ------------------------------------------------
    const cells = {};
    const row = (key, label, cls) => {
      cells[key + 'L'] = el('td'); cells[key + 'S'] = el('td');
      return el('tr', { class: cls }, [el('th', { scope: 'row', text: label }), cells[key + 'L'], cells[key + 'S']]);
    };
    const tbl = el('table', { class: 'tr-tbl' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: '' }), el('th', { class: 'num', text: OX.t('长块 1 × 960', 'Long block 1\u00a0×\u00a0960') }), el('th', { class: 'num', text: OX.t('短块 8 × 120', 'Short blocks 8\u00a0×\u00a0120') })])),
      el('tbody', {}, [
        row('reach', OX.t('误差最远扩散到敲击前', 'Earliest error before the hit')),
        row('pre', OX.t(`敲击前 ${fmtMs(REACH_LONG, '毫秒')}内的误差，相对原始信号`, `Error in the ${fmtMs(REACH_LONG)} before the hit, relative to the original signal`)),
        row('far', OX.t(`其中离敲击 ${fmtMs(REACH_SHORT, '毫秒')}以外的部分`, `Of that, more than ${fmtMs(REACH_SHORT)} before the hit`), 'sub'),
        row('snr', OX.t(`整段 ${ms(SHOW)} 毫秒的信噪比`, `Signal-to-noise ratio over all ${ms(SHOW)} ms`)),
        row('bits', OX.t('形状用的比特（这一帧）', 'Bits for the shapes (this frame)')),
      ]),
    ]);
    const tileHost = el('div');
    const tileNote = el('p', { class: 'ox-note tr-tile-note', text: OX.t('两种做法都是 960 个系数（图中每格代表 60 个）。短块分得清“什么时候”，但分不清相近的频率，所以 CELT 只在检测到瞬态的帧里用短块。', 'Both approaches give 960 coefficients (each cell in the sketch stands for 60). Short blocks resolve “when” well but cannot separate nearby frequencies, so CELT uses them only in frames where it detects a transient.') });

    body.append(
      el('div', { class: 'ox-controls' }, [viewCtl, qCtl]),
      host,
      el('div', { class: 'ox-controls' }, [playRow]),
      legendEl = OX.ui.legend([
        { label: OX.t('原始信号', 'Original signal'), color: 'var(--ink)' },
        { label: OX.t('解码结果', 'Decoded'), color: 'var(--accent)' },
        { label: OX.t('量化误差', 'Quantization error'), color: 'var(--bad)', kind: 'box' },
        { label: OX.t('预回声范围', 'Pre-echo range'), color: 'color-mix(in srgb, var(--bad) 22%, transparent)', kind: 'box' },
        { label: OX.t('变换窗（块）', 'Transform windows (blocks)'), color: 'var(--celt)' },
      ]),
      el('div', { class: 'tr-below' }, [
        el('div', {}, [tbl, el('p', { class: 'ox-note', text: OX.t(`示意。两种做法用同样的 21 个频带，每个频带的能量按精确值保留，形状用同样多的 PVQ 脉冲（K = 频带系数个数 n 的一个比例，最少 ${KMIN} 个），所以比特数完全一样，只有变换不同。短块的系数像 CELT 那样交错排列，每个频带的系数个数不变。省略了能量量化等其他步骤。`, `Illustration. Both approaches use the same 21 bands. Each band keeps its exact energy, and its shape gets the same number of pyramid vector quantization (PVQ) pulses (K is a fixed fraction of n, the number of coefficients in the band, and at least ${KMIN}). So the bit counts are identical, and only the transform differs. The short-block coefficients are interleaved as in CELT, so every band keeps the same number of coefficients. Energy quantization and the other steps are left out.`) })]),
        el('div', {}, [tileHost, tileNote]),
      ]));

    /** Samples as milliseconds: '16.5 ms' for readouts and canvas labels, or with unit '毫秒' inside sentences. */
    function fmtMs(samples, unit) { const v = ms(samples); return (Math.abs(v - Math.round(v)) < 1e-9 ? v.toFixed(0) : v.toFixed(1)) + ' ' + (unit || 'ms'); }
    const fmtDb = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + ' dB';

    // --- main canvas ----------------------------------------------------
    const FR_LABELS = [OX.t('上一帧', 'Previous frame'), OX.t('这一帧（有敲击）', 'This frame (with the hit)'), OX.t('下一帧', 'Next frame')];
    let playPos = -1; // sample index (array coordinates) of the playback cursor, or -1
    const cv = OX.view.canvas(host, {
      height: (w) => (w < 560 ? 360 : 400),
      label: OX.t('长块与短块编码后的量化误差：长块的误差铺满整个窗口，敲击之前也有；短块的误差只在敲击附近', "Quantization error after coding with one long block and with short blocks: the long block's error fills the whole window, including the time before the hit; the short blocks' error stays near the hit"),
    }, draw);

    function lanesFor(view, top, bottom, narrow) {
      const gap = narrow ? 12 : 14;
      if (view === 'cmp') {
        const hA = (bottom - top - 2 * gap) * 0.3, hE = (bottom - top - 2 * gap - hA) / 2;
        return [
          { kind: 'orig', y0: top, y1: top + hA },
          { kind: 'err', v: 'L', y0: top + hA + gap, y1: top + hA + gap + hE, strip: true },
          { kind: 'err', v: 'S', y0: top + hA + 2 * gap + hE, y1: bottom, strip: true },
        ];
      }
      const hA = (bottom - top - gap) * 0.55;
      return [
        { kind: 'mix', v: view, y0: top, y1: top + hA, strip: true },
        { kind: 'err', v: view, y0: top + hA + gap, y1: bottom },
      ];
    }

    function draw(ctx, w, h) {
      const T = OX.theme;
      const narrow = w < 560;
      const left = narrow ? 30 : 40, right = 8, top = st.view === 'cmp' ? 22 : 28, bottom = h - 30;
      const xs = OX.view.linear(PAD, PAD + SHOW, left, w - right);
      const lanes = lanesFor(st.view, top, bottom, narrow);
      const xA = xs(ATT);

      // frame boundaries and frame labels
      ctx.font = OX.font(narrow ? 10.5 : 11, 'body', 500); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'center';
      for (let f = 0; f < 3; f++) {
        const a = xs(PAD + f * N), b = xs(PAD + (f + 1) * N);
        ctx.fillStyle = f === TF ? T.ink : T.ink3;
        const label = narrow && f === TF ? OX.t('这一帧', 'This frame') : FR_LABELS[f];
        ctx.fillText(label, (a + b) / 2, top - 8);
        if (f > 0) {
          ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.moveTo(Math.round(a) + 0.5, top - 4); ctx.lineTo(Math.round(a) + 0.5, bottom); ctx.stroke(); ctx.setLineDash([]);
        }
      }

      lanes.forEach((ln) => drawLane(ctx, ln, xs, left, w - right, narrow, T));

      // the click: a vertical marker across all lanes
      ctx.strokeStyle = T.ink2; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(Math.round(xA) + 0.5, top - 2); ctx.lineTo(Math.round(xA) + 0.5, bottom); ctx.stroke(); ctx.setLineDash([]);

      // time axis
      const ticks = [];
      const stepMs = narrow ? 20 : 10;
      for (let t = 0; t <= 60; t += stepMs) ticks.push({ v: PAD + (t / 1000) * FS, label: String(t) });
      OX.view.axis(ctx, xs, { side: 'bottom', at: bottom, ticks });
      ctx.font = OX.font(10.5, 'body', 500); ctx.fillStyle = T.ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText(OX.t('时间 ms', narrow ? 'ms' : 'Time (ms)'), w - right, bottom + 18);
      ctx.textAlign = 'left'; ctx.fillStyle = T.ink;
      ctx.fillText(OX.t(`敲击 ${ms(ATT - PAD).toFixed(0)} ms`, `Hit at ${ms(ATT - PAD).toFixed(0)} ms`), Math.min(xA + 4, w - right - 70), bottom + 18);

      if (playPos >= 0) {
        const X = Math.round(xs(playPos)) + 0.5;
        ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(X, top - 2); ctx.lineTo(X, bottom); ctx.stroke();
      }
    }

    function drawLane(ctx, ln, xs, x0, x1, narrow, T) {
      const stripH = ln.strip ? 14 : 0;
      const titleH = 16;
      const wy0 = ln.y0 + stripH + titleH, wy1 = ln.y1;
      const isErr = ln.kind === 'err';
      const lim = isErr ? 0.5 : 1;
      const ys = OX.view.linear(-lim * 1.08, lim * 1.08, wy1, wy0);

      // pre-echo range (where this variant's error can reach before the click)
      if (isErr || ln.kind === 'mix') {
        const reach = ln.v === 'L' ? REACH_LONG : REACH_SHORT;
        const a = xs(ATT - reach), b = xs(ATT);
        ctx.fillStyle = OX.alpha(T.bad, T.dark ? 0.09 : 0.07);
        ctx.fillRect(a, wy0 - 2, b - a, wy1 - wy0 + 2);
      }

      // grid: zero line and +-lim
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
      for (const v of [-lim, lim]) { const y = Math.round(ys(v)) + 0.5; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); }
      ctx.strokeStyle = T.rule;
      ctx.beginPath(); ctx.moveTo(x0, Math.round(ys(0)) + 0.5); ctx.lineTo(x1, Math.round(ys(0)) + 0.5); ctx.stroke();
      const tl = (v) => (v < 0 ? '−' : '') + String(Math.abs(v));
      OX.view.axis(ctx, ys, { side: 'left', at: x0, ticks: [{ v: lim, label: tl(lim) }, { v: 0, label: '0' }, { v: -lim, label: tl(-lim) }], fontSize: 10 });

      // block windows strip
      if (ln.strip) {
        drawStrip(ctx, ln.v === 'S', xs, x0, x1, ln.y0 + 1, ln.y0 + stripH - 2, T);
        ctx.font = OX.font(10, 'body', 400); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(OX.t('窗', 'Win.'), x0 - 6, ln.y0 + stripH / 2);
      }

      // data
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, wy0 - 2, x1 - x0, wy1 - wy0 + 4); ctx.clip();
      if (ln.kind === 'orig') {
        OX.view.wave(ctx, x, PAD, PAD + SHOW, xs, ys, T.ink, 1.2);
      } else if (ln.kind === 'mix') {
        OX.view.wave(ctx, ln.v === 'L' ? R.recL : R.recS, PAD, PAD + SHOW, xs, ys, T.accent, 1.4);
        OX.view.wave(ctx, x, PAD, PAD + SHOW, xs, ys, OX.alpha(T.ink, 0.85), 1);
      } else {
        fillBand(ctx, (ln.v === 'L' ? R.L : R.S).err, xs, x0, x1, ys, OX.alpha(T.bad, T.dark ? 0.85 : 0.75));
      }
      ctx.restore();

      // pre-echo label, low in the lane where the error rarely reaches
      if (isErr) {
        const reach = ln.v === 'L' ? REACH_LONG : REACH_SHORT;
        const a = xs(ATT - reach), b = xs(ATT);
        ctx.font = OX.font(10.5, 'body', 600); ctx.textBaseline = 'bottom';
        const label = ln.v === 'L' || !narrow ? OX.t(`预回声 ${fmtMs(reach)}`, `Pre-echo ${fmtMs(reach)}`) : fmtMs(reach);
        const tw = ctx.measureText(label).width;
        const lx = tw + 10 < b - a ? a + 5 : b - 4 - tw, ly = ys(-lim) - 2;
        ctx.fillStyle = OX.alpha(T.surface, 0.85);
        ctx.fillRect(lx - 3, ly - 14, tw + 6, 14);
        ctx.fillStyle = T.bad; ctx.textAlign = 'left';
        ctx.fillText(label, lx, ly);
      }

      // title
      const titles = {
        orig: [OX.t('原始信号', 'Original signal'), narrow ? '' : OX.t('前面是很轻的 300 Hz 低音', 'starts with a very soft 300 Hz tone')],
        mixL: [OX.t('长块：原始与解码', 'Long block: original and decoded'), ''],
        mixS: [OX.t('短块：原始与解码', 'Short blocks: original and decoded'), ''],
        errL: [OX.t('长块的误差', 'Long-block error'), narrow ? OX.t('纵轴放大 2 倍', 'magnified 2× vertically') : OX.t('1 个 MDCT，纵轴放大 2 倍', '1 MDCT, magnified 2× vertically')],
        errS: [OX.t('短块的误差', 'Short-block error'), narrow ? OX.t('纵轴放大 2 倍', 'magnified 2× vertically') : OX.t('8 个 MDCT，纵轴放大 2 倍', '8 MDCTs, magnified 2× vertically')],
      };
      const tkey = ln.kind === 'orig' ? 'orig' : ln.kind + ln.v;
      const [t1, t2] = titles[tkey];
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      const ty = ln.y0 + stripH + titleH / 2 - 1;
      ctx.font = OX.font(narrow ? 11 : 11.5, 'body', 600);
      const w1 = ctx.measureText(t1).width;
      ctx.font = OX.font(10.5, 'body', 400);
      const w2 = t2 ? ctx.measureText(t2).width + 8 : 0;
      ctx.fillStyle = OX.alpha(T.surface, 0.85);
      ctx.fillRect(x0 + 2, ty - 8, w1 + w2 + 8, 16);
      ctx.font = OX.font(narrow ? 11 : 11.5, 'body', 600); ctx.fillStyle = T.ink;
      ctx.fillText(t1, x0 + 6, ty);
      if (t2) { ctx.font = OX.font(10.5, 'body', 400); ctx.fillStyle = T.ink3; ctx.fillText(t2, x0 + 6 + w1 + 8, ty); }
    }

    /** The windows of every block in view; the transient frame's blocks in the CELT color. */
    function drawStrip(ctx, short, xs, x0, x1, y0, y1, T) {
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, y0 - 2, x1 - x0, y1 - y0 + 4); ctx.clip();
      const curve = (start, n, color, lw) => {
        ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
        const steps = Math.max(8, Math.min(120, Math.round(xs(start + n + OV) - xs(start))));
        for (let s = 0; s <= steps; s++) {
          const i = (s / steps) * (n + OV);
          const v = winAt(Math.min(n + OV - 1, Math.floor(i)), n);
          const X = xs(start + i), Y = y1 - (y1 - y0) * v;
          if (s === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      };
      for (let f = 0; f < NF; f++) {
        const s0 = frameStart(f);
        if (f === TF && short) for (let b = 0; b < B; b++) curve(s0 + b * NS, NS, T.celt, 1.2);
        else curve(s0, N, f === TF ? T.celt : OX.alpha(T.ink3, 0.7), f === TF ? 1.4 : 1);
      }
      ctx.restore();
    }

    /** Error as a filled band: per pixel column, from its minimum to its maximum (and through zero). */
    function fillBand(ctx, e, xs, x0, x1, ys, color) {
      ctx.fillStyle = color;
      const from = PAD, to = PAD + SHOW;
      const cols = Math.max(1, Math.round(x1 - x0)), spp = (to - from) / cols;
      for (let c = 0; c < cols; c++) {
        const i0 = Math.floor(from + c * spp), i1 = Math.max(i0 + 1, Math.floor(from + (c + 1) * spp));
        let mn = 0, mx = 0;
        for (let i = i0; i < i1 && i < to; i++) { const v = e[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        const ya = ys(mx), yb = ys(mn);
        ctx.fillRect(x0 + c, ya, 1, Math.max(0.6, yb - ya));
      }
    }

    // --- time-frequency tiling sketch -----------------------------------
    const tiles = OX.view.canvas(tileHost, { height: (w) => (w < 380 ? 150 : 140), label: OX.t('时间与频率的划分：长块是 1 列 960 个窄频率格，短块是 8 列、每列 120 个宽频率格', 'How time and frequency are divided: the long block is 1 column of 960 narrow frequency cells; the short blocks are 8 columns of 120 wide frequency cells each') }, drawTiles);
    function drawTiles(ctx, w, h) {
      const T = OX.theme;
      const gap = 22, padL = 14, padR = 2, top = 30, bottom = h - 26;
      const pw = (w - padL - gap - padR) / 2;
      const panels = [
        { x0: padL, cols: 1, rows: 16, t: OX.t('长块：1 × 960', 'Long block: 1 × 960'), s: OX.t('每个系数 20 ms × 25 Hz', 'Each coefficient: 20 ms × 25 Hz'), sShort: OX.t('每个系数 20 ms × 25 Hz', '20 ms × 25 Hz each') },
        { x0: padL + pw + gap, cols: 8, rows: 2, t: OX.t('短块：8 × 120', 'Short blocks: 8 × 120'), s: OX.t('每个系数 2.5 ms × 200 Hz', 'Each coefficient: 2.5 ms × 200 Hz'), sShort: OX.t('每个系数 2.5 ms × 200 Hz', '2.5 ms × 200 Hz each') },
      ];
      const att = (ATT - PAD - TF * N) / N; // 0.7 of the frame
      ctx.font = OX.font(10.5, 'body', 400);
      const subFits = panels.every((p) => ctx.measureText(p.s).width <= pw);
      panels.forEach((p) => {
        const cw = pw / p.cols, rh = (bottom - top) / p.rows;
        const hitCol = Math.floor(att * p.cols);
        for (let c = 0; c < p.cols; c++) for (let r = 0; r < p.rows; r++) {
          ctx.fillStyle = c === hitCol ? OX.alpha(T.bad, T.dark ? 0.26 : 0.16) : OX.alpha(T.celt, T.dark ? 0.12 : 0.08);
          ctx.fillRect(p.x0 + c * cw + 0.5, top + r * rh + 0.5, cw - 1, rh - 1);
        }
        ctx.strokeStyle = OX.alpha(T.celt, 0.55); ctx.lineWidth = 1;
        for (let c = 0; c <= p.cols; c++) { const X = Math.round(p.x0 + c * cw) + 0.5; ctx.beginPath(); ctx.moveTo(X, top); ctx.lineTo(X, bottom); ctx.stroke(); }
        for (let r = 0; r <= p.rows; r++) { const Y = Math.round(top + r * rh) + 0.5; ctx.beginPath(); ctx.moveTo(p.x0, Y); ctx.lineTo(p.x0 + pw, Y); ctx.stroke(); }
        const X = Math.round(p.x0 + att * pw) + 0.5;
        ctx.strokeStyle = T.ink2; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X, top - 3); ctx.lineTo(X, bottom + 3); ctx.stroke(); ctx.setLineDash([]);
        ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
        ctx.font = OX.font(11.5, 'body', 600); ctx.fillStyle = T.ink;
        ctx.fillText(p.t, p.x0, top - 16);
        ctx.font = OX.font(10.5, 'body', 400); ctx.fillStyle = T.ink3;
        ctx.fillText(subFits ? p.s : p.sShort, p.x0, top - 4);
        ctx.font = OX.font(10, 'body', 400); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
        ctx.fillText('0', p.x0, bottom + 4);
        ctx.textAlign = 'right'; ctx.fillText('20 ms', p.x0 + pw, bottom + 4);
        ctx.textAlign = 'center'; ctx.fillStyle = T.ink2; ctx.fillText(OX.t('敲击', 'Hit'), X, bottom + 4);
      });
      // frequency axis hint
      ctx.save(); ctx.translate(6, (top + bottom) / 2); ctx.rotate(-Math.PI / 2);
      ctx.font = OX.font(10, 'body', 400); ctx.fillStyle = T.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(OX.t('频率', 'Frequency'), 0, 0); ctx.restore();
    }

    // --- audio ----------------------------------------------------------
    const GAP = Math.round(0.25 * FS), FADE = Math.round(0.002 * FS);
    const loopLen = SHOW + GAP;
    function buffer(src) {
      const out = new Float32Array(REPS * loopLen);
      for (let r = 0; r < REPS; r++) {
        for (let i = 0; i < SHOW; i++) {
          const g = Math.min(1, i / FADE, (SHOW - 1 - i) / FADE);
          out[r * loopLen + i] = 0.6 * g * src[PAD + i];
        }
      }
      return out;
    }
    const playLoop = OX.loop(host, () => {
      const hnd = st.playing;
      if (!hnd || !hnd.playing) { playPos = -1; cv.redraw(); return false; }
      const p = hnd.position() % loopLen;
      playPos = p < SHOW ? PAD + p : -1;
      cv.redraw();
      return true;
    });
    function setPlayUi(kind) {
      Object.keys(playBtns).forEach((k) => {
        const on = k === kind;
        playBtns[k].setAttribute('aria-pressed', on ? 'true' : 'false');
        playBtns[k].setIcon(on ? 'stop' : 'sound');
      });
    }
    function play(kind) {
      if (st.playing && st.playKind === kind) { OX.audio.stop(); return; }
      const src = kind === 'orig' ? x : kind === 'L' ? R.recL : R.recS;
      // OX.audio.play() stops the previous sound first, and that runs the previous onEnd synchronously.
      // So the new state is written only after play() has returned; writing it before would be cleared again.
      const hnd = OX.audio.play(buffer(src), FS, {
        gain: 0.5,
        onEnd: () => { if (st.playing === hnd) { st.playing = null; st.playKind = null; setPlayUi(null); playPos = -1; cv.redraw(); } },
      });
      if (!hnd) return;
      st.playing = hnd; st.playKind = kind;
      setPlayUi(kind);
      if (!OX.reducedMotion()) playLoop.start();
    }

    function update() {
      cells.reachL.textContent = fmtMs(REACH_LONG);
      cells.reachS.textContent = fmtMs(REACH_SHORT);
      cells.preL.textContent = fmtDb(R.L.preDb);
      cells.preS.textContent = fmtDb(R.S.preDb);
      cells.preL.classList.toggle('hot', R.L.preDb > 0);
      cells.preS.classList.toggle('hot', R.S.preDb > 0);
      cells.farL.textContent = fmtDb(R.L.farDb);
      cells.farS.textContent = fmtDb(R.S.farDb);
      cells.farL.classList.toggle('hot', R.L.farDb > 0);
      cells.farS.classList.toggle('hot', R.S.farDb > 0);
      cells.snrL.textContent = R.L.snrDb.toFixed(1) + ' dB';
      cells.snrS.textContent = R.S.snrDb.toFixed(1) + ' dB';
      cells.bitsL.textContent = Math.round(R.bits) + OX.t(' 比特', ' bits');
      cells.bitsS.textContent = Math.round(R.bits) + OX.t(' 比特', ' bits');
      cv.redraw();
    }
    syncLegend();
    update();
  });
})();
