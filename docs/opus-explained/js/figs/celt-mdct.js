/* Figure 7: one frame of music through CELT's analysis, in five steps.
 * 1-2: the 1080 PCM samples one 20 ms MDCT reads, and the low-overlap window (2.5 ms tapers, flat top).
 * 3:   the 960 MDCT coefficients decoded from the real 64 kb/s opus1 stream (music.celt_detail[].mdct); for a
 *      transient frame they are 8 interleaved short MDCTs (coefficient i: block i % 8, bin i / 8), shown de-interleaved.
 * 4:   band energies sqrt(sum x^2) over the 21 bands, against the bitstream's band_log2_amp.
 * 5:   the shape: each coefficient divided by its band's bitstream energy sqrt(sum x^2), so every band has unit norm.
 */
OX.fig('celt-mdct', function (root, body) {
  const { el, button, segmented, readout } = OX.ui;
  const D = OX.data.raw;
  const music = D && D.music, celt = D && D.meta && D.meta.celt;
  if (!music || !music.celt_detail || !music.celt_frames || !music.pcm || !celt) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js，这张图无法显示。', 'The data file data/opus-demo-data.js is missing, so this figure cannot be shown.') }));
    return;
  }

  const FS = music.sample_rate; // 48000
  const N = music.frame_samples; // 960 coefficients per 20 ms frame
  const OV = celt.overlap; // 120 samples = 2.5 ms
  const BLOCK = N + OV; // 1080 samples read by one frame's MDCT
  const SHORT = celt.short_mdct_size; // 120 coefficients per short MDCT
  const NSHORT = N / SHORT; // 8 short blocks in a transient frame
  const E = celt.band_edges_bins_20ms, EHZ = celt.band_edges_hz;
  const NB = E.length - 1; // 21 bands
  const CODED = E[NB]; // 800 coefficients below 20 kHz
  const NYQ = FS / 2;
  const HZ = NYQ / N; // 25 Hz per coefficient
  const SHORT_HZ = NYQ / SHORT; // 200 Hz per short-block bin
  const DB_PER_LOG2 = 20 * Math.log10(2);
  const ms = (n) => +((n * 1000) / FS).toFixed(3); // samples -> milliseconds, printed without float noise
  const pcm = OX.data.audio('music.pcm', music.pcm);
  const taper = OX.dsp.celtWindow(OV);
  const ROLES = [
    { role: 'tonal', label: OX.t('音高稳定的帧', 'Tonal frame') },
    { role: 'transient', label: OX.t('瞬态帧', 'Transient frame') },
    { role: 'hihat', label: OX.t('镲片帧', 'Cymbal frame') },
  ].filter((r) => music.celt_detail.some((d) => d.role === r.role));
  if (!ROLES.length) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('数据文件里没有 celt_detail，这张图无法显示。', 'The data file has no celt_detail, so this figure cannot be shown.') }));
    return;
  }

  /** Everything one frame needs, computed once. */
  function model(role) {
    return OX.data.memo('celt-mdct.' + role, () => {
      const d = music.celt_detail.find((x) => x.role === role);
      const fr = music.celt_frames.frames[d.index];
      const X = Float32Array.from(d.mdct);
      const ampC = new Float64Array(NB), ampB = new Float64Array(NB), sumSq = new Float64Array(NB);
      const shape = new Float32Array(N);
      let maxDiffDb = 0, sMin = Infinity, sMax = -Infinity;
      for (let b = 0; b < NB; b++) {
        let s = 0;
        for (let i = E[b]; i < E[b + 1]; i++) s += X[i] * X[i];
        ampC[b] = Math.sqrt(s);
        ampB[b] = Math.pow(2, fr.band_log2_amp[b]);
        let q = 0;
        for (let i = E[b]; i < E[b + 1]; i++) { shape[i] = X[i] / ampB[b]; q += shape[i] * shape[i]; }
        sumSq[b] = q; sMin = Math.min(sMin, q); sMax = Math.max(sMax, q);
        maxDiffDb = Math.max(maxDiffDb, Math.abs(20 * Math.log10(ampC[b] / ampB[b])));
      }
      // short blocks, de-interleaved: block b, bin k  <-  X[k * NSHORT + b]
      const blocks = d.transient ? Array.from({ length: NSHORT }, (_, b) => Float32Array.from({ length: SHORT }, (_, k) => X[k * NSHORT + b])) : null;
      // the short block with the most energy above 2 kHz: where the click is
      let loud = -1;
      if (blocks) {
        let best = -1;
        blocks.forEach((blk, b) => { let s = 0; for (let k = 2000 / SHORT_HZ; k < SHORT; k++) s += blk[k] * blk[k]; if (s > best) { best = s; loud = b; } });
      }
      // time signal: the 1080-sample block plus 2.5 ms of context on each side
      const T0 = -OV, T1 = BLOCK + OV;
      const seg = new Float32Array(T1 - T0);
      let peak = 1e-6;
      for (let j = T0; j < T1; j++) { const n = d.start_sample + j; const v = n >= 0 && n < pcm.length ? pcm[n] : 0; seg[j - T0] = v; peak = Math.max(peak, Math.abs(v)); }
      const win = new Float32Array(BLOCK);
      for (let j = 0; j < BLOCK; j++) win[j] = j < OV ? taper[j] : j >= N ? taper[BLOCK - 1 - j] : 1;
      return { d, fr, X, ampC, ampB, sumSq, sMin, sMax, shape, maxDiffDb, blocks, loud, T0, T1, seg, peak, win, short: !!d.transient };
    });
  }

  // shared vertical scales over the frames on offer, so switching frames compares like with like
  let xMaxDb = -Infinity, aMin = Infinity, aMax = -Infinity;
  for (const r of ROLES) {
    const m = model(r.role);
    for (let i = 0; i < N; i++) if (m.X[i] !== 0) xMaxDb = Math.max(xMaxDb, 20 * Math.log10(Math.abs(m.X[i])));
    for (let b = 0; b < NB; b++) { const v = 20 * Math.log10(m.ampC[b]); aMin = Math.min(aMin, v); aMax = Math.max(aMax, v); }
  }
  const XHI = Math.ceil(xMaxDb / 10) * 10, XLO = XHI - 80;
  const HEAT_RANGE = 60; // dB shown by the short-block colour scale, below XHI
  const AHI = Math.ceil(aMax / 10) * 10, ALO = Math.floor(aMin / 10) * 10 - 10;

  const NSTEPS = 5;
  const st = { role: ROLES[0].role, axis: 'band', step: NSTEPS - 1, band: 0 }; // band: null = none, -1 = the uncoded 20-24 kHz
  let anim = 1;

  function stepText(m) {
    return [
      OX.t(`一帧信号：CELT 每 20 毫秒处理一帧。这一帧的 MDCT 读入 ${BLOCK} 个样本（${ms(BLOCK)} 毫秒）：开头 ${ms(OV)} 毫秒与上一帧重叠，结尾 ${ms(OV)} 毫秒与下一帧重叠。两侧浅色的部分属于相邻的帧。`,
        `One frame: CELT processes the signal in 20 ms frames. The MDCT of this frame reads ${BLOCK} samples (${ms(BLOCK)} ms): the first ${ms(OV)} ms overlap the previous frame, and the last ${ms(OV)} ms overlap the next frame. The faint parts on either side belong to the neighboring frames.`),
      m.short
        ? OX.t(`加窗：瞬态帧不用一个长变换（长块），改用 ${NSHORT} 个短变换（短块），每个短块有自己的短窗。每个短窗长 ${ms(2 * SHORT)} 毫秒（${2 * SHORT} 个样本），从 0 升到 1 再降回 0，相邻两个重叠 ${ms(SHORT)} 毫秒。最外两端的斜坡和长块的窗一样，所以和相邻帧照样能拼接。`,
          `Windowing: instead of one long transform (a long block), a transient frame uses ${NSHORT} short transforms (short blocks), and each short block has its own short window. Each short window is ${ms(2 * SHORT)} ms long (${2 * SHORT} samples). It rises from 0 to 1 and falls back to 0, and neighboring windows overlap by ${ms(SHORT)} ms. The two outermost slopes are the same as in the long-block window, so the frame still joins up with its neighbors.`)
        : OX.t(`加窗：这一帧用一个长变换（长块），乘上 CELT 的低重叠窗。窗在开头用 ${ms(OV)} 毫秒（${OV} 个样本）从 0 升到 1，在结尾用 ${ms(OV)} 毫秒降回 0，中间 ${ms(N - OV)} 毫秒是平的。相邻两帧的窗在重叠区平方相加等于 1，逆变换后才能无缝拼接。`,
          `Windowing: this frame uses one long transform (a long block), multiplied by CELT's low-overlap window. The window rises from 0 to 1 over the first ${ms(OV)} ms (${OV} samples), falls back to 0 over the last ${ms(OV)} ms, and is flat for the ${ms(N - OV)} ms in between. In the overlap, the squares of two neighboring windows add up to 1. This is what lets the frames join without a seam after the inverse transform.`),
      m.short
        ? OX.t(`MDCT：${NSHORT} 个短块各做一次 MDCT，每个得到 ${SHORT} 个系数，间隔 ${SHORT_HZ} Hz。码流里 ${NSHORT} 个短块的系数交错排列：第 1 个系数属于块 1，第 2 个属于块 2，……，第 ${NSHORT + 1} 个又回到块 1。这里把它们拆开画：每一行是一个短块，从上到下每行晚 ${ms(SHORT)} 毫秒，颜色越浓系数越大。2 kHz 以上颜色最浓的是块 ${m.loud + 1}（${ms(m.loud * SHORT)}–${ms(m.loud * SHORT + 2 * SHORT)} 毫秒），敲击就落在这里。`,
          `MDCT: each of the ${NSHORT} short blocks gets its own MDCT, which gives ${SHORT} coefficients spaced ${SHORT_HZ} Hz apart. In the bitstream, the coefficients of the ${NSHORT} short blocks are interleaved: coefficient 1 belongs to block 1, coefficient 2 to block 2, and so on, until coefficient ${NSHORT + 1} is back in block 1. Here they are drawn apart: each row is one short block, each row starts ${ms(SHORT)} ms later than the one above it, and a stronger color means a larger coefficient. Above 2 kHz, the strongest color is in block ${m.loud + 1} (${ms(m.loud * SHORT)}–${ms(m.loud * SHORT + 2 * SHORT)} ms): this is where the hit is.`)
        : OX.t(`MDCT：加窗后的 ${BLOCK} 个样本变成 ${N} 个系数，第 k 个系数代表 ${HZ}·k 到 ${HZ}·(k+1) Hz。20 kHz 以上的 ${N - CODED} 个系数不编码。这里画的是从码流解出来的系数，也就是解码端在逆 MDCT 之前得到的值。`,
          `MDCT: the ${BLOCK} windowed samples become ${N} coefficients, and coefficient k covers ${HZ}·k to ${HZ}·(k+1) Hz. The ${N - CODED} coefficients above 20 kHz are not coded. The plot shows the coefficients decoded from the bitstream, that is, the values the decoder has just before the inverse MDCT.`),
      OX.t(`分带与能量：把系数按 ${NB} 个频带分组，每组算一个能量：带内系数平方和的平方根 √(Σx²)。它比带内任何一个系数都大，频带越宽高出越多。蓝点是码流里解码出的能量，和从系数算出的值重合。`,
        `Bands and energy: the coefficients are grouped into ${NB} bands, and each band gets one energy value, the square root of the sum of its squared coefficients, √(Σx²). This value is larger than any single coefficient in the band, and the wider the band, the larger the gap. The blue dots are the energies decoded from the bitstream; they coincide with the values computed from the coefficients.`),
      OX.t(`归一化形状：每个系数除以码流里这个频带的能量（上一步的蓝点，和算出的 √(Σx²) 相同）。每个频带剩下一个长度为 1 的向量（平方和 = 1），值都在 −1 到 1 之间，表示能量在带内怎么分布。后面的 PVQ 编码的就是它。${m.short ? `瞬态帧里，带内是 ${NSHORT} 个短块的系数交错排在一起。` : ''}`,
        `Normalized shape: each coefficient is divided by its band's energy from the bitstream (the blue dot of the previous step, equal to the computed √(Σx²)). Each band is left with a vector of length 1 (sum of squares = 1), with values between −1 and 1, that shows how the energy is spread within the band. This vector is what PVQ codes next.${m.short ? ` In a transient frame, each band holds the coefficients of the ${NSHORT} short blocks, interleaved.` : ''}`),
    ];
  }

  // ---- DOM ----
  OX.ui.css('celt-mdct', `
    .cm-step { margin: 2px 0 10px; min-height: 5.1em; font-size: var(--small); color: var(--ink); line-height: 1.7; }
    .cm-step b { font-family: var(--font-mono); color: var(--accent); margin-right: 6px; }
    .cm-dots { display: inline-flex; gap: 6px; align-items: center; }
    .cm-dot { width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid var(--ink-3); background: transparent; padding: 0; cursor: pointer; }
    .cm-dot.on { background: var(--accent); border-color: var(--accent); }
    .cm-canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    @media (max-width: 560px) { .cm-step { min-height: 8.6em; } }
  `);
  const caption = el('p', { class: 'cm-step', 'aria-live': 'polite' });
  const host = el('div');
  const prev = button({ label: OX.t('上一步', 'Previous'), onClick: () => go(st.step - 1) });
  const next = button({ icon: 'step', label: OX.t('下一步', 'Next'), kind: 'primary', onClick: () => go(st.step + 1) });
  const dots = el('div', { class: 'cm-dots' }, Array.from({ length: NSTEPS }, (_, i) => i).map((i) => el('button', { type: 'button', class: 'cm-dot', 'aria-label': OX.t(`第 ${i + 1} 步`, `Step ${i + 1}`), onclick: () => go(i) })));
  const roFrame = readout(OX.t('这一帧', 'This frame'), '');
  const roMatch = readout(OX.t('能量：系数算出 vs 码流', 'Energy: computed vs. bitstream'), '');
  const roNorm = readout(OX.t('21 个频带形状的平方和', 'Sum of squares of the 21 band shapes'), '');
  const roBand = readout(OX.t('频带', 'Band'), '');
  const roBandE = readout(OX.t('频带能量：log₂ 值 → dB', 'Band energy: log₂ value → dB'), '');
  const roBandS = readout(OX.t('形状平方和 · PVQ 比特', 'Shape sum of squares · PVQ bits'), '');
  const note = el('p', { class: 'ox-note' });
  body.append(
    el('div', { class: 'ox-controls' }, [
      segmented({ id: 'cm-frame', label: OX.t('帧', 'Frame'), value: st.role, options: ROLES.map((r) => ({ value: r.role, label: r.label })), onChange: (v) => { st.role = v; update(); } }),
      segmented({ id: 'cm-axis', label: OX.t('频率轴（第 3–5 步）', 'Frequency axis (steps 3–5)'), value: st.axis, options: [{ value: 'band', label: OX.t('每个频带等宽', 'Equal width per band') }, { value: 'lin', label: OX.t('线性频率', 'Linear frequency') }], onChange: (v) => { st.axis = v; update(); } }),
    ]),
    caption, host,
    el('div', { class: 'ox-controls' }, [prev, next, dots]),
    el('div', { class: 'ox-readouts', 'aria-live': 'polite' }, [roFrame, roMatch, roNorm, roBand, roBandE, roBandS]),
    OX.ui.legend([
      { label: OX.t('原始信号', 'Original signal'), color: 'var(--ink)' },
      { label: OX.t('窗', 'Window'), color: 'var(--celt)', dash: true },
      { label: OX.t('加窗后的信号、MDCT 系数、算出的能量、形状', 'Windowed signal, MDCT coefficients, computed energy, shape'), color: 'var(--celt)' },
      { label: OX.t('码流里解码出的频带能量（蓝点）', 'Band energy decoded from the bitstream (blue dots)'), color: 'var(--accent)', kind: 'box' },
      { label: OX.t('20 kHz 以上：不编码', 'Above 20 kHz: not coded'), color: 'var(--rule)', kind: 'box' },
    ]),
    note,
  );

  // ---- geometry ----
  function layout(w) {
    const narrow = w < 560;
    const L = narrow ? 34 : 46, R = narrow ? 6 : 10;
    let y = 2;
    const A = { t: y, top: y + 18, bot: y + 18 + (narrow ? 80 : 100) };
    y = A.bot + 26;
    const num = y + 6;
    y += 14;
    const B = { t: y, top: y + 18, bot: y + 18 + (narrow ? 72 : 88) };
    y = B.bot + 8;
    const C = { t: y, top: y + 18, bot: y + 18 + (narrow ? 50 : 60) };
    y = C.bot + 8;
    const S = { t: y, top: y + 18, bot: y + 18 + (narrow ? 66 : 82) };
    y = S.bot + 22;
    return { narrow, L, R, A, num, B, C, S, H: y };
  }
  /** Frequency (Hz) -> x. 'band': each of the 21 bands, and the uncoded 20-24 kHz, gets one equal slot. */
  function freqScale(g, w) {
    const x0 = g.L, x1 = w - g.R;
    if (st.axis === 'lin') return OX.view.linear(0, NYQ, x0, x1);
    const slots = NB + 1, sw = (x1 - x0) / slots;
    const s = (hz) => {
      if (hz >= EHZ[NB]) return x0 + (NB + Math.min(1, (hz - EHZ[NB]) / (NYQ - EHZ[NB]))) * sw;
      let b = 0;
      while (b < NB - 1 && hz >= EHZ[b + 1]) b++;
      return x0 + (b + Math.max(0, hz - EHZ[b]) / (EHZ[b + 1] - EHZ[b])) * sw;
    };
    s.invert = (px) => {
      const u = (px - x0) / sw;
      if (u >= NB) return EHZ[NB] + (u - NB) * (NYQ - EHZ[NB]);
      const b = Math.max(0, Math.floor(u));
      return EHZ[b] + (u - b) * (EHZ[b + 1] - EHZ[b]);
    };
    return s;
  }

  const cv = OX.view.canvas(host, { height: (w) => layout(w).H, label: OX.t('一帧音乐的时间信号、窗、MDCT 系数、频带能量和归一化形状', 'Time signal, window, MDCT coefficients, band energies, and normalized shape of one frame of music') }, draw);
  cv.canvas.classList.add('cm-canvas');
  cv.canvas.tabIndex = 0;
  // The canvas picks a band with the arrow keys, so expose it as a slider (as silk-frame does), not as an image.
  cv.canvas.setAttribute('role', 'slider');
  cv.canvas.setAttribute('aria-label', OX.t('选择频带：一帧音乐的 MDCT 系数、频带能量和归一化形状', 'Choose a band: MDCT coefficients, band energies, and normalized shape of one frame of music'));
  cv.canvas.setAttribute('aria-valuemin', '0');
  cv.canvas.setAttribute('aria-valuemax', String(NB - 1));

  function draw(ctx, w) {
    const T = OX.theme, g = layout(w), m = model(st.role), fx = freqScale(g, w);
    const vis = (i) => (i < st.step ? 1 : i === st.step ? anim : 0.12);
    const fs = g.narrow ? 9.5 : 10.5;

    const laneTitle = (lane, num, text, a) => {
      ctx.globalAlpha = Math.max(0.45, a);
      ctx.font = OX.font(fs, 'mono', 600); ctx.fillStyle = T.accent; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(num, g.L, lane.top - 5);
      const nw = ctx.measureText(num).width;
      ctx.font = OX.font(g.narrow ? 10.5 : 11.5, 'body', 600); ctx.fillStyle = a > 0.5 ? T.ink : T.ink3;
      ctx.fillText(text, g.L + nw + 6, lane.top - 5);
      ctx.globalAlpha = 1;
    };
    const hline = (y, x0, x1, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };

    // ---------------- lane A: time signal and window ----------------
    {
      const A = g.A;
      const xs = OX.view.linear(m.T0, m.T1, g.L, w - g.R);
      const ymax = niceCeil(m.peak * 1.08);
      const ys = OX.view.linear(-ymax, ymax, A.bot, A.top);
      laneTitle(A, '1–2', m.short ? OX.t('时间信号，与 8 个短窗', 'Time signal and the 8 short windows') : OX.t('时间信号与窗', 'Time signal and window'), Math.max(vis(0), vis(1)));
      // overlap zones
      ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.1 : 0.06);
      ctx.fillRect(xs(0), A.top, xs(OV) - xs(0), A.bot - A.top);
      ctx.fillRect(xs(N), A.top, xs(BLOCK) - xs(N), A.bot - A.top);
      // overlap labels: drawn last, on a backing, so the short windows do not run through them
      const zoneLabels = g.narrow ? [] : [[OX.t('← 与上一帧重叠', '← Overlap with the previous frame'), xs(OV) + 4, 'left'], [OX.t('与下一帧重叠 →', 'Overlap with the next frame →'), xs(N) - 4, 'right']];
      hline(ys(0), g.L, w - g.R, T.grid);
      OX.view.axis(ctx, ys, { side: 'left', at: g.L, ticks: [{ v: -ymax, label: '−' + fmtAmp(ymax) }, { v: 0, label: '0' }, { v: ymax, label: fmtAmp(ymax) }], fontSize: fs });
      const ticks = (g.narrow ? [0, 10, 20] : [0, 2.5, 5, 10, 15, 20, 22.5]).map((t) => ({ v: (t * FS) / 1000, label: String(t) }));
      OX.view.axis(ctx, xs, { side: 'bottom', at: A.bot, ticks, fontSize: fs });
      ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText('ms', w - g.R, A.bot + 6);
      ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.L, Math.round(A.bot) + 0.5); ctx.lineTo(w - g.R, Math.round(A.bot) + 0.5); ctx.stroke();

      ctx.save();
      ctx.beginPath(); ctx.rect(g.L, A.top - 2, w - g.L - g.R, A.bot - A.top + 4); ctx.clip();
      const a1 = vis(1);
      const sig = (from, to, color, lw, alpha) => {
        ctx.globalAlpha = alpha;
        OX.view.wave(ctx, m.seg, from - m.T0, to - m.T0, (i) => xs(i + m.T0), ys, color, lw);
        ctx.globalAlpha = 1;
      };
      // context outside the block, faint; the block itself fades back once the window is applied
      sig(m.T0, 1, T.ink, 1, 0.28 * vis(0));
      sig(BLOCK - 1, m.T1, T.ink, 1, 0.28 * vis(0));
      sig(0, BLOCK, T.ink, 1.1, vis(0) * (1 - 0.65 * (st.step >= 1 ? a1 : 0)));
      // window(s)
      ctx.globalAlpha = a1;
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
      if (!m.short) {
        for (const sgn of [1, -1]) {
          ctx.strokeStyle = T.celt; ctx.beginPath();
          for (let j = 0; j < BLOCK; j += 2) { const X = xs(j), Y = ys(sgn * ymax * m.win[j]); if (j === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
          ctx.lineTo(xs(BLOCK - 1), ys(0)); ctx.stroke();
        }
      } else {
        for (let b = 0; b < NSHORT; b++) {
          ctx.strokeStyle = b % 2 ? T.ink3 : T.celt; ctx.beginPath();
          for (let j = 0; j < 2 * OV; j += 2) {
            const wv = j < OV ? taper[j] : taper[2 * OV - 1 - j];
            const X = xs(b * SHORT + j), Y = ys(ymax * wv);
            if (j === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.lineTo(xs(b * SHORT + 2 * OV - 1), ys(0)); ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      // windowed signal (long block only: in a short-block frame every instant sits in two overlapping windows)
      if (!m.short) {
        ctx.strokeStyle = T.celt; ctx.lineWidth = 1.2; ctx.beginPath();
        const step = Math.max(1, Math.floor(BLOCK / (w - g.L - g.R) / 2));
        for (let j = 0; j < BLOCK; j += step) { const X = xs(j), Y = ys(m.seg[j - m.T0] * m.win[j]); if (j === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.restore();
      ctx.font = OX.font(10, 'body', 500); ctx.textBaseline = 'top';
      for (const [text, x, align] of zoneLabels) {
        const tw = ctx.measureText(text).width, x0 = align === 'left' ? x : x - tw;
        ctx.fillStyle = OX.alpha(T.surface, 0.85); ctx.fillRect(x0 - 3, A.top + 1, tw + 6, 14);
        ctx.fillStyle = T.ink3; ctx.textAlign = align; ctx.fillText(text, x, A.top + 2);
      }
    }

    // ---------------- frequency lanes: shared decorations ----------------
    const xu0 = fx(EHZ[NB]), xu1 = fx(NYQ);
    for (const lane of [g.B, g.C, g.S]) {
      ctx.fillStyle = OX.alpha(T.ink3, T.dark ? 0.2 : 0.13);
      ctx.fillRect(xu0, lane.top, xu1 - xu0, lane.bot - lane.top);
    }
    if (st.band != null && st.band >= 0) {
      const b0 = fx(EHZ[st.band]), b1 = fx(EHZ[st.band + 1]);
      for (const lane of [g.B, g.C, g.S]) {
        ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.16 : 0.08);
        ctx.fillRect(b0, lane.top, b1 - b0, lane.bot - lane.top);
      }
    }
    // band edges: in lane B only once bands have been introduced (step 4); always in C and S
    const edgeLines = (lane, a, all) => {
      ctx.globalAlpha = a; ctx.strokeStyle = T.dark ? OX.alpha(T.ink3, 0.55) : OX.alpha(T.ink3, 0.4); ctx.lineWidth = 1;
      for (let b = 0; b <= NB; b++) {
        const x = Math.round(fx(EHZ[b])) + 0.5;
        if (!all && b > 0 && fx(EHZ[b]) - fx(EHZ[b - 1]) < 2.5) continue;
        ctx.beginPath(); ctx.moveTo(x, lane.top); ctx.lineTo(x, lane.bot); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };
    // band numbers
    {
      const a = Math.max(0.3, vis(3));
      ctx.globalAlpha = a;
      ctx.font = OX.font(g.narrow ? 9 : 10, 'mono', 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      let lastRight = -Infinity;
      for (let b = 0; b < NB; b++) {
        const c = (fx(EHZ[b]) + fx(EHZ[b + 1])) / 2, tw = ctx.measureText(String(b)).width;
        const sel = b === st.band;
        if (!sel && (c - tw / 2 < lastRight + 3)) continue;
        ctx.fillStyle = sel ? T.accent : T.ink3;
        ctx.fillText(String(b), c, g.num);
        lastRight = c + tw / 2;
      }
      ctx.font = OX.font(g.narrow ? 9 : 10, 'body', 500); ctx.fillStyle = T.ink3; ctx.textAlign = 'right';
      ctx.fillText(OX.t('频带', 'Band'), g.L - 6, g.num);
      if (xu1 - xu0 > 30) { ctx.textAlign = 'center'; ctx.fillText(OX.t('不编码', 'Not coded'), (xu0 + xu1) / 2, g.num); }
      ctx.globalAlpha = 1;
    }
    const coefX = (i) => [fx(i * HZ), fx((i + 1) * HZ)];

    // ---------------- lane B: MDCT coefficients ----------------
    {
      const B = g.B, a = vis(2);
      laneTitle(B, '3', m.short ? OX.t(`MDCT 系数 |X|：${NSHORT} 个短块，每行一个`, `MDCT coefficients |X|: ${NSHORT} short blocks, one per row`) : OX.t('MDCT 系数 |X|（dB）', 'MDCT coefficients |X| (dB)'), a);
      ctx.globalAlpha = a;
      if (!m.short) {
        const ys = OX.view.linear(XLO, XHI, B.bot, B.top);
        const ticks = [];
        for (let v = XLO + 20; v <= XHI; v += g.narrow ? 40 : 20) ticks.push(v);
        for (const v of ticks) hline(ys(v), g.L, xu0, T.grid);
        OX.view.axis(ctx, ys, { side: 'left', at: g.L, ticks: ticks.map((v) => ({ v, label: String(v) })), fontSize: fs });
        ctx.fillStyle = OX.alpha(T.celt, 0.9);
        for (let i = 0; i < N; i++) {
          const v = m.X[i];
          if (v === 0) continue;
          const db = 20 * Math.log10(Math.abs(v));
          if (db <= XLO) continue;
          const [x0, x1] = coefX(i), bw = x1 - x0, ww = bw >= 3 ? bw * 0.6 : Math.max(1, bw);
          const y = ys(Math.min(XHI, db));
          ctx.fillRect(x0 + (bw - ww) / 2, y, ww, B.bot - y);
        }
      } else {
        const rowH = (B.bot - B.top) / NSHORT;
        for (let b = 0; b < NSHORT; b++) {
          const y0 = B.top + b * rowH;
          for (let k = 0; k < SHORT; k++) {
            const v = m.blocks[b][k];
            if (v === 0) continue;
            const t = Math.max(0, Math.min(1, (20 * Math.log10(Math.abs(v)) - (XHI - HEAT_RANGE)) / HEAT_RANGE));
            const x0 = fx(k * SHORT_HZ), x1 = fx((k + 1) * SHORT_HZ);
            ctx.fillStyle = OX.alpha(T.celt, 0.04 + 0.96 * t);
            ctx.fillRect(x0, y0, Math.max(1, x1 - x0), rowH + 0.5);
          }
        }
        ctx.font = OX.font(g.narrow ? 8.5 : 9.5, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let b = 0; b < NSHORT; b++) {
          if (g.narrow && b !== 0 && b !== NSHORT - 1 && b !== m.loud) continue;
          ctx.fillStyle = b === m.loud ? T.ink : T.ink3;
          ctx.fillText(g.narrow ? String(b + 1) : OX.t(`块 ${b + 1}`, `Block ${b + 1}`), g.L - 5, B.top + (b + 0.5) * rowH);
        }
      }
      ctx.globalAlpha = 1;
      edgeLines(B, vis(3) * a, false);
      ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.L, Math.round(B.bot) + 0.5); ctx.lineTo(w - g.R, Math.round(B.bot) + 0.5); ctx.stroke();
    }

    // ---------------- lane C: band energies ----------------
    {
      const C = g.C, a = vis(3);
      laneTitle(C, '4', OX.t('频带能量 √(Σx²)（dB）', 'Band energy √(Σx²) (dB)'), a);
      ctx.globalAlpha = a;
      const ys = OX.view.linear(ALO, AHI, C.bot, C.top);
      const ticks = [];
      for (let v = Math.ceil(ALO / 20) * 20; v <= AHI; v += g.narrow ? 40 : 20) if (v > ALO) ticks.push(v);
      for (const v of ticks) hline(ys(v), g.L, xu0, T.grid);
      OX.view.axis(ctx, ys, { side: 'left', at: g.L, ticks: ticks.map((v) => ({ v, label: String(v) })), fontSize: fs });
      edgeLines(C, a, true);
      ctx.globalAlpha = a;
      for (let b = 0; b < NB; b++) {
        const x0 = fx(EHZ[b]), x1 = fx(EHZ[b + 1]);
        const db = 20 * Math.log10(m.ampC[b]), y = ys(Math.max(ALO, db));
        const pad = x1 - x0 > 5 ? 1.5 : 0.25;
        ctx.fillStyle = OX.alpha(T.celt, T.dark ? 0.22 : 0.16);
        ctx.fillRect(x0 + pad, y, x1 - x0 - 2 * pad, C.bot - y);
        ctx.fillStyle = b === st.band ? T.accent : T.celt;
        ctx.fillRect(x0 + pad, y - 1, x1 - x0 - 2 * pad, 2);
        // the bitstream's value
        const yb = ys(Math.max(ALO, m.fr.band_log2_amp[b] * DB_PER_LOG2));
        ctx.fillStyle = T.accent;
        ctx.beginPath(); ctx.arc((x0 + x1) / 2, yb, g.narrow ? 2 : 2.6, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.L, Math.round(C.bot) + 0.5); ctx.lineTo(w - g.R, Math.round(C.bot) + 0.5); ctx.stroke();
    }

    // ---------------- lane S: normalised shape ----------------
    {
      const S = g.S, a = vis(4);
      laneTitle(S, '5', OX.t('归一化形状：系数 ÷ 频带能量', 'Normalized shape: coefficient ÷ band energy'), a);
      ctx.globalAlpha = a;
      const ys = OX.view.linear(-1, 1, S.bot, S.top);
      hline(ys(0), g.L, w - g.R, T.rule);
      hline(ys(1), g.L, xu0, T.grid); hline(ys(-1), g.L, xu0, T.grid);
      OX.view.axis(ctx, ys, { side: 'left', at: g.L, ticks: [{ v: 1, label: '1' }, { v: 0, label: '0' }, { v: -1, label: '−1' }], fontSize: fs });
      edgeLines(S, a, true);
      ctx.globalAlpha = a;
      ctx.fillStyle = T.celt;
      const y0 = ys(0);
      for (let i = 0; i < CODED; i++) {
        const v = m.shape[i];
        if (v === 0) continue;
        const [x0, x1] = coefX(i), bw = x1 - x0, ww = bw >= 3 ? bw * 0.6 : Math.max(1, bw);
        const y = ys(v);
        ctx.fillRect(x0 + (bw - ww) / 2, Math.min(y, y0), ww, Math.max(1, Math.abs(y - y0)));
      }
      ctx.globalAlpha = 1;
    }

    // frequency axis under the last lane
    const fticks = st.axis === 'lin'
      ? (g.narrow ? [0, 4000, 8000, 12000, 16000, 20000, 24000] : [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000])
      : (g.narrow ? [0, 1, 2, 4, 8, 12, 20].map((k) => k * 1000) : [0, 400, 800, 1200, 1600, 2400, 3200, 4800, 6800, 9600, 15600, 20000, 24000]);
    OX.view.axis(ctx, fx, { side: 'bottom', at: g.S.bot, ticks: fticks.map((v) => ({ v, label: String(+(v / 1000).toFixed(1)), align: v === 0 ? 'left' : v === NYQ ? 'right' : 'center' })), fontSize: fs });
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText('kHz', g.L - 6, g.S.bot + 6);
  }

  function niceCeil(v) {
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }
  function fmtAmp(v) { return String(+v.toPrecision(2)); }

  // ---- interaction ----
  const bandAt = (e) => {
    const g = layout(cv.w);
    if (e.offsetY < g.B.t || e.offsetY > g.S.bot || e.offsetX < g.L || e.offsetX > cv.w - g.R) return null;
    const hz = freqScale(g, cv.w).invert(e.offsetX);
    if (hz >= EHZ[NB]) return -1;
    for (let b = 0; b < NB; b++) if (hz < EHZ[b + 1]) return b;
    return -1;
  };
  const pick = (e) => { const b = bandAt(e); if (b !== null && b !== st.band) { st.band = b; update(); } };
  cv.canvas.addEventListener('pointermove', pick);
  cv.canvas.addEventListener('pointerdown', pick);
  cv.canvas.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    let b;
    if (d) b = st.band == null || st.band < 0 ? (d > 0 ? 0 : NB - 1) : Math.max(0, Math.min(NB - 1, st.band + d));
    else if (e.key === 'Home') b = 0;
    else if (e.key === 'End') b = NB - 1;
    else return;
    e.preventDefault();
    if (b !== st.band) { st.band = b; update(); }
  });

  const fade = OX.loop(host, (dt) => { anim = Math.min(1, anim + dt * 2.5); cv.redraw(); return anim < 1; });
  function go(s) {
    st.step = Math.max(0, Math.min(NSTEPS - 1, s));
    if (OX.reducedMotion()) anim = 1; else { anim = 0; fade.start(); }
    update();
  }

  function update() {
    const m = model(st.role), fr = m.fr, texts = stepText(m);
    caption.innerHTML = `<b>${st.step + 1}/${NSTEPS}</b>${texts[st.step]}`;
    const atStart = st.step === 0, atEnd = st.step === NSTEPS - 1;
    // move focus off a step button before it becomes disabled, so keyboard focus is not dropped to <body>
    if (atEnd && document.activeElement === next) prev.focus();
    if (atStart && document.activeElement === prev) next.focus();
    prev.disabled = atStart; next.disabled = atEnd;
    [...dots.children].forEach((d, i) => {
      d.classList.toggle('on', i === st.step);
      if (i === st.step) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
    });
    roFrame.set(OX.t(`第 ${m.d.index} 帧 · ${(m.d.start_sample / FS).toFixed(3)} s · ${m.short ? `${NSHORT} 个短块` : '1 个长块'}`,
      `Frame ${m.d.index} · ${(m.d.start_sample / FS).toFixed(3)} s · ${m.short ? `${NSHORT} short blocks` : '1 long block'}`));
    roMatch.set(st.step >= 3 ? (m.maxDiffDb < 0.01 ? OX.t('最大相差不到 0.01 dB', 'Max difference below 0.01 dB') : OX.t(`最大相差 ${m.maxDiffDb.toFixed(2)} dB`, `Max difference ${m.maxDiffDb.toFixed(2)} dB`)) : '—');
    roNorm.set(st.step >= 4 ? (m.sMin.toFixed(2) === m.sMax.toFixed(2) ? OX.t(`全部 = ${m.sMin.toFixed(2)}`, `All = ${m.sMin.toFixed(2)}`) : `${m.sMin.toFixed(2)}–${m.sMax.toFixed(2)}`) : '—');
    const b = st.band;
    if (b != null && b >= 0) {
      const lo = EHZ[b], hi = EHZ[b + 1];
      const range = hi <= 1000 ? `${lo}–${hi} Hz` : `${+(lo / 1000).toFixed(1)}–${+(hi / 1000).toFixed(1)} kHz`;
      roBand.set(OX.t(`${b} 号 · ${range} · ${E[b + 1] - E[b]} 个系数`, `No. ${b} · ${range} · ${E[b + 1] - E[b]} coefficients`));
      cv.canvas.setAttribute('aria-valuenow', String(b));
      cv.canvas.setAttribute('aria-valuetext', OX.t(`${b} 号频带，${range}`, `Band ${b}, ${range}`));
      const v = fr.band_log2_amp[b];
      roBandE.set(`${v.toFixed(2)} → ${(v * DB_PER_LOG2).toFixed(1)} dB`);
      const pv = fr.pvq_bits ? fr.pvq_bits[b] : null;
      roBandS.set(`${m.sumSq[b].toFixed(2)} · ${pv == null ? '—' : pv === 0 ? OX.t('0 比特（形状由折叠或噪声补出）', '0 bits (shape filled in by folding or noise)') : OX.t(`${pv} 比特`, `${pv} ${pv === 1 ? 'bit' : 'bits'}`)}`);
    } else {
      roBand.set(b === -1 ? OX.t('20–24 kHz：不编码', '20–24 kHz: not coded') : OX.t('把指针移到下面三条上，或点一下', 'Hover over the lower three plots or tap them'));
      roBandE.set('—'); roBandS.set('—');
      cv.canvas.removeAttribute('aria-valuenow');
      cv.canvas.setAttribute('aria-valuetext', b === -1 ? OX.t('20–24 kHz，不编码', '20–24 kHz, not coded') : OX.t('未选频带，按左右方向键选择', 'No band selected; use the left and right arrow keys to choose one'));
    }
    const pf = fr.postfilter;
    note.textContent =
      OX.t(`系数和频带能量都来自 opus1 以 ${music.celt_frames.bitrate / 1000} kb/s 编码这段音乐的真实码流。系数是解码端在逆 MDCT 之前得到的值，已经带着量化误差。`,
        `The coefficients and band energies both come from the real bitstream that opus1 produced when it encoded this music at ${music.celt_frames.bitrate / 1000} kb/s. The coefficients are the values the decoder has before the inverse MDCT, so they already contain quantization error. `) +
      OX.t('系数对应的是预加重之后的信号（预加重见本节开头）；图里最上面一条（第 1、2 步）画的是预加重之前的原始信号。',
        'The coefficients belong to the signal after pre-emphasis (described at the start of this section); the top plot (steps 1 and 2) shows the original signal before pre-emphasis. ') +
      (pf ? OX.t(`这一帧还开启了基音预滤波器（周期 ${pf.period} 个样本，约 ${Math.round(FS / pf.period)} Hz，增益 ${pf.gain.toFixed(2)}）：MDCT 之前谐波先被削弱，解码端的后滤波器再把它们加回来。这样做的好处是，后滤波器在加回谐波的同时，也压低了谐波之间的量化噪声（见本节末“深入一点”）。`,
        `This frame also has the pitch pre-filter turned on (period ${pf.period} samples, about ${Math.round(FS / pf.period)} Hz, gain ${pf.gain.toFixed(2)}): the harmonics are weakened before the MDCT, and the decoder's post-filter adds them back. The benefit is that, while the post-filter restores the harmonics, it also lowers the quantization noise between them (see “Going deeper” at the end of this section). `) : '') +
      OX.t('把指针移到下面三条中的一个频带上，或点一下，可以查看它的数值；图获得焦点后，也可以用左右方向键切换频带。',
        'Hover over a band in the lower three plots, or tap it, to see its values. When the figure has keyboard focus, the left and right arrow keys also move between bands.');
    cv.redraw();
  }
  update();
});
