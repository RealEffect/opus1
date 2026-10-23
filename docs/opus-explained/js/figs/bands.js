/* Figure 2: CELT's 21 bands over the spectrum of one frame of real music.
 * Main plot: the spectrum (dB) of the original 48 kHz PCM around the chosen frame (2048-point FFT, Hann window),
 * with the 21 band edges (EBAND_5MS x 200 Hz, src/modes.rs) as alternating columns. 20-24 kHz is never coded.
 * Lane: the band energies decoded from the real 64 kb/s opus1 CELT stream (band_log2_amp; 6.02 dB per unit).
 * Rows: where each audio bandwidth stops (end band 13/17/19/21, Bandwidth::celt_end_band in src/lib.rs) and, for
 * Hybrid mode, where SILK hands over to CELT (start band 17 = 8 kHz, HYBRID_START_BAND in src/lib.rs). Hybrid CELT
 * stops at the same end band, so
 * Hybrid FB codes bands 17-20 and Hybrid SWB only 17-18.
 * The main canvas is a keyboard slider over the bands (role="slider", arrow keys, Home/End).
 */
OX.fig('bands', function (root, body) {
  const { el, slider, segmented, readout } = OX.ui;
  const D = OX.data.raw;
  const music = D && D.music, celt = D && D.meta && D.meta.celt;
  if (!music || !music.celt_frames || !music.pcm || !celt || !D.meta.bandwidths) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js，这张图无法显示。', 'The data file data/opus-demo-data.js is missing, so this figure cannot be shown.') }));
    return;
  }

  const FS = music.sample_rate; // 48000
  const NYQ = FS / 2; // 24 kHz
  const EDGES = celt.band_edges_hz; // 22 edges, 0 ... 20000 Hz
  const BINS = celt.band_edges_bins_20ms; // the same edges in 25 Hz MDCT bins of a 20 ms frame
  const NB = EDGES.length - 1; // 21 bands
  const TOP_HZ = EDGES[NB]; // 20 kHz: nothing above is coded
  const HYBRID_START = 17; // in Hybrid mode CELT codes bands 17..20 only (HYBRID_START_BAND in src/lib.rs), i.e. from 8 kHz up
  const FMIN_LOG = 50;
  const NFFT = 2048;
  const BLOCK = music.frame_samples + celt.overlap; // 1080 samples read by one 20 ms MDCT
  const DB_PER_LOG2 = 20 * Math.log10(2); // 6.0206 dB per log2 unit of amplitude
  const frames = music.celt_frames.frames;
  const NF = frames.length;
  const pcm = OX.data.audio('music.pcm', music.pcm);
  const hann = OX.dsp.hann(NFFT);
  const BW_NAMES = { NB: OX.t('窄带', 'Narrowband'), WB: OX.t('宽带', 'Wideband'), SWB: OX.t('超宽带', 'Super-wideband'), FB: OX.t('全带', 'Fullband') };
  const BW = ['NB', 'WB', 'SWB', 'FB'].map((key) => {
    const hz = D.meta.bandwidths[key].audio_hz;
    return { key, name: BW_NAMES[key], hz, end: EDGES.indexOf(hz) }; // end band (exclusive): 13, 17, 19, 21
  });
  const SWB = BW[2]; // Hybrid SWB: CELT codes bands HYBRID_START .. SWB.end - 1 only
  const KBPS = music.celt_frames.bitrate / 1000; // 64
  const delaySamples = D.meta.codec_delay && D.meta.codec_delay.music_48k_audio_samples; // 312
  const DELAY_MS = delaySamples ? +((delaySamples / FS) * 1000).toFixed(1) : null; // 6.5
  const SPEC_LO = -130, SPEC_HI = -10; // dB, fixed for every frame so frames can be compared
  const EVENT_LABEL = { castanet: OX.t('响板', 'castanets'), hihat: OX.t('镲片', 'hi-hat') };
  const events = (music.events || []).filter((e) => e.kind === 'pluck' || EVENT_LABEL[e.kind]);

  // Fixed scale for the energy lane: the range over every frame that lies inside the 2.5 s of music.
  const inSignal = (f) => f && f.start_sample >= 0 && f.start_sample + BLOCK <= music.n_samples;
  // The floor is the 1st percentile, so the two fading frames at the very end do not squash the scale.
  const allE = [];
  frames.forEach((f) => { if (inSignal(f)) allE.push(...f.band_log2_amp); });
  allE.sort((a, b) => a - b);
  const eLo = allE[Math.floor(allE.length * 0.01)], eMax = allE[allE.length - 1];
  const LANE_LO = Math.floor((eLo * DB_PER_LOG2) / 10) * 10, LANE_HI = Math.ceil((eMax * DB_PER_LOG2) / 10) * 10;

  // Facts shown in the axis note, computed from the band table.
  const narrowBands = EDGES.slice(1).filter((hi, b) => hi - EDGES[b] === 200).length; // 8 bands of 200 Hz
  const ratios = [];
  for (let b = 0; b < NB; b++) if (EDGES[b] >= 1000) ratios.push(EDGES[b + 1] / EDGES[b]);
  const rMin = Math.min(...ratios), rMax = Math.max(...ratios);

  const st = { k: 20, scale: 'lin', band: NB - 1 };

  /** Spectrum (dB, 1025 bins) of NFFT samples centred on frame k's MDCT block, or null for an all-zero window. */
  function spectrum(k) {
    return OX.data.memo('bands.spec.' + k, () => {
      const c = frames[k].start_sample + BLOCK / 2;
      const seg = new Float32Array(NFFT);
      let any = false;
      for (let i = 0; i < NFFT; i++) {
        const n = c - NFFT / 2 + i;
        const v = n >= 0 && n < pcm.length ? pcm[n] : 0;
        seg[i] = v;
        if (v !== 0) any = true;
      }
      return any ? OX.dsp.spectrumDb(seg, NFFT, hann, -200) : null;
    });
  }

  const khz = (v) => String(+(v / 1000).toFixed(1));
  const rangeHz = (lo, hi) => (hi <= 1000 ? `${lo}–${hi} Hz` : `${khz(lo)}–${khz(hi)} kHz`);
  const frameTime = (k) => frames[k].start_sample / FS;

  // ---- DOM ----
  OX.ui.css('bands', `
    .bd-note { margin: 4px 0 8px; font-size: var(--small); color: var(--ink-2); line-height: 1.65; min-height: 3.3em; }
    .bd-tl { margin-top: 2px; }
    .bd-canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  `);
  const axisNote = el('p', { class: 'bd-note', 'aria-live': 'polite' });
  const frameSlider = slider({
    id: 'bd-frame', label: OX.t('帧', 'Frame'), min: 0, max: NF - 1, step: 1, value: st.k,
    format: (v) => OX.t(`第 ${v} 帧`, `${v}`), onInput: (v) => setFrame(v),
  });
  const scaleCtl = segmented({
    id: 'bd-scale', label: OX.t('频率轴', 'Frequency axis'), value: st.scale,
    options: [{ value: 'lin', label: OX.t('线性频率', 'Linear') }, { value: 'log', label: OX.t('对数频率', 'Logarithmic') }],
    onChange: (v) => { st.scale = v; update(); },
  });
  const tlHost = el('div', { class: 'bd-tl' });
  const host = el('div');
  const roBand = readout(OX.t('频带', 'Band'), '');
  const roWidth = readout(OX.t('宽度', 'Width'), '');
  const roBins = readout(OX.t('频带里的频率系数（每个 25 Hz）', 'Frequency coefficients in the band (25 Hz each)'), '');
  const roE = readout(OX.t('码流里的能量（log₂ 值 → dB）', 'Energy in the bitstream (log₂ value → dB)'), '');
  const roBw = readout(OX.t('哪些带宽会编码它', 'Bandwidths that code it'), '');
  const roFrame = readout(OX.t('这一帧覆盖的时间', 'Time span of this frame'), '');
  body.append(
    el('div', { class: 'ox-controls' }, [scaleCtl, frameSlider]),
    axisNote,
    host,
    tlHost,
    el('div', { class: 'ox-readouts', 'aria-live': 'polite' }, [roBand, roWidth, roBins, roE, roBw, roFrame]),
    OX.ui.legend([
      { label: OX.t('原始信号的频谱', 'Spectrum of the original signal'), color: 'var(--ink)' },
      { label: OX.t('频带（相邻频带交替加底色）', 'Bands (every other band shaded)'), color: 'var(--celt-soft)', kind: 'box' },
      { label: OX.t('码流里的频带能量', 'Band energy in the bitstream'), color: 'var(--celt)', kind: 'box' },
      { label: OX.t('20 kHz 以上：不编码', 'Above 20 kHz: not coded'), color: 'var(--rule)', kind: 'box' },
    ]),
    el('p', { class: 'ox-note', text:
      OX.t(`频谱取自原始 ${FS / 1000} kHz 信号在这一帧中心附近的 ${NFFT} 个样本（${((NFFT / FS) * 1000).toFixed(1)} 毫秒，汉宁窗）。0 dB 约为满幅正弦波的电平。`,
        `The spectrum is computed from ${NFFT} samples of the original ${FS / 1000} kHz signal around the center of this frame (${((NFFT / FS) * 1000).toFixed(1)} ms, Hann window). 0 dB is about the level of a full-scale sine wave. `) +
      OX.t(`条形是 opus1 以 ${KBPS} kb/s 编码这段音乐时，码流里这一帧 ${NB} 个频带的能量，换算成 dB（以 2 为底的对数值 × 6.02）。`,
        `The bars show the energies of the ${NB} bands in this frame, as stored in the bitstream that opus1 produced when it encoded this music at ${KBPS} kb/s. The energies are converted to dB (base-2 logarithm × 6.02). `) +
      OX.t('编码器先做预加重（压低低频、略抬高高频的滤波，见第 6 节），再算能量。每个数又是整个频带的总能量，频带越宽、系数越多，数值越大。',
        'The encoder applies pre-emphasis (a filter that lowers the low frequencies and slightly raises the high ones; see Section 6) before it computes the energies. Each value is also the total energy of a whole band, so a wider band, with more coefficients, gets a larger value. ') +
      OX.t('所以条形只能和频谱比较起伏，不能比较绝对数值。',
        'So the bars can be compared with the spectrum only in how they rise and fall, not in absolute value. ') +
      OX.t('把指针移到一个频带上，或点一下，查看它的详情；聚焦这张图后，也可以用左右方向键切换频带。',
        'Hover over a band or tap it to see its details. When the figure has focus, the left and right arrow keys also move between bands.') }),
  );

  // ---- main canvas: geometry ----
  function layout(w) {
    const narrow = w < 560;
    const L = narrow ? 34 : 44, R = narrow ? 6 : 10;
    const pTop = 18, pBot = pTop + (narrow ? 150 : 190);
    const laneTitleY = pBot + 24, lTop = laneTitleY + 16, lBot = lTop + (narrow ? 56 : 76);
    const rowH = 15, bwTop = lBot + 12;
    return { narrow, L, R, pTop, pBot, laneTitleY, lTop, lBot, rowH, bwTop, H: bwTop + rowH * 5 + 4 };
  }
  const xScale = (g, w) => (st.scale === 'log' ? OX.view.log(FMIN_LOG, NYQ, g.L, w - g.R) : OX.view.linear(0, NYQ, g.L, w - g.R));
  const xAt = (xs, f) => xs(Math.max(f, xs.domain[0]));

  const cv = OX.view.canvas(host, { height: (w) => layout(w).H, label: OX.t('一帧音乐的频谱、CELT 的 21 个频带边界和码流里的频带能量；用左右方向键选择频带', 'Spectrum of one frame of music, the 21 CELT band edges, and the band energies in the bitstream. Use the left and right arrow keys to choose a band.') }, draw);
  cv.canvas.classList.add('bd-canvas');
  cv.canvas.tabIndex = 0;
  // A focusable picker, so expose it as a slider over the band index (screen readers then pass the arrow keys on).
  cv.canvas.setAttribute('role', 'slider');
  cv.canvas.setAttribute('aria-valuemin', '0');
  cv.canvas.setAttribute('aria-valuemax', String(NB - 1));

  function draw(ctx, w) {
    const T = OX.theme, g = layout(w), xs = xScale(g, w);
    const f = frames[st.k], spec = spectrum(st.k);
    const ys = OX.view.linear(SPEC_LO, SPEC_HI, g.pBot, g.pTop);
    const yl = OX.view.linear(LANE_LO, LANE_HI, g.lBot, g.lTop);
    const colTop = g.pTop, colBot = g.lBot;

    // band columns (alternating), the uncoded region, the selected band
    for (let b = 0; b < NB; b += 2) {
      const x0 = xAt(xs, EDGES[b]), x1 = xAt(xs, EDGES[b + 1]);
      ctx.fillStyle = OX.alpha(T.celt, T.dark ? 0.13 : 0.075);
      ctx.fillRect(x0, colTop, x1 - x0, colBot - colTop);
    }
    const xu0 = xAt(xs, TOP_HZ), xu1 = xs(NYQ);
    ctx.fillStyle = OX.alpha(T.ink3, T.dark ? 0.22 : 0.15);
    ctx.fillRect(xu0, colTop, xu1 - xu0, colBot - colTop);
    const sb = st.band, sx0 = xAt(xs, EDGES[sb]), sx1 = xAt(xs, EDGES[sb + 1]);
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.15 : 0.07);
    ctx.fillRect(sx0, colTop, sx1 - sx0, colBot - colTop);
    ctx.strokeStyle = OX.alpha(T.accent, 0.7); ctx.lineWidth = 1;
    for (const x of [sx0, sx1]) { ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, colTop); ctx.lineTo(Math.round(x) + 0.5, colBot); ctx.stroke(); }

    // spectrum grid and y axis
    const dbTicks = [-120, -90, -60, -30];
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
    for (const v of dbTicks) { const y = Math.round(ys(v)) + 0.5; ctx.beginPath(); ctx.moveTo(g.L, y); ctx.lineTo(w - g.R, y); ctx.stroke(); }
    OX.view.axis(ctx, ys, { side: 'left', at: g.L, ticks: dbTicks.map((v) => ({ v, label: String(v).replace('-', '−') })), fontSize: g.narrow ? 9.5 : 10.5 });
    ctx.strokeStyle = T.rule;
    ctx.beginPath(); ctx.moveTo(g.L + 0.5, g.pTop); ctx.lineTo(g.L + 0.5, g.pBot); ctx.lineTo(w - g.R, g.pBot + 0.5); ctx.stroke();
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText('dB', g.L - 6, g.pTop + 2); // unit of the spectrum axis, above its top tick

    // the spectrum itself
    ctx.save();
    ctx.beginPath(); ctx.rect(g.L, g.pTop, w - g.L - g.R, g.pBot - g.pTop); ctx.clip();
    if (spec) {
      const df = FS / NFFT;
      const k0 = st.scale === 'log' ? Math.floor(FMIN_LOG / df) : 0;
      ctx.beginPath();
      ctx.moveTo(xAt(xs, k0 * df), g.pBot);
      for (let k = k0; k < spec.length; k++) ctx.lineTo(xAt(xs, k * df), ys(Math.max(SPEC_LO - 5, spec[k])));
      ctx.lineTo(xs(NYQ), g.pBot); ctx.closePath();
      ctx.fillStyle = OX.alpha(T.ink, T.dark ? 0.1 : 0.06); ctx.fill();
      ctx.beginPath();
      for (let k = k0; k < spec.length; k++) {
        const X = xAt(xs, k * df), Y = ys(Math.max(SPEC_LO - 5, spec[k]));
        if (k === k0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.1; ctx.lineJoin = 'round'; ctx.stroke();
    } else {
      // A flush frame after the end of the music: its whole FFT window is zero.
      const msg = [OX.t('音乐已经结束，这是编码器补编的静音帧', 'The music has ended. This is a silent frame that the encoder added.')];
      if (DELAY_MS) msg.push(OX.t(`编码器有 ${DELAY_MS} 毫秒延迟，输入结束后要多编几帧，才能把最后一段信号送出来`, `The encoder has a ${DELAY_MS} ms delay, so after the input ends it must code a few more frames to get the last part of the signal out.`));
      ctx.font = OX.font(g.narrow ? 11 : 12, 'body', 500); ctx.fillStyle = T.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const lines = [];
      // The English text is longer, so its box stays over the coded bands, clear of the "Not coded" label.
      const mx1 = OX.lang === 'en' ? xu0 : w - g.R;
      msg.forEach((m) => lines.push(...wrap(ctx, m, mx1 - g.L - 16)));
      const lh = g.narrow ? 16 : 18, y0 = (g.pTop + g.pBot) / 2 - ((lines.length - 1) * lh) / 2, xc = (g.L + mx1) / 2;
      const bw = Math.max(...lines.map((ln) => ctx.measureText(ln).width)) + 16;
      ctx.fillStyle = OX.alpha(T.surface, 0.9);
      ctx.fillRect(xc - bw / 2, y0 - lh / 2 - 4, bw, lines.length * lh + 8);
      ctx.fillStyle = T.ink2;
      lines.forEach((ln, i) => ctx.fillText(ln, xc, y0 + i * lh));
    }
    ctx.restore();

    // "not coded" label
    ctx.font = OX.font(g.narrow ? 10 : 11, 'body', 500); ctx.fillStyle = T.ink2;
    const nc = OX.t('不编码', 'Not coded'), ncw = ctx.measureText(nc).width;
    if (xu1 - xu0 >= ncw + 8) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(nc, (xu0 + xu1) / 2, g.pTop + 6); }
    else { ctx.save(); ctx.translate((xu0 + xu1) / 2, g.pTop + 6); ctx.rotate(Math.PI / 2); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(nc, 0, 0); ctx.restore(); }

    // x axis
    const fticks = st.scale === 'log'
      ? (g.narrow ? [100, 1000, 10000] : [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
      : (g.narrow ? [0, 4000, 8000, 12000, 16000, 20000, 24000] : [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000]);
    const flabel = (v) => (st.scale === 'log' ? (v >= 1000 ? v / 1000 + 'k' : String(v)) : String(v / 1000));
    OX.view.axis(ctx, xs, { side: 'bottom', at: g.pBot, ticks: fticks.map((v) => ({ v, label: flabel(v), align: v === NYQ ? 'right' : v === xs.domain[0] ? 'left' : 'center' })), fontSize: g.narrow ? 9.5 : 10.5 });
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText(st.scale === 'log' ? 'Hz' : 'kHz', g.L - 6, g.pBot + 6);

    // band numbers: widest bands first, skip a label that would overlap one already placed. The row is named "频带".
    ctx.font = OX.font(10, 'body', 500); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(OX.t('频带', 'Band'), g.L - 6, 9);
    ctx.font = OX.font(g.narrow ? 9.5 : 10.5, 'mono', 500); ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    const colW = (b) => xAt(xs, EDGES[b + 1]) - xAt(xs, EDGES[b]);
    const order = [...Array(NB).keys()].sort((a, b) => { const d = colW(b) - colW(a); return Math.abs(d) < 0.5 ? a - b : d; }); // equal widths: lower band first
    const placed = [];
    const center = (b) => (xAt(xs, EDGES[b]) + xAt(xs, EDGES[b + 1])) / 2;
    const fits = (x0, x1) => placed.every(([p0, p1]) => x1 + 3 <= p0 || x0 >= p1 + 3);
    const sw = ctx.measureText(String(sb)).width;
    placed.push([center(sb) - sw / 2 - 2, center(sb) + sw / 2 + 2]);
    for (const b of order) {
      if (b === sb) continue;
      const tw = ctx.measureText(String(b)).width, c = center(b);
      if (c - tw / 2 < g.L - 2) continue;
      if (fits(c - tw / 2, c + tw / 2)) { placed.push([c - tw / 2, c + tw / 2]); ctx.fillStyle = T.ink3; ctx.fillText(String(b), c, 9); }
    }
    ctx.fillStyle = T.accent; ctx.font = OX.font(g.narrow ? 10 : 11, 'mono', 600);
    ctx.fillText(String(sb), center(sb), 9);

    // energy lane
    ctx.font = OX.font(g.narrow ? 10.5 : 11.5, 'body', 600); ctx.fillStyle = T.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(g.narrow ? OX.t('码流里的频带能量（相对，dB）', 'Band energy in the bitstream (relative, dB)')
      : OX.t('码流里的频带能量（相对单位，dB）', 'Band energy in the bitstream (relative units, dB)'), g.L, g.lTop - 3);
    const laneTicks = [];
    for (let v = Math.ceil(LANE_LO / 40) * 40; v <= LANE_HI; v += 40) laneTicks.push(v);
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
    for (const v of laneTicks) { const y = Math.round(yl(v)) + 0.5; ctx.beginPath(); ctx.moveTo(g.L, y); ctx.lineTo(xu0, y); ctx.stroke(); }
    OX.view.axis(ctx, yl, { side: 'left', at: g.L, ticks: laneTicks.map((v) => ({ v, label: String(v).replace('-', '−') })), fontSize: g.narrow ? 9.5 : 10 });
    for (let b = 0; b < NB; b++) {
      const db = f.band_log2_amp[b] * DB_PER_LOG2;
      const x0 = xAt(xs, EDGES[b]), x1 = xAt(xs, EDGES[b + 1]);
      const bw = Math.max(1, x1 - x0 - (x1 - x0 > 4 ? 2 : 0.5));
      const y = yl(Math.max(LANE_LO, Math.min(LANE_HI, db)));
      ctx.fillStyle = b === sb ? T.accent : OX.alpha(T.celt, T.dark ? 0.8 : 0.75);
      ctx.fillRect(x0 + (x1 - x0 - bw) / 2, y, bw, g.lBot - y);
    }
    ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.L, g.lBot + 0.5); ctx.lineTo(w - g.R, g.lBot + 0.5); ctx.stroke();

    // bandwidth rows: how far each audio bandwidth codes, and the Hybrid split
    const xStart = xs(xs.domain[0]);
    ctx.font = OX.font(g.narrow ? 10 : 10.5, 'body', 500); ctx.textBaseline = 'middle';
    const label = (text, xEnd, y, color) => {
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = color;
      if (xEnd + 6 + tw <= w - g.R) { ctx.textAlign = 'left'; ctx.fillText(text, xEnd + 6, y); return; }
      const x = Math.max(xStart + 2, xEnd - 8 - tw);
      ctx.fillStyle = T.surface; ctx.fillRect(x - 4, y - 7, tw + 8, 14);
      ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.fillText(text, x, y);
    };
    // On a phone the English names make the labels long. If any row's label would cover its end tick, every row
    // drops the name and keeps the key (in Chinese both forms are the same text).
    const narrowText = (bw, named) => (named ? OX.t(`${bw.key} ${bw.name} ${khz(bw.hz)} kHz`, `${bw.key} ${bw.name.toLowerCase()} ${khz(bw.hz)} kHz`)
      : OX.t(`${bw.key} ${bw.name} ${khz(bw.hz)} kHz`, `${bw.key} ${khz(bw.hz)} kHz`));
    const named = BW.every((bw) => {
      const tw = ctx.measureText(narrowText(bw, true)).width, xe = xAt(xs, bw.hz);
      return xe + 6 + tw <= w - g.R || xe - 8 - tw >= xStart + 2;
    });
    BW.forEach((bw, i) => {
      const y = g.bwTop + i * g.rowH + g.rowH / 2, xe = xAt(xs, bw.hz);
      ctx.strokeStyle = OX.alpha(T.ink2, 0.5); ctx.lineWidth = 3; ctx.lineCap = 'butt';
      ctx.beginPath(); ctx.moveTo(xStart, y); ctx.lineTo(xe, y); ctx.stroke();
      ctx.lineWidth = 1; ctx.strokeStyle = T.ink2;
      ctx.beginPath(); ctx.moveTo(Math.round(xe) + 0.5, y - 5); ctx.lineTo(Math.round(xe) + 0.5, y + 5); ctx.stroke();
      const text = g.narrow ? narrowText(bw, named)
        : OX.t(`${bw.key} ${bw.name}：编到 ${khz(bw.hz)} kHz（频带 0–${bw.end - 1}）`, `${bw.name} (${bw.key}): coded up to ${khz(bw.hz)} kHz (bands 0–${bw.end - 1})`);
      label(text, xe, y, T.ink2);
    });
    {
      // Hybrid: SILK below 8 kHz, CELT from band 17 up to the end band (20 for FB, 18 for SWB: tick at 12 kHz).
      const y = g.bwTop + 4 * g.rowH + g.rowH / 2, xm = xAt(xs, EDGES[HYBRID_START]), xe = xAt(xs, TOP_HZ), xw = xAt(xs, SWB.hz);
      ctx.lineWidth = 3;
      ctx.strokeStyle = OX.alpha(T.silk, 0.75); ctx.beginPath(); ctx.moveTo(xStart, y); ctx.lineTo(xm, y); ctx.stroke();
      ctx.strokeStyle = OX.alpha(T.celt, 0.85); ctx.beginPath(); ctx.moveTo(xm, y); ctx.lineTo(xe, y); ctx.stroke();
      ctx.lineWidth = 1; ctx.strokeStyle = T.ink2;
      for (const x of [xm, xe]) { ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, y - 5); ctx.lineTo(Math.round(x) + 0.5, y + 5); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(Math.round(xw) + 0.5, y - 3); ctx.lineTo(Math.round(xw) + 0.5, y + 3); ctx.stroke();
      // Labels sit on the row with a background. Prefer one label per part, so the 8 kHz split stays visible.
      const lo = `${HYBRID_START}–${NB - 1}`, swbLast = SWB.end - 1, split = khz(EDGES[HYBRID_START]);
      const options = [
        [OX.t(`混合模式（Hybrid）：${split} kHz 以下用 SILK`, `Hybrid mode: SILK below ${split} kHz`),
          OX.t(`CELT 编 ${lo} 号频带；超宽带只编到 ${swbLast} 号`, `CELT codes bands ${lo}; SWB stops at band ${swbLast}`)],
        [OX.t(`混合模式（Hybrid）：${split} kHz 以下用 SILK，CELT 编 ${lo} 号频带；超宽带只编到 ${swbLast} 号`,
          `Hybrid mode: SILK below ${split} kHz, CELT codes bands ${lo}; SWB stops at band ${swbLast}`), null],
        [OX.t('混合：SILK', 'Hybrid: SILK'), OX.t(`CELT ${lo}（SWB 到 ${swbLast}）`, `CELT ${lo} (SWB to ${swbLast})`)],
        [OX.t(`混合：SILK + CELT ${lo}（SWB 到 ${swbLast}）`, `Hybrid: SILK + CELT ${lo} (SWB to ${swbLast})`), null],
        [OX.t('混合：SILK', 'Hybrid: SILK'), `CELT ${lo}`], // narrowest screens: the 12 kHz tick still marks where SWB stops
      ];
      const fitsIn = (t, x0, x1) => t == null || ctx.measureText(t).width + 8 <= x1 - x0 - 4;
      // the left label must stay inside the SILK part; if nothing fits, the last (shortest) one is used anyway
      const pickd = options.find(([a, b]) => fitsIn(a, xStart, xm) && fitsIn(b, xm, xe)) || options[options.length - 1];
      const tag = (t, x) => {
        const tw = ctx.measureText(t).width;
        ctx.fillStyle = T.surface; ctx.fillRect(x - 4, y - 7, tw + 8, 14);
        ctx.fillStyle = T.ink2; ctx.textAlign = 'left'; ctx.fillText(t, x, y);
      };
      tag(pickd[0], xStart + 4);
      if (pickd[1]) {
        // right-align the CELT label when that leaves the 12 kHz tick visible
        const tw = ctx.measureText(pickd[1]).width, xr = xe - 6 - tw;
        tag(pickd[1], xr - 4 > xw + 4 ? xr : xm + 6);
      }
    }
  }

  /** Break `text` into lines no wider than maxW (Chinese text: any character may end a line). */
  function wrap(ctx, text, maxW) {
    const out = [];
    let line = '';
    if (OX.lang === 'en') {
      // English: break only between words.
      for (const word of text.split(' ')) {
        const next = line ? line + ' ' + word : word;
        if (line && ctx.measureText(next).width > maxW) { out.push(line); line = word; }
        else line = next;
      }
      if (line) out.push(line);
      return out;
    }
    for (const ch of text) {
      if (line && ctx.measureText(line + ch).width > maxW) { out.push(line); line = ch === '\uFF0C' ? '' : ch; } // a full-width comma does not start a line
      else line += ch;
    }
    if (line) out.push(line);
    return out;
  }

  // ---- timeline: the 2.5 s of music, its events, the chosen frame ----
  const TL_H = 46;
  const tl = OX.view.canvas(tlHost, { height: TL_H, label: OX.t(`${+(music.n_samples / FS).toFixed(2)} 秒音乐的波形和事件；点一下或拖动选择一帧，键盘用上方的“帧”滑块`,
    `Waveform and events of ${+(music.n_samples / FS).toFixed(2)} s of music. Tap or drag to choose a frame; with the keyboard, use the “Frame” slider above.`) }, drawTimeline);
  function drawTimeline(ctx, w, h) {
    const T = OX.theme, g = layout(w);
    const xs = OX.view.linear(0, music.n_samples, g.L, w - g.R);
    const mid = 26, amp = 11;
    // waveform envelope (memoised per width)
    const env = OX.data.memo('bands.env.' + Math.round(w), () => {
      const px = Math.max(1, Math.round(w - g.L - g.R)), out = new Float32Array(px * 2);
      for (let p = 0; p < px; p++) {
        const a = Math.floor((p / px) * pcm.length), b = Math.floor(((p + 1) / px) * pcm.length);
        let mn = 0, mx = 0;
        for (let i = a; i < b; i++) { const v = pcm[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        out[2 * p] = mn; out[2 * p + 1] = mx;
      }
      return out;
    });
    let peak = 1e-6;
    for (let i = 0; i < env.length; i++) peak = Math.max(peak, Math.abs(env[i]));
    ctx.fillStyle = OX.alpha(T.ink, T.dark ? 0.45 : 0.35);
    for (let p = 0; p < env.length / 2; p++) {
      const y0 = mid - (env[2 * p + 1] / peak) * amp, y1 = mid - (env[2 * p] / peak) * amp;
      ctx.fillRect(g.L + p, y0, 1, Math.max(1, y1 - y0));
    }
    // events
    ctx.font = OX.font(9.5, 'mono'); ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    for (const e of events) {
      const x = Math.round(xs(e.t_s * FS)) + 0.5;
      ctx.strokeStyle = OX.alpha(T.ink3, 0.9); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, 14); ctx.stroke();
      if (!g.narrow) {
        ctx.fillStyle = T.ink3;
        ctx.fillText(e.kind === 'pluck' ? String(e.detail).split(' ')[0] : EVENT_LABEL[e.kind], x, 10);
      }
    }
    // chosen frame: FFT window (light) and the MDCT block (solid)
    const f = frames[st.k], c = f.start_sample + BLOCK / 2;
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.22 : 0.16);
    ctx.fillRect(xs(c - NFFT / 2), 12, Math.max(2, xs(c + NFFT / 2) - xs(c - NFFT / 2)), h - 24);
    ctx.fillStyle = OX.alpha(T.accent, 0.7);
    ctx.fillRect(xs(f.start_sample), 12, Math.max(2, xs(f.start_sample + BLOCK) - xs(f.start_sample)), h - 24);
    // time labels
    ctx.font = OX.font(9.5, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
    const dur = music.n_samples / FS;
    for (let t = 0; t <= dur + 1e-9; t += 0.5) {
      ctx.textAlign = t === 0 ? 'left' : t >= dur - 1e-9 ? 'right' : 'center';
      ctx.fillText(t === 0 ? '0' : t >= dur - 1e-9 ? `${t} s` : String(t), xs(t * FS), h - 10);
    }
  }
  tl.canvas.style.touchAction = 'pan-y';
  const frameAtX = (x) => {
    const g = layout(tl.w);
    const s = ((x - g.L) / (tl.w - g.L - g.R)) * music.n_samples;
    const k0 = frames[0].start_sample + BLOCK / 2; // centre of frame 0
    return Math.max(0, Math.min(NF - 1, Math.round((s - k0) / music.frame_samples)));
  };
  let dragging = false;
  tl.canvas.addEventListener('pointerdown', (e) => { dragging = true; try { tl.canvas.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } setFrame(frameAtX(e.offsetX)); });
  tl.canvas.addEventListener('pointermove', (e) => { if (dragging) setFrame(frameAtX(e.offsetX)); });
  tl.canvas.addEventListener('pointerup', () => { dragging = false; });
  tl.canvas.addEventListener('pointercancel', () => { dragging = false; });

  // ---- band picking on the main canvas ----
  const bandAtX = (x) => {
    const g = layout(cv.w), xs = xScale(g, cv.w);
    if (x < g.L || x > cv.w - g.R) return -1;
    const f = xs.invert(x);
    for (let b = 0; b < NB; b++) if (f < EDGES[b + 1]) return b;
    return -1; // 20-24 kHz
  };
  const pick = (e) => { const b = bandAtX(e.offsetX); if (b >= 0 && b !== st.band) { st.band = b; update(); } };
  cv.canvas.addEventListener('pointermove', pick);
  cv.canvas.addEventListener('pointerdown', pick);
  cv.canvas.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    let b;
    if (d) b = st.band + d;
    else if (e.key === 'Home') b = 0;
    else if (e.key === 'End') b = NB - 1;
    else return;
    e.preventDefault();
    b = Math.max(0, Math.min(NB - 1, b));
    if (b !== st.band) { st.band = b; update(); }
  });

  function setFrame(k) {
    if (k === st.k) return;
    st.k = k; frameSlider.set(k); update();
  }

  function update() {
    const b = st.band, lo = EDGES[b], hi = EDGES[b + 1], f = frames[st.k];
    roBand.set(OX.t(`${b} 号 · ${rangeHz(lo, hi)}`, `${b} · ${rangeHz(lo, hi)}`));
    cv.canvas.setAttribute('aria-valuenow', String(b));
    cv.canvas.setAttribute('aria-valuetext', OX.t(`${b} 号频带，${rangeHz(lo, hi)}`, `Band ${b}, ${rangeHz(lo, hi)}`));
    roWidth.set(lo > 0 ? OX.t(`${hi - lo} Hz · ${Math.log2(hi / lo).toFixed(2)} 个倍频程`, `${hi - lo} Hz · ${Math.log2(hi / lo).toFixed(2)} octaves`) : `${hi - lo} Hz`);
    roBins.set(OX.t(`${BINS[b + 1] - BINS[b]} 个`, `${BINS[b + 1] - BINS[b]}`));
    const v = f.band_log2_amp[b];
    roE.set(`${v.toFixed(2)} → ${(v * DB_PER_LOG2).toFixed(1)} dB`);
    roBw.set(BW.filter((x) => b < x.end).map((x) => x.key).join(OX.t('、', ', ')));
    const t0 = frameTime(st.k), t1 = (f.start_sample + BLOCK) / FS;
    roFrame.set(`${Math.max(0, t0).toFixed(3)}–${t1.toFixed(3)} s`);
    const note = st.scale === 'lin'
      ? OX.t(`线性频率轴：1.6 kHz 以下挤着 ${narrowBands} 个频带，每个只有 200 Hz 宽；越往上频带越宽，最后一个（${NB - 1} 号）从 ${khz(EDGES[NB - 1])} kHz 到 ${khz(TOP_HZ)} kHz，宽 ${TOP_HZ - EDGES[NB - 1]} Hz。`,
        `Linear frequency axis: ${narrowBands} bands are packed below 1.6 kHz, each only 200 Hz wide. Higher up, the bands get wider; the last one (band ${NB - 1}) runs from ${khz(EDGES[NB - 1])} kHz to ${khz(TOP_HZ)} kHz and is ${TOP_HZ - EDGES[NB - 1]} Hz wide.`)
      : OX.t(`对数频率轴：1 kHz 以上，每个频带的上边界都比下边界高 ${Math.round((rMin - 1) * 100)}%–${Math.round((rMax - 1) * 100)}%（约 ${Math.log2(rMin).toFixed(1)}–${Math.log2(rMax).toFixed(1)} 个倍频程），所以宽度看起来差不多。1.6 kHz 以下的频带固定 200 Hz 宽，其中 600 Hz 以下的几个在对数轴上反而显得更宽。`,
        `Logarithmic frequency axis: above 1 kHz, the upper edge of every band is ${Math.round((rMin - 1) * 100)}%–${Math.round((rMax - 1) * 100)}% higher than its lower edge (about ${Math.log2(rMin).toFixed(1)}–${Math.log2(rMax).toFixed(1)} octaves), so the bands look about equally wide. Below 1.6 kHz the bands are a fixed 200 Hz wide, and the few below 600 Hz even look wider on the logarithmic axis.`);
    if (axisNote.textContent !== note) axisNote.textContent = note; // a live region: rewrite only on a real change
    cv.redraw(); tl.redraw();
  }
  update();
});
