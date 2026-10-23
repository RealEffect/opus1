/* Figure 3: framing. Three seconds of real speech (speech.pcm, 16 kHz) and the packets opus1 made from it.
 * Top lane: the waveform with the 20 ms packet boundaries; one frame is selected (hover, tap, drag or arrow keys).
 * Bottom lane: bytes per packet for one target bitrate, colored by the SILK signal type of the 16 kb/s stream.
 * Zoom: the selected SILK frame (320 samples) with its four 5 ms subframes, original and decoded.
 * Timing: packet k decodes to aligned samples [320k - delay, 320k - delay + 320). SILK frame k starts
 * 13 samples later (silk_frames.core_to_output_offset), at silk_frames.frames[k].start_sample; the subframes are
 * placed from that start.
 * Keyboard: the overview canvas is a slider (role="slider"): arrows, PageUp/PageDown, Home and End change the frame.
 */
OX.fig('framing', function (root, body) {
  const { el, segmented, button, readout, legend } = OX.ui;
  const D = OX.data.raw, S = D && D.speech;
  if (!S || !S.pcm || !S.encodes || !S.encodes.length) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js，这张图无法显示。', 'The data file data/opus-demo-data.js is missing, so this figure cannot be shown.') }));
    return;
  }
  const FS = S.sample_rate, F = S.frame_samples, NS = S.n_samples;
  const FRAME_MS = (F / FS) * 1000; // 20
  const SUB = F / 4; // one SILK subframe: 80 samples = 5 ms at 16 kHz
  const SF = (S.silk_frames && S.silk_frames.frames) || [];
  const TYPE_NAME = [OX.t('无语音', 'Inactive'), OX.t('清音', 'Unvoiced'), OX.t('浊音', 'Voiced')];
  const YMAX = 120; // bytes; the largest packet of any stream here is 116 bytes (32 kb/s)

  const pcm = OX.data.audio('speech.pcm', S.pcm);
  const peak = OX.data.memo('speech.pcm.peak', () => { let m = 0; for (let i = 0; i < pcm.length; i++) m = Math.max(m, Math.abs(pcm[i])); return m || 1; });
  const encOf = (r) => S.encodes.find((e) => e.bitrate === r) || S.encodes[0];
  const decodedAt = (r) => OX.data.audio('speech.encodes.' + r + '.decoded', encOf(r).decoded);
  const stats = (e) => OX.data.memo('framing.stats.' + e.bitrate, () => {
    const s = e.sizes.slice(0, e.n_frames);
    return { min: Math.min(...s), max: Math.max(...s) };
  });

  let rate = encOf(16000).bitrate;
  let enc = encOf(rate);
  const nPk = enc.n_frames; // the 150 coded packets; the flush packets at the end are left out
  const frameStart = (k) => (SF[k] ? SF[k].start_sample : k * F + 13 - enc.delay_samples); // SILK frame k
  const pkStart = (k) => k * F - enc.delay_samples; // what packet k decodes to, on the aligned timeline
  let sel = pickDefault();
  let cursor = null; // aligned sample index of the playback position
  let play = null; // { kind: 'orig'|'dec', token, h }
  let dragging = false;

  function pickDefault() {
    if (SF[127] && SF[127].signal_type === 2) return 127; // the voiced frame used across the page
    let best = Math.floor(nPk / 2);
    for (let d = 0; d < nPk; d++) for (const k of [best + d, best - d]) if (SF[k] && SF[k].signal_type === 2) return k;
    return best;
  }

  OX.ui.css('framing', `
    .fr-over .ox-canvas { cursor: pointer; }
    .fr-over .ox-canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .fr-zhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; margin: 16px 0 6px; font-size: var(--small); color: var(--ink-2); }
    .fr-zhead b { color: var(--accent); font: 600 var(--small)/1.3 var(--font-mono); }
    .fr-zhead .mono { font-variant-numeric: tabular-nums; }
    .fr-zhead .fr-type { display: inline-flex; align-items: center; gap: 6px; color: var(--ink); }
    .fr-zhead .fr-type i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
    .fr-zhead .fr-mut { color: var(--ink-3); font-size: var(--tiny); }
    .fr-legend-row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 16px; }
    .fr-legend-row .ox-legend { margin: 0; }
    .fr-legend-row .ox-note { margin: 0; }
  `);

  /* ---------- controls ---------- */
  const rateSeg = segmented({
    id: 'fr-rate', label: OX.t('目标码率（kb/s）', 'Target bitrate (kb/s)'), value: String(rate),
    options: S.encodes.map((e) => ({ value: String(e.bitrate), label: String(e.bitrate / 1000) })),
    onChange: (v) => setRate(+v),
  });
  const btnOrig = button({ icon: 'sound', label: OX.t('播放原始', 'Play original'), onClick: () => toggle('orig') });
  const btnDec = button({ icon: 'sound', label: OX.t('播放解码', 'Play decoded'), onClick: () => toggle('dec') });

  const hostO = el('div', { class: 'fr-over' });
  const zHead = el('div', { class: 'fr-zhead', 'aria-live': 'polite' });
  const hostZ = el('div', { class: 'fr-zoom' });
  const roKbps = readout(OX.t('实际平均码率', 'Actual average bitrate'), '');
  const roBytes = readout(OX.t('平均每包', 'Average per packet'), '');
  const roMinMax = readout(OX.t('最大 / 最小包', 'Largest / smallest packet'), '');
  const roN = readout(OX.t('包数', 'Packets'), '');
  const roRatio = readout(OX.t(`压缩倍数（相对 ${FS / 1000} kHz、16 位 PCM 的 ${(FS * 16) / 1000} kb/s）`, `Compression ratio (vs. ${FS / 1000} kHz, 16-bit PCM at ${(FS * 16) / 1000} kb/s)`), '');
  const typeColorsCss = ['color-mix(in srgb, var(--ink-3) 55%, transparent)', 'color-mix(in srgb, var(--silk) 42%, transparent)', 'var(--silk)'];
  const note = el('p', { class: 'ox-note' });

  body.append(
    el('div', { class: 'ox-controls' }, [rateSeg, btnOrig, btnDec]),
    hostO,
    el('div', { class: 'fr-legend-row' }, [
      legend([
        { label: OX.t('无语音（没有检测到语音活动）', 'Inactive (no voice activity detected)'), color: typeColorsCss[0], kind: 'box' },
        { label: OX.t('清音（如 s、f）', 'Unvoiced (such as s, f)'), color: typeColorsCss[1], kind: 'box' },
        { label: OX.t('浊音（声带振动，如元音）', 'Voiced (the vocal folds vibrate, as in vowels)'), color: typeColorsCss[2], kind: 'box' },
      ]),
      el('span', { class: 'ox-note', text: OX.t('柱子的颜色是 16 kb/s 码流里 SILK 给每帧标的信号类型。', 'The bar colors show the signal type that SILK gave each frame in the 16 kb/s stream.') }),
    ]),
    zHead,
    hostZ,
    legend([
      { label: OX.t('原始', 'Original'), color: 'var(--ink)' },
      { label: OX.t('解码', 'Decoded'), color: 'var(--accent)' },
      { label: OX.t('子帧分界', 'Subframe boundaries'), color: 'var(--ink-3)', dash: true },
    ]),
    el('div', { class: 'ox-readouts' }, [roKbps, roBytes, roMinMax, roN, roRatio]),
    note);

  /* ---------- overview: waveform lane + bytes lane ---------- */
  function geoO(w, h) {
    const narrow = w < 560;
    const L = narrow ? 33 : 40, R = 8;
    const aTop = 20, aBot = aTop + (narrow ? 62 : 88);
    const bTop = aBot + (narrow ? 30 : 34), bBot = h - 22;
    return { narrow, L, R, aTop, aBot, bTop, bBot, xs: OX.view.linear(0, NS, L, w - R) };
  }
  const typeColor = (t, T) => (t === 2 ? T.silk : t === 1 ? OX.alpha(T.silk, T.dark ? 0.5 : 0.42) : OX.alpha(T.ink3, T.dark ? 0.7 : 0.55));
  const niceStep = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m))); const r = m / p; return (r >= 5 ? 5 : r >= 2 ? 2 : 1) * p; };
  const ampLabel = (v) => (v === 0 ? '0' : (v < 0 ? '−' : '') + String(+Math.abs(v).toPrecision(2)));

  function drawO(ctx, w, h) {
    const T = OX.theme, g = geoO(w, h), xs = g.xs;
    const plotW = w - g.L - g.R;
    // lane titles
    ctx.font = OX.font(11, 'body', 500); ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = T.ink2; ctx.textAlign = 'left';
    ctx.fillText(g.narrow ? OX.t('原始语音', 'Original speech') : OX.t(`原始语音（${S.dur_s} 秒，${FS / 1000} kHz）`, `Original speech (${S.dur_s} s, ${FS / 1000} kHz)`), g.L, g.aTop - 7);
    ctx.fillText(OX.t(`每个包的字节数（目标 ${rate / 1000} kb/s）`, `Bytes per packet (target ${rate / 1000} kb/s)`), g.L, g.bTop - 9);
    if (!g.narrow) {
      ctx.fillStyle = T.ink3; ctx.textAlign = 'right';
      ctx.fillText(OX.t(`浅色竖线：包的分界，每 ${FRAME_MS} 毫秒一个`, `Light vertical lines: packet boundaries, one every ${FRAME_MS} ms`), w - g.R, g.aTop - 7);
    }

    // selected frame band, behind everything else
    const s0 = Math.max(0, pkStart(sel)), s1 = Math.min(NS, pkStart(sel) + F);
    const bx0 = xs(s0), bw = Math.max(2, xs(s1) - bx0);
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.2 : 0.12);
    ctx.fillRect(bx0, g.aTop, bw, g.aBot - g.aTop);
    ctx.fillRect(bx0, g.bTop, bw, g.bBot - g.bTop);

    // lane A: packet boundaries, zero line, waveform
    const pxPerPk = (plotW * F) / NS;
    if (pxPerPk >= 4) {
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.beginPath();
      for (let k = 1; k <= nPk; k++) {
        const x = Math.round(xs(pkStart(k))) + 0.5;
        if (x > g.L && x < w - g.R) { ctx.moveTo(x, g.aTop); ctx.lineTo(x, g.aBot); }
      }
      ctx.stroke();
    }
    const ysA = OX.view.linear(-peak * 1.05, peak * 1.05, g.aBot, g.aTop);
    const stA = niceStep(peak);
    OX.view.axis(ctx, ysA, { side: 'left', at: g.L, ticks: [-stA, 0, stA].map((v) => ({ v, label: ampLabel(v) })), fontSize: 10 });
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(g.L, Math.round(ysA(0)) + 0.5); ctx.lineTo(w - g.R, Math.round(ysA(0)) + 0.5); ctx.stroke();
    OX.view.wave(ctx, pcm, 0, NS, xs, ysA, T.ink, 1);
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1; ctx.beginPath();
    for (const x of [bx0, bx0 + bw]) { const X = Math.round(x) + 0.5; ctx.moveTo(X, g.aTop); ctx.lineTo(X, g.aBot); }
    ctx.stroke();

    // lane B: byte axis, bars, average line
    const ysB = OX.view.linear(0, YMAX, g.bBot, g.bTop);
    OX.view.axis(ctx, ysB, { side: 'left', at: g.L, ticks: (g.narrow ? [0, 60, 120] : [0, 30, 60, 90, 120]).map((v) => ({ v, label: String(v) })), grid: plotW, fontSize: 10 });
    const gap = pxPerPk > 3.5 ? 0.6 : 0;
    for (let k = 0; k < nPk; k++) {
      const a = Math.max(0, pkStart(k)), b = Math.min(NS, pkStart(k) + F);
      const x0 = xs(a) + gap, x1 = xs(b) - gap;
      const y = ysB(enc.sizes[k]);
      ctx.fillStyle = typeColor(SF[k] ? SF[k].signal_type : 0, T);
      ctx.fillRect(x0, y, Math.max(0.8, x1 - x0), g.bBot - y);
    }
    // the selected bar: outline and its size
    {
      const a = Math.max(0, pkStart(sel)), b = Math.min(NS, pkStart(sel) + F);
      const x0 = xs(a), x1 = xs(b), y = ysB(enc.sizes[sel]);
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 - 0.75, y - 0.75, Math.max(1.5, x1 - x0) + 1.5, g.bBot - y + 0.75);
      ctx.font = OX.font(10.5, 'body', 600); ctx.textBaseline = 'bottom';
      const label = OX.t(`${enc.sizes[sel]} 字节`, `${enc.sizes[sel]} bytes`);
      const tw = ctx.measureText(label).width;
      const cx = Math.min(w - g.R - tw / 2 - 3, Math.max(g.L + tw / 2 + 3, (x0 + x1) / 2));
      // a backing box, like the average label below, so the text stays readable over taller neighbouring bars
      ctx.fillStyle = OX.alpha(T.surface, 0.9);
      ctx.fillRect(cx - tw / 2 - 3, y - 17, tw + 6, 14);
      ctx.fillStyle = T.accent; ctx.textAlign = 'center'; ctx.fillText(label, cx, y - 3);
    }
    // stream average
    const ya = Math.round(ysB(enc.avg_bytes)) + 0.5;
    ctx.strokeStyle = T.ink2; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(g.L, ya); ctx.lineTo(w - g.R, ya); ctx.stroke(); ctx.setLineDash([]);
    {
      const txt = OX.t(`平均 ${enc.avg_bytes.toFixed(1)} 字节`, `Average ${enc.avg_bytes.toFixed(1)} bytes`);
      ctx.font = OX.font(10.5, 'body', 500);
      const tw = ctx.measureText(txt).width;
      // keep the label clear of the selected bar's size label
      const selX = xs(Math.max(0, pkStart(sel)));
      const right = selX > w - g.R - tw - 40 ? g.L + tw + 8 : w - g.R;
      ctx.fillStyle = OX.alpha(T.surface, 0.9);
      ctx.fillRect(right - tw - 6, ya - 16, tw + 6, 14);
      ctx.fillStyle = T.ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(txt, right - 3, ya - 3);
    }

    // time axis
    const ticks = [];
    const step = g.narrow ? 1 : 0.5;
    for (let t = 0; t <= S.dur_s + 1e-9; t += step) {
      const last = t + step > S.dur_s + 1e-9;
      ticks.push({ v: t * FS, label: last ? t.toFixed(1) + ' s' : t === 0 ? '0' : t.toFixed(1), align: last ? 'right' : 'center' });
    }
    OX.view.axis(ctx, xs, { side: 'bottom', at: g.bBot, ticks, fontSize: 10 });

    // playback cursor across both lanes
    if (cursor != null) {
      const X = Math.round(xs(cursor)) + 0.5;
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X, g.aTop - 2); ctx.lineTo(X, g.bBot); ctx.stroke();
    }
  }

  /* ---------- zoom: one SILK frame and its subframes ---------- */
  function zoomWindow(narrow) {
    const s0 = frameStart(sel), C = narrow ? 20 : 40;
    return { s0, w0: s0 - C, w1: s0 + F + C };
  }
  function zoomPeak(z, dec) {
    let m = 0;
    for (let i = Math.max(0, z.w0); i < Math.min(NS, z.w1); i++) m = Math.max(m, Math.abs(pcm[i]), Math.abs(dec[i]));
    return Math.max(m, 0.002);
  }
  function drawZ(ctx, w, h) {
    const T = OX.theme, narrow = w < 560;
    const L = narrow ? 33 : 40, R = 8, top = 24, bot = h - 22;
    const z = zoomWindow(narrow), s0 = z.s0;
    const xs = OX.view.linear(z.w0, z.w1, L, w - R);
    const dec = decodedAt(rate);
    const m = zoomPeak(z, dec);
    const ys = OX.view.linear(-m * 1.1, m * 1.1, bot, top);
    const fx0 = xs(s0), fx1 = xs(s0 + F);
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.09 : 0.05);
    ctx.fillRect(fx0, top, fx1 - fx0, bot - top);
    const st = niceStep(m);
    OX.view.axis(ctx, ys, { side: 'left', at: L, ticks: [-st, 0, st].map((v) => ({ v, label: ampLabel(v) })), fontSize: 10 });
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(L, Math.round(ys(0)) + 0.5); ctx.lineTo(w - R, Math.round(ys(0)) + 0.5); ctx.stroke();
    // subframe boundaries and names
    for (let j = 0; j <= 4; j++) {
      const X = Math.round(xs(s0 + j * SUB)) + 0.5;
      ctx.strokeStyle = j === 0 || j === 4 ? T.accent : T.ink3;
      ctx.lineWidth = 1; ctx.setLineDash(j === 0 || j === 4 ? [] : [3, 3]);
      ctx.beginPath(); ctx.moveTo(X, top - 4); ctx.lineTo(X, bot); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.font = OX.font(narrow ? 10.5 : 11, 'body', 500); ctx.fillStyle = T.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    // "Subframe N" is wider than the Chinese label: on very narrow screens the English falls back to the number.
    const subNames = [1, 2, 3, 4].map((n) => OX.t(`子帧 ${n}`, `Subframe ${n}`));
    const subFit = subNames.every((t) => ctx.measureText(t).width + 5 <= xs(SUB) - xs(0));
    for (let j = 0; j < 4; j++) ctx.fillText(subFit ? subNames[j] : OX.t(`子帧 ${j + 1}`, `${j + 1}`), xs(s0 + j * SUB + SUB / 2), top - 8);
    // signals: faint outside the frame, full inside it
    ctx.globalAlpha = 0.3;
    OX.view.wave(ctx, pcm, z.w0, z.w1, xs, ys, T.ink, 1.2);
    OX.view.wave(ctx, dec, z.w0, z.w1, xs, ys, T.accent, 1.1);
    ctx.globalAlpha = 1;
    ctx.save(); ctx.beginPath(); ctx.rect(fx0, 0, fx1 - fx0, h); ctx.clip();
    OX.view.wave(ctx, pcm, z.w0, z.w1, xs, ys, T.ink, 1.3);
    OX.view.wave(ctx, dec, z.w0, z.w1, xs, ys, T.accent, 1.1);
    ctx.restore();
    OX.view.axis(ctx, xs, { side: 'bottom', at: bot, fontSize: 10, ticks: [0, 1, 2, 3, 4].map((j) => ({ v: s0 + j * SUB, label: j === 4 ? FRAME_MS + ' ms' : String((j * FRAME_MS) / 4) })) });
  }

  const cvO = OX.view.canvas(hostO, { height: (w) => (w < 560 ? 236 : 290), label: OX.t('选择一帧：3 秒语音的波形和每个数据包的字节数，按信号类型着色。用左右方向键换帧', 'Choose a frame: the waveform of 3 s of speech and the bytes in each packet, colored by signal type. Use the left and right arrow keys to change the frame.') }, drawO);
  const cvZ = OX.view.canvas(hostZ, { height: (w) => (w < 560 ? 150 : 176), label: OX.t('选中的一帧：320 个样本，分成 4 个 5 毫秒的子帧，原始和解码波形', 'The selected frame: 320 samples in four 5 ms subframes, original and decoded waveforms') }, drawZ);

  /* ---------- selection ---------- */
  const fmtS = (s) => (s < 0 ? '−' : '') + (Math.abs(s) / FS).toFixed(3);
  function updateHead() {
    const s0 = frameStart(sel), t = SF[sel] ? SF[sel].signal_type : null;
    zHead.innerHTML = '';
    const sw = el('i'); if (t != null) sw.style.background = typeColorsCss[t];
    zHead.append(
      el('b', { text: OX.t(`第 ${sel} 帧`, `Frame ${sel}`) }),
      el('span', { class: 'mono', text: `${fmtS(s0)}–${fmtS(s0 + F)} s` }),
      t != null ? el('span', { class: 'fr-type' }, [sw, TYPE_NAME[t]]) : null,
      el('span', { class: 'mono', text: OX.t(`${enc.sizes[sel]} 字节（目标 ${rate / 1000} kb/s）`, `${enc.sizes[sel]} bytes (target ${rate / 1000} kb/s)`) }),
      el('span', { class: 'fr-mut', text: OX.t('帧号从 0 数起。纵轴按这一帧的幅度缩放。', 'Frames are numbered from 0. The y-axis is scaled to this frame’s amplitude.') }));
    // the overview canvas is a slider over the frames (cv is declared below, so use cvO.canvas here)
    cvO.canvas.setAttribute('aria-valuenow', String(sel));
    cvO.canvas.setAttribute('aria-valuetext', OX.t(`第 ${sel} 帧，${t != null ? TYPE_NAME[t] + '，' : ''}${enc.sizes[sel]} 字节`,
      `Frame ${sel}, ${t != null ? TYPE_NAME[t].toLowerCase() + ', ' : ''}${enc.sizes[sel]} bytes`));
  }
  function setSel(k) {
    k = Math.max(0, Math.min(nPk - 1, k));
    if (k === sel) return;
    sel = k;
    cvO.redraw(); cvZ.redraw(); updateHead();
  }
  function frameAt(clientX) {
    const r = cvO.canvas.getBoundingClientRect();
    const g = geoO(cvO.w, cvO.h);
    const s = g.xs.invert(clientX - r.left);
    return Math.floor((s + enc.delay_samples) / F);
  }
  const cv = cvO.canvas;
  cv.tabIndex = 0;
  // A focusable canvas that answers arrow keys is a slider, not an image: screen readers then pass the keys through.
  cv.setAttribute('role', 'slider');
  cv.setAttribute('aria-valuemin', '0');
  cv.setAttribute('aria-valuemax', String(nPk - 1));
  cv.addEventListener('pointerdown', (e) => { dragging = true; try { cv.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } setSel(frameAt(e.clientX)); });
  cv.addEventListener('pointermove', (e) => { if (dragging || e.pointerType === 'mouse') setSel(frameAt(e.clientX)); });
  const endDrag = () => { dragging = false; };
  cv.addEventListener('pointerup', endDrag);
  cv.addEventListener('pointercancel', endDrag);
  cv.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: -1, ArrowRight: 1, PageDown: 10, PageUp: -10 }[e.key];
    if (d) { setSel(sel + d); e.preventDefault(); }
    else if (e.key === 'Home') { setSel(0); e.preventDefault(); }
    else if (e.key === 'End') { setSel(nPk - 1); e.preventDefault(); }
  });

  /* ---------- bitrate and readouts ---------- */
  function updateReadouts() {
    const s = stats(enc);
    roKbps.set(`${enc.avg_kbps.toFixed(2)} kb/s`);
    roBytes.set(OX.t(`${enc.avg_bytes.toFixed(1)} 字节`, `${enc.avg_bytes.toFixed(1)} bytes`));
    roMinMax.set(OX.t(`${s.max} / ${s.min} 字节`, `${s.max} / ${s.min} bytes`));
    roN.set(OX.t(`${nPk}（${S.dur_s} 秒 ÷ ${FRAME_MS} 毫秒）`, `${nPk} (${S.dur_s} s ÷ ${FRAME_MS} ms)`));
    roRatio.set(OX.t(`${((FS * 16) / 1000 / enc.avg_kbps).toFixed(1)} 倍`, `${((FS * 16) / 1000 / enc.avg_kbps).toFixed(1)}×`));
    note.textContent = OX.t(`柱子已按编解码延迟平移，每个包正好对着它解码出的那 ${FRAME_MS} 毫秒声音。`,
      `The bars are shifted by the codec delay, so each packet lines up with the ${FRAME_MS} ms of sound it decodes to. `)
      + OX.t('因为有这段延迟，输入结束时最后一小段信号还留在编码器里。',
        'Because of this delay, the last short piece of the signal is still inside the encoder when the input ends. ')
      + OX.t(`编码器又补编了 ${enc.flush_frames} 个静音帧，把这段信号送出来。统计只算前面的 ${nPk} 个包。`,
        `The encoder then codes ${enc.flush_frames} extra silent frames to get it out. The statistics count only the first ${nPk} packets.`);
  }
  function setRate(r) {
    rate = r; enc = encOf(r);
    if (play && play.kind === 'dec') start('dec', cursor == null ? 0 : cursor);
    syncButtons(); updateReadouts(); updateHead();
    cvO.redraw(); cvZ.redraw();
  }

  /* ---------- playback ---------- */
  const loop = OX.loop(hostO, () => {
    if (!play || !play.h) return false;
    cursor = play.h.position();
    cvO.redraw();
  });
  function syncButtons() {
    const on = (k) => play && play.kind === k;
    btnOrig.setIcon(on('orig') ? 'stop' : 'sound', on('orig') ? OX.t('停止', 'Stop') : OX.t('播放原始', 'Play original'));
    btnDec.setIcon(on('dec') ? 'stop' : 'sound', on('dec') ? OX.t('停止', 'Stop') : OX.t(`播放解码（${rate / 1000} kb/s）`, `Play decoded (${rate / 1000} kb/s)`));
    btnOrig.setAttribute('aria-pressed', on('orig') ? 'true' : 'false');
    btnDec.setAttribute('aria-pressed', on('dec') ? 'true' : 'false');
  }
  function start(kind, from) {
    if (from >= NS - FS * 0.1) from = 0;
    const token = {};
    play = { kind, token, h: null };
    const h = OX.audio.play(kind === 'orig' ? pcm : decodedAt(rate), FS, {
      from: Math.max(0, Math.round(from)),
      onEnd: () => { if (play && play.token === token) { play = null; cursor = null; loop.stop(); syncButtons(); cvO.redraw(); } },
    });
    if (!h) { play = null; syncButtons(); return; }
    play.h = h;
    syncButtons();
    loop.start();
  }
  function toggle(kind) {
    if (play && play.kind === kind) { OX.audio.stop(); return; }
    // switching between original and decoded keeps the position, so the two are easy to compare
    start(kind, play && cursor != null ? cursor : 0);
  }

  syncButtons(); updateReadouts(); updateHead();
});
