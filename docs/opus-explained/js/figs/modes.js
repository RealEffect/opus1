/* Figure 4: which mode and audio bandwidth the encoder uses at each target bitrate.
 * Real data: `mode_map` in data/opus-demo-data.js. Three signals (speech as Voip, speech as Audio, synthetic music
 * as Audio; all 48 kHz mono, 20 ms frames, VBR) were encoded at 13 target bitrates by libopus 1.6.1 and by opus1.
 * Each cell shows the mode and bandwidth that most packets used, the actual average bitrate and, as a thin strip,
 * the share of packets in each mode when the stream mixes modes.
 * Below: the frequency range each mode + bandwidth pair codes. Selecting a cell highlights its row there.
 */
OX.fig('modes', function (root, body) {
  const { el, readout } = OX.ui;
  const D = OX.data.raw;
  const MM = D && D.mode_map;
  if (!MM || !Array.isArray(MM.series) || !Array.isArray(MM.bitrates_kbps)) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js（或其中的 mode_map），这张图无法显示。', 'The data file data/opus-demo-data.js (or its mode_map) is missing, so this figure cannot be shown.') }));
    return;
  }
  const meta = D.meta || {};
  const RATES = MM.bitrates_kbps;
  const BWHZ = { NB: 4000, MB: 6000, WB: 8000, SWB: 12000, FB: 20000 };
  if (meta.bandwidths) for (const k in BWHZ) if (meta.bandwidths[k] && meta.bandwidths[k].audio_hz) BWHZ[k] = meta.bandwidths[k].audio_hz;
  // Hybrid: SILK codes up to 8 kHz, CELT starts at band 17 (8 kHz). Read the band edge from the data when present.
  const SPLIT = (meta.celt && meta.celt.band_edges_hz && meta.celt.band_edges_hz[17]) || 8000;
  const BWNAME = {
    NB: OX.t('窄带', 'narrowband'), MB: OX.t('中带', 'mediumband'), WB: OX.t('宽带', 'wideband'),
    SWB: OX.t('超宽带', 'super-wideband'), FB: OX.t('全带', 'fullband'),
  };
  const MODENAME = { SILK: 'SILK', Hybrid: OX.t('混合', 'Hybrid'), CELT: 'CELT' }; // tight cells
  const MODEFULL = { SILK: 'SILK-only', Hybrid: OX.t('混合模式（Hybrid）', 'Hybrid'), CELT: 'CELT-only' };
  const MODESAY = { SILK: 'SILK-only', Hybrid: OX.t('混合模式', 'hybrid mode'), CELT: 'CELT-only' }; // inside sentences and labels
  const SERIES = {
    speech_voip: { what: OX.t('语音', 'Speech'), app: 'Voip', src: OX.t('语音（16 kHz 电话频带录音，升采样到 48 kHz）', 'Speech (telephone-band recording at 16 kHz, upsampled to 48 kHz)') },
    speech_audio: { what: OX.t('语音', 'Speech'), app: 'Audio', src: OX.t('语音（16 kHz 电话频带录音，升采样到 48 kHz）', 'Speech (telephone-band recording at 16 kHz, upsampled to 48 kHz)') },
    music_audio: { what: OX.t('音乐', 'Music'), app: 'Audio', src: OX.t('合成音乐（48 kHz）', 'Synthetic music (48 kHz)') },
  };
  const serInfo = (s) => SERIES[s.name] || {
    what: /music/.test(s.signal || '') ? OX.t('音乐', 'Music') : OX.t('语音', 'Speech'),
    app: s.application === 'voip' ? 'Voip' : s.application === 'audio' ? 'Audio' : String(s.application || ''),
    src: s.signal || s.name,
  };
  const ENC = [
    { key: 'libopus', label: 'libopus', full: meta.libopus_version || 'libopus' },
    { key: 'opus1', label: 'opus1', full: 'opus1' + (meta.crate_version ? ' ' + meta.crate_version : '') },
  ];
  // Rows: grouped by series; inside each group libopus first (what Opus is designed to do), then opus1.
  const rows = [];
  MM.series.forEach((s, si) => ENC.forEach((e, ei) => {
    const cells = Array.isArray(s[e.key]) ? s[e.key] : [];
    rows.push({ s, si, e, ei, info: serInfo(s), cells });
  }));
  const OFF = 0.25; // same rule as the data's rate_note: actual rate more than 25 % away from the target
  const offTarget = (c) => c && c.bitrate_kbps > 0 && Math.abs(c.avg_kbps / c.bitrate_kbps - 1) > OFF;
  const famKey = (mode, bw) => mode + '-' + bw;
  const nf = (v, d) => (+v).toFixed(d == null ? 1 : d);

  // Default selection: libopus, first series, lowest bitrate (SILK narrowband).
  const st = { r: 0, c: 0, hover: null, layout: null };

  OX.ui.css('modes', `
    .md-scroll { margin: 6px 0 2px; }
    .md-grid { display: grid; gap: 3px; min-width: 0; }
    .md-h { font: 500 var(--tiny)/1.2 var(--font-mono); color: var(--ink-2); text-align: center; align-self: end; padding: 0 0 3px; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .md-corner { text-align: left; font-family: var(--font-body); color: var(--ink-3); font-size: 0.74rem; }
    .md-ser { align-self: center; font-size: var(--tiny); line-height: 1.35; color: var(--ink); font-weight: 600; min-width: 0; }
    .md-ser small { display: block; font: 500 0.72rem/1.3 var(--font-mono); color: var(--ink-3); }
    .md-serh { text-align: center; align-self: end; font-size: 0.74rem; line-height: 1.3; font-weight: 600; color: var(--ink); white-space: nowrap; border-bottom: 1px solid var(--rule); padding-bottom: 3px; }
    .md-enc { align-self: center; font: 500 0.72rem/1 var(--font-mono); color: var(--ink-2); white-space: nowrap; }
    .md-ench { text-align: center; align-self: end; font: 500 0.64rem/1.1 var(--font-body); color: var(--ink-2); white-space: nowrap; padding-bottom: 2px; overflow: hidden; }
    .md-rate { align-self: center; text-align: right; padding-right: 6px; font: 500 0.72rem/1 var(--font-mono); color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
    .md-cell { position: relative; border: 0; margin: 0; padding: 5px 1px 8px; border-radius: 5px; min-height: 42px; min-width: 0;
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 3px; cursor: pointer; color: var(--ink); font: inherit; overflow: hidden; }
    .md-grid.tr .md-cell { min-height: 34px; padding: 3px 1px 7px; gap: 2px; }
    .md-cell:hover { box-shadow: inset 0 0 0 1px var(--ink-3); }
    .md-cell[aria-pressed="true"] { box-shadow: inset 0 0 0 2px var(--accent); }
    .md-bw { font: 600 0.76rem/1 var(--font-mono); letter-spacing: 0.02em; }
    .md-kb { font: 500 0.66rem/1 var(--font-mono); color: var(--ink); font-variant-numeric: tabular-nums; }
    /* an off-target number gets its own backing so it never sits on the split SILK/CELT colors */
    .md-kb.over { color: var(--bad); font-weight: 600; background: var(--surface); padding: 1px 3px; border-radius: 3px; }
    .md-mix { position: absolute; left: 4px; right: 4px; bottom: 3px; height: 3px; display: flex; border-radius: 2px; overflow: hidden; }
    .md-mix i { display: block; height: 100%; }
    .md-why { margin: 8px 0 0; font-size: var(--small); color: var(--ink-2); line-height: 1.65; max-width: 52rem; }
    .md-why b { color: var(--ink); font-weight: 600; }
    .md-legend .ox-sw.box { width: 12px; height: 12px; }
    .md-ramp { display: inline-flex; gap: 2px; }
    .md-ramp i { width: 9px; height: 12px; border-radius: 2px; display: block; }
    .md-legend .over { color: var(--bad); font: 600 0.72rem/1 var(--font-mono); }
    .md-mixsw { width: 16px; height: 3px; border-radius: 2px; display: inline-flex; overflow: hidden; }
    .md-mixsw i { display: block; height: 100%; }
    .md-sel { margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--rule); }
    .md-sel .ox-readouts { margin-top: 2px; gap: 8px 20px; }
    .md-sel .ox-ro-val.bad { color: var(--bad); }
    .md-sel .md-txt .ox-ro-val { font-family: var(--font-body); font-size: 0.94rem; }
    .md-sub { margin: 16px 0 4px; font-size: var(--small); font-weight: 600; color: var(--ink); }
    .md-sub span { font-weight: 400; color: var(--ink-3); font-size: var(--tiny); margin-left: 8px; }
  `);

  /* ---------------- chart (DOM grid of buttons) ---------------- */
  const scroll = el('div', { class: 'ox-scroll md-scroll' });
  const grid = el('div', { class: 'md-grid', role: 'group', 'aria-label': OX.t('各目标码率下的模式与带宽。方向键在格子之间移动。', 'Mode and bandwidth at each target bitrate. Use the arrow keys to move between cells.') });
  scroll.appendChild(grid);
  const why = el('p', { class: 'md-why' });
  // The two rows are the same in this data: mode, bandwidth and packets per mode in every cell, and average
  // bitrates within 0.4 kb/s. opus1 ports libopus's mode and bandwidth rules (decide_mode_and_bandwidth in
  // src/lib.rs) and the float build's speech/music and bandwidth analysis (src/analysis/, feature `analysis`,
  // on by default), which runs from complexity 7. The data is encoded at complexity 9.
  why.innerHTML = OX.t('<b>为什么两行一样：</b>opus1 移植了 libopus 选模式和带宽的规则。',
    '<b>Why the two rows are the same:</b> opus1 ports the rules that libopus uses to choose the mode and the bandwidth. ')
    + OX.t('它还移植了 libopus 的语音/音乐分析和带宽检测。这两项在复杂度 7 及以上运行，这张图用的是复杂度 9。',
      'It also ports the speech/music analysis and the bandwidth detection of libopus. They run at complexity 7 and above, and this figure uses complexity 9. ')
    + OX.t('所以每个格子的模式、带宽和各模式的包数都和 libopus 相同，平均码率相差不到 0.4 kb/s。',
      'So every cell has the same mode, the same bandwidth, and the same number of packets in each mode as libopus, and the average bitrates differ by less than 0.4 kb/s.');

  const swatch = (bg) => { const s = el('span', { class: 'ox-sw box' }); s.style.background = bg; return s; };
  const legSilk = swatch(''), legHyb = swatch(''), legCelt = swatch('');
  const ramp = el('span', { class: 'md-ramp', 'aria-hidden': 'true' }, ['NB', 'WB', 'SWB', 'FB'].map(() => el('i')));
  const mixSw = el('span', { class: 'md-mixsw', 'aria-hidden': 'true' }, [el('i'), el('i'), el('i')]);
  const legend = el('div', { class: 'ox-legend md-legend' }, [
    el('span', { class: 'ox-legend-item' }, [legSilk, 'SILK-only']),
    el('span', { class: 'ox-legend-item' }, [legHyb, OX.t('混合模式（Hybrid）', 'Hybrid')]),
    el('span', { class: 'ox-legend-item' }, [legCelt, 'CELT-only']),
    el('span', { class: 'ox-legend-item' }, [ramp, OX.t('颜色越深，带宽越宽（窄带 NB → 全带 FB）', 'Darker color = wider bandwidth (narrowband NB → fullband FB)')]),
    // The red numbers are explained only when some cell has one (none does in the current data).
    el('span', { class: 'ox-legend-item' }, [el('span', {}, rows.some((r) => r.cells.some(offTarget))
      ? [OX.t('格内小字：实际平均码率（kb/s），', 'Number in each cell: actual average bitrate (kb/s); '), el('span', { class: 'over', text: OX.t('红色', 'red') }), OX.t('表示偏离目标超过 25%', ' means more than 25% off the target')]
      : [OX.t('格内小字：实际平均码率（kb/s）', 'Number in each cell: actual average bitrate (kb/s)')])]),
    el('span', { class: 'ox-legend-item' }, [mixSw, el('span', { text: OX.t('底部细条：一个码流混用几种模式时，各模式包的比例', 'Thin strip at the bottom: share of packets in each mode, when a stream mixes modes') })]),
  ]);

  /* ---------------- readout of the selected / hovered cell ---------------- */
  const TXT = { cls: 'md-txt' };
  const roSrc = readout(OX.t('素材 · 应用类型', 'Signal · application'), '', TXT);
  const roEnc = readout(OX.t('编码器', 'Encoder'), '');
  const roTarget = readout(OX.t('目标码率', 'Target bitrate'), '');
  const roActual = readout(OX.t('实际平均码率', 'Actual average bitrate'), '');
  const roMode = readout(OX.t('多数包的模式 · 带宽', 'Mode · bandwidth of most packets'), '', TXT);
  const roCounts = readout(OX.t('各模式的包数', 'Packets per mode'), '', TXT);
  const roBytes = readout(OX.t('平均每包字节', 'Average bytes per packet'), '');
  const sel = el('div', { class: 'md-sel', 'aria-live': 'polite' }, [
    el('div', { class: 'ox-readouts' }, [roSrc, roEnc, roTarget, roActual, roMode, roCounts, roBytes]),
  ]);

  /* ---------------- frequency coverage (canvas) ---------------- */
  const FAM = [
    { mode: 'SILK', bw: 'NB', segs: [['silk', 0, BWHZ.NB]] },
    { mode: 'SILK', bw: 'MB', segs: [['silk', 0, BWHZ.MB]] },
    { mode: 'SILK', bw: 'WB', segs: [['silk', 0, BWHZ.WB]] },
    { mode: 'Hybrid', bw: 'SWB', segs: [['silk', 0, SPLIT], ['celt', SPLIT, BWHZ.SWB]] },
    { mode: 'Hybrid', bw: 'FB', segs: [['silk', 0, SPLIT], ['celt', SPLIT, BWHZ.FB]] },
    { mode: 'CELT', bw: 'NB', segs: [['celt', 0, BWHZ.NB]] },
    { mode: 'CELT', bw: 'WB', segs: [['celt', 0, BWHZ.WB]] },
    { mode: 'CELT', bw: 'SWB', segs: [['celt', 0, BWHZ.SWB]] },
    { mode: 'CELT', bw: 'FB', segs: [['celt', 0, BWHZ.FB]] },
  ];
  FAM.forEach((f) => { f.key = famKey(f.mode, f.bw); });
  const sub = el('div', { class: 'md-sub' }, [OX.t('每种模式和带宽编码的频率范围', 'Frequency range coded by each mode and bandwidth'), el('span', { text: OX.t('选中的格子对应的那一行会高亮', 'The row for the selected cell is highlighted') })]);
  const host = el('div');

  body.append(scroll, why, legend, sel, sub, host);

  // Everything draw() reads is declared before OX.view.canvas runs it for the first time.
  const ROWH = (w) => (w < 560 ? 18 : 20);
  const GAPG = 9; // extra gap between the SILK / Hybrid / CELT groups
  const TOP = 22, AXIS = 30;
  const covHeight = (w) => TOP + FAM.length * (ROWH(w) + 4) + 2 * GAPG + AXIS;
  const cv = OX.view.canvas(host, { height: covHeight, label: OX.t('频率覆盖图：SILK 窄带 0–4 kHz，中带 0–6 kHz，宽带 0–8 kHz；混合超宽带为 SILK 0–8 kHz 加 CELT 8–12 kHz，混合全带为 SILK 0–8 kHz 加 CELT 8–20 kHz；CELT 窄带、宽带、超宽带、全带分别到 4、8、12、20 kHz；20 kHz 以上不编码。', 'Frequency coverage: SILK narrowband 0–4 kHz, mediumband 0–6 kHz, wideband 0–8 kHz; Hybrid super-wideband is SILK 0–8 kHz plus CELT 8–12 kHz, and Hybrid fullband is SILK 0–8 kHz plus CELT 8–20 kHz; CELT narrowband, wideband, super-wideband, and fullband go up to 4, 8, 12, and 20 kHz; nothing above 20 kHz is coded.') }, drawCoverage);

  function current() {
    const p = st.hover || { r: st.r, c: st.c };
    const row = rows[p.r];
    return { row, cell: row && row.cells[p.c], r: p.r, c: p.c };
  }

  function drawCoverage(ctx, w, h) {
    const T = OX.theme;
    const narrow = w < 560;
    const rh = ROWH(w);
    const L = narrow ? 70 : 118, R = 10;
    const xs = OX.view.linear(0, 24000, L, w - R);
    const cur = current();
    const selKey = cur.cell ? famKey(cur.cell.mode, cur.cell.bandwidth) : null;
    const y0 = TOP, yEnd = h - AXIS;
    // row positions
    const ys = [];
    let y = y0;
    FAM.forEach((f, i) => {
      if (i > 0 && f.mode !== FAM[i - 1].mode) y += GAPG;
      ys.push(y);
      y += rh + 4;
    });
    // vertical grid
    ctx.lineWidth = 1;
    for (let k = 4000; k < 20000; k += 4000) {
      ctx.strokeStyle = T.grid;
      ctx.beginPath(); ctx.moveTo(Math.round(xs(k)) + 0.5, y0 - 4); ctx.lineTo(Math.round(xs(k)) + 0.5, yEnd); ctx.stroke();
    }
    // 20-24 kHz: never coded
    const gx0 = xs(20000), gx1 = xs(24000);
    ctx.fillStyle = OX.alpha(T.ink3, T.dark ? 0.12 : 0.1);
    ctx.fillRect(gx0, y0 - 4, gx1 - gx0, yEnd - y0 + 4);
    ctx.save();
    ctx.beginPath(); ctx.rect(gx0, y0 - 4, gx1 - gx0, yEnd - y0 + 4); ctx.clip();
    ctx.strokeStyle = OX.alpha(T.ink3, 0.28); ctx.lineWidth = 1;
    for (let d = -h; d < gx1 - gx0 + h; d += 7) { ctx.beginPath(); ctx.moveTo(gx0 + d, y0 - 4); ctx.lineTo(gx0 + d + (yEnd - y0 + 4), yEnd); ctx.stroke(); }
    ctx.restore();
    ctx.font = OX.font(narrow ? 10 : 10.5, 'body', 500); ctx.fillStyle = T.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(OX.t('不编码', 'Not coded'), (gx0 + gx1) / 2, y0 - 6);
    // Hybrid split line
    ctx.strokeStyle = OX.alpha(T.ink3, 0.7); ctx.setLineDash([3, 3]);
    const sx = Math.round(xs(SPLIT)) + 0.5;
    ctx.beginPath(); ctx.moveTo(sx, ys[3] - 3); ctx.lineTo(sx, ys[4] + rh + 3); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = T.ink3; ctx.font = OX.font(10, 'mono'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText((narrow ? '' : OX.t('SILK | CELT 分界 ', 'SILK | CELT split at ')) + SPLIT / 1000 + ' kHz', sx + 4, y0 - 6);

    FAM.forEach((f, i) => {
      const yy = ys[i];
      const on = f.key === selKey;
      if (on) {
        ctx.fillStyle = T.accentSoft;
        ctx.fillRect(0, yy - 2, w, rh + 4);
      }
      // label
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.font = OX.font(narrow ? 10.5 : 11, 'mono', on ? 600 : 500);
      ctx.fillStyle = on ? T.accent : f.mode === 'SILK' ? T.silk : f.mode === 'CELT' ? T.celt : T.ink;
      const modeTxt = MODENAME[f.mode];
      ctx.fillText(modeTxt, 4, yy + rh / 2);
      const mw = ctx.measureText(modeTxt + ' ').width;
      ctx.fillStyle = on ? T.accent : T.ink2;
      // In English only the abbreviation (NB, WB, ...) fits in the label column; the legend and readout spell it out.
      ctx.fillText(narrow || OX.lang === 'en' ? f.bw : f.bw + ' ' + BWNAME[f.bw], 4 + mw, yy + rh / 2);
      // bars
      const dim = selKey && !on;
      for (const [layer, a, b] of f.segs) {
        const x0 = xs(a), x1 = xs(b);
        const col = layer === 'silk' ? T.silk : T.celt;
        ctx.fillStyle = dim ? OX.alpha(col, T.dark ? 0.3 : 0.22) : col;
        ctx.fillRect(Math.round(x0), yy, Math.round(x1) - Math.round(x0) - (b < 24000 && f.segs.length > 1 && a === 0 ? 1 : 0), rh);
        // text inside the bar when it fits
        ctx.font = OX.font(narrow ? 9.5 : 10.5, 'mono', 500);
        const txt = f.segs.length > 1 ? `${layer === 'silk' ? 'SILK' : 'CELT'} ${a / 1000}–${b / 1000} kHz` : `0–${b / 1000} kHz`;
        const short = `${a / 1000}–${b / 1000}`;
        const tw = ctx.measureText(txt).width, sw = ctx.measureText(short).width;
        ctx.fillStyle = dim ? col : T.surface; ctx.textAlign = 'left';
        if (tw + 10 < x1 - x0) ctx.fillText(txt, x0 + 6, yy + rh / 2 + 0.5);
        else if (sw + 8 < x1 - x0) ctx.fillText(short, x0 + 4, yy + rh / 2 + 0.5);
      }
      if (on) {
        // marker at the bar's top edge
        const top = f.segs[f.segs.length - 1][2];
        const mx = xs(top);
        ctx.fillStyle = T.accent;
        ctx.beginPath(); ctx.moveTo(mx + 3, yy + rh / 2); ctx.lineTo(mx + 9, yy + rh / 2 - 4); ctx.lineTo(mx + 9, yy + rh / 2 + 4); ctx.closePath(); ctx.fill();
      }
    });
    // axis
    // On narrow screens the 20 and 24 kHz labels would collide: label 20 kHz and leave 24 unlabeled.
    const ticks = [0, 4000, 8000, 12000, 16000, 20000, 24000].map((v) => (narrow
      ? { v, label: v === 24000 ? null : v === 20000 ? '20 kHz' : String(v / 1000) }
      : { v, label: v === 24000 ? '24 kHz' : String(v / 1000), align: v === 24000 ? 'right' : 'center' }));
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, yEnd + 0.5); ctx.lineTo(w - R, yEnd + 0.5); ctx.stroke();
    OX.view.axis(ctx, xs, { side: 'bottom', at: yEnd + 0.5, ticks, fontSize: 10 });
  }

  /* ---------------- building the grid ---------------- */
  let cellEls = []; // cellEls[r][c]
  function cellBg(c) {
    const T = OX.theme;
    const A = T.dark ? { NB: 0.2, MB: 0.25, WB: 0.31, SWB: 0.4, FB: 0.5 } : { NB: 0.2, MB: 0.28, WB: 0.36, SWB: 0.5, FB: 0.64 };
    const a = A[c.bandwidth] || 0.4;
    const s = OX.alpha(T.silk, a), k = OX.alpha(T.celt, a);
    if (c.mode === 'SILK') return s;
    if (c.mode === 'CELT') return k;
    return `linear-gradient(135deg, ${s} 50%, ${k} 50%)`;
  }
  function mixBg(mode) {
    const T = OX.theme;
    if (mode === 'SILK') return T.silk;
    if (mode === 'CELT') return T.celt;
    return `linear-gradient(to bottom, ${T.silk} 50%, ${T.celt} 50%)`;
  }
  function cellLabel(row, c) {
    if (!c) return OX.t('无数据', 'No data');
    const parts = [OX.t(`${row.e.label}，${row.info.what} · ${row.info.app}，目标 ${c.bitrate_kbps} kb/s`, `${row.e.label}, ${row.info.what} · ${row.info.app}, target ${c.bitrate_kbps} kb/s`),
      OX.t(`${MODESAY[c.mode] || c.mode}，${BWNAME[c.bandwidth] || ''} ${c.bandwidth}`, `${MODESAY[c.mode] || c.mode}, ${BWNAME[c.bandwidth] || ''} (${c.bandwidth})`),
      OX.t(`实际 ${nf(c.avg_kbps)} kb/s`, `actual ${nf(c.avg_kbps)} kb/s`) + (offTarget(c) ? OX.t('，偏离目标超过 25%', ', more than 25% off the target') : '')];
    return parts.join(OX.t('，', ', '));
  }

  function build() {
    const narrow = body.clientWidth < 720;
    const layout = narrow ? 'tr' : 'std';
    if (layout === st.layout && cellEls.length) return;
    st.layout = layout;
    grid.innerHTML = '';
    grid.className = 'md-grid' + (narrow ? ' tr' : '');
    cellEls = rows.map(() => []);
    const nS = MM.series.length;
    const place = (e, row, col, rs, cs) => {
      e.style.gridRow = rs ? `${row} / span ${rs}` : String(row);
      e.style.gridColumn = cs ? `${col} / span ${cs}` : String(col);
      grid.appendChild(e);
      return e;
    };
    if (!narrow) {
      grid.style.gridTemplateColumns = `4.6em 4.6em repeat(${RATES.length}, minmax(42px, 1fr))`;
      grid.style.gridTemplateRows = 'auto' + ' 6px auto auto'.repeat(nS);
      place(el('div', { class: 'md-h md-corner', text: OX.t('目标码率（kb/s）', 'Target bitrate (kb/s)') }), 1, 1, 0, 2);
      RATES.forEach((r, j) => place(el('div', { class: 'md-h', text: String(r) }), 1, 3 + j));
      MM.series.forEach((s, si) => {
        const info = serInfo(s);
        place(el('div', { class: 'md-ser' }, [info.what, el('small', { text: info.app })]), 3 + 3 * si, 1, 2);
      });
      rows.forEach((row, ri) => {
        const gr = 3 + 3 * row.si + row.ei;
        place(el('div', { class: 'md-enc', text: row.e.label }), gr, 2);
        RATES.forEach((_, j) => { cellEls[ri][j] = place(makeCell(row, ri, j), gr, 3 + j); });
      });
    } else {
      grid.style.gridTemplateColumns = '38px' + ' 3px minmax(36px, 1fr) minmax(36px, 1fr)'.repeat(nS);
      grid.style.gridTemplateRows = '';
      place(el('div', { class: 'md-h md-corner', text: 'kb/s' }), 2, 1);
      MM.series.forEach((s, si) => {
        const info = serInfo(s);
        place(el('div', { class: 'md-serh', text: `${info.what} · ${info.app}` }), 1, 3 + 3 * si, 0, 2);
      });
      rows.forEach((row, ri) => {
        const gc = 3 + 3 * row.si + row.ei;
        place(el('div', { class: 'md-ench', text: row.e.label }), 2, gc);
        RATES.forEach((_, j) => { cellEls[ri][j] = place(makeCell(row, ri, j), 3 + j, gc); });
      });
      RATES.forEach((r, j) => place(el('div', { class: 'md-rate', text: String(r) }), 3 + j, 1));
    }
    paint();
  }

  function makeCell(row, ri, j) {
    const c = row.cells[j];
    const b = el('button', { type: 'button', class: 'md-cell', 'aria-label': cellLabel(row, c), 'data-r': ri, 'data-c': j });
    if (!c) { b.disabled = true; b.textContent = '—'; return b; }
    b.appendChild(el('span', { class: 'md-bw', text: c.bandwidth }));
    b.appendChild(el('span', { class: 'md-kb' + (offTarget(c) ? ' over' : ''), text: nf(c.avg_kbps) }));
    const mc = c.mode_counts || {};
    const tot = (mc.SILK || 0) + (mc.Hybrid || 0) + (mc.CELT || 0);
    const used = ['SILK', 'Hybrid', 'CELT'].filter((m) => mc[m] > 0);
    if (tot > 0 && used.length > 1) {
      const mix = el('span', { class: 'md-mix', 'aria-hidden': 'true' });
      used.forEach((m) => { const i = el('i', { 'data-m': m }); i.style.width = (100 * mc[m]) / tot + '%'; mix.appendChild(i); });
      b.appendChild(mix);
    }
    b.addEventListener('click', () => select(ri, j, false));
    b.addEventListener('pointerenter', (ev) => { if (ev.pointerType === 'mouse') { st.hover = { r: ri, c: j }; show(); } });
    b.addEventListener('keydown', onKey);
    return b;
  }

  function paint() {
    const T = OX.theme;
    rows.forEach((row, ri) => row.cells.forEach((c, j) => {
      const b = cellEls[ri][j];
      if (!b || !c) return;
      b.style.background = cellBg(c);
      b.querySelectorAll('.md-mix i').forEach((i) => { i.style.background = mixBg(i.getAttribute('data-m')); });
      const on = ri === st.r && j === st.c;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
    }));
    legSilk.style.background = OX.alpha(T.silk, T.dark ? 0.5 : 0.64);
    legCelt.style.background = OX.alpha(T.celt, T.dark ? 0.5 : 0.64);
    legHyb.style.background = `linear-gradient(135deg, ${OX.alpha(T.silk, T.dark ? 0.5 : 0.64)} 50%, ${OX.alpha(T.celt, T.dark ? 0.5 : 0.64)} 50%)`;
    const A = T.dark ? [0.2, 0.31, 0.4, 0.5] : [0.2, 0.36, 0.5, 0.64];
    ramp.querySelectorAll('i').forEach((i, k) => { i.style.background = OX.alpha(T.celt, A[k]); });
    const ms = mixSw.querySelectorAll('i');
    [['SILK', 30], ['Hybrid', 20], ['CELT', 50]].forEach(([m, wPct], k) => { ms[k].style.width = wPct + '%'; ms[k].style.background = mixBg(m); });
  }

  function select(r, c, focus) {
    st.r = r; st.c = c; st.hover = null;
    paint();
    show();
    if (focus && cellEls[r] && cellEls[r][c]) cellEls[r][c].focus();
  }

  function onKey(ev) {
    const nR = rows.length, nC = RATES.length;
    let { r, c } = st;
    const tr = st.layout === 'tr';
    // In the transposed layout, columns are rows of the data and vice versa.
    const k = ev.key;
    const dRate = tr ? (k === 'ArrowDown' ? 1 : k === 'ArrowUp' ? -1 : 0) : (k === 'ArrowRight' ? 1 : k === 'ArrowLeft' ? -1 : 0);
    const dRow = tr ? (k === 'ArrowRight' ? 1 : k === 'ArrowLeft' ? -1 : 0) : (k === 'ArrowDown' ? 1 : k === 'ArrowUp' ? -1 : 0);
    if (k === 'Home') c = 0;
    else if (k === 'End') c = nC - 1;
    else if (dRate || dRow) { c = Math.max(0, Math.min(nC - 1, c + dRate)); r = Math.max(0, Math.min(nR - 1, r + dRow)); }
    else return;
    ev.preventDefault();
    select(r, c, true);
  }

  grid.addEventListener('pointerleave', () => { if (st.hover) { st.hover = null; show(); } });

  function show() {
    const { row, cell } = current();
    if (!row || !cell) return;
    roSrc.set(`${row.info.src} · ${row.info.app}`);
    roEnc.set(row.e.full);
    roTarget.set(`${cell.bitrate_kbps} kb/s`);
    const pct = Math.round((cell.avg_kbps / cell.bitrate_kbps - 1) * 100);
    roActual.set(OX.t(`${nf(cell.avg_kbps)} kb/s（${pct >= 0 ? '+' : '−'}${Math.abs(pct)}%）`, `${nf(cell.avg_kbps)} kb/s (${pct >= 0 ? '+' : '−'}${Math.abs(pct)}%)`));
    roActual.querySelector('.ox-ro-val').classList.toggle('bad', offTarget(cell));
    roMode.set(OX.t(`${MODEFULL[cell.mode] || cell.mode} · ${BWNAME[cell.bandwidth] || ''} ${cell.bandwidth}`, `${MODEFULL[cell.mode] || cell.mode} · ${BWNAME[cell.bandwidth] || ''} (${cell.bandwidth})`));
    const mc = cell.mode_counts || {};
    const tot = (mc.SILK || 0) + (mc.Hybrid || 0) + (mc.CELT || 0);
    const parts = ['SILK', 'Hybrid', 'CELT'].filter((m) => mc[m] > 0).map((m) => `${MODENAME[m]} ${mc[m]}`);
    roCounts.set(parts.length ? OX.t(`${parts.join(' · ')}（共 ${tot}）`, `${parts.join(' · ')} (${tot} in total)`) : '—');
    roBytes.set(OX.t(`${nf(cell.avg_bytes)} 字节`, `${nf(cell.avg_bytes)} bytes`));
    cv.redraw();
  }

  build();
  show();
  OX.onTheme(() => { paint(); });
  let ro = null;
  if (window.ResizeObserver) { ro = new ResizeObserver(() => build()); ro.observe(body); }
  else window.addEventListener('resize', build);
  // Undo the resize hook when the figure is rebuilt (for example on a language switch).
  return () => { if (ro) ro.disconnect(); else window.removeEventListener('resize', build); };
});
