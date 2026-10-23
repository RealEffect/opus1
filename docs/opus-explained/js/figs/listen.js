/* Figure 15: listen to the original and the opus1 decoded signal, and compare their spectrograms.
 * Real data: speech (16 kHz, Voip; SILK NB at 6 and 9 kb/s, SILK WB at 12 and 16 kb/s, WB packets mostly CELT at 24
 * and 32 kb/s) and synthetic music (48 kHz, Audio; SILK WB at 12 kb/s, CELT FB from 24 kb/s, with Hybrid FB for the
 * first packets at 24 and 48 kb/s) from data/opus-demo-data.js, each encoded by opus1 at six target bitrates. `decoded` is already aligned to `pcm` (the codec delay is removed),
 * so both spectrograms share one time axis and A/B switching keeps the playback position.
 * Optional third lane: the spectral difference original − decoded in dB, smoothed a little so that the
 * frame-to-frame speckle of noise-like content does not hide the systematic differences.
 */
OX.fig('listen', function (root, body) {
  const { el, segmented, button, readout } = OX.ui;
  const D = OX.data.raw;
  const ok = (M) => M && Array.isArray(M.encodes) && M.encodes.length && M.pcm && M.sample_rate;
  if (!D || !ok(D.speech) || !ok(D.music)) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js（或其中的语音、音乐数据），这张图无法显示。', 'The data file data/opus-demo-data.js (or the speech or music data in it) is missing, so this figure cannot be shown.') }));
    return;
  }
  const MATS = {
    speech: {
      M: D.speech, label: OX.t('语音（16 kHz）', 'Speech (16 kHz)'), nfft: 512, minDb: -100, maxDb: -20, diffFloor: -90, def: 2,
      ticks: [0, 2000, 4000, 6000, 8000],
    },
    music: {
      M: D.music, label: OX.t('音乐（48 kHz）', 'Music (48 kHz)'), nfft: 1024, minDb: -100, maxDb: -20, diffFloor: -80, def: 1,
      ticks: [0, 4000, 8000, 12000, 16000, 20000, 24000],
    },
  };
  const DIFF_RANGE = 20; // dB at full color in the difference lane
  const BWNAME = { NB: OX.t('窄带', 'narrowband'), MB: OX.t('中带', 'mediumband'), WB: OX.t('宽带', 'wideband'), SWB: OX.t('超宽带', 'super-wideband'), FB: OX.t('全带', 'fullband') };
  const MODEFULL = { SILK: 'SILK-only', Hybrid: OX.t('混合模式（Hybrid）', 'Hybrid'), CELT: 'CELT-only' }; // readouts
  const MODENAME = { SILK: 'SILK', Hybrid: OX.t('混合', 'Hybrid'), CELT: 'CELT' }; // tight places and inside 用…模式
  const modeBw = (m, bw) => (m === 'Hybrid' ? OX.t('混合', 'hybrid ') : m + ' ') + (BWNAME[bw] || bw || ''); // "混合全带", "SILK 宽带"
  const st = { mat: 'speech', idx: { speech: MATS.speech.def, music: MATS.music.def }, diff: false, pos: 0, handle: null, src: null, sr: 16000 };
  let themeStamp = 0, base = null, baseKey = '';

  const enc = (mat, i) => MATS[mat].M.encodes[i];
  const cur = () => enc(st.mat, st.idx[st.mat]);
  const pcm = (mat) => OX.data.audio(`listen:${mat}:pcm`, MATS[mat].M.pcm);
  const decAt = (mat, i) => OX.data.audio(`listen:${mat}:dec:${enc(mat, i).bitrate}`, enc(mat, i).decoded);
  const durOf = (mat) => MATS[mat].M.n_samples / MATS[mat].M.sample_rate || pcm(mat).length / MATS[mat].M.sample_rate;

  /* ---------- spectrograms, memoised per (material, bitrate) ---------- */
  // The signal is padded by nfft/2 on both sides so that column c is centred on sample c·hop.
  function stftOf(mat, which) {
    return OX.data.memo(`listen:stft:${mat}:${which}`, () => {
      const C = MATS[mat];
      const i = which === 'orig' ? -1 : C.M.encodes.findIndex((e) => e.bitrate === which);
      const x = which === 'orig' ? pcm(mat) : decAt(mat, i);
      const nfft = C.nfft, hop = nfft / 4;
      // reflect the ends instead of padding with zeros, so the cut at 0 s and at the end does not read as a click
      const half = nfft / 2, xp = new Float32Array(x.length + nfft);
      xp.set(x, half);
      for (let i = 1; i <= half; i++) {
        xp[half - i] = x[Math.min(i, x.length - 1)];
        xp[half + x.length - 1 + i] = x[Math.max(0, x.length - 1 - i)];
      }
      return OX.dsp.stft(xp, { nfft, hop });
    });
  }
  const imgOf = (mat, which) => OX.data.memo(`listen:img:${mat}:${which}`, () =>
    OX.view.spectrogramImage(stftOf(mat, which), { minDb: MATS[mat].minDb, maxDb: MATS[mat].maxDb }));

  /** Smoothed spectral difference (original − decoded) in dB, as an image. Red: decoded has less energy. Blue: more. */
  function diffImgOf(mat, rate) {
    return OX.data.memo(`listen:diff:${mat}:${rate}`, () => {
      const C = MATS[mat];
      const So = stftOf(mat, 'orig'), Sd = stftOf(mat, rate);
      const cols = Math.min(So.cols, Sd.cols), bins = So.bins;
      const fl = C.diffFloor;
      const pw = (S) => {
        const p = new Float32Array(cols * bins);
        for (let i = 0; i < cols * bins; i++) p[i] = Math.pow(10, Math.max(S.data[i], fl) / 10);
        return blur(p, cols, bins, 1, 2);
      };
      const po = pw(So), pd = pw(Sd);
      const c = document.createElement('canvas');
      c.width = cols; c.height = bins;
      const g = c.getContext('2d');
      const img = g.createImageData(cols, bins);
      const C0 = [10, 10, 22], RED = [240, 84, 66], BLUE = [80, 150, 255];
      for (let col = 0; col < cols; col++) {
        for (let k = 0; k < bins; k++) {
          const a = po[col * bins + k], b = pd[col * bins + k];
          const d = 10 * Math.log10(a / b);
          const level = 10 * Math.log10(Math.max(a, b));
          const wgt = Math.max(0, Math.min(1, (level - fl - 3) / 15));
          let t = Math.max(-1, Math.min(1, d / DIFF_RANGE)) * wgt;
          const end = t > 0 ? RED : BLUE;
          t = Math.pow(Math.abs(t), 0.8);
          const p = ((bins - 1 - k) * cols + col) * 4;
          img.data[p] = C0[0] + (end[0] - C0[0]) * t;
          img.data[p + 1] = C0[1] + (end[1] - C0[1]) * t;
          img.data[p + 2] = C0[2] + (end[2] - C0[2]) * t;
          img.data[p + 3] = 255;
        }
      }
      g.putImageData(img, 0, 0);
      return c;
    });
  }
  /** Separable box blur over (cols × bins), radius rc along time and rb along frequency. */
  function blur(p, cols, bins, rc, rb) {
    const t = new Float32Array(p.length), o = new Float32Array(p.length);
    for (let c = 0; c < cols; c++) {
      for (let k = 0; k < bins; k++) {
        let s = 0, n = 0;
        for (let j = Math.max(0, k - rb); j <= Math.min(bins - 1, k + rb); j++) { s += p[c * bins + j]; n++; }
        t[c * bins + k] = s / n;
      }
    }
    for (let c = 0; c < cols; c++) {
      for (let k = 0; k < bins; k++) {
        let s = 0, n = 0;
        for (let j = Math.max(0, c - rc); j <= Math.min(cols - 1, c + rc); j++) { s += t[j * bins + k]; n++; }
        o[c * bins + k] = s / n;
      }
    }
    return o;
  }

  /* ---------- controls ---------- */
  OX.ui.css('listen', `
    .ls-canvas { margin-top: 10px; }
    .ls-canvas canvas { cursor: pointer; }
    .ls-play { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center; margin: 10px 0 2px; }
    .ls-play .ls-gap { width: 6px; }
    .ls-status { font-size: var(--tiny); color: var(--ink-3); margin-left: 4px; }
    .ls-status b { color: var(--accent); font-weight: 600; }
    .ls-scale { display: flex; flex-wrap: wrap; gap: 6px 22px; align-items: center; font-size: var(--tiny); color: var(--ink-2); margin-top: 8px; }
    .ls-bar { display: inline-flex; align-items: center; gap: 6px; font: 400 0.72rem/1 var(--font-mono); color: var(--ink-3); white-space: nowrap; }
    .ls-dscale { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; }
    .ls-dtxt { color: var(--ink-2); }
    .ls-bar i { display: block; width: 110px; height: 8px; border-radius: 2px; }
    .ls-ro .ox-ro-val.bad { color: var(--bad); }
    .ls-ro .ls-txt .ox-ro-val { font-family: var(--font-body); font-size: 0.94rem; }
    .ls-warn { margin: 8px 0 0; font-size: var(--small); line-height: 1.65; color: var(--ink-2); }
    .ls-warn .ls-warn-head { color: var(--bad); }
    .ls-note { margin: 6px 0 0; font-size: var(--small); line-height: 1.65; color: var(--ink-2); }
    @media (max-width: 560px) {
      .ls-bar i { width: 80px; }
      .ls-ro { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px 14px; }
    }
  `);

  const matSeg = segmented({
    id: 'ls-mat', label: OX.t('素材', 'Clip'), value: st.mat,
    options: Object.keys(MATS).map((k) => ({ value: k, label: MATS[k].label })),
    onChange: (v) => setMat(v),
  });
  const rateHolder = el('div');
  const controls = el('div', { class: 'ox-controls' }, [matSeg, rateHolder]);
  let rateSeg = null;
  function buildRateSeg() {
    const M = MATS[st.mat].M;
    rateSeg = segmented({
      id: 'ls-rate-' + st.mat, label: OX.t('目标码率（kb/s）', 'Target bitrate (kb/s)'), value: String(st.idx[st.mat]),
      options: M.encodes.map((e, i) => ({ value: String(i), label: e.bitrate / 1000 + (e.rate_note ? ' *' : '') })),
      onChange: (v) => setRate(+v),
    });
    rateHolder.innerHTML = '';
    rateHolder.appendChild(rateSeg);
  }

  const btnOrig = button({ icon: 'sound', label: OX.t('播放原始', 'Play original'), onClick: () => toggle('orig') });
  const btnDec = button({ icon: 'sound', label: OX.t('播放解码', 'Play decoded'), onClick: () => toggle('dec') });
  const btnAB = button({ label: OX.t('A/B 切换', 'A/B switch'), onClick: switchAB });
  btnAB.title = OX.t('播放时在原始和解码之间切换，位置不变', 'During playback, switch between the original and the decoded signal at the same position');
  const btnDiff = button({ label: OX.t('显示差值', 'Show difference'), onClick: () => { st.diff = !st.diff; btnDiff.setAttribute('aria-pressed', st.diff ? 'true' : 'false'); diffScale.style.display = st.diff ? '' : 'none'; cv.redraw(); } });
  btnDiff.setAttribute('aria-pressed', 'false');
  const status = el('span', { class: 'ls-status', 'aria-live': 'polite' });
  const play = el('div', { class: 'ls-play' }, [btnOrig, btnDec, btnAB, el('span', { class: 'ls-gap' }), btnDiff, status]);

  const host = el('div', { class: 'ls-canvas' });

  // color scales
  const cbar = el('i');
  const cbarLo = el('span'), cbarHi = el('span');
  const stops = [];
  for (let i = 0; i <= 10; i++) { const [r, g, b] = OX.view.cmap(i / 10); stops.push(`rgb(${r | 0},${g | 0},${b | 0}) ${i * 10}%`); }
  cbar.style.background = `linear-gradient(to right, ${stops.join(',')})`;
  const dbar = el('i');
  dbar.style.background = 'linear-gradient(to right, rgb(80,150,255), rgb(10,10,22) 50%, rgb(240,84,66))';
  const diffScale = el('span', { class: 'ls-dscale', style: { display: 'none' } }, [
    el('span', { class: 'ls-bar' }, [OX.t('差值', 'Difference'), el('span', { text: `−${DIFF_RANGE}` }), dbar, el('span', { text: `+${DIFF_RANGE} dB` })]),
    el('span', { class: 'ls-dtxt', text: OX.t('蓝：解码后多出的能量；红：解码后少了的能量', 'Blue: extra energy after decoding. Red: missing energy after decoding.') })]);
  const scale = el('div', { class: 'ls-scale' }, [
    el('span', { class: 'ls-bar' }, [OX.t('能量', 'Energy'), cbarLo, cbar, cbarHi]),
    diffScale,
    el('span', { text: OX.t('点一下频谱图，播放位置就跳到那个时刻', 'Click the spectrogram to move the playback position to that time') }),
  ]);

  const TXT = { cls: 'ls-txt' };
  const roTarget = readout(OX.t('目标码率', 'Target bitrate'), '');
  const roActual = readout(OX.t('实际平均码率', 'Actual average bitrate'), '');
  const roBytes = readout(OX.t('平均每包字节', 'Average bytes per packet'), '');
  const roMode = readout(OX.t('模式 / 带宽', 'Mode / bandwidth'), '', TXT);
  const roRatio = readout(OX.t('压缩倍数', 'Compression ratio'), '');
  const roSnr = readout(OX.t('波形信噪比（仅供参考）', 'Waveform SNR (for reference only)'), '');
  const roLib = readout(OX.t('libopus 同目标码率', 'libopus at the same target bitrate'), '', TXT);
  const ros = el('div', { class: 'ox-readouts ls-ro' }, [roTarget, roActual, roBytes, roMode, roRatio, roSnr, roLib]);
  const warn = el('p', { class: 'ls-warn', role: 'note' });
  const note = el('p', { class: 'ls-note' });

  body.append(controls, play, host, scale, ros, warn, note);

  /* ---------- canvas ---------- */
  const laneH = (w) => (w < 560 ? 104 : 150);
  const LGAP = 10, TOPPAD = 20, AXIS = 26;
  const nLanes = () => (st.diff ? 3 : 2);
  const cvHeight = (w) => TOPPAD + nLanes() * laneH(w) + (nLanes() - 1) * LGAP + AXIS;
  function geom(w) {
    const narrow = w < 560;
    const L = narrow ? 30 : 40, R = 8;
    const dur = durOf(st.mat);
    const xs = OX.view.linear(0, dur, L, w - R);
    const lh = laneH(w);
    const lanes = [];
    for (let i = 0; i < nLanes(); i++) lanes.push({ y: TOPPAD + i * (lh + LGAP), h: lh, kind: i === 0 ? 'orig' : i === 1 ? 'dec' : 'diff' });
    return { narrow, L, R, xs, dur, lanes, lh };
  }

  OX.onTheme(() => { themeStamp++; });
  const cv = OX.view.canvas(host, { height: cvHeight, label: OX.t('原始与解码信号的频谱图，横轴为时间（秒），纵轴为频率（kHz）', 'Spectrograms of the original and decoded signals, with time (s) on the horizontal axis and frequency (kHz) on the vertical axis') }, draw);
  cv.canvas.addEventListener('pointerdown', (ev) => {
    const r = cv.canvas.getBoundingClientRect();
    const g = geom(cv.w);
    const x = ev.clientX - r.left;
    if (x < g.L - 4 || x > cv.w - g.R + 4) return;
    seek(Math.max(0, Math.min(g.dur, g.xs.invert(x))));
  });

  function draw(ctx, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const e = cur();
    const key = [w, h, dpr, st.mat, e.bitrate, st.diff, themeStamp].join('|');
    if (key !== baseKey || !base) {
      base = base || document.createElement('canvas');
      base.width = Math.round(w * dpr); base.height = Math.round(h * dpr);
      const b = base.getContext('2d');
      b.setTransform(dpr, 0, 0, dpr, 0, 0);
      b.clearRect(0, 0, w, h);
      drawBase(b, w, h);
      baseKey = key;
    }
    ctx.drawImage(base, 0, 0, w, h);
    drawCursor(ctx, w, h);
  }

  function drawBase(ctx, w, h) {
    const T = OX.theme;
    const C = MATS[st.mat], M = C.M, sr = M.sample_rate;
    const e = cur();
    const g = geom(w);
    const { L, R, xs, lanes, narrow } = g;
    const pw = w - L - R;
    const nfft = C.nfft, hop = nfft / 4, df = sr / nfft;
    const fmax = sr / 2;
    const titles = {
      orig: OX.t('原始', 'Original'),
      dec: OX.t(`解码 · 目标 ${e.bitrate / 1000} kb/s（实际 ${e.avg_kbps.toFixed(1)} kb/s）`, `Decoded · target ${e.bitrate / 1000} kb/s (actual ${e.avg_kbps.toFixed(1)} kb/s)`),
      diff: OX.t('差值：原始 − 解码（dB，略作平滑）', 'Difference: original − decoded (dB, lightly smoothed)'),
    };
    for (const ln of lanes) {
      const ys = OX.view.linear(0, fmax, ln.y + ln.h, ln.y);
      ctx.fillStyle = 'rgb(6,6,20)';
      ctx.fillRect(L, ln.y, pw, ln.h);
      ctx.save();
      ctx.beginPath(); ctx.rect(L, ln.y, pw, ln.h); ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      const S = stftOf(st.mat, ln.kind === 'orig' ? 'orig' : e.bitrate);
      const img = ln.kind === 'diff' ? diffImgOf(st.mat, e.bitrate) : imgOf(st.mat, ln.kind === 'orig' ? 'orig' : e.bitrate);
      // column c is centred on time c·hop/sr; bin k on frequency k·df
      const x0 = xs((-0.5 * hop) / sr), x1 = xs(((img.width - 0.5) * hop) / sr);
      const yTop = ys((S.bins - 0.5) * df), yBot = ys(-0.5 * df);
      ctx.drawImage(img, x0, yTop, x1 - x0, yBot - yTop);
      // 20 kHz: the top of the fullband; nothing above it is coded
      if (fmax > 20000) {
        const y20 = Math.round(ys(20000)) + 0.5;
        ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(L, y20); ctx.lineTo(L + pw, y20); ctx.stroke();
        ctx.setLineDash([]);
        if (ln.kind === 'dec' && !narrow) {
          ctx.font = OX.font(10, 'body', 500); ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
          ctx.fillText(OX.t('20 kHz 以上不编码', 'Not coded above 20 kHz'), L + pw - 6, y20 - 3);
        }
      }
      ctx.restore();
      // lane title chip
      ctx.font = OX.font(narrow ? 10.5 : 11.5, 'body', 600);
      const tt = narrow && ln.kind === 'dec' ? OX.t(`解码 · ${e.bitrate / 1000} kb/s（实际 ${e.avg_kbps.toFixed(1)}）`, `Decoded · ${e.bitrate / 1000} kb/s (actual ${e.avg_kbps.toFixed(1)})`) : narrow && ln.kind === 'diff' ? OX.t('差值：原始 − 解码', 'Difference: original − decoded') : titles[ln.kind];
      const tw = ctx.measureText(tt).width;
      ctx.fillStyle = 'rgba(6,6,20,0.72)';
      ctx.fillRect(L + 1, ln.y + 1, tw + 14, narrow ? 19 : 21);
      ctx.fillStyle = ln.kind === 'dec' && e.rate_note ? '#ff9b8f' : '#ffffff';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(tt, L + 8, ln.y + 1 + (narrow ? 9.5 : 10.5));
      // frame
      ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
      ctx.strokeRect(L + 0.5, ln.y + 0.5, pw - 1, ln.h - 1);
      // frequency axis
      let ticks = C.ticks;
      if ((ln.h / (ticks.length - 1)) < 18) ticks = ticks.filter((_, i) => i % 2 === 0);
      // the 0 label is left out so that it does not collide with the top label of the lane below
      OX.view.axis(ctx, ys, { side: 'left', at: L, ticks: ticks.map((v) => ({ v, label: v === 0 ? null : String(v / 1000) })), fontSize: narrow ? 9.5 : 10 });
    }
    // unit of the frequency axis
    ctx.font = OX.font(narrow ? 9.5 : 10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('kHz', L - 6, 7);
    // time axis
    const last = lanes[lanes.length - 1];
    const yAx = last.y + last.h;
    // ticks every 0.5 s; on narrow screens only whole seconds get a label
    const tks = [];
    for (let t = 0; t <= g.dur + 1e-9; t += 0.5) tks.push(t);
    const every = (xs(0.5) - xs(0)) < 56 ? 1 : 0.5;
    const fmtT = (t) => (every === 1 ? String(t) : t === 0 ? '0' : t.toFixed(1));
    OX.view.axis(ctx, xs, {
      side: 'bottom', at: yAx,
      ticks: tks.map((t, i) => {
        const lastT = i === tks.length - 1;
        const show = lastT || Math.abs(t / every - Math.round(t / every)) < 1e-6;
        // the last tick sits on the right edge: right-align it; drop a whole-second label that would touch it
        const crowded = !lastT && (xs(tks[tks.length - 1]) - xs(t)) < 40;
        return { v: t, label: show && !crowded ? (lastT ? `${fmtT(t)} s` : fmtT(t)) : null, align: lastT ? 'right' : i === 0 ? 'left' : 'center' };
      }),
      fontSize: narrow ? 9.5 : 10,
    });
  }

  function drawCursor(ctx, w) {
    const T = OX.theme;
    const g = geom(w);
    const playing = !!st.handle;
    // which lane is being heard
    if (playing) {
      const ln = g.lanes[st.src === 'orig' ? 0 : 1];
      ctx.strokeStyle = T.accent; ctx.lineWidth = 2;
      ctx.strokeRect(g.L + 1, ln.y + 1, w - g.L - g.R - 2, ln.h - 2);
    }
    if (!playing && st.pos <= 0) return;
    const x = Math.round(g.xs(st.pos)) + 0.5;
    const yTop = g.lanes[0].y, yBot = g.lanes[g.lanes.length - 1].y + g.lh;
    ctx.strokeStyle = '#ffffff'; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = T.accent;
    ctx.beginPath(); ctx.moveTo(x - 5, yTop - 7); ctx.lineTo(x + 5, yTop - 7); ctx.lineTo(x, yTop - 1); ctx.closePath(); ctx.fill();
  }

  /* ---------- playback ---------- */
  const loop = OX.loop(host, () => {
    if (!st.handle) { cv.redraw(); return false; }
    st.pos = st.handle.position() / st.sr;
    cv.redraw();
  });
  const nowSec = () => (st.handle ? st.handle.position() / st.sr : st.pos);

  function startAt(src, sec) {
    const C = MATS[st.mat], sr = C.M.sample_rate;
    const x = src === 'orig' ? pcm(st.mat) : decAt(st.mat, st.idx[st.mat]);
    if (!x) return;
    let from = Math.max(0, Math.round(sec * sr));
    if (from >= x.length - Math.round(0.05 * sr)) from = 0;
    st.handle = null; // the old handle's onEnd (fired inside play) must not count as "playback ended"
    const h = OX.audio.play(x, sr, { from, onEnd: () => { if (st.handle === h) ended(h, x.length); } });
    if (!h) { update(); return; }
    st.handle = h; st.src = src; st.sr = sr;
    loop.start();
    updatePlay();
  }
  function ended(h, len) {
    const p = h.position();
    st.handle = null;
    st.pos = p >= len - 1 ? 0 : p / st.sr; // natural end rewinds; a stop from elsewhere keeps the position
    updatePlay();
    cv.redraw();
  }
  function stopPlay() {
    const h = st.handle;
    if (!h) return;
    st.pos = h.position() / st.sr;
    st.handle = null;
    h.stop();
    updatePlay();
    cv.redraw();
  }
  function toggle(src) {
    if (st.handle && st.src === src) stopPlay();
    else startAt(src, nowSec());
  }
  function switchAB() {
    if (!st.handle) return;
    startAt(st.src === 'orig' ? 'dec' : 'orig', nowSec());
  }
  function seek(t) {
    if (st.handle) startAt(st.src, t);
    else { st.pos = t; cv.redraw(); }
  }
  function updatePlay() {
    const p = !!st.handle;
    const po = p && st.src === 'orig', pd = p && st.src === 'dec';
    btnOrig.setIcon(po ? 'stop' : 'sound', po ? OX.t('停止', 'Stop') : OX.t('播放原始', 'Play original'));
    btnDec.setIcon(pd ? 'stop' : 'sound', pd ? OX.t('停止', 'Stop') : OX.t('播放解码', 'Play decoded'));
    btnOrig.setAttribute('aria-pressed', po ? 'true' : 'false');
    btnDec.setAttribute('aria-pressed', pd ? 'true' : 'false');
    btnAB.disabled = !p;
    status.innerHTML = p ? OX.t(`正在听：<b>${po ? '原始' : '解码 · ' + cur().bitrate / 1000 + ' kb/s'}</b>`, `Now playing: <b>${po ? 'original' : 'decoded · ' + cur().bitrate / 1000 + ' kb/s'}</b>`) : OX.t('A/B 切换在播放时可用', 'A/B switch works during playback');
  }

  /* ---------- state changes ---------- */
  function setMat(m) {
    if (m === st.mat) return;
    stopPlay();
    st.mat = m; st.pos = 0;
    buildRateSeg();
    update();
  }
  function setRate(i) {
    st.idx[st.mat] = i;
    if (st.handle && st.src === 'dec') startAt('dec', nowSec());
    update();
  }

  function update() {
    const C = MATS[st.mat], M = C.M, e = cur();
    const sr = M.sample_rate;
    cbarLo.textContent = `${C.minDb}`.replace('-', '−');
    cbarHi.textContent = `${C.maxDb} dB`.replace('-', '−');
    roTarget.set(`${e.bitrate / 1000} kb/s`);
    const pct = Math.round((e.avg_kbps / (e.bitrate / 1000) - 1) * 100);
    roActual.set(`${e.avg_kbps.toFixed(1)} kb/s` + (e.rate_note ? OX.t(`（+${pct}%）`, ` (+${pct}%)`) : ''));
    roActual.querySelector('.ox-ro-val').classList.toggle('bad', !!e.rate_note);
    roBytes.set(OX.t(`${e.avg_bytes.toFixed(1)} 字节`, `${e.avg_bytes.toFixed(1)} bytes`));
    const mc = e.mode_counts || {};
    const modes = ['SILK', 'Hybrid', 'CELT'].filter((m) => mc[m] > 0);
    const modeTxt = OX.t(`${MODEFULL[e.mode] || e.mode} · ${BWNAME[e.bandwidth] || ''} ${e.bandwidth}`, `${MODEFULL[e.mode] || e.mode} · ${BWNAME[e.bandwidth] || ''} (${e.bandwidth})`);
    roMode.set(modes.length > 1 ? OX.t(`${modeTxt}（${modes.map((m) => `${MODENAME[m]} ${mc[m]}`).join('，')}）`, `${modeTxt}; packets: ${modes.map((m) => `${MODENAME[m]} ${mc[m]}`).join(', ')}`) : modeTxt);
    const pcmKbps = (sr * 16) / 1000;
    roRatio.set(OX.t(`${(pcmKbps / e.avg_kbps).toFixed(1)} 倍`, `${(pcmKbps / e.avg_kbps).toFixed(1)}×`));
    roRatio.querySelector('.ox-ro-label').textContent = OX.t(`压缩倍数（相对 ${sr / 1000} kHz、16 位 PCM 的 ${pcmKbps} kb/s）`, `Compression ratio (vs. 16-bit PCM at ${sr / 1000} kHz, ${pcmKbps} kb/s)`);
    roSnr.set(`${e.snr_db.toFixed(1)} dB`);
    const lr = e.libopus_reference;
    roLib.set(lr ? OX.t(`${MODEFULL[lr.mode] || lr.mode} · ${BWNAME[lr.bandwidth] || ''} ${lr.bandwidth}，${lr.avg_kbps.toFixed(1)} kb/s`, `${MODEFULL[lr.mode] || lr.mode} · ${BWNAME[lr.bandwidth] || ''} (${lr.bandwidth}), ${lr.avg_kbps.toFixed(1)} kb/s`) : '—');
    warn.innerHTML = '';
    warn.style.display = e.rate_note ? '' : 'none';
    if (e.rate_note) {
      const head = OX.t(`* 目标是 ${e.bitrate / 1000} kb/s，但 opus1 在这里用了${e.mode === 'Hybrid' ? '' : ' '}${modeBw(e.mode, e.bandwidth)}模式，实际平均 ${e.avg_kbps.toFixed(1)} kb/s，比目标高 ${pct}%。`, `* The target is ${e.bitrate / 1000} kb/s, but opus1 uses ${modeBw(e.mode, e.bandwidth)} mode here, at an actual average of ${e.avg_kbps.toFixed(1)} kb/s, ${pct}% above the target. `);
      let why = lr ? OX.t(`libopus 在同一目标码率下用${lr.mode === 'Hybrid' ? '' : ' '}${modeBw(lr.mode, lr.bandwidth)}，实际 ${lr.avg_kbps.toFixed(1)} kb/s。`, `At the same target bitrate, libopus uses ${modeBw(lr.mode, lr.bandwidth)} and averages ${lr.avg_kbps.toFixed(1)} kb/s. `) : '';
      warn.append(el('span', { class: 'ls-warn-head', text: head }), why);
    }
    const NOTE_20K = OX.t('虚线标出 20 kHz。Opus 不编码 20 kHz 以上的声音，所以解码结果在虚线以上是空的。', 'The dashed line marks 20 kHz. Opus does not code sound above 20 kHz, so the decoded spectrogram is empty above the line. ');
    note.textContent = st.mat === 'speech'
      ? OX.t('语音素材是电话频带的录音，原始信号在约 3.8 kHz 以上几乎没有能量，所以原始频谱图的上半部分是暗的。这是素材的原因，不是编解码器造成的。', 'The speech clip is a telephone-band recording: the original has almost no energy above about 3.8 kHz, so the upper half of the original spectrogram is dark. This comes from the recording, not from the codec. ')
        + (e.bandwidth === 'NB'
          ? OX.t('这一档是窄带，只编码到 4 kHz，所以解码频谱图在 4 kHz 以上几乎是空的。', 'This bitrate uses narrowband, which codes only up to 4 kHz, so the decoded spectrogram is almost empty above 4 kHz.')
          : OX.t('这一档是宽带，一直编码到 8 kHz。解码频谱图在约 3.8–8 kHz 的微弱颜色是编码带来的噪声，比低频的语音能量低约 50 dB（12 kb/s）到 60 dB（32 kb/s）。码率越高，这层噪声越弱。', 'This bitrate uses wideband, which codes all the way up to 8 kHz. The faint color at about 3.8–8 kHz in the decoded spectrogram is coding noise. It is about 50 dB (12 kb/s) to 60 dB (32 kb/s) below the speech energy at low frequencies. The higher the bitrate, the weaker this noise.'))
      : NOTE_20K + (e.mode === 'SILK'
        ? OX.t('这一档只用 SILK 宽带：只编码 8 kHz 以下，所以解码频谱图在 8 kHz 以上几乎是空的。', 'This bitrate uses SILK-only wideband, which codes only the band below 8 kHz, so the decoded spectrogram is almost empty above 8 kHz.')
        : OX.t('码率越低，CELT 分给高频频带的比特越少，这些频带里的细节就越多地用折叠或噪声填充的近似内容代替，但每个频带的能量仍然接近原来。', 'The lower the bitrate, the fewer bits CELT gives the high bands, and the more of their detail is replaced by an approximation made by folding or noise filling. The energy of each band still stays close to the original.'));
    updatePlay();
    cv.redraw();
  }

  buildRateSeg();
  update();
});
