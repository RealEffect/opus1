/* Figure 14: packet loss concealment (PLC) on the real 16 kb/s SILK stream.
 * speech.plc.decoded is the same packets decoded with six packets dropped (frames 12, 44-45, 100-102): for each
 * lost packet the decoder got an empty payload and had to produce 20 ms on its own. Top: the whole 3 s with the
 * lost packets in red. Below: a zoom on one loss event with the original, the normal decode and the decode with
 * loss. Every number in the readouts is computed here from those three aligned signals.
 */
OX.fig('plc', function (root, body) {
  const { el, segmented, button, readout } = OX.ui;
  const D = OX.data.raw;
  const sp = D && D.speech;
  const P = sp && sp.plc;
  const enc = sp && (sp.encodes || []).find((e) => P && e.bitrate === P.bitrate);
  if (!P || !P.decoded || !enc || !sp.pcm) {
    body.appendChild(el('p', { class: 'fig-error', text: OX.t('缺少数据文件 data/opus-demo-data.js（或其中的 speech.plc），这张图无法显示。', 'The data file data/opus-demo-data.js (or its speech.plc part) is missing, so this figure cannot be shown.') }));
    return;
  }

  const FS = sp.sample_rate; // 16000
  const N = sp.frame_samples; // 320 samples = one 20 ms packet
  const NS = sp.n_samples;
  const pcm = OX.data.audio('speech.pcm', sp.pcm);
  const normal = OX.data.audio('speech.encodes.' + enc.bitrate + '.decoded', enc.decoded);
  const lossy = OX.data.audio('speech.plc.decoded', P.decoded);
  const delay = enc.delay_samples; // output packet k covers aligned samples [N*k - delay, N*k - delay + N)
  // detail view: 70 ms either side of the lost region (40 ms on narrow screens)
  const viewOf = (e, w) => {
    const mg = Math.round((w < 560 ? 0.04 : 0.07) * FS);
    return [Math.max(0, e.start - mg), Math.min(NS, e.end + mg)];
  };
  const PLAY_MARGIN = Math.round(0.4 * FS);

  /* Group consecutive lost packets into events. */
  const events = [];
  P.lost_frames.forEach((k, i) => {
    const s = P.lost_start_samples[i];
    const last = events[events.length - 1];
    if (last && k === last.frames[last.frames.length - 1] + 1) { last.frames.push(k); last.starts.push(s); last.end = s + N; }
    else events.push({ frames: [k], starts: [s], start: s, end: s + N });
  });
  const LABELS = { 1: OX.t('丢 1 个包', '1 packet lost'), 2: OX.t('连丢 2 个', '2 lost in a row'), 3: OX.t('连丢 3 个', '3 lost in a row') };
  events.forEach((e, i) => {
    e.id = String(i);
    e.label = LABELS[e.frames.length] || OX.t(`连丢 ${e.frames.length} 个`, `${e.frames.length} lost in a row`);
  });

  const sumSq = (x, a, b) => { let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return s; };
  const errSq = (x, y, a, b) => { let s = 0; for (let i = a; i < b; i++) { const d = x[i] - y[i]; s += d * d; } return s; };
  const snr = (ref, test, a, b) => 10 * Math.log10((sumSq(ref, a, b) + 1e-20) / (errSq(ref, test, a, b) + 1e-20));
  /** Numbers for one event, from the aligned signals. */
  function stats(e) {
    return OX.data.memo('plc:stats:' + e.id, () => {
      const lostSnr = snr(normal, lossy, e.start, e.end);
      const rmsRatio = e.starts.map((s) => Math.sqrt(sumSq(lossy, s, s + N) / Math.max(1e-20, sumSq(normal, s, s + N))));
      // packets after the loss until the decode with loss is back within -20 dB of the normal decode
      let after = 0;
      for (let s = e.end; s + N <= NS; s += N, after++) if (snr(normal, lossy, s, s + N) >= 20) break;
      return { lostSnr, rmsRatio, after };
    });
  }

  let cur = events.length > 2 ? events[2] : events[events.length - 1];
  let range = 'event'; // or 'all'
  let playing = null; // { which, handle }

  /* ---------------- DOM ---------------- */
  OX.ui.css('plc', `
    .plc-ov canvas { cursor: pointer; }
    .plc-detail { margin-top: 10px; }
    .plc-say { margin: 8px 0 0; font-size: var(--small); color: var(--ink); line-height: 1.7; max-width: 52rem; }
    .plc-play { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  `);
  const evCtl = segmented({ id: 'plc-ev', label: OX.t('丢包事件', 'Loss event'), value: cur.id, options: events.map((e) => ({ value: e.id, label: e.label })), onChange: (v) => setEvent(events[+v]) });
  const ovHost = el('div', { class: 'plc-ov' });
  const dtHost = el('div', { class: 'plc-detail' });
  const ro = {
    dur: readout(OX.t('丢失时长', 'Duration lost'), ''),
    snr: readout(OX.t('丢失区域的信噪比（以正常解码为参照）', 'Signal-to-noise ratio (SNR) in the lost region, against the normal decode'), ''),
    rms: readout(OX.t('补出信号的均方根幅度（RMS），相对正常解码', 'Root-mean-square (RMS) level of the concealed signal, relative to the normal decode'), ''),
    rec: readout(OX.t('丢包后恢复', 'Recovery after the loss'), ''),
  };
  const say = el('p', { class: 'plc-say', 'aria-live': 'polite' });
  const SOURCES = [
    { key: 'orig', label: OX.t('原始', 'Original'), x: pcm },
    { key: 'normal', label: OX.t('正常解码', 'Normal decode'), x: normal },
    { key: 'lossy', label: OX.t('丢包后解码', 'Decode with loss'), x: lossy },
  ];
  const playBtns = SOURCES.map((s) => {
    const b = button({ icon: 'sound', label: s.label, onClick: () => togglePlay(s) });
    b.setAttribute('aria-pressed', 'false');
    return b;
  });
  const rangeCtl = segmented({ id: 'plc-range', label: OX.t('范围', 'Range'), value: range, options: [{ value: 'event', label: OX.t('这次丢包 ±0.4 s', 'This loss ±0.4 s') }, { value: 'all', label: OX.t('整段 3 s', 'All 3 s') }], onChange: (v) => { range = v; stopPlay(); } });
  body.append(
    el('div', { class: 'ox-controls' }, [evCtl]),
    ovHost, dtHost,
    OX.ui.legend([
      { label: OX.t('原始语音', 'Original speech'), color: 'color-mix(in srgb, var(--ink) 30%, transparent)' },
      { label: OX.t('正常解码（没有丢包）', 'Normal decode (no loss)'), color: 'var(--ink)', dash: true },
      { label: OX.t('丢包后的解码', 'Decode with loss'), color: 'var(--accent)' },
      { label: OX.t('丢掉的包', 'Lost packets'), color: 'color-mix(in srgb, var(--bad) 30%, transparent)', kind: 'box' },
    ]),
    el('div', { class: 'ox-readouts' }, Object.values(ro)),
    say,
    el('div', { class: 'ox-controls' }, [el('div', { class: 'plc-play' }, [el('span', { class: 'ox-ctl-label', text: OX.t('试听', 'Listen') }), ...playBtns]), rangeCtl]),
    el('p', { class: 'ox-note', text: OX.t(`这段码流全是 SILK（${enc.bitrate / 1000} kb/s，宽带）。“丢包”的做法是把空数据交给 opus1 的解码器，让它自己补出这 20 毫秒。帧号从 0 数起。整段 3 秒里，丢包后的解码相对原始语音的信噪比是 ${P.snr_db.toFixed(1)} dB，正常解码是 ${enc.snr_db.toFixed(1)} dB。`, `This stream is all SILK (${enc.bitrate / 1000} kb/s, wideband). A packet is “lost” by handing the opus1 decoder an empty payload, so that it has to produce those 20 ms on its own. Frames are numbered from 0. Over the whole 3 s, the decode with loss has a signal-to-noise ratio of ${P.snr_db.toFixed(1)} dB against the original speech; the normal decode has ${enc.snr_db.toFixed(1)} dB.`) }),
    el('p', { class: 'ox-note', text: OX.t('用 libopus 1.6.1 解码同一串包、丢掉同样的包，补出的每个样本都和 opus1 相同，每次丢包后的第一个正常包也一样。', 'Decoding the same packets with libopus 1.6.1, with the same packets lost, gives exactly the same concealed samples as opus1, and the same first good packet after each loss.') }),
  );

  /* ---------------- overview ---------------- */
  const ov = OX.view.canvas(ovHost, { height: (w) => (w < 560 ? 84 : 96), label: OX.t('3 秒语音的波形，红色是丢掉的包；点一下选择一次丢包', 'Waveform of the 3 s of speech, with the lost packets in red; click or tap near a loss to select it') }, drawOverview);
  ov.canvas.addEventListener('pointerdown', (e) => {
    const r = ov.canvas.getBoundingClientRect();
    const s = ((e.clientX - r.left) / ov.w) * NS;
    let best = cur, bd = Infinity;
    for (const ev of events) { const [v0, v1] = viewOf(ev, ov.w); const d = s < v0 ? v0 - s : s > v1 ? s - v1 : 0; if (d < bd) { bd = d; best = ev; } }
    setEvent(best);
  });

  function cursorAt() {
    if (!playing || !playing.handle || !playing.handle.playing) return null;
    return playing.handle.position();
  }

  function drawOverview(ctx, w, h) {
    const T = OX.theme;
    const xs = OX.view.linear(0, NS, 0, w);
    const y0 = 4, y1 = h - 22;
    const mid = (y0 + y1) / 2, half = (y1 - y0) / 2;
    let pk = OX.data.memo('plc:peak', () => { let m = 1e-6; for (let i = 0; i < NS; i++) m = Math.max(m, Math.abs(pcm[i]), Math.abs(lossy[i])); return m; });
    const ys = (v) => mid - (v / pk) * half;
    // zoom window
    const [v0, v1] = viewOf(cur, w);
    ctx.fillStyle = OX.alpha(T.ink, T.dark ? 0.07 : 0.045);
    ctx.fillRect(xs(v0), y0 - 2, xs(v1) - xs(v0), y1 - y0 + 4);
    // lost packets
    for (const e of events) for (const s of e.starts) {
      ctx.fillStyle = OX.alpha(T.bad, e === cur ? 0.32 : 0.2);
      ctx.fillRect(xs(s), y0 - 2, Math.max(2, xs(s + N) - xs(s)), y1 - y0 + 4);
    }
    OX.view.wave(ctx, pcm, 0, NS, xs, ys, OX.alpha(T.ink, T.dark ? 0.3 : 0.25), 1);
    OX.view.wave(ctx, lossy, 0, NS, xs, ys, OX.alpha(T.accent, 0.85), 1);
    ctx.strokeStyle = T.ink3; ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(xs(v0)) + 0.5, y0 - 2 + 0.5, Math.round(xs(v1) - xs(v0)), Math.round(y1 - y0 + 3));
    // time axis
    ctx.font = OX.font(10.5, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
    const step = w < 560 ? 1 : 0.5, dur = NS / FS;
    for (let t = 0; t <= dur + 1e-9; t += step) {
      ctx.textAlign = t === 0 ? 'left' : t >= dur - 1e-9 ? 'right' : 'center';
      ctx.fillText(t === 0 ? '0' : t.toFixed(step < 1 ? 1 : 0) + (t >= dur - 1e-9 ? ' s' : ''), xs(t * FS), h - 15);
    }
    const c = cursorAt();
    if (c != null) {
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(Math.round(xs(c)) + 0.5, y0 - 2); ctx.lineTo(Math.round(xs(c)) + 0.5, y1 + 2); ctx.stroke();
    }
  }

  /* ---------------- detail ---------------- */
  const dt = OX.view.canvas(dtHost, { height: (w) => (w < 560 ? 230 : 270), label: OX.t('选中的丢包前后的波形：原始、正常解码和丢包后的解码', 'Waveforms around the selected loss: original, normal decode, and decode with loss') }, drawDetail);

  function drawDetail(ctx, w, h) {
    const T = OX.theme;
    const narrow = w < 560;
    const ML = narrow ? 34 : 40;
    const x0 = ML, x1 = w, yTop = 26, yBot = h - 32;
    const [a, b] = viewOf(cur, w);
    const xs = OX.view.linear(a, b, x0, x1);
    let m = 1e-4;
    for (let i = a; i < b; i++) m = Math.max(m, Math.abs(pcm[i]), Math.abs(normal[i]), Math.abs(lossy[i]));
    const ys = OX.view.linear(-m * 1.06, m * 1.06, yBot, yTop);
    // frame (packet) boundaries
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
    const k0 = Math.ceil((a + delay) / N), k1 = Math.floor((b + delay) / N);
    for (let k = k0; k <= k1; k++) { const x = Math.round(xs(k * N - delay)) + 0.5; ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke(); }
    // lost region
    ctx.fillStyle = OX.alpha(T.bad, T.dark ? 0.2 : 0.13);
    ctx.fillRect(xs(cur.start), yTop, xs(cur.end) - xs(cur.start), yBot - yTop);
    ctx.strokeStyle = OX.alpha(T.bad, 0.7); ctx.lineWidth = 1;
    for (const s of cur.starts.concat([cur.end])) { const x = Math.round(xs(s)) + 0.5; ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke(); }
    // label of the lost region
    const n = cur.frames.length;
    const lbl = n === 1 ? OX.t(`丢掉第 ${cur.frames[0]} 帧的包（20 ms）`, `Lost packet: frame ${cur.frames[0]} (20 ms)`) : OX.t(`丢掉第 ${cur.frames[0]}–${cur.frames[n - 1]} 帧的包（${n * 20} ms）`, `Lost packets: frames ${cur.frames[0]}–${cur.frames[n - 1]} (${n * 20} ms)`);
    ctx.font = OX.font(narrow ? 11 : 11.5, 'body', 600); ctx.fillStyle = T.bad; ctx.textBaseline = 'alphabetic';
    const cx = (xs(cur.start) + xs(cur.end)) / 2;
    const tw = ctx.measureText(lbl).width;
    ctx.textAlign = 'center';
    ctx.fillText(lbl, Math.max(x0 + tw / 2, Math.min(x1 - tw / 2, cx)), yTop - 9);
    // amplitude grid and labels
    const st = niceStep(m, 2);
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of [-st, 0, st]) {
      const y = Math.round(ys(v)) + 0.5;
      ctx.strokeStyle = v === 0 ? T.rule : T.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText(v === 0 ? '0' : fmtAmp(v), x0 - 6, y);
    }
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, yTop + 0.5, x1 - x0 - 1, yBot - yTop);
    // signals
    ctx.save(); ctx.beginPath(); ctx.rect(x0, yTop, x1 - x0, yBot - yTop); ctx.clip();
    poly(ctx, pcm, a, b, xs, ys, OX.alpha(T.ink, T.dark ? 0.32 : 0.26), 1.6);
    ctx.setLineDash([3, 2.5]);
    poly(ctx, normal, a, b, xs, ys, OX.alpha(T.ink, T.dark ? 0.8 : 0.7), 1);
    ctx.setLineDash([]);
    poly(ctx, lossy, a, b, xs, ys, T.accent, 1.5);
    const c = cursorAt();
    if (c != null && c >= a && c <= b) {
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(Math.round(xs(c)) + 0.5, yTop); ctx.lineTo(Math.round(xs(c)) + 0.5, yBot); ctx.stroke();
    }
    ctx.restore();
    // time axis: ms from the start of the loss
    const ms = (s) => ((s - cur.start) / FS) * 1000;
    const tick = narrow ? 40 : 20;
    const ticks = [];
    for (let t = Math.ceil(ms(a) / tick) * tick; t <= ms(b) + 1e-9; t += tick) {
      const v = cur.start + (t / 1000) * FS, X = xs(v);
      ticks.push({ v, label: t === 0 ? '0' : (t > 0 ? '+' : '−') + Math.abs(t), align: X - x0 < 14 ? 'left' : x1 - X < 14 ? 'right' : 'center' });
    }
    OX.view.axis(ctx, xs, { side: 'bottom', at: yBot, ticks, fontSize: 10 });
    ctx.font = OX.font(10, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText(narrow ? OX.t(`ms，0 = ${(cur.start / FS).toFixed(3)} s`, `ms, 0 = ${(cur.start / FS).toFixed(3)} s`) : OX.t(`毫秒，0 = 丢包开始的时刻（${(cur.start / FS).toFixed(3)} s）`, `ms from the start of the loss (at ${(cur.start / FS).toFixed(3)} s)`), x1, yBot + 19);
  }

  function poly(ctx, x, a, b, xs, ys, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = a; i < b; i++) { const X = xs(i), Y = ys(x[i]); if (i === a) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
    ctx.stroke();
  }
  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const q = raw / p;
    return (q < 1.5 ? 1 : q < 3.5 ? 2 : q < 7.5 ? 5 : 10) * p;
  }
  function fmtAmp(v) {
    const s = Math.abs(v) >= 0.1 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toFixed(3);
    return s.replace('-', '−');
  }

  /* ---------------- readouts ---------------- */
  const pct = (r) => Math.round(r * 100) + '%';
  function update() {
    const s = stats(cur);
    const n = cur.frames.length;
    ro.dur.set(OX.t(`${n} 个包 · ${n * 20} ms`, `${n} ${n === 1 ? 'packet' : 'packets'} · ${n * 20} ms`));
    ro.snr.set(`${s.lostSnr.toFixed(1)} dB`);
    ro.rms.set(s.rmsRatio.map(pct).join(' → '));
    ro.rec.set(s.after === 0 ? OX.t('下一个包就恢复', 'At the next packet') : OX.t(`再过 ${s.after} 个包（${s.after * 20} ms）`, `${s.after} ${s.after === 1 ? 'packet' : 'packets'} later (${s.after * 20} ms)`));
    const fade = n >= 2 && s.rmsRatio[n - 1] < s.rmsRatio[0];
    say.textContent = OX.t('补出的波形和正常解码对不上（所以信噪比接近 0 dB），解码器只是沿用前面的音高和频谱包络接着合成', 'The concealed waveform does not match the normal decode (hence an SNR near 0 dB): the decoder only keeps synthesizing with the earlier pitch and spectral envelope')
      + (fade ? OX.t('；连续丢包时，补出的声音一帧比一帧弱。', '. When packets are lost in a row, the concealed sound gets weaker with every frame.') : OX.t('。', '.'));
  }

  function setEvent(e) {
    if (!e || e === cur) return;
    cur = e;
    evCtl.set(e.id);
    if (playing && range === 'event') stopPlay();
    update();
    ov.redraw(); dt.redraw();
  }

  /* ---------------- playback ---------------- */
  const loop = OX.loop(dtHost, () => {
    ov.redraw(); dt.redraw();
    if (!playing) return false;
    return true;
  });
  function stopPlay() {
    if (playing && playing.handle) playing.handle.stop();
  }
  function togglePlay(src) {
    if (playing && playing.key === src.key) { stopPlay(); return; }
    const from = range === 'event' ? Math.max(0, cur.start - PLAY_MARGIN) : 0;
    const to = range === 'event' ? Math.min(NS, cur.end + PLAY_MARGIN) : NS;
    const me = { key: src.key, handle: null };
    playing = me;
    me.handle = OX.audio.play(src.x, FS, {
      from, to,
      onEnd: () => {
        if (playing === me) playing = null;
        playBtns.forEach((b, i) => { if (SOURCES[i].key === src.key) { b.setAttribute('aria-pressed', 'false'); b.setIcon('sound'); } });
        ov.redraw(); dt.redraw();
      },
    });
    if (!me.handle) { playing = null; return; }
    playBtns.forEach((b, i) => {
      const on = SOURCES[i].key === src.key;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.setIcon(on ? 'stop' : 'sound');
    });
    loop.start();
  }

  update();
});
