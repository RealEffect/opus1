/* Figure 1: sampling and quantization.
 * A pure tone, its samples at the chosen rate, the samples rounded to the chosen bit depth, and the band-limited
 * curve an ideal DAC would rebuild from those samples (sinc interpolation). Above fs/2 that curve is the alias.
 * The quantizer is symmetric with no zero level ("mid-rise"): 2^b levels at ±0.5/L … ±(L−0.5)/L, L = 2^(b−1).
 * So the 0.9-amplitude tone never overloads, the error stays within ±half a step at every bit depth, and the
 * 6.02·b + 1.76 dB readout matches what is drawn. A two's-complement quantizer (with a 0 level) would clip the
 * positive peaks at 2 and 3 bits.
 */
OX.fig('sampling', function (root, body) {
  const { el, slider, segmented, readout, button, fmt } = OX.ui;
  const st = { f: 1000, fs: 16000, bits: 4 };
  const AMP = 0.9;

  const fToSlider = (f) => Math.log(f / 50) / Math.log(24000 / 50) * 1000;
  const sliderToF = (v) => Math.round(50 * Math.pow(24000 / 50, v / 1000) / 10) * 10;

  const alias = (f, fs) => { let a = f % fs; if (a > fs / 2) a = fs - a; return a; };

  const controls = el('div', { class: 'ox-controls' }, [
    slider({ id: 'smp-f', label: OX.t('信号频率', 'Signal frequency'), min: 0, max: 1000, step: 1, value: fToSlider(st.f), format: (v) => fmt.hz(sliderToF(v)), onInput: (v) => { st.f = sliderToF(v); update(); } }),
    segmented({ id: 'smp-fs', label: OX.t('采样率', 'Sample rate'), value: String(st.fs), options: [{ value: '8000', label: '8 kHz' }, { value: '16000', label: '16 kHz' }, { value: '48000', label: '48 kHz' }], onChange: (v) => { st.fs = +v; update(); } }),
    slider({ id: 'smp-bits', label: OX.t('位深', 'Bit depth'), min: 2, max: 16, step: 1, value: st.bits, format: (v) => OX.t(`${v} 位`, `${v} bits`), onInput: (v) => { st.bits = v; update(); } }),
  ]);
  const host = el('div');
  const host2 = el('div', { style: { marginTop: '6px' } });
  const roNyq = readout(OX.t('奈奎斯特频率 fs/2', 'Nyquist frequency fs/2'), '');
  const roHear = readout(OX.t('采样后还原出的频率', 'Frequency reconstructed from the samples'), '');
  const roSnr = readout(OX.t('理论量化信噪比（满幅正弦波）', 'Theoretical quantization signal-to-noise ratio (full-scale sine wave)'), '');
  const roRate = readout(OX.t('PCM 码率（单声道）', 'PCM bitrate (mono)'), '');
  const play1 = button({ icon: 'sound', label: OX.t('听原始正弦波', 'Play the original sine wave'), onClick: () => playTone(false) });
  const play2 = button({ icon: 'sound', label: OX.t('听采样、量化之后', 'Play it sampled and quantized'), onClick: () => playTone(true) });
  body.append(controls, host, host2,
    el('div', { class: 'ox-readouts' }, [roNyq, roHear, roSnr, roRate]),
    el('div', { class: 'ox-controls' }, [play1, play2]),
    OX.ui.legend([
      { label: OX.t('原始声波', 'Original sound wave'), color: 'var(--ink)' },
      { label: OX.t('量化后的采样点', 'Quantized samples'), color: 'var(--accent)', kind: 'box' },
      { label: OX.t('由采样点还原出的波形', 'Waveform reconstructed from the samples'), color: 'var(--accent)', dash: true },
      { label: OX.t('量化误差', 'Quantization error'), color: 'var(--bad)', kind: 'box' },
    ]));

  function windowSec() {
    const a = alias(st.f, st.fs);
    const fv = Math.max(60, Math.min(st.f, a || st.f));
    return Math.min(0.01, Math.max(0.0006, 4 / fv));
  }
  // Mid-rise quantizer: level k (−L … L−1) sits at (k + 0.5) / L; one step is 1 / L = 2 / 2^bits.
  const q = (v) => {
    const L = Math.pow(2, st.bits - 1);
    const k = Math.max(-L, Math.min(L - 1, Math.floor(v * L)));
    return (k + 0.5) / L;
  };
  const x = (t) => AMP * Math.sin(2 * Math.PI * st.f * t + 0.35);

  function geom(w, h) {
    const left = w < 520 ? 34 : 44, right = 10, top = 12, bottom = h - 38;
    return { left, right, top, bottom, xs: OX.view.linear(0, windowSec(), left, w - right), ys: OX.view.linear(-1.05, 1.05, bottom, top) };
  }

  const main = OX.view.canvas(host, { height: (w) => (w < 560 ? 210 : 260), label: OX.t('正弦波、采样点和还原波形', 'Sine wave, samples, and reconstructed waveform') }, (ctx, w, h) => {
    const T = OX.theme;
    const g = geom(w, h);
    const W = windowSec(), fs = st.fs, Ts = 1 / fs;
    const levels = Math.pow(2, st.bits);
    // quantization levels as faint lines when they are few enough to see
    if (levels <= 64) {
      ctx.strokeStyle = OX.alpha(T.accent, T.dark ? 0.14 : 0.1);
      ctx.lineWidth = 1;
      const L = levels / 2;
      for (let k = -L; k < L; k++) { const y = Math.round(g.ys((k + 0.5) / L)) + 0.5; ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(w - g.right, y); ctx.stroke(); }
    }
    // axes
    const ticks = [];
    const stepMs = niceStep(W * 1000 / 5);
    for (let v = 0; v <= W * 1000 + 1e-9; v += stepMs) ticks.push({ v: v / 1000, label: (+v.toFixed(3)) + '' });
    OX.view.axis(ctx, g.xs, { side: 'bottom', at: g.bottom, ticks, title: OX.t('时间 ms', 'Time (ms)'), titleX: w - g.right });
    OX.view.axis(ctx, g.ys, { side: 'left', at: g.left, ticks: [{ v: -1, label: '−1' }, { v: 0, label: '0' }, { v: 1, label: '1' }] });
    ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.left, Math.round(g.ys(0)) + 0.5); ctx.lineTo(w - g.right, Math.round(g.ys(0)) + 0.5); ctx.stroke();

    ctx.save();
    ctx.beginPath(); ctx.rect(g.left, 0, w - g.left - g.right, h); ctx.clip();
    // original tone
    ctx.strokeStyle = T.ink; ctx.lineWidth = 1.2; ctx.beginPath();
    const px = w - g.left - g.right;
    const per = Math.max(0.25, Math.min(1, px / (W * st.f * 40)));
    for (let p = 0; p <= px; p += per) { const t = g.xs.invert(g.left + p); const y = g.ys(x(t)); if (p === 0) ctx.moveTo(g.left + p, y); else ctx.lineTo(g.left + p, y); }
    ctx.stroke();

    // samples, quantized
    const n0 = -48, n1 = Math.ceil(W / Ts) + 48;
    const smp = [];
    for (let n = n0; n <= n1; n++) smp.push(q(x(n * Ts)));
    // band-limited reconstruction (sinc interpolation of the quantized samples)
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath();
    for (let p = 0; p <= px; p += 1) {
      const t = g.xs.invert(g.left + p), u = t * fs;
      let s = 0;
      const c = Math.round(u);
      for (let n = c - 40; n <= c + 40; n++) {
        const v = smp[n - n0]; if (v === undefined) continue;
        const d = u - n;
        const sinc = Math.abs(d) < 1e-9 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d);
        const win = 0.5 + 0.5 * Math.cos((Math.PI * d) / 41);
        s += v * sinc * win;
      }
      const y = g.ys(s); if (p === 0) ctx.moveTo(g.left + p, y); else ctx.lineTo(g.left + p, y);
    }
    ctx.stroke(); ctx.setLineDash([]);
    // stems and dots
    const pxPerSample = px / (W * fs);
    for (let n = 0; n * Ts <= W; n++) {
      const X = g.xs(n * Ts), v = smp[n - n0], Y = g.ys(v);
      if (pxPerSample >= 5) { ctx.strokeStyle = OX.alpha(T.accent, 0.45); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X, g.ys(0)); ctx.lineTo(X, Y); ctx.stroke(); }
      ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(X, Y, pxPerSample >= 5 ? 2.8 : 1.8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    const a = alias(st.f, st.fs);
    if (st.f > st.fs / 2) {
      const msg = OX.t(`发生混叠：${fmt.hz(st.f)} 被记录成了 ${fmt.hz(a)}`, `Aliasing: ${fmt.hz(st.f)} is recorded as ${fmt.hz(a)}`);
      ctx.font = OX.font(12, 'body', 600); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const tw = ctx.measureText(msg).width;
      ctx.fillStyle = OX.alpha(T.surface, 0.92);
      ctx.fillRect(w - g.right - tw - 12, g.top - 2, tw + 12, 22);
      ctx.fillStyle = T.bad;
      ctx.fillText(msg, w - g.right - 6, g.top + 9);
    }
  });

  // Half a quantization step is exactly 1 / 2^bits of full scale: it doubles for every bit removed.
  function halfStepText() {
    const half = 1 / Math.pow(2, st.bits), hv = half.toPrecision(2);
    return `±1/${Math.pow(2, st.bits)} ${+hv === half ? '=' : '≈'} ±${hv}`;
  }

  const errCv = OX.view.canvas(host2, { height: (w) => (w < 560 ? 92 : 84), label: OX.t('每个采样点的量化误差', 'Quantization error of each sample') }, (ctx, w, h) => {
    const T = OX.theme;
    const g = geom(w, h + 38);
    const W = windowSec(), Ts = 1 / st.fs;
    const step = 2 / Math.pow(2, st.bits); // one quantization step in full-scale units
    // The title says that the y range follows the step size (so the bars keep their height) and gives the real
    // size of half a step. One line when it fits, else two.
    ctx.font = OX.font(11, 'body', 500);
    const one = OX.t(`量化误差（纵轴随台阶大小缩放；浅红带是 ±半个量化台阶 = ${halfStepText()}）`,
      `Quantization error (the y-axis scales with the step size; the light red band is ±half a quantization step = ${halfStepText()})`);
    const lines = ctx.measureText(one).width <= w - g.left - g.right ? [one]
      : [OX.t('量化误差（纵轴随台阶大小缩放）', 'Quantization error (y-axis scales with the step size)'),
        OX.t(`浅红带是 ±半个量化台阶 = ${halfStepText()}`, `Light red band: ±half a step = ${halfStepText()}`)];
    const top = lines.length * 15 + 2;
    const ys = OX.view.linear(-step * 0.75, step * 0.75, h - 8, top);
    ctx.fillStyle = OX.alpha(T.bad, 0.07);
    ctx.fillRect(g.left, ys(step / 2), w - g.left - g.right, ys(-step / 2) - ys(step / 2));
    ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(g.left, Math.round(ys(0)) + 0.5); ctx.lineTo(w - g.right, Math.round(ys(0)) + 0.5); ctx.stroke();
    const px = w - g.left - g.right, pxPerSample = px / (W * st.fs);
    const bw = Math.max(1, Math.min(6, pxPerSample * 0.6));
    for (let n = 0; n * Ts <= W; n++) {
      const t = n * Ts, e = q(x(t)) - x(t);
      const X = g.xs(t);
      ctx.fillStyle = T.bad;
      const y0 = ys(0), y1 = ys(e);
      ctx.fillRect(X - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0)));
    }
    ctx.fillStyle = T.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    lines.forEach((s, i) => ctx.fillText(s, g.left, i * 15));
  });

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }

  function update() {
    const a = alias(st.f, st.fs);
    roNyq.set(fmt.hz(st.fs / 2));
    roHear.set(st.f > st.fs / 2 ? OX.t(`${fmt.hz(a)}（混叠）`, `${fmt.hz(a)} (aliased)`) : fmt.hz(st.f));
    roSnr.set(`≈ ${(6.02 * st.bits + 1.76).toFixed(1)} dB`);
    roRate.set(fmt.kbps(st.fs * st.bits));
    main.canvas.setAttribute('aria-label', OX.t(`${fmt.hz(st.f)} 正弦波，以 ${fmt.hz(st.fs)} 采样、${st.bits} 位量化后的采样点和还原波形`,
      `${fmt.hz(st.f)} sine wave, its samples taken at ${fmt.hz(st.fs)} and quantized to ${st.bits} bits, and the reconstructed waveform`) +
      (st.f > st.fs / 2 ? OX.t(`；发生混叠，还原出的是 ${fmt.hz(a)}`, `; aliasing: the reconstructed tone is ${fmt.hz(a)}`) : ''));
    errCv.canvas.setAttribute('aria-label', OX.t(`每个采样点的量化误差，都在 ±半个量化台阶（${halfStepText()}）以内`,
      `Quantization error of each sample, always within ±half a quantization step (${halfStepText()})`));
    main.redraw(); errCv.redraw();
  }

  function playTone(sampled) {
    const dur = 0.9;
    if (!sampled) {
      const sr = 48000, n = Math.round(sr * dur), y = new Float32Array(n);
      for (let i = 0; i < n; i++) y[i] = 0.25 * Math.sin(2 * Math.PI * st.f * i / sr) * fade(i, n, sr);
      OX.audio.play(y, sr);
    } else {
      // The quantized samples at the chosen rate; the browser's own resampler plays the role of the DAC.
      const sr = st.fs, n = Math.round(sr * dur), y = new Float32Array(n);
      for (let i = 0; i < n; i++) y[i] = 0.25 * q(AMP * Math.sin(2 * Math.PI * st.f * i / sr)) / AMP * fade(i, n, sr);
      OX.audio.play(y, sr);
    }
  }
  const fade = (i, n, sr) => Math.min(1, i / (0.02 * sr), (n - i) / (0.02 * sr));

  update();
});
