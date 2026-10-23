/* Figure 13: inverse MDCT and overlap-add (time-domain aliasing cancellation, TDAC).
 * Two overlapping frames A and B of a smooth test signal. Each frame alone comes back from MDCT -> IMDCT with
 * time-domain aliasing (red); in the overlap the two aliasing terms have opposite signs, so A + B equals the input.
 * The window can be a sine window (50% overlap) or a CELT-style low-overlap window (flat top, short tapers).
 */
OX.fig('tdac', function (root, body) {
  const { el, button, segmented, readout } = OX.ui;
  const N = 64; // MDCT size: N coefficients, 2N samples per frame
  const L = 3 * N; // samples shown: frame A = [0, 2N), frame B = [N, 3N)
  const OV_LOW = N / 4; // taper length of the low-overlap window
  const STEPS = [
    OX.t(`原信号被切成两帧：帧 A 和帧 B 各 2N 个样本（这里 N = ${N}），中间重叠。每帧先乘一个窗（虚线）。`, `The signal is cut into two frames, A and B, each 2N samples long (here N = ${N}) and overlapping in the middle. Each frame is first multiplied by a window (dashed).`),
    OX.t('帧 A 做 MDCT，得到 N 个系数；再做逆 MDCT，乘同一个窗。结果等于原信号乘了两次窗，再加上红色的部分。红色部分就是时域混叠：窗口每一半里的信号前后翻转后叠加上来的部分。', "Frame A goes through the MDCT, which gives N coefficients, then the inverse MDCT (IMDCT), and is multiplied by the same window. The result is the signal multiplied by the window twice, plus the red part. The red part is the time-domain aliasing: the signal in each half of the window, reversed in time and added on top."),
    OX.t('帧 B 也一样带着时域混叠。注意重叠区：B 的时域混叠和 A 的形状相同、符号相反。', "Frame B also comes back with time-domain aliasing. Look at the overlap: B's aliasing has the same shape as A's, but the opposite sign."),
    OX.t('把两帧在重叠区相加：时域混叠互相抵消，两个窗的平方和为 1，得到的正好是原信号。', 'Add the two frames in the overlap: the time-domain aliasing cancels, and the squares of the two windows sum to 1, so the result is exactly the original signal.'),
  ];
  let step = 0, winKind = 'sine', anim = 1;

  const x = new Float32Array(L);
  for (let n = 0; n < L; n++) {
    const env = 0.75 + 0.25 * Math.sin((2 * Math.PI * n) / (L * 1.3) + 0.6);
    x[n] = env * (0.62 * Math.sin((2 * Math.PI * n) / 41 + 0.3) + 0.3 * Math.sin((2 * Math.PI * n) / 11.5 + 1.1));
  }

  function makeWindow(kind) {
    const w = new Float32Array(2 * N);
    if (kind === 'sine') {
      for (let n = 0; n < 2 * N; n++) w[n] = Math.sin((Math.PI * (n + 0.5)) / (2 * N));
      return w;
    }
    // Low overlap, as in CELT: zeros, a short power-complementary taper, a flat top, the mirrored taper, zeros.
    const ov = OV_LOW, z = (N - ov) / 2, taper = OX.dsp.celtWindow(ov);
    for (let n = 0; n < 2 * N; n++) {
      if (n < z) w[n] = 0;
      else if (n < z + ov) w[n] = taper[n - z];
      else if (n < z + ov + (N - ov)) w[n] = 1;
      else if (n < z + 2 * ov + (N - ov)) w[n] = taper[ov - 1 - (n - (z + ov + (N - ov)))];
      else w[n] = 0;
    }
    return w;
  }
  /** Samples [b0, b1) where both windows are nonzero: all of [N, 2N) for the sine window, only the taper for low overlap. */
  function overlapBand(kind) {
    if (kind !== 'low') return [N, 2 * N];
    const z = (N - OV_LOW) / 2;
    return [N + z, N + z + OV_LOW];
  }

  let model;
  function compute() {
    const w = makeWindow(winKind);
    const frame = (off) => {
      const seg = new Float32Array(2 * N);
      for (let i = 0; i < 2 * N; i++) seg[i] = x[off + i] * w[i];
      const y = OX.dsp.imdct(OX.dsp.mdct(seg, N), N);
      const out = new Float32Array(2 * N), alias = new Float32Array(2 * N);
      for (let i = 0; i < 2 * N; i++) { out[i] = y[i] * w[i]; alias[i] = out[i] - x[off + i] * w[i] * w[i]; }
      return { out, alias };
    };
    const A = frame(0), B = frame(N);
    const sum = new Float32Array(L);
    for (let i = 0; i < 2 * N; i++) { sum[i] += A.out[i]; sum[N + i] += B.out[i]; }
    const band = overlapBand(winKind);
    let maxErr = 0;
    for (let n = band[0]; n < band[1]; n++) maxErr = Math.max(maxErr, Math.abs(sum[n] - x[n]));
    model = { w, A, B, sum, band, maxErr };
  }
  compute();

  /** 6.0e-8 -> "6.0×10⁻⁸" */
  function sci(v) {
    if (!(v > 0)) return '0';
    let e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e);
    if (+m.toFixed(1) >= 10) { m /= 10; e += 1; }
    const sup = String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
    return `${m.toFixed(1)}×10${sup}`;
  }

  const caption = el('p', { class: 'tdac-step', 'aria-live': 'polite' });
  const host = el('div');
  const prev = button({ label: OX.t('上一步', 'Previous'), onClick: () => { stopAuto(); go(step - 1); } });
  const next = button({ icon: 'step', label: OX.t('下一步', 'Next'), kind: 'primary', onClick: () => { stopAuto(); go(step + 1); } });
  const auto = button({ icon: 'play', label: OX.t('自动播放', 'Autoplay'), onClick: toggleAuto });
  const roErr = readout(OX.t('重叠区最大误差', 'Largest error in the overlap'), '');
  const dots = el('div', { class: 'tdac-dots' }, STEPS.map((_, i) => el('button', { type: 'button', class: 'tdac-dot', 'aria-label': OX.t(`第 ${i + 1} 步`, `Step ${i + 1}`), onclick: () => { stopAuto(); go(i); } })));
  OX.ui.css('tdac', `
    .tdac-step { margin: 2px 0 10px; min-height: 3.4em; font-size: var(--small); color: var(--ink); line-height: 1.7; }
    .tdac-step b { font-family: var(--font-mono); color: var(--accent); margin-right: 6px; }
    .tdac-dots { display: inline-flex; gap: 6px; align-items: center; }
    .tdac-dot { width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid var(--ink-3); background: transparent; padding: 0; cursor: pointer; }
    .tdac-dot.on { background: var(--accent); border-color: var(--accent); }
    @media (max-width: 560px) { .tdac-step { min-height: 5.1em; } }
    html[data-lang="en"] .tdac-step { min-height: 5.1em; }
    @media (max-width: 560px) { html[data-lang="en"] .tdac-step { min-height: 6.8em; } }
  `);
  body.append(
    el('div', { class: 'ox-controls' }, [
      segmented({ id: 'tdac-win', label: OX.t('窗', 'Window'), value: winKind, options: [{ value: 'sine', label: OX.t('正弦窗（重叠一半）', 'Sine (50% overlap)') }, { value: 'low', label: OX.t('低重叠窗（CELT 的做法）', 'Low overlap (as in CELT)') }], onChange: (v) => { winKind = v; compute(); update(); } }),
    ]),
    caption, host,
    el('div', { class: 'ox-controls' }, [prev, next, auto, dots, roErr]),
    OX.ui.legend([
      { label: OX.t('原信号', 'Original signal'), color: 'var(--ink)' },
      { label: OX.t('帧 A 的窗', 'Frame A window'), color: 'var(--accent)', dash: true },
      { label: OX.t('帧 B 的窗', 'Frame B window'), color: 'var(--celt)', dash: true },
      { label: OX.t('帧 A 的输出', 'Frame A output'), color: 'var(--accent)' },
      { label: OX.t('帧 B 的输出', 'Frame B output'), color: 'var(--celt)' },
      { label: OX.t('时域混叠', 'Time-domain aliasing'), color: 'var(--bad)', kind: 'box' },
    ]));

  // An English lane name may hold a line break: two lines in the wide left margin, one line on narrow screens.
  const LANES = [OX.t('原信号与窗', 'Original signal\nand windows'), OX.t('帧 A：逆变换输出', 'Frame A:\nIMDCT output'), OX.t('帧 B：逆变换输出', 'Frame B:\nIMDCT output'), OX.t('A + B 重叠相加', 'A + B:\noverlap-add')];
  const cv = OX.view.canvas(host, { height: (w) => (w < 560 ? 360 : 420), label: OX.t('逆 MDCT 与重叠相加的四个步骤：单独一帧带着时域混叠，两帧在重叠区相加后时域混叠抵消', 'The four steps of the inverse MDCT and overlap-add: a single frame comes back with time-domain aliasing; adding the two frames in the overlap cancels it') }, draw);

  function draw(ctx, w, h) {
    const T = OX.theme;
    const narrow = w < 560;
    const left = narrow ? 8 : 112, right = 8, top = 6, laneGap = 10;
    const laneH = (h - top - 22 - laneGap * 3) / 4;
    const xs = OX.view.linear(0, L - 1, left, w - right);
    const lanes = LANES.map((name, i) => {
      const y0 = top + i * (laneH + laneGap);
      return { name, y0, y1: y0 + laneH, ys: OX.view.linear(-1.25, 1.25, y0 + laneH - 4, y0 + 4) };
    });
    const vis = (i) => (i <= step ? (i === step ? anim : 1) : 0.12);

    // overlap region band: where both windows are nonzero
    const [b0, b1] = model.band;
    ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.08 : 0.05);
    ctx.fillRect(xs(b0), top, xs(b1) - xs(b0), h - top - 22);
    ctx.font = OX.font(10.5, 'body', 500); ctx.fillStyle = T.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(OX.t('重叠区', 'Overlap'), (xs(b0) + xs(b1)) / 2, h - 18);
    ctx.textAlign = 'left'; ctx.fillText(OX.t('只有帧 A', 'Frame A only'), xs(0) + 2, h - 18);
    ctx.textAlign = 'right'; ctx.fillText(OX.t('只有帧 B', 'Frame B only'), xs(L - 1) - 2, h - 18);

    const line = (arr, off, color, lw, ys, alpha, dash) => {
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) { const X = xs(off + i), Y = ys(arr[i]); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.stroke(); ctx.restore();
    };
    const fillAlias = (arr, off, ys, alpha) => {
      ctx.save(); ctx.globalAlpha = alpha * 0.85; ctx.fillStyle = OX.alpha(T.bad, 0.5);
      ctx.beginPath(); ctx.moveTo(xs(off), ys(0));
      for (let i = 0; i < arr.length; i++) ctx.lineTo(xs(off + i), ys(arr[i]));
      ctx.lineTo(xs(off + arr.length - 1), ys(0)); ctx.closePath(); ctx.fill(); ctx.restore();
    };

    lanes.forEach((ln, i) => {
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, Math.round(ln.ys(0)) + 0.5); ctx.lineTo(w - right, Math.round(ln.ys(0)) + 0.5); ctx.stroke();
      ctx.globalAlpha = Math.max(0.35, vis(i));
      ctx.font = OX.font(narrow ? 10.5 : 11.5, 'body', 600); ctx.fillStyle = i <= step ? T.ink : T.ink3;
      const parts = ln.name.split('\n');
      if (narrow) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(parts.join(' '), left + 2, ln.y0); }
      else {
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        parts.forEach((p, k) => ctx.fillText(p, left - 12, (ln.y0 + ln.y1) / 2 + (k - (parts.length - 1) / 2) * 15));
      }
      ctx.globalAlpha = 1;
    });

    const { w: win, A, B, sum } = model;
    // lane 0: signal and the two windows
    const l0 = lanes[0];
    line(x, 0, T.ink, 1.4, l0.ys, vis(0));
    line(win, 0, T.accent, 1.2, l0.ys, vis(0) * 0.9, [4, 3]);
    line(win, N, T.celt, 1.2, l0.ys, vis(0) * 0.9, [4, 3]);
    // lane 1: frame A
    const l1 = lanes[1];
    fillAlias(A.alias, 0, l1.ys, vis(1));
    line(A.out, 0, T.accent, 1.6, l1.ys, vis(1));
    // lane 2: frame B
    const l2 = lanes[2];
    fillAlias(B.alias, N, l2.ys, vis(2));
    line(B.out, N, T.celt, 1.6, l2.ys, vis(2));
    // lane 3: sum. Only [N, 2N] has both frames.
    const l3 = lanes[3];
    const a3 = vis(3);
    line(x, 0, T.ink, 3.2, l3.ys, a3 * 0.18);
    line(sum.subarray(N, 2 * N + 1), N, T.ink, 1.6, l3.ys, a3);
    line(A.out.subarray(0, N + 1), 0, T.accent, 1.2, l3.ys, a3 * 0.6, [3, 3]);
    line(B.out.subarray(N - 1), 2 * N - 1, T.celt, 1.2, l3.ys, a3 * 0.6, [3, 3]);
  }

  let timer = 0;
  function go(s) {
    step = Math.max(0, Math.min(STEPS.length - 1, s));
    if (OX.reducedMotion()) anim = 1;
    else { anim = 0; fade.start(); }
    update();
  }
  const fade = OX.loop(host, (dt) => { anim = Math.min(1, anim + dt * 2.5); cv.redraw(); return anim < 1; });
  function stopAuto() {
    if (!timer) return;
    clearInterval(timer); timer = 0; auto.setIcon('play', OX.t('自动播放', 'Autoplay'));
  }
  function toggleAuto() {
    if (timer) { stopAuto(); return; }
    auto.setIcon('pause', OX.t('暂停', 'Pause'));
    if (step === STEPS.length - 1) go(0);
    timer = setInterval(() => {
      if (step >= STEPS.length - 1) { stopAuto(); return; }
      go(step + 1);
    }, 2600);
  }
  function update() {
    caption.innerHTML = `<b>${step + 1}/${STEPS.length}</b>${STEPS[step]}`;
    const atStart = step === 0, atEnd = step === STEPS.length - 1;
    // move focus off a step button before it becomes disabled, so keyboard focus is not dropped to <body>
    if (atEnd && document.activeElement === next) prev.focus();
    if (atStart && document.activeElement === prev) next.focus();
    prev.disabled = atStart; next.disabled = atEnd;
    [...dots.children].forEach((d, i) => {
      d.classList.toggle('on', i === step);
      if (i === step) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
    });
    roErr.set(atEnd ? sci(model.maxErr) : '—');
    cv.redraw();
  }
  // Start on the last step so the page shows the result at rest; the reader can step back through it.
  step = STEPS.length - 1;
  update();
  return () => { if (timer) clearInterval(timer); timer = 0; };
});
