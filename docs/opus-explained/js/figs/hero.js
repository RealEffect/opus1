/* Hero: the whole signal path as one conveyor.
 * A synthetic voiced signal moves left to right. Each zone draws the same stretch of signal the way that stage
 * holds it: a continuous wave, samples, per-frame spectra (the encoder's parameters), packets on the network,
 * spectra again in the decoder, and finally a continuous wave.
 */
OX.fig('hero', function (root, body) {
  const { el } = OX.ui;
  // English text is wider than Chinese. `short` replaces `label` when the label is wider than its zone, and a
  // `sub` with a line break is drawn on two lines when it does not fit on one. The Chinese strings have no line
  // break and `short` equals `label`, so they are drawn as before.
  const ZONES = [
    { key: 'wave', label: OX.t('声波', 'Sound wave'), short: OX.t('声波', 'Sound'), sub: OX.t('空气压强连续变化', 'Air pressure varies\ncontinuously'), w: 1.15 },
    { key: 'samp', label: OX.t('采样 · 量化', 'Sample · quantize'), short: OX.t('采样 · 量化', 'Sample'), sub: OX.t('48 000 次/秒，16 位', '48 000 samples/s,\n16 bits'), w: 1.0 },
    { key: 'enc', label: OX.t('编码', 'Encode'), short: OX.t('编码', 'Encode'), sub: OX.t('变换或预测，得到参数', 'Transform or predict\n→ parameters'), w: 1.05 },
    { key: 'net', label: OX.t('传输', 'Transmit'), short: OX.t('传输', 'Transmit'), sub: OX.t('每 20 毫秒一个包', 'One packet\nevery 20 ms'), w: 1.25 },
    { key: 'dec', label: OX.t('解码', 'Decode'), short: OX.t('解码', 'Decode'), sub: OX.t('参数还原成样本', 'Parameters back\nto samples'), w: 1.05 },
    { key: 'out', label: OX.t('声波', 'Sound wave'), short: OX.t('声波', 'Sound'), sub: OX.t('重建的声音', 'Reconstructed\nsound'), w: 1.15 },
  ];
  const SPEED = 34; // px per second of animation
  const FRAME = 1.25; // seconds of animation time per codec frame (purely visual)
  let T = 18; // animation clock in seconds

  // Synthetic "speech": a few harmonics under a moving formant, with syllable-like loudness.
  const F0 = 0.95; // cycles per animation second
  const loud = (t) => {
    const syl = Math.max(0, Math.sin((2 * Math.PI * t) / 9.5));
    const gap = 0.5 + 0.5 * Math.sin((2 * Math.PI * t) / 23 + 1.3);
    return 0.12 + 0.88 * Math.pow(syl, 0.7) * (0.45 + 0.55 * gap);
  };
  const formant = (t) => 2.2 + 1.6 * Math.sin((2 * Math.PI * t) / 13 + 0.4);
  const harm = (h, t) => Math.exp(-Math.pow(h - formant(t), 2) / 2.2) / Math.sqrt(h);
  const sig = (t) => {
    let s = 0;
    for (let h = 1; h <= 6; h++) s += harm(h, t) * Math.sin(2 * Math.PI * h * F0 * t + h * 0.7);
    return 0.62 * loud(t) * s;
  };
  // Per-frame parameters, as the encoder would send them: 8 "band energies" and a packet size.
  const frameParams = (n) => {
    const tc = (n + 0.5) * FRAME;
    const L = loud(tc);
    const bands = [];
    for (let h = 1; h <= 8; h++) bands.push(Math.min(1, Math.sqrt((0.06 + harm(h, tc) * 1.5) * L) * 0.95 + 0.05));
    return { bands, bytes: Math.round(6 + 52 * L) };
  };

  const host = el('div', { class: 'hero-stage' });
  body.appendChild(host);
  const pauseBtn = OX.ui.button({ icon: 'pause', label: OX.t('暂停', 'Pause'), kind: 'ghost', onClick: toggle });
  pauseBtn.classList.add('hero-pause');
  body.appendChild(pauseBtn);
  OX.ui.css('hero', `
    .hero-stage { position: relative; }
    .hero-pause { position: absolute; right: -6px; bottom: -32px; font-size: 0.72rem; padding: 5px 9px; }
    #fig-hero { padding-bottom: 40px; }
  `);

  const cv = OX.view.canvas(host, { height: (w) => (w < 560 ? 200 : 230), label: OX.t('Opus 信号通路：声波、采样、编码、传输、解码、声波', 'Opus signal path: sound wave, sampling, encoding, transmission, decoding, sound wave') }, draw);

  function layout(w) {
    const total = ZONES.reduce((a, z) => a + z.w, 0);
    let x = 0;
    return ZONES.map((z) => { const zw = (z.w / total) * w; const r = { ...z, x0: x, x1: x + zw }; x += zw; return r; });
  }

  function draw(ctx, w, h) {
    const Th = OX.theme;
    const narrow = w < 560;
    const zones = layout(w);
    const top = narrow ? 26 : 30, bottom = h - (narrow ? 22 : 34);
    const mid = (top + bottom) / 2, amp = (bottom - top) * 0.36;
    const tOf = (x) => T - x / SPEED; // signal time at pixel x (right = older)
    const xOf = (t) => (T - t) * SPEED;

    // zone frames and labels
    ctx.font = OX.font(narrow ? 11 : 12.5, 'body', 600);
    ctx.textBaseline = 'alphabetic';
    zones.forEach((z, i) => {
      if (i > 0) {
        ctx.strokeStyle = Th.rule; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(Math.round(z.x0) + 0.5, top - 6); ctx.lineTo(Math.round(z.x0) + 0.5, bottom + 6); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = z.key === 'enc' || z.key === 'dec' ? Th.accent : Th.ink;
      ctx.textAlign = 'center';
      const zw = z.x1 - z.x0;
      ctx.fillText(ctx.measureText(z.label).width <= zw - 4 ? z.label : z.short, (z.x0 + z.x1) / 2, top - 12);
      if (!narrow) {
        ctx.fillStyle = Th.ink3; ctx.font = OX.font(11, 'body', 400);
        const one = z.sub.replace('\n', ' ');
        if (!z.sub.includes('\n') || ctx.measureText(one).width <= zw - 12) ctx.fillText(one, (z.x0 + z.x1) / 2, h - 10);
        else z.sub.split('\n').forEach((s, j) => ctx.fillText(s, (z.x0 + z.x1) / 2, h - 17 + j * 13));
        ctx.font = OX.font(12.5, 'body', 600);
      }
    });
    // light background behind the encoder and decoder zones
    ctx.fillStyle = OX.alpha(Th.accent, Th.dark ? 0.08 : 0.05);
    for (const z of [zones[2], zones[4]]) ctx.fillRect(z.x0, top - 4, z.x1 - z.x0, bottom - top + 8);

    const clip = (z, fn) => { ctx.save(); ctx.beginPath(); ctx.rect(z.x0 + 1, top - 4, z.x1 - z.x0 - 2, bottom - top + 8); ctx.clip(); fn(); ctx.restore(); };

    // 1. analog wave
    const smooth = (z, color, width) => clip(z, () => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let x = z.x0; x <= z.x1; x += 1) { const y = mid - amp * sig(tOf(x)); if (x === z.x0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
    });
    smooth(zones[0], Th.ink, 1.6);

    // 2. samples (fixed on the tape, so they travel with the wave)
    clip(zones[1], () => {
      const dt = 1 / (SPEED / (narrow ? 5 : 6.5));
      const levels = 7;
      const n0 = Math.ceil(tOf(zones[1].x1) / dt), n1 = Math.floor(tOf(zones[1].x0) / dt);
      ctx.strokeStyle = OX.alpha(Th.ink3, 0.5); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(zones[1].x0, mid); ctx.lineTo(zones[1].x1, mid); ctx.stroke(); ctx.setLineDash([]);
      for (let n = n0; n <= n1; n++) {
        const t = n * dt, x = xOf(t);
        const q = Math.round((sig(t) / 0.75) * levels) / levels * 0.75;
        const y = mid - amp * q;
        ctx.strokeStyle = Th.ink2; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x, y); ctx.stroke();
        ctx.fillStyle = Th.ink; ctx.beginPath(); ctx.arc(x, y, narrow ? 1.8 : 2.2, 0, Math.PI * 2); ctx.fill();
      }
    });

    // frames that are inside zones 2..4
    const nOld = Math.floor(tOf(zones[4].x1) / FRAME) - 1, nNew = Math.ceil(tOf(zones[2].x0) / FRAME) + 1;
    const blockY0 = top + 6, blockY1 = bottom - 6;

    const spectrumBlock = (n, color) => {
      const x1 = xOf(n * FRAME), x0 = xOf((n + 1) * FRAME);
      const p = frameParams(n);
      const pad = 3, bw = (x1 - x0 - 2 * pad) / p.bands.length;
      ctx.strokeStyle = OX.alpha(color, 0.35); ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(x0 + 1) + 0.5, blockY0 + 0.5, Math.round(x1 - x0 - 2), blockY1 - blockY0);
      p.bands.forEach((b, i) => {
        const bh = (blockY1 - blockY0 - 8) * b;
        ctx.fillStyle = OX.alpha(color, 0.25 + 0.75 * b);
        ctx.fillRect(x0 + pad + i * bw + 0.5, blockY1 - 3 - bh, Math.max(1, bw - 1.2), bh);
      });
    };
    clip(zones[2], () => { for (let n = nOld; n <= nNew; n++) spectrumBlock(n, Th.accent); });
    clip(zones[4], () => { for (let n = nOld; n <= nNew; n++) spectrumBlock(n, Th.accent); });

    // 4. packets on the network
    clip(zones[3], () => {
      ctx.strokeStyle = Th.rule; ctx.lineWidth = 1; ctx.setLineDash([1, 5]);
      ctx.beginPath(); ctx.moveTo(zones[3].x0, mid); ctx.lineTo(zones[3].x1, mid); ctx.stroke(); ctx.setLineDash([]);
      for (let n = nOld; n <= nNew; n++) {
        const x1 = xOf(n * FRAME), x0 = xOf((n + 1) * FRAME);
        const p = frameParams(n);
        const ph = 10 + (blockY1 - blockY0 - 30) * (p.bytes / 60);
        const px0 = x0 + 4, pw = x1 - x0 - 8, py = mid - ph / 2;
        ctx.fillStyle = Th.accentSoft; ctx.strokeStyle = Th.accent; ctx.lineWidth = 1.2;
        roundRect(ctx, px0, py, pw, ph, 4); ctx.fill(); ctx.stroke();
        // TOC byte stripe
        ctx.fillStyle = Th.accent; roundRect(ctx, px0, py, Math.min(6, pw * 0.2), ph, 3); ctx.fill();
        if (pw > 30) {
          ctx.fillStyle = Th.accent; ctx.font = OX.font(10, 'mono', 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(p.bytes + 'B', px0 + pw / 2 + 3, mid);
        }
      }
    });

    // 6. reconstructed wave
    smooth(zones[5], Th.dec, 1.8);
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
  }

  const loop = OX.loop(host, (dt) => { T += dt; cv.redraw(); });
  function toggle() {
    if (loop.running) { loop.stop(); pauseBtn.setIcon('play', OX.t('播放', 'Play')); }
    else { loop.start(); pauseBtn.setIcon('pause', OX.t('暂停', 'Pause')); }
  }
  if (OX.reducedMotion()) { pauseBtn.setIcon('play', OX.t('播放', 'Play')); }
  else loop.start();
});
