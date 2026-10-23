/* Figure 5: the source-filter model behind SILK.
 * Excitation (a pulse train for voiced sounds, noise for unvoiced) goes through an all-pole "vocal tract" filter
 * built from vowel formants. Time signals on top, spectra below; the filter's response is the envelope.
 * The filter response and the output spectrum are drawn down to FLOOR_DB without clamping: the 4-formant, order-8
 * filter falls to about -100 dB at 8 kHz, and a clamp at a higher floor looked like a brick-wall cut-off.
 */
OX.fig('silk-model', function (root, body) {
  const { el, segmented, slider, button } = OX.ui;
  const FS = 16000;
  const FLOOR_DB = -115; // lowest level on the filter and output-spectrum panels (the /u/ response reaches -112 dB)
  const VOWELS = {
    a: { label: OX.t('“啊” /a/', '/a/ “father”'), f: [[730, 90], [1090, 110], [2440, 170], [3400, 250]] },
    i: { label: OX.t('“衣” /i/', '/i/ “see”'), f: [[270, 60], [2290, 100], [3010, 170], [3600, 250]] },
    u: { label: OX.t('“乌” /u/', '/u/ “boot”'), f: [[300, 60], [870, 90], [2240, 170], [3400, 250]] },
  };
  const st = { src: 'voiced', f0: 140, vowel: 'a' };
  let sig = null;

  /** All-pole filter from formants: A(z) = prod (1 - 2 r cos(theta) z^-1 + r^2 z^-2). Returns prediction coefs a_k. */
  function formantLpc(formants) {
    let poly = [1];
    for (const [f, bw] of formants) {
      const r = Math.exp((-Math.PI * bw) / FS), th = (2 * Math.PI * f) / FS;
      const sec = [1, -2 * r * Math.cos(th), r * r];
      const out = new Array(poly.length + 2).fill(0);
      for (let i = 0; i < poly.length; i++) for (let j = 0; j < 3; j++) out[i + j] += poly[i] * sec[j];
      poly = out;
    }
    return Float32Array.from(poly.slice(1).map((c) => -c)); // x[n] = e[n] + sum a_k x[n-k]
  }

  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };

  function build() {
    const n = FS; // one second
    const exc = new Float32Array(n);
    if (st.src === 'voiced') {
      const P = FS / st.f0;
      let next = 0;
      for (let i = 0; i < n; i++) {
        if (i >= next) { exc[i] = 1; next += P; }
        exc[i] += 0.01 * rnd();
      }
    } else {
      seed = 777;
      for (let i = 0; i < n; i++) exc[i] = 0.35 * rnd();
    }
    const a = formantLpc(VOWELS[st.vowel].f);
    const out = OX.dsp.synthFilter(exc, a);
    // normalise both for display and playback
    const norm = (x, from, to) => { let m = 0; for (let i = from; i < to; i++) m = Math.max(m, Math.abs(x[i])); return m || 1; };
    const s0 = 4800, len = 512; // 32 ms window after the filter has settled
    const ge = norm(exc, s0, s0 + len), go = norm(out, s0, s0 + len);
    const excN = exc.map((v) => v / ge), outN = out.map((v) => v / go);
    const win = OX.dsp.hann(len);
    const specE = OX.dsp.spectrumDb(excN.subarray(s0, s0 + len), 1024, win);
    const specO = OX.dsp.spectrumDb(outN.subarray(s0, s0 + len), 1024, win);
    const env = OX.dsp.lpcEnvelopeDb(a, 1, 513);
    sig = { exc: excN, out: outN, s0, len, specE, specO, env, a };
  }

  const host = el('div');
  /* Audio: each button plays 1 s of its signal; pressing it again (or the other button) stops it. */
  let playing = null; // { key, handle }
  const playBtns = {
    exc: button({ icon: 'sound', label: OX.t('听激励', 'Listen to the excitation'), onClick: () => togglePlay('exc') }),
    out: button({ icon: 'sound', label: OX.t('听滤波器输出', 'Listen to the filter output'), onClick: () => togglePlay('out') }),
  };
  Object.values(playBtns).forEach((b) => b.setAttribute('aria-pressed', 'false'));
  function togglePlay(key) {
    if (playing && playing.key === key) { playing.handle.stop(); return; }
    const x = key === 'exc' ? sig.exc.map((v) => v * 0.35) : sig.out.map((v) => v * 0.5);
    const me = { key, handle: null };
    playing = me;
    me.handle = OX.audio.play(x, FS, {
      onEnd: () => {
        if (playing === me) playing = null;
        playBtns[key].setIcon('sound'); playBtns[key].setAttribute('aria-pressed', 'false');
      },
    });
    if (!me.handle) { playing = null; return; }
    for (const k in playBtns) { playBtns[k].setIcon(k === key ? 'stop' : 'sound'); playBtns[k].setAttribute('aria-pressed', k === key ? 'true' : 'false'); }
  }
  const ctlVoice = segmented({ id: 'sm-src', label: OX.t('声源', 'Source'), value: st.src, options: [{ value: 'voiced', label: OX.t('浊音：周期脉冲', 'Voiced: periodic pulses') }, { value: 'unvoiced', label: OX.t('清音：噪声', 'Unvoiced: noise') }], onChange: (v) => { st.src = v; f0Ctl.style.opacity = v === 'voiced' ? '1' : '0.4'; f0Ctl.input.disabled = v !== 'voiced'; build(); cv.redraw(); } });
  const f0Ctl = slider({ id: 'sm-f0', label: OX.t('基音频率', 'Pitch frequency'), min: 80, max: 320, step: 5, value: st.f0, format: (v) => v + ' Hz', onInput: (v) => { st.f0 = v; build(); cv.redraw(); } });
  const ctlVowel = segmented({ id: 'sm-vowel', label: OX.t('声道形状', 'Vocal-tract shape'), value: st.vowel, options: Object.keys(VOWELS).map((k) => ({ value: k, label: VOWELS[k].label })), onChange: (v) => { st.vowel = v; build(); cv.redraw(); } });
  body.append(
    el('div', { class: 'ox-controls' }, [ctlVoice, f0Ctl, ctlVowel]),
    host,
    el('div', { class: 'ox-controls' }, Object.values(playBtns)),
    OX.ui.legend([
      { label: OX.t('激励（声源）', 'Excitation (source)'), color: 'var(--silk)' },
      { label: OX.t('输出（语音）', 'Output (speech)'), color: 'var(--ink)' },
      { label: OX.t('声道滤波器的频率响应（包络）', 'Frequency response of the vocal-tract filter (envelope)'), color: 'var(--accent)' },
    ]));

  build();
  const cv = OX.view.canvas(host, { height: (w) => (w < 640 ? 560 : 330), label: OX.t('激励、声道滤波器和输出的波形与频谱', 'Waveforms and spectra of the excitation, the vocal-tract filter, and the output') }, draw);

  function draw(ctx, w, h) {
    const T = OX.theme;
    const narrow = w < 640;
    // panels: [exc wave, filter, out wave] over [exc spec, (arrow), out spec]
    const P = [];
    if (!narrow) {
      const gap = 26, colW = (w - 2 * gap) / 3, rowH = (h - 30) / 2;
      const col = (i) => i * (colW + gap);
      P.push({ k: 'excW', x: col(0), y: 18, w: colW, h: rowH - 26 });
      P.push({ k: 'filt', x: col(1), y: 18, w: colW, h: h - 50 });
      P.push({ k: 'outW', x: col(2), y: 18, w: colW, h: rowH - 26 });
      P.push({ k: 'excS', x: col(0), y: rowH + 26, w: colW, h: rowH - 30 });
      P.push({ k: 'outS', x: col(2), y: rowH + 26, w: colW, h: rowH - 30 });
    } else {
      const rows = [['excW', 'excS'], ['filt'], ['outW', 'outS']];
      const rowH = (h - 20) / 3;
      rows.forEach((r, i) => {
        const y = 18 + i * rowH;
        if (r.length === 1) P.push({ k: r[0], x: 0, y, w, h: rowH - 34 });
        else { const cw = (w - 14) / 2; P.push({ k: r[0], x: 0, y, w: cw, h: rowH - 34 }); P.push({ k: r[1], x: cw + 14, y, w: cw, h: rowH - 34 }); }
      });
    }
    /** Panel title above the box; the font shrinks a little when the panel is too narrow for it. */
    const title = (p, text, color, reserve) => {
      let size = 11.5;
      ctx.font = OX.font(size, 'body', 600);
      while (size > 9.5 && ctx.measureText(text).width > p.w - (reserve || 0)) { size -= 0.5; ctx.font = OX.font(size, 'body', 600); }
      ctx.fillStyle = color || T.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(text, p.x, p.y - 4);
    };
    const frameBox = (p) => { ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.strokeRect(Math.round(p.x) + 0.5, Math.round(p.y) + 0.5, Math.round(p.w), Math.round(p.h)); };
    const wave = (p, x, color) => {
      frameBox(p);
      const xs = OX.view.linear(sig.s0, sig.s0 + sig.len, p.x + 4, p.x + p.w - 4);
      const ys = OX.view.linear(-1.1, 1.1, p.y + p.h - 4, p.y + 4);
      ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(p.x, Math.round(ys(0)) + 0.5); ctx.lineTo(p.x + p.w, Math.round(ys(0)) + 0.5); ctx.stroke();
      OX.view.wave(ctx, x, sig.s0, sig.s0 + sig.len, xs, ys, color, 1.2);
      ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText('32 ms', p.x + p.w - 4, p.y + p.h + 3);
    };
    const spec = (p, s, color, withEnv) => {
      frameBox(p);
      const xs = OX.view.linear(0, 8000, p.x + 4, p.x + p.w - 4);
      // the flat excitation spectrum needs less range than the filtered output
      const ys = OX.view.linear(withEnv ? FLOOR_DB : -70, 5, p.y + p.h - 2, p.y + 4);
      const top = Math.max(...s);
      ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = 1.1; ctx.beginPath();
      for (let k = 0; k < s.length; k++) { const X = xs((k * FS) / 2 / (s.length - 1)), Y = ys(s[k] - top); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.stroke();
      if (withEnv) {
        const e = sig.env, eTop = Math.max(...e);
        ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath();
        for (let k = 0; k < e.length; k++) { const X = xs((k * FS) / 2 / (e.length - 1)), Y = ys(e[k] - eTop); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
        ctx.stroke();
      }
      ctx.restore();
      ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
      ctx.textAlign = 'left'; ctx.fillText('0', p.x, p.y + p.h + 3);
      ctx.textAlign = 'right'; ctx.fillText('8 kHz', p.x + p.w, p.y + p.h + 3);
    };
    const voiced = st.src === 'voiced';
    for (const p of P) {
      // English titles are longer: a panel narrower than 250 px (two panels in a row on a phone, or three columns on
      // a small screen) gets a shorter wording. The Chinese title is the same in both cases.
      const short = p.w < 250;
      if (p.k === 'excW') { title(p, voiced ? (short ? OX.t(`激励：每 ${(1000 / st.f0).toFixed(1)} 毫秒一个脉冲`, `Excitation: pulse every ${(1000 / st.f0).toFixed(1)} ms`) : OX.t(`激励：每 ${(1000 / st.f0).toFixed(1)} 毫秒一个脉冲`, `Excitation: one pulse every ${(1000 / st.f0).toFixed(1)} ms`)) : OX.t('激励：白噪声', 'Excitation: white noise'), T.silk); wave(p, sig.exc, T.silk); }
      if (p.k === 'outW') { title(p, short ? OX.t('输出：合成的语音波形', 'Output: synthesized speech') : OX.t('输出：合成的语音波形', 'Output: synthesized speech waveform')); wave(p, sig.out, T.ink); }
      if (p.k === 'excS') { title(p, voiced ? (short ? OX.t(`激励频谱：间隔 ${st.f0} Hz 的谐波`, `Harmonics ${st.f0} Hz apart`) : OX.t(`激励频谱：间隔 ${st.f0} Hz 的谐波`, `Excitation spectrum: harmonics ${st.f0} Hz apart`)) : OX.t('激励频谱：平坦', 'Excitation spectrum: flat'), T.silk); spec(p, sig.specE, T.silk, false); }
      if (p.k === 'outS') { title(p, short ? OX.t('输出频谱 = 激励频谱 × 包络', 'Spectrum: excitation × envelope') : OX.t('输出频谱 = 激励频谱 × 包络', 'Output spectrum = excitation spectrum × envelope')); spec(p, sig.specO, OX.alpha(T.ink, 0.75), true); }
      if (p.k === 'filt') {
        // the note goes on the right of the title row (the panel is at least ~196 px wide, so both fit)
        ctx.font = OX.font(10.5, 'body');
        const noteW = ctx.measureText(OX.t('F1–F4：共振峰', 'F1–F4: formants')).width;
        title(p, OX.t('声道滤波器 1/A(z)', 'Vocal-tract filter 1/A(z)'), T.accent, noteW + 10);
        ctx.font = OX.font(10.5, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText(OX.t('F1–F4：共振峰', 'F1–F4: formants'), p.x + p.w, p.y - 4);
        frameBox(p);
        const xs = OX.view.linear(0, 8000, p.x + 10, p.x + p.w - 10);
        const yAxis = p.y + p.h - 26;
        const ys = OX.view.linear(FLOOR_DB, 5, yAxis, p.y + 18); // headroom for the F1-F4 labels
        // dB grid, 0 dB = the highest peak; labels at the right end, where the response is far below each line
        ctx.font = OX.font(9.5, 'mono'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.lineWidth = 1;
        for (const v of [0, -40, -80]) {
          const y = Math.round(ys(v)) + 0.5;
          ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(xs(0), y); ctx.lineTo(xs(8000), y); ctx.stroke();
          ctx.fillStyle = T.ink3; ctx.fillText((v < 0 ? '−' + -v : '0') + ' dB', xs(8000), y - 2);
        }
        const e = sig.env, eTop = Math.max(...e);
        const fx = (k) => xs((k * 8000) / (e.length - 1));
        ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, yAxis - p.y); ctx.clip();
        ctx.fillStyle = OX.alpha(T.accent, T.dark ? 0.16 : 0.1);
        ctx.beginPath(); ctx.moveTo(xs(0), ys(FLOOR_DB));
        for (let k = 0; k < e.length; k++) ctx.lineTo(fx(k), ys(e[k] - eTop));
        ctx.lineTo(xs(8000), ys(FLOOR_DB)); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath();
        for (let k = 0; k < e.length; k++) { if (k === 0) ctx.moveTo(fx(k), ys(e[k] - eTop)); else ctx.lineTo(fx(k), ys(e[k] - eTop)); }
        ctx.stroke();
        ctx.restore();
        // formant labels above their peaks, pushed apart when two peaks are close (F1 and F2 of /a/)
        ctx.font = OX.font(10.5, 'mono', 500); ctx.fillStyle = T.accent; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        const labs = VOWELS[st.vowel].f.map(([f], i) => {
          const k = Math.round((f / 8000) * (e.length - 1)), t = 'F' + (i + 1);
          return { t, x: xs(f), y: ys(e[k] - eTop) - 4, w: ctx.measureText(t).width };
        });
        for (let i = 1; i < labs.length; i++) {
          const a = labs[i - 1], b = labs[i], need = (a.w + b.w) / 2 + 4 - (b.x - a.x);
          if (need > 0) { a.x -= need / 2; b.x += need / 2; }
        }
        for (const l of labs) ctx.fillText(l.t, l.x, l.y);
        OX.view.axis(ctx, xs, { side: 'bottom', at: yAxis, ticks: [0, 2000, 4000, 6000, 8000].map((v) => ({ v, label: v === 8000 ? '8 kHz' : String(v / 1000), align: v === 8000 ? 'right' : v === 0 ? 'left' : 'center' })), fontSize: 10 });
        if (!narrow) {
          // arrows from excitation to filter and filter to output
          ctx.strokeStyle = T.ink3; ctx.fillStyle = T.ink3; ctx.lineWidth = 1.2;
          const ay = P[0].y + P[0].h / 2;
          arrow(ctx, P[0].x + P[0].w + 4, ay, p.x - 4, ay);
          arrow(ctx, p.x + p.w + 4, ay, P[2].x - 4, ay);
        }
      }
    }
  }
  function arrow(ctx, x0, y0, x1, y1) {
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 7 * Math.cos(a - 0.4), y1 - 7 * Math.sin(a - 0.4)); ctx.lineTo(x1 - 7 * Math.cos(a + 0.4), y1 - 7 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
  }
});
