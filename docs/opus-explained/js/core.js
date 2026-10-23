/* Opus 图解 — shared runtime for every figure.
 *
 * Everything lives on one global, `OX`:
 *   OX.dsp    signal processing (FFT, STFT, MDCT, LPC, filters, base64 audio)
 *   OX.data   the real opus1 data from data/opus-demo-data.js, decoded on first use
 *   OX.theme  colors and fonts read from the CSS tokens, refreshed on theme change
 *   OX.t      the page language ('zh' or 'en'): OX.t('中文', 'English'), OX.setLang, OX.onLang
 *   OX.view   canvas setup, scales, axes, colormap
 *   OX.ui     small DOM builders for the shared controls (sliders, segmented, buttons)
 *   OX.audio  one shared Web Audio player
 *   OX.fig    figure registry; figures are initialised when they come near the viewport
 *
 * No library is used. Figures live in js/figs/*.js and call OX.fig(name, init).
 */
(function () {
  'use strict';
  const OX = (window.OX = window.OX || {});

  /* ------------------------------------------------------------------ */
  /* DSP                                                                 */
  /* ------------------------------------------------------------------ */
  const dsp = (OX.dsp = {});

  /** In-place radix-2 complex FFT. `re` and `im` have the same power-of-two length. */
  dsp.fft = function (re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const ang = (sign * 2 * Math.PI) / len;
      for (let k = 0; k < half; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        for (let i = k; i < n; i += len) {
          const b = i + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[i] - tr; im[b] = im[i] - ti;
          re[i] += tr; im[i] += ti;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  };

  dsp.nextPow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };

  dsp.hann = function (N) {
    const w = new Float32Array(N);
    for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / N);
    return w;
  };

  /** The CELT overlap window (power complementary): w(i) = sin(pi/2 * sin^2(pi*(i+0.5)/(2*overlap))). */
  dsp.celtWindow = function (overlap) {
    const w = new Float32Array(overlap);
    for (let i = 0; i < overlap; i++) {
      const s = Math.sin((Math.PI * (i + 0.5)) / (2 * overlap));
      w[i] = Math.sin((Math.PI / 2) * s * s);
    }
    return w;
  };

  /** Power spectrum in dB of one frame (zero padded to nfft). Returns nfft/2+1 bins. */
  dsp.spectrumDb = function (frame, nfft, win, floorDb) {
    nfft = nfft || dsp.nextPow2(frame.length);
    const re = new Float32Array(nfft), im = new Float32Array(nfft);
    const n = Math.min(frame.length, nfft);
    let wsum = 0;
    for (let i = 0; i < n; i++) {
      const w = win ? win[i] : 1;
      re[i] = frame[i] * w;
      wsum += w;
    }
    dsp.fft(re, im);
    const out = new Float32Array(nfft / 2 + 1);
    const norm = 2 / Math.max(wsum, 1e-9);
    const fl = floorDb == null ? -120 : floorDb;
    for (let k = 0; k <= nfft / 2; k++) {
      const m = Math.hypot(re[k], im[k]) * norm;
      out[k] = Math.max(fl, 20 * Math.log10(m + 1e-12));
    }
    return out;
  };

  /**
   * Short-time Fourier transform magnitude in dB.
   * Returns { cols, bins, hop, nfft, data } where data[c*bins + k] is dB of column c, bin k.
   */
  dsp.stft = function (x, opts) {
    const nfft = (opts && opts.nfft) || 512;
    const hop = (opts && opts.hop) || nfft / 4;
    const win = dsp.hann(nfft);
    const bins = nfft / 2 + 1;
    const cols = Math.max(1, Math.floor((x.length - nfft) / hop) + 1);
    const data = new Float32Array(cols * bins);
    const re = new Float32Array(nfft), im = new Float32Array(nfft);
    let wsum = 0;
    for (let i = 0; i < nfft; i++) wsum += win[i];
    const norm = 2 / wsum;
    for (let c = 0; c < cols; c++) {
      const off = c * hop;
      for (let i = 0; i < nfft; i++) { re[i] = (x[off + i] || 0) * win[i]; im[i] = 0; }
      dsp.fft(re, im);
      for (let k = 0; k < bins; k++) {
        const m = Math.hypot(re[k], im[k]) * norm;
        data[c * bins + k] = 20 * Math.log10(m + 1e-9);
      }
    }
    return { cols, bins, hop, nfft, data };
  };

  /**
   * MDCT (modified discrete cosine transform), direct O(N^2) form.
   * x has 2N samples (already windowed). X[k] = sum_n x[n] cos(pi/N (n + 1/2 + N/2)(k + 1/2)).
   */
  dsp.mdct = function (x, N) {
    const X = new Float32Array(N);
    const n0 = 0.5 + N / 2;
    for (let k = 0; k < N; k++) {
      let s = 0;
      const f = (Math.PI / N) * (k + 0.5);
      for (let n = 0; n < 2 * N; n++) s += x[n] * Math.cos(f * (n + n0));
      X[k] = s;
    }
    return X;
  };

  /** Inverse MDCT: 2N output samples, y[n] = (2/N) sum_k X[k] cos(...). With a Princen-Bradley window on both sides, overlap-add rebuilds the input. */
  dsp.imdct = function (X, N) {
    const y = new Float32Array(2 * N);
    const n0 = 0.5 + N / 2;
    for (let n = 0; n < 2 * N; n++) {
      let s = 0;
      const t = n + n0;
      for (let k = 0; k < N; k++) s += X[k] * Math.cos((Math.PI / N) * t * (k + 0.5));
      y[n] = (2 / N) * s;
    }
    return y;
  };

  /** Autocorrelation r[0..p]. */
  dsp.autocorr = function (x, p) {
    const r = new Float64Array(p + 1);
    for (let l = 0; l <= p; l++) {
      let s = 0;
      for (let n = l; n < x.length; n++) s += x[n] * x[n - l];
      r[l] = s;
    }
    return r;
  };

  /**
   * Levinson-Durbin recursion.
   * Returns prediction coefficients a[0..p-1] with x̂[n] = sum_k a[k] x[n-k-1],
   * the reflection coefficients and the final prediction error energy.
   */
  dsp.levinson = function (r, p) {
    const a = new Float64Array(p + 1); // a[0] unused (A(z) = 1 - sum a_k z^-k)
    const k = new Float64Array(p);
    let err = r[0] * (1 + 1e-9) + 1e-12;
    for (let i = 1; i <= p; i++) {
      let acc = r[i];
      for (let j = 1; j < i; j++) acc -= a[j] * r[i - j];
      const ki = acc / err;
      k[i - 1] = ki;
      const prev = a.slice();
      a[i] = ki;
      for (let j = 1; j < i; j++) a[j] = prev[j] - ki * prev[i - j];
      err *= 1 - ki * ki;
    }
    return { a: Float32Array.from(a.subarray(1)), k: Float32Array.from(k), err };
  };

  /** LPC analysis of one frame with a Hann window and a little lag windowing. */
  dsp.lpc = function (frame, p) {
    const w = dsp.hann(frame.length);
    const xw = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) xw[i] = frame[i] * w[i];
    const r = dsp.autocorr(xw, p);
    for (let l = 1; l <= p; l++) r[l] *= Math.exp(-0.5 * Math.pow((2 * Math.PI * 60 * l) / 16000, 2));
    return dsp.levinson(r, p);
  };

  /** dB magnitude of the LPC synthesis filter gain / |1 - sum a_k e^{-jwk}| on nbins points in [0, pi]. */
  dsp.lpcEnvelopeDb = function (a, gain, nbins) {
    const out = new Float32Array(nbins);
    for (let b = 0; b < nbins; b++) {
      const w = (Math.PI * b) / (nbins - 1);
      let re = 1, im = 0;
      for (let k = 0; k < a.length; k++) {
        re -= a[k] * Math.cos(w * (k + 1));
        im += a[k] * Math.sin(w * (k + 1));
      }
      out[b] = 20 * Math.log10((gain || 1) / Math.max(Math.hypot(re, im), 1e-9));
    }
    return out;
  };

  /** Prediction error (residual) e[n] = x[n] - sum a_k x[n-k]. `hist` gives samples before x (optional). */
  dsp.inverseFilter = function (x, a, hist) {
    const p = a.length;
    const e = new Float32Array(x.length);
    const get = (i) => (i >= 0 ? x[i] : hist ? hist[hist.length + i] || 0 : 0);
    for (let n = 0; n < x.length; n++) {
      let pred = 0;
      for (let k = 0; k < p; k++) pred += a[k] * get(n - k - 1);
      e[n] = x[n] - pred;
    }
    return e;
  };

  /** All-pole synthesis y[n] = e[n] + sum a_k y[n-k]. */
  dsp.synthFilter = function (e, a, hist) {
    const p = a.length;
    const y = new Float32Array(e.length);
    const get = (i) => (i >= 0 ? y[i] : hist ? hist[hist.length + i] || 0 : 0);
    for (let n = 0; n < e.length; n++) {
      let s = e[n];
      for (let k = 0; k < p; k++) s += a[k] * get(n - k - 1);
      y[n] = s;
    }
    return y;
  };

  dsp.rms = function (x, from, to) {
    from = from || 0; to = to == null ? x.length : to;
    let s = 0;
    for (let i = from; i < to; i++) s += x[i] * x[i];
    return Math.sqrt(s / Math.max(1, to - from));
  };

  dsp.snrDb = function (ref, test) {
    let s = 0, e = 0;
    const n = Math.min(ref.length, test.length);
    for (let i = 0; i < n; i++) { s += ref[i] * ref[i]; const d = ref[i] - test[i]; e += d * d; }
    return 10 * Math.log10((s + 1e-20) / (e + 1e-20));
  };

  dsp.db = (v) => 20 * Math.log10(Math.abs(v) + 1e-12);

  /** base64 -> Uint8Array */
  dsp.b64ToBytes = function (s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  };

  /** base64 of little-endian int16 -> Float32Array in [-1, 1) */
  dsp.b64ToFloat = function (s) {
    const bytes = dsp.b64ToBytes(s);
    const n = bytes.length >> 1;
    const out = new Float32Array(n);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < n; i++) out[i] = dv.getInt16(2 * i, true) / 32768;
    return out;
  };

  /* ------------------------------------------------------------------ */
  /* Real data (window.OPUS_DEMO_DATA), decoded lazily and cached        */
  /* ------------------------------------------------------------------ */
  const cache = new Map();
  OX.data = {
    /** The raw object, or null when the data file is missing. */
    get raw() { return window.OPUS_DEMO_DATA || null; },
    /** Decode a base64 int16 field once; `key` names the cache slot. */
    audio(key, b64) {
      if (!cache.has(key)) cache.set(key, b64 ? dsp.b64ToFloat(b64) : null);
      return cache.get(key);
    },
    bytes(key, b64) {
      if (!cache.has(key)) cache.set(key, b64 ? dsp.b64ToBytes(b64) : null);
      return cache.get(key);
    },
    /** Memoise any derived value (spectrograms etc.). */
    memo(key, fn) {
      if (!cache.has(key)) cache.set(key, fn());
      return cache.get(key);
    },
  };

  /* ------------------------------------------------------------------ */
  /* Theme: colors and fonts from CSS custom properties                  */
  /* ------------------------------------------------------------------ */
  const TOKENS = ['paper', 'surface', 'sunken', 'ink', 'ink-2', 'ink-3', 'rule', 'grid', 'accent', 'accent-soft',
    'silk', 'silk-soft', 'celt', 'celt-soft', 'bad', 'good', 'orig', 'dec', 'font-body', 'font-mono', 'font-display'];
  const themeCbs = new Set();
  OX.theme = {};
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    for (const t of TOKENS) OX.theme[t.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase())] = cs.getPropertyValue('--' + t).trim();
    OX.theme.dark = isDark();
  }
  function isDark() {
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'dark') return true;
    if (attr === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  OX.onTheme = (cb) => { themeCbs.add(cb); return figureOwns(() => themeCbs.delete(cb)); };
  function themeChanged() { readTheme(); themeCbs.forEach((cb) => { try { cb(OX.theme); } catch (e) { console.error(e); } }); }
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(themeChanged);
  }
  new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  readTheme();

  /** Canvas font string: OX.font(12, 'mono', 500) */
  OX.font = function (size, role, weight) {
    const fam = role === 'mono' ? OX.theme.fontMono : role === 'display' ? OX.theme.fontDisplay : OX.theme.fontBody;
    return `${weight || 400} ${size}px ${fam || 'sans-serif'}`;
  };

  /** Mix a CSS color with alpha: OX.alpha('#123456', 0.3) (hex or rgb()). */
  OX.alpha = function (color, a) {
    if (!color) return `rgba(0,0,0,${a})`;
    if (color[0] === '#') {
      let h = color.slice(1);
      if (h.length === 3) h = h.split('').map((c) => c + c).join('');
      const v = parseInt(h, 16);
      return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${a})`;
    }
    const m = color.match(/rgba?\(([^)]+)\)/);
    if (m) { const p = m[1].split(',').map((s) => s.trim()); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; }
    return color;
  };

  OX.reducedMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------------------------------------------ */
  /* Language: Chinese ('zh') or English ('en')                          */
  /* ------------------------------------------------------------------ */
  // The script in <head> has already picked the language and set html[data-lang]. Text in the HTML comes in pairs,
  // <span lang="zh-CN">…</span><span lang="en">…</span>, and the CSS shows one of each pair. Figures write their
  // text with OX.t when they are built, so OX.setLang rebuilds them.
  OX.lang = document.documentElement.getAttribute('data-lang') === 'en' ? 'en' : 'zh';
  /** The string for the current language: OX.t('中文', 'English'). */
  OX.t = (zh, en) => (OX.lang === 'en' && en != null ? en : zh);
  const langCbs = new Set();
  /** Called after every language switch, with the new language. */
  OX.onLang = (cb) => { langCbs.add(cb); return figureOwns(() => langCbs.delete(cb)); };

  // Things a figure set up that must be undone before the figure is rebuilt: { el, dispose, fig }. `fig` is the
  // figure that was being built when the thing was made; things made later are found by `el` inside its .fig-body.
  const owned = new Set();
  const own = (el, dispose) => { owned.add({ el, dispose, fig: initing }); };
  // Callbacks a figure registers while it is being built are removed when it is rebuilt.
  let initing = null;
  function figureOwns(off) { if (initing) initing.__oxOff.push(off); return off; }

  /* ------------------------------------------------------------------ */
  /* View helpers: canvas, scales, colormap, axes                        */
  /* ------------------------------------------------------------------ */
  const view = (OX.view = {});

  /**
   * Create a HiDPI canvas inside `host` that redraws on resize and theme change.
   * opts.height: CSS px height, or opts.aspect: width/height ratio. `draw(ctx, w, h)` gets CSS pixels.
   * Returns { canvas, ctx, redraw(), get w, get h }.
   */
  view.canvas = function (host, opts, draw) {
    const canvas = document.createElement('canvas');
    canvas.className = 'ox-canvas';
    if (opts && opts.label) { canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', opts.label); }
    host.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    const st = { canvas, ctx, w: 0, h: 0 };
    function size() {
      const w = Math.max(10, Math.round(host.clientWidth));
      const h = opts && opts.height ? (typeof opts.height === 'function' ? opts.height(w) : opts.height)
        : Math.round(w / ((opts && opts.aspect) || 2.5));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
      }
      st.w = w; st.h = h;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    st.redraw = function () {
      size();
      ctx.save();
      ctx.clearRect(0, 0, st.w, st.h);
      try { draw(ctx, st.w, st.h); } catch (e) { console.error(e); }
      ctx.restore();
    };
    let raf = 0;
    const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; st.redraw(); }); };
    const ro = window.ResizeObserver ? new ResizeObserver(schedule) : null;
    if (ro) ro.observe(host);
    else window.addEventListener('resize', schedule);
    const offTheme = OX.onTheme(schedule);
    own(canvas, () => {
      if (ro) ro.disconnect(); else window.removeEventListener('resize', schedule);
      offTheme();
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    });
    st.schedule = schedule;
    st.redraw();
    return st;
  };

  view.linear = function (d0, d1, r0, r1) {
    const s = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    s.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
    s.domain = [d0, d1]; s.range = [r0, r1];
    return s;
  };

  view.log = function (d0, d1, r0, r1) {
    const l0 = Math.log(d0), l1 = Math.log(d1);
    const s = (v) => r0 + ((Math.log(Math.max(v, 1e-12)) - l0) / (l1 - l0)) * (r1 - r0);
    s.invert = (p) => Math.exp(l0 + ((p - r0) / (r1 - r0)) * (l1 - l0));
    s.domain = [d0, d1]; s.range = [r0, r1];
    return s;
  };

  /** Perceptual dark-to-bright colormap (inferno-like). t in [0,1] -> [r,g,b]. */
  const CMAP = [
    [0.0, [6, 6, 20]], [0.15, [34, 12, 72]], [0.32, [92, 18, 110]], [0.5, [158, 40, 100]],
    [0.66, [214, 76, 62]], [0.8, [243, 134, 32]], [0.92, [250, 200, 60]], [1.0, [252, 250, 200]],
  ];
  view.cmap = function (t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    for (let i = 1; i < CMAP.length; i++) {
      if (t <= CMAP[i][0]) {
        const [t0, c0] = CMAP[i - 1], [t1, c1] = CMAP[i];
        const u = (t - t0) / (t1 - t0);
        return [c0[0] + (c1[0] - c0[0]) * u, c0[1] + (c1[1] - c0[1]) * u, c0[2] + (c1[2] - c0[2]) * u];
      }
    }
    return CMAP[CMAP.length - 1][1];
  };

  /**
   * Render an STFT (from dsp.stft) to an offscreen canvas, frequency axis upward.
   * opts: { minDb, maxDb, maxBin } . Returns a canvas you can drawImage() scaled.
   */
  view.spectrogramImage = function (S, opts) {
    const minDb = opts && opts.minDb != null ? opts.minDb : -90;
    const maxDb = opts && opts.maxDb != null ? opts.maxDb : -10;
    const maxBin = Math.min(S.bins, (opts && opts.maxBin) || S.bins);
    const c = document.createElement('canvas');
    c.width = S.cols; c.height = maxBin;
    const g = c.getContext('2d');
    const img = g.createImageData(S.cols, maxBin);
    for (let col = 0; col < S.cols; col++) {
      for (let k = 0; k < maxBin; k++) {
        const v = (S.data[col * S.bins + k] - minDb) / (maxDb - minDb);
        const [r, gg, b] = view.cmap(v);
        const p = ((maxBin - 1 - k) * S.cols + col) * 4;
        img.data[p] = r; img.data[p + 1] = gg; img.data[p + 2] = b; img.data[p + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  };

  /**
   * Axis helper. ctx, scale, { side: 'bottom'|'left', at: px, ticks: [{v, label}], from, to, grid: px length }.
   */
  view.axis = function (ctx, scale, o) {
    const T = OX.theme;
    ctx.save();
    ctx.font = OX.font(o.fontSize || 10.5, 'mono');
    ctx.fillStyle = T.ink3 || T.ink2;
    ctx.strokeStyle = T.rule;
    ctx.lineWidth = 1;
    for (const t of o.ticks) {
      const p = Math.round(scale(t.v)) + 0.5;
      if (o.side === 'bottom') {
        if (o.grid) { ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(p, o.at); ctx.lineTo(p, o.at - o.grid); ctx.stroke(); }
        ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(p, o.at); ctx.lineTo(p, o.at + 4); ctx.stroke();
        ctx.textAlign = t.align || 'center'; ctx.textBaseline = 'top';
        if (t.label != null) ctx.fillText(t.label, p, o.at + 6);
      } else {
        if (o.grid) { ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(o.at, p); ctx.lineTo(o.at + o.grid, p); ctx.stroke(); }
        ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(o.at - 4, p); ctx.lineTo(o.at, p); ctx.stroke();
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        if (t.label != null) ctx.fillText(t.label, o.at - 6, p);
      }
    }
    if (o.title) {
      ctx.fillStyle = T.ink2;
      ctx.font = OX.font(11, 'body', 500);
      if (o.side === 'bottom') { ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX, o.at + 22); }
      else { ctx.save(); ctx.translate(o.at - (o.titleOffset || 36), o.titleY); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.title, 0, 0); ctx.restore(); }
    }
    ctx.restore();
  };

  /** Draw a waveform polyline (min/max per pixel when dense). */
  view.wave = function (ctx, x, from, to, xs, ys, color, width) {
    from = Math.max(0, Math.floor(from)); to = Math.min(x.length, Math.ceil(to));
    const n = to - from;
    const px0 = xs(from), px1 = xs(to);
    const pxCount = Math.max(1, Math.abs(px1 - px0));
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width || 1.25; ctx.lineJoin = 'round';
    ctx.beginPath();
    if (n > pxCount * 2) {
      const per = n / pxCount;
      for (let p = 0; p < pxCount; p++) {
        const a = from + Math.floor(p * per), b = Math.min(to, from + Math.floor((p + 1) * per));
        let mn = Infinity, mx = -Infinity;
        for (let i = a; i < b; i++) { const v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        const X = px0 + p + 0.5;
        if (p === 0) ctx.moveTo(X, ys(mx)); else ctx.lineTo(X, ys(mx));
        ctx.lineTo(X, ys(mn));
      }
    } else {
      for (let i = from; i < to; i++) { const X = xs(i), Y = ys(x[i]); if (i === from) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
    }
    ctx.stroke();
    ctx.restore();
  };

  /* ------------------------------------------------------------------ */
  /* Small DOM builders for shared controls                              */
  /* ------------------------------------------------------------------ */
  const ui = (OX.ui = {});
  let uid = 0;
  ui.id = (p) => `${p || 'ox'}-${++uid}`;

  /** OX.ui.el('div', {class:'x', onclick: fn}, [children or text]) */
  ui.el = function (tag, attrs, kids) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    if (kids != null) (Array.isArray(kids) ? kids : [kids]).forEach((c) => {
      if (c == null || c === false) return;
      e.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    });
    return e;
  };

  /**
   * Labeled range slider.
   * { id, label, min, max, step, value, format(v) -> string, onInput(v) }
   * Returns the wrapper element; wrapper.input is the <input>, wrapper.set(v) updates it.
   */
  ui.slider = function (o) {
    const id = o.id || ui.id('sl');
    const out = ui.el('output', { class: 'ox-slider-val', for: id });
    const input = ui.el('input', { type: 'range', id, min: o.min, max: o.max, step: o.step || 1, value: o.value });
    const fmt = o.format || ((v) => String(v));
    const upd = () => { const v = +input.value; out.textContent = fmt(v); input.style.setProperty('--p', ((v - o.min) / (o.max - o.min)) * 100 + '%'); };
    input.addEventListener('input', () => { upd(); o.onInput && o.onInput(+input.value); });
    upd();
    const wrap = ui.el('div', { class: 'ox-slider' }, [ui.el('label', { for: id, text: o.label }), input, out]);
    wrap.input = input;
    wrap.set = (v) => { input.value = v; upd(); };
    return wrap;
  };

  /**
   * Segmented control (radio group).
   * { id, label, options: [{value, label}], value, onChange(value) }
   */
  ui.segmented = function (o) {
    const name = o.id || ui.id('seg');
    const group = ui.el('div', { class: 'ox-seg', role: 'radiogroup', 'aria-label': o.label || '' });
    let current = o.value;
    const btns = o.options.map((opt) => {
      const b = ui.el('button', { type: 'button', class: 'ox-seg-btn', role: 'radio', id: `${name}-${opt.value}`, 'data-value': opt.value, text: opt.label });
      b.addEventListener('click', () => set(opt.value, true));
      group.appendChild(b);
      return b;
    });
    function set(v, fire) {
      current = v;
      let any = false;
      btns.forEach((b) => {
        const on = b.getAttribute('data-value') === String(v);
        any = any || on;
        b.setAttribute('aria-checked', on ? 'true' : 'false');
        b.classList.toggle('on', on);
        b.tabIndex = on ? 0 : -1; // one Tab stop per group, as in the ARIA radio-group pattern
      });
      if (!any && btns.length) btns[0].tabIndex = 0;
      if (fire && o.onChange) o.onChange(v);
    }
    group.addEventListener('keydown', (e) => {
      const n = btns.length;
      const i = btns.findIndex((b) => b.getAttribute('data-value') === String(current));
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      let j;
      if (d) j = i < 0 ? (d > 0 ? 0 : n - 1) : (i + d + n) % n;
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = n - 1;
      else return;
      e.preventDefault();
      set(o.options[j].value, true);
      btns[j].focus();
    });
    set(current, false);
    const wrap = o.label ? ui.el('div', { class: 'ox-seg-wrap' }, [ui.el('span', { class: 'ox-ctl-label', text: o.label }), group]) : group;
    wrap.set = set;
    wrap.get = () => current;
    return wrap;
  };

  /** Button: { label, icon: 'play'|'pause'|'step'|'reset'|null, kind: 'primary'|'ghost', onClick } */
  const ICONS = {
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
    stop: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3.5h9v9h-9z" fill="currentColor"/></svg>',
    step: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2.5v11l7-5.5zM11 2.5h2v11h-2z" fill="currentColor"/></svg>',
    reset: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3a5 5 0 1 1-4.6 3H1.8A6.5 6.5 0 1 0 8 1.5V0L4.5 2.3 8 4.6z" fill="currentColor"/></svg>',
    sound: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6h3l4-3.5v11L5 10H2z" fill="currentColor"/><path d="M11 5.2a4 4 0 0 1 0 5.6M12.8 3.4a6.5 6.5 0 0 1 0 9.2" stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"/></svg>',
  };
  ui.icon = (name) => ICONS[name] || '';
  ui.button = function (o) {
    const b = ui.el('button', { type: 'button', class: 'ox-btn' + (o.kind ? ' ox-btn-' + o.kind : ''), id: o.id, 'aria-label': o.aria || null });
    b.innerHTML = (o.icon ? ICONS[o.icon] : '') + (o.label ? `<span>${o.label}</span>` : '');
    if (o.onClick) b.addEventListener('click', o.onClick);
    b.setIcon = (name, label) => { b.innerHTML = (name ? ICONS[name] : '') + (label != null ? `<span>${label}</span>` : o.label ? `<span>${o.label}</span>` : ''); };
    return b;
  };

  /** Key-value readout: returns element with .set(value) */
  ui.readout = function (label, value, opts) {
    const v = ui.el('span', { class: 'ox-ro-val', text: value == null ? '—' : String(value) });
    const e = ui.el('div', { class: 'ox-ro' + (opts && opts.cls ? ' ' + opts.cls : '') }, [ui.el('span', { class: 'ox-ro-label', text: label }), v]);
    e.set = (x) => { v.textContent = x; };
    return e;
  };

  /** Legend: [{ label, color, dash, kind: 'line'|'box' }] */
  ui.legend = function (items) {
    return ui.el('div', { class: 'ox-legend' }, items.map((it) => {
      const sw = ui.el('span', { class: 'ox-sw' + (it.kind === 'box' ? ' box' : '') + (it.dash ? ' dash' : '') });
      sw.style.setProperty('--c', it.color);
      return ui.el('span', { class: 'ox-legend-item' }, [sw, it.label]);
    }));
  };

  /** Inject figure-specific CSS once. */
  const injected = new Set();
  ui.css = function (key, text) {
    if (injected.has(key)) return;
    injected.add(key);
    document.head.appendChild(ui.el('style', { 'data-fig-css': key, text }));
  };

  ui.fmt = {
    int: (v) => Math.round(v).toLocaleString('en-US'),
    num: (v, d) => (+v).toFixed(d == null ? 1 : d),
    hz: (v) => (v >= 1000 ? (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + ' kHz' : Math.round(v) + ' Hz'),
    kbps: (v) => (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + ' kb/s',
    ms: (v) => (+v).toFixed(v % 1 === 0 ? 0 : 1) + ' ms',
  };

  /* ------------------------------------------------------------------ */
  /* Animation loop that runs only while the element is on screen        */
  /* ------------------------------------------------------------------ */
  /**
   * OX.loop(el, tick) -> { start(), stop(), running }
   * tick(dtSeconds, tSeconds) returns false to stop. The loop pauses while `el` is off screen.
   */
  OX.loop = function (el, tick) {
    let raf = 0, last = 0, t = 0, want = false, visible = true;
    const st = { get running() { return want; } };
    function frame(now) {
      raf = 0;
      if (!want || !visible) return;
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now; t += dt;
      if (tick(dt, t) === false) { want = false; return; }
      if (want && !raf) raf = requestAnimationFrame(frame); // a tick may already have scheduled the next frame
    }
    st.start = () => { if (!want) last = 0; want = true; if (!raf && visible) raf = requestAnimationFrame(frame); };
    st.stop = () => { want = false; if (raf) cancelAnimationFrame(raf); raf = 0; };
    const io = window.IntersectionObserver ? new IntersectionObserver((ents) => {
      visible = ents[ents.length - 1].isIntersecting; // the newest entry wins after a fast scroll
      if (visible && want && !raf) { last = 0; raf = requestAnimationFrame(frame); }
    }) : null;
    if (io) io.observe(el);
    own(el, () => { st.stop(); if (io) io.disconnect(); });
    return st;
  };

  /* ------------------------------------------------------------------ */
  /* Audio: one shared player                                            */
  /* ------------------------------------------------------------------ */
  let actx = null, current = null;
  OX.audio = {
    /**
     * Play a Float32Array at `sampleRate`. Stops whatever was playing.
     * opts: { gain, onEnd, from, to } (from/to in samples). Returns a handle { stop(), position() } where
     * position() is the current sample index.
     */
    play(samples, sampleRate, opts) {
      opts = opts || {};
      OX.audio.stop();
      try {
        if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
        if (actx.state === 'suspended') actx.resume();
      } catch (e) { console.error('Web Audio unavailable', e); return null; }
      const from = opts.from || 0, to = opts.to || samples.length;
      const buf = actx.createBuffer(1, to - from, sampleRate);
      buf.getChannelData(0).set(samples.subarray(from, to));
      const src = actx.createBufferSource();
      src.buffer = buf;
      const g = actx.createGain();
      g.gain.value = opts.gain == null ? 0.9 : opts.gain;
      src.connect(g).connect(actx.destination);
      if (actx.state !== 'running' && actx.state !== 'suspended') return null;
      const t0 = actx.currentTime;
      const h = {
        stopped: false,
        stop() { if (h.stopped) return; h.stopped = true; try { src.stop(); } catch (e) { /* already stopped */ } if (current === h) current = null; opts.onEnd && opts.onEnd(); },
        position() { return from + Math.min(to - from, Math.round((actx.currentTime - t0) * sampleRate)); },
        get playing() { return !h.stopped; },
      };
      src.onended = () => { if (!h.stopped) h.stop(); };
      src.start();
      current = h;
      return h;
    },
    stop() { if (current) current.stop(); current = null; },
    get current() { return current; },
  };

  /* ------------------------------------------------------------------ */
  /* Figure registry                                                     */
  /* ------------------------------------------------------------------ */
  const registry = new Map();
  /**
   * OX.fig('name', function init(root, body) { ... })
   * `root` is the <figure data-fig="name"> element; build into `body`, its .fig-body. Canvases, loops and
   * OX.onTheme / OX.onLang callbacks are cleaned up by the runtime when the figure is rebuilt. If init sets up
   * anything else that outlives the figure (its own ResizeObserver, a timer, a window listener), it returns a
   * function that undoes it.
   */
  OX.fig = function (name, init) { registry.set(name, init); };

  function initFigure(root) {
    if (root.__oxInit) return;
    const name = root.getAttribute('data-fig');
    const init = registry.get(name);
    if (!init) return;
    root.__oxInit = true;
    root.__oxOff = [];
    const body = root.querySelector('.fig-body');
    if (!body) { console.error(`figure "${name}" has no .fig-body`); return; }
    if (!root.__oxBlank) root.__oxBlank = body.cloneNode(false);
    initing = root;
    try {
      const dispose = init(root, body);
      if (typeof dispose === 'function') root.__oxOff.push(dispose);
      root.classList.add('ready');
    } catch (e) {
      console.error(`figure "${name}" failed`, e);
      body.appendChild(ui.el('p', { class: 'fig-error', text: OX.t(`这张图没能加载：${e.message}`, `This figure failed to load: ${e.message}`) }));
    } finally {
      initing = null;
    }
  }

  /** Undo initFigure: run the figure's clean-up and give it an empty .fig-body again. */
  function resetFigure(root) {
    if (!root.__oxInit) return;
    const body = root.querySelector('.fig-body');
    root.__oxOff.forEach((off) => { try { off(); } catch (e) { console.error(e); } });
    root.__oxOff = [];
    for (const r of owned) {
      if (r.fig !== root && !(body && body.contains(r.el))) continue;
      try { r.dispose(); } catch (e) { console.error(e); }
      owned.delete(r);
    }
    if (body) body.replaceWith(root.__oxBlank.cloneNode(false));
    root.classList.remove('ready');
    root.__oxInit = false;
  }

  /* Switching language ------------------------------------------------ */
  // Attributes cannot hold two languages, so an element keeps the Chinese value in the attribute and the English
  // value in data-en-<name>. <title data-en="…"> does the same for the page title.
  const LANG_ATTRS = ['title', 'aria-label', 'alt', 'content'];
  function applyLang() {
    const html = document.documentElement;
    html.setAttribute('data-lang', OX.lang);
    html.lang = OX.lang === 'en' ? 'en' : 'zh-CN';
    for (const a of LANG_ATTRS) {
      document.querySelectorAll(`[data-en-${a}]`).forEach((e) => {
        if (!e.hasAttribute(`data-zh-${a}`)) e.setAttribute(`data-zh-${a}`, e.getAttribute(a) || '');
        e.setAttribute(a, e.getAttribute(OX.lang === 'en' ? `data-en-${a}` : `data-zh-${a}`));
      });
    }
    const title = document.querySelector('title[data-en]');
    if (title) {
      if (!title.hasAttribute('data-zh')) title.setAttribute('data-zh', title.textContent);
      document.title = title.getAttribute(OX.lang === 'en' ? 'data-en' : 'data-zh');
    }
  }

  // The text changes length, so remember the first block that starts below the top bar and put it back at the same
  // height. If no block starts in view (a tall figure fills it), use the block that covers the top.
  const ANCHORS = 'main h1, main h2, main h3, main p, main li, main tr, main summary, main figcaption, main figure';
  function scrollAnchor() {
    const top = document.querySelector('.topbar');
    const y0 = top ? top.getBoundingClientRect().bottom : 0;
    let covering = null;
    for (const e of document.querySelectorAll(ANCHORS)) {
      if (e.closest('.fig-body')) continue; // rebuilt below
      const r = e.getBoundingClientRect();
      if (!r.height || r.bottom <= y0) continue;
      if (r.top >= y0) return r.top < window.innerHeight ? { e, top: r.top } : covering || { e, top: r.top };
      if (!covering) covering = { e, top: r.top };
    }
    return covering;
  }

  /** Show the page in 'zh' or 'en' without reloading, and remember the choice. */
  OX.setLang = function (lang) {
    lang = lang === 'en' ? 'en' : 'zh';
    try { localStorage.setItem('ox-lang', lang); } catch (e) { /* storage blocked */ }
    if (lang === OX.lang) return;
    const anchor = scrollAnchor();
    OX.lang = lang;
    applyLang();
    OX.audio.stop();
    const built = [...document.querySelectorAll('[data-fig]')].filter((r) => r.__oxInit);
    built.forEach(resetFigure);
    built.forEach(initFigure);
    langCbs.forEach((cb) => { try { cb(lang); } catch (e) { console.error(e); } });
    if (anchor) window.scrollBy({ top: anchor.e.getBoundingClientRect().top - anchor.top, behavior: 'instant' });
  };

  /** Build every figure that comes before `target` in the page, so in-page jumps land where they should. */
  OX.initBefore = function (target) {
    document.querySelectorAll('[data-fig]').forEach((r) => {
      if (r.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING) initFigure(r);
    });
  };

  function boot() {
    applyLang();
    const params = new URLSearchParams(location.search);
    const solo = params.get('solo');
    if (solo) {
      document.documentElement.classList.add('solo');
      const f = document.querySelector(`[data-fig="${solo}"]`);
      if (f) { f.classList.add('solo-show'); f.closest('section') && f.closest('section').classList.add('solo-show'); }
    }
    const roots = [...document.querySelectorAll('[data-fig]')].filter((r) => !solo || r.getAttribute('data-fig') === solo);
    const eager = params.has('eager') || solo || !window.IntersectionObserver;
    if (eager) { roots.forEach(initFigure); return; }
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.isIntersecting) { io.unobserve(en.target); initFigure(en.target); } });
    }, { rootMargin: '900px 0px' });
    roots.forEach((r) => io.observe(r));
    if (location.hash) {
      const t = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (t) { OX.initBefore(t); requestAnimationFrame(() => t.scrollIntoView()); }
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else setTimeout(boot, 0);
})();
