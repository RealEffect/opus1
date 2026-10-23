/* Figure 8: pyramid vector quantization in three dimensions.
 * Codewords are all integer vectors y with |y1|+|y2|+|y3| = K. They sit on an octahedron (the "pyramid");
 * normalised, they become points on the unit sphere. The encoder picks the codeword closest in angle to the
 * target shape, with CELT's search: a projection to get close, then pulses added one at a time.
 */
OX.fig('pvq', function (root, body) {
  const { el, slider, segmented, readout, button } = OX.ui;
  // Target directions the "换一个目标" button cycles through. The second one shows a miss of the fast search at K = 4.
  const PRESETS = [[0.63, 0.31, 0.71], [0.3, 0.3, 1], [-0.55, 0.8, 0.3], [0.25, -0.9, -0.45]].map(norm);
  let preset = 0;
  const st = { K: 4, view: 'sphere', yaw: -0.65, pitch: 0.42, target: PRESETS[0] };
  let book = [], picked = null, best = null, geo = null;

  function norm(v) { const n = Math.hypot(...v) || 1; return v.map((c) => c / n); }

  /** All integer vectors of dimension 3 with L1 norm K. */
  function codebook(K) {
    const out = [];
    for (let a = -K; a <= K; a++) {
      const r = K - Math.abs(a);
      for (let b = -r; b <= r; b++) {
        const c = r - Math.abs(b);
        out.push([a, b, c]);
        if (c !== 0) out.push([a, b, -c]);
      }
    }
    return out;
  }
  /** V(N,K) by the recurrence V(N,K) = V(N-1,K) + V(N,K-1) + V(N-1,K-1). */
  function V(N, K) {
    const t = [];
    for (let n = 0; n <= N; n++) { t[n] = []; for (let k = 0; k <= K; k++) t[n][k] = k === 0 ? 1 : n === 0 ? 0 : t[n - 1][k] + t[n][k - 1] + t[n - 1][k - 1]; }
    return t[N][K];
  }

  /**
   * CELT-style PVQ search (pvq_search_scalar in src/pvq.rs): work on |x|, project to K' <= K pulses when K > N/2,
   * then add the remaining pulses greedily, each time maximising (x.y)^2 / (y.y). Signs are restored at the end.
   */
  function pvqSearch(x, K) {
    const N = x.length, ax = x.map(Math.abs), y = new Array(N).fill(0);
    let xy = 0, yy = 0, left = K;
    if (K > N / 2) {
      const sum = ax.reduce((a, b) => a + b, 0) || 1;
      const r = (K + 0.8) / sum; // as in opus1 pvq_search_scalar / libopus op_pvq_search_c (float build)
      for (let i = 0; i < N; i++) { y[i] = Math.floor(r * ax[i]); left -= y[i]; xy += ax[i] * y[i]; yy += y[i] * y[i]; }
    }
    while (left > 0) {
      let best = 0, bestNum = -1, bestDen = 1;
      for (let i = 0; i < N; i++) {
        const num = Math.pow(xy + ax[i], 2), den = yy + 2 * y[i] + 1;
        if (num * bestDen > bestNum * den) { best = i; bestNum = num; bestDen = den; }
      }
      xy += ax[best]; yy += 2 * y[best] + 1; y[best] += 1; left--;
    }
    return y.map((v, i) => (x[i] < 0 ? -v : v));
  }
  const cosAng = (a, b) => { const na = Math.hypot(...a), nb = Math.hypot(...b); return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb); };

  function solve() {
    book = codebook(st.K);
    picked = pvqSearch(st.target, st.K);
    // exhaustive search, to show when the fast search misses the closest codeword
    let bc = -2;
    for (const y of book) { const c = cosAng(st.target, y); if (c > bc + 1e-12) { bc = c; best = y; } }
    if (cosAng(st.target, picked) >= bc - 1e-12) best = picked;
  }

  // ---- UI ----
  const host = el('div', { class: 'pvq-stage' });
  const roV = readout(OX.t('码本大小 V(3, K)', 'Codebook size V(3, K)'), '');
  const roBits = readout(OX.t('传一个序号需要', 'Sending one index takes'), '');
  const roY = readout(OX.t('选中的整数向量 y', 'Chosen integer vector y'), '');
  const roErr = readout(OX.t('与目标的夹角', 'Angle to the target'), '');
  const roBest = readout(OX.t('逐个比较得到的最近码字', 'Closest codeword by exhaustive search'), '');
  const nextTarget = button({ label: OX.t('换一个目标', 'New target'), onClick: () => { preset = (preset + 1) % PRESETS.length; st.target = PRESETS[preset]; solve(); update(); } });
  const legend = OX.ui.legend([
    { label: OX.t('码字', 'Codewords'), color: 'var(--ink-3)', kind: 'box' },
    { label: OX.t('目标形状', 'Target shape'), color: 'var(--accent)' },
    { label: OX.t('选中的码字', 'Chosen codeword'), color: 'var(--silk)', kind: 'box' },
    { label: OX.t('真正最近的码字（和选中的不同时才画）', 'Actual closest codeword (drawn only when it differs from the chosen one)'), color: 'var(--accent)', kind: 'box' },
  ]);
  legend.lastChild.classList.add('pvq-ring');
  body.append(
    el('div', { class: 'ox-controls' }, [
      slider({ id: 'pvq-k', label: OX.t('脉冲数 K', 'Number of pulses K'), min: 1, max: 12, step: 1, value: st.K, format: (v) => String(v), onInput: (v) => { st.K = v; solve(); update(); } }),
      segmented({ id: 'pvq-view', label: OX.t('显示', 'Show'), value: st.view, options: [{ value: 'sphere', label: OX.t('归一化到单位球面', 'Unit sphere') }, { value: 'pyramid', label: OX.t('金字塔表面（整数点）', 'Pyramid (integer points)') }], onChange: (v) => { st.view = v; cv.redraw(); } }),
      nextTarget,
    ]),
    host,
    el('div', { class: 'ox-readouts', 'aria-live': 'polite' }, [roV, roBits, roY, roErr, roBest]),
    el('p', { class: 'ox-note', text: OX.t('拖动画面可以旋转视角；点一下球面上的一点，可以把它设为要编码的形状。用键盘时，让图获得焦点后按方向键旋转；“换一个目标”按钮会轮流换几个预设的形状。', 'Drag the figure to rotate the view. Click or tap a point on the sphere to make it the shape to code. With a keyboard, give the figure focus and use the arrow keys to rotate. The “New target” button cycles through a few preset shapes.') }),
    legend);
  OX.ui.css('pvq', `
    .pvq-stage canvas { cursor: grab; touch-action: none; }
    .pvq-stage canvas:active { cursor: grabbing; }
    .pvq-stage canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .pvq-ring .ox-sw.box { background: transparent; border: 1.5px solid var(--c); border-radius: 50%; }
  `);

  solve();
  const cv = OX.view.canvas(host, { height: (w) => Math.min(420, Math.max(280, Math.round(w * 0.5))), label: OX.t('三维 PVQ 码本与选中的码字。按方向键可以旋转视角', '3D PVQ codebook and the chosen codeword. Use the arrow keys to rotate the view') }, draw);
  cv.canvas.tabIndex = 0;

  // rotation: world -> view
  function rot(p) {
    const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
    const x1 = cy * p[0] + sy * p[1], y1 = -sy * p[0] + cy * p[1], z1 = p[2];
    // pitch around the view x axis
    return [x1, cp * y1 - sp * z1, sp * y1 + cp * z1]; // [screen x, depth, screen up]
  }
  function unrot(v) {
    const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
    const x1 = v[0], y1 = cp * v[1] + sp * v[2], z1 = -sp * v[1] + cp * v[2];
    return [cy * x1 - sy * y1, sy * x1 + cy * y1, z1];
  }

  function draw(ctx, w, h) {
    const T = OX.theme;
    const R = Math.min(w * 0.42, h * 0.42), cx = w / 2, cy = h / 2;
    geo = { R, cx, cy };
    const P = (p, scale) => { const v = rot(p); return { x: cx + v[0] * R * scale, y: cy - v[2] * R * scale, d: v[1] }; };
    const pyr = st.view === 'pyramid';
    const S = pyr ? 1 / st.K : 1; // pyramid: integer points scaled so that vertices sit at radius 1

    // sphere outline + a few great circles
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    const circle = (fn, color, dashFront) => {
      for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i <= 96; i++) {
          const q = P(fn((i / 96) * Math.PI * 2), 1);
          const front = q.d < 0;
          if ((pass === 0) !== front) { started = false; continue; }
          if (!started) { ctx.moveTo(q.x, q.y); started = true; } else ctx.lineTo(q.x, q.y);
        }
        ctx.strokeStyle = color; ctx.globalAlpha = pass === 1 ? 0.9 : 0.35; ctx.setLineDash(pass === 0 && dashFront ? [2, 4] : []);
        ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    };
    if (!pyr) {
      circle((t) => [Math.cos(t), Math.sin(t), 0], T.grid, true);
      circle((t) => [Math.cos(t), 0, Math.sin(t)], T.grid, true);
      circle((t) => [0, Math.cos(t), Math.sin(t)], T.grid, true);
    } else {
      // octahedron edges
      const V6 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      ctx.lineWidth = 1;
      for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) {
        if (V6[i].some((c, k) => c !== 0 && V6[j][k] !== 0)) continue;
        const a = P(V6[i], 1), b = P(V6[j], 1);
        ctx.strokeStyle = OX.alpha(T.ink3, (a.d + b.d) / 2 < 0 ? 0.7 : 0.25);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    // axes (their names are drawn last, so codewords do not cover them)
    const AXES = [[1.18, 0, 0, 'x₁'], [0, 1.18, 0, 'x₂'], [0, 0, 1.18, 'x₃']].map(([a, b, c, name]) => ({ e: P([a, b, c], 1), name }));
    {
      const o = P([0, 0, 0], 1);
      ctx.strokeStyle = OX.alpha(T.ink3, 0.5); ctx.setLineDash([3, 3]);
      for (const { e } of AXES) { ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(e.x, e.y); ctx.stroke(); }
      ctx.setLineDash([]);
    }

    // codewords, back to front
    const pts = book.map((y) => {
      const n = pyr ? y.map((c) => c * S) : norm(y);
      return { y, q: P(n, 1) };
    }).sort((a, b) => b.q.d - a.q.d);
    const sel = picked.join(',');
    const rDot = Math.max(1.6, Math.min(3.4, 9 / Math.sqrt(st.K + 1)));
    for (const p of pts) {
      if (p.y.join(',') === sel) continue;
      const front = p.q.d < 0;
      ctx.fillStyle = OX.alpha(T.ink3, front ? 0.95 : 0.3);
      ctx.beginPath(); ctx.arc(p.q.x, p.q.y, front ? rDot : rDot * 0.8, 0, Math.PI * 2); ctx.fill();
    }
    // target vector and chosen codeword
    const o = P([0, 0, 0], 1);
    const tq = P(pyr ? scaleToL1(st.target) : st.target, 1);
    ctx.strokeStyle = T.accent; ctx.fillStyle = T.accent; ctx.lineWidth = 2.2;
    const len = Math.hypot(tq.x - o.x, tq.y - o.y);
    if (len > 14) {
      // arrow: the shaft stops at the base of the head
      const ux = (tq.x - o.x) / len, uy = (tq.y - o.y) / len, hl = 11, hw = 4.5;
      const bx = tq.x - ux * hl, by = tq.y - uy * hl;
      ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(bx, by); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tq.x, tq.y); ctx.lineTo(bx - uy * hw, by + ux * hw); ctx.lineTo(bx + uy * hw, by - ux * hw); ctx.closePath(); ctx.fill();
    } else {
      // seen almost end-on: no room for a shaft, mark the tip
      ctx.beginPath(); ctx.arc(tq.x, tq.y, 4.2, 0, Math.PI * 2); ctx.fill();
    }
    const pn = pyr ? picked.map((c) => c * S) : norm(picked);
    const pq = P(pn, 1);
    ctx.strokeStyle = T.silk; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(pq.x, pq.y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = T.silk; ctx.strokeStyle = T.surface; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(pq.x, pq.y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // label of the chosen codeword: right of the point, or left of it when it would leave the canvas
    const lab = `(${picked.join(', ')})`;
    ctx.font = OX.font(11.5, 'mono', 500); ctx.textBaseline = 'bottom';
    const lw = ctx.measureText(lab).width, toRight = pq.x + 9 + lw <= w - 4;
    const lx = toRight ? pq.x + 9 : Math.max(4 + lw, pq.x - 9), ly = Math.max(16, pq.y - 6);
    ctx.textAlign = toRight ? 'left' : 'right';
    ctx.lineJoin = 'round'; ctx.lineWidth = 3; ctx.strokeStyle = OX.alpha(T.surface, 0.9);
    ctx.strokeText(lab, lx, ly);
    ctx.fillStyle = T.silk; ctx.fillText(lab, lx, ly);
    if (best !== picked) {
      const bq = P(pyr ? best.map((c) => c * S) : norm(best), 1);
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(bq.x, bq.y, 8, 0, Math.PI * 2); ctx.stroke();
    }
    // axis names, with a halo in the surface colour
    ctx.font = OX.font(11, 'mono'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round'; ctx.lineWidth = 3; ctx.strokeStyle = OX.alpha(T.surface, 0.9); ctx.fillStyle = T.ink3;
    for (const { e, name } of AXES) { ctx.strokeText(name, e.x, e.y); ctx.fillText(name, e.x, e.y); }
  }
  function scaleToL1(v) { const s = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]); return v.map((c) => c / s); }

  function update() {
    const v = V(3, st.K);
    roV.set(OX.t(`${v.toLocaleString('en-US')} 个`, `${v.toLocaleString('en-US')} codewords`));
    roBits.set(OX.t(`${Math.log2(v).toFixed(2)} 比特`, `${Math.log2(v).toFixed(2)} bits`));
    roY.set(`(${picked.join(', ')})`);
    const deg = (y) => (Math.acos(Math.min(1, cosAng(st.target, y))) * 180 / Math.PI).toFixed(1) + '°';
    roErr.set(deg(picked));
    roBest.set(best === picked ? OX.t('同一个', 'Same as the chosen one') : OX.t(`(${best.join(', ')})，${deg(best)}`, `(${best.join(', ')}), ${deg(best)}`));
    cv.redraw();
  }

  // ---- interaction: drag rotates, click sets the target, arrow keys rotate ----
  let drag = null;
  const canvas = cv.canvas;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: st.yaw, pitch: st.pitch, moved: false }; canvas.setPointerCapture(e.pointerId); spin.stop(); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (!drag.moved) return;
    st.yaw = drag.yaw + dx * 0.01;
    st.pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + dy * 0.01));
    cv.redraw();
  });
  canvas.addEventListener('pointerup', (e) => {
    if (drag && !drag.moved && geo) {
      const r = canvas.getBoundingClientRect();
      const sx = (e.clientX - r.left - geo.cx) / geo.R, sz = -(e.clientY - r.top - geo.cy) / geo.R;
      const s2 = sx * sx + sz * sz;
      if (s2 <= 1) {
        const v = [sx, -Math.sqrt(1 - s2), sz]; // front hemisphere (depth < 0 faces the viewer)
        st.target = norm(unrot(v));
        solve(); update();
      }
    }
    drag = null;
  });
  canvas.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!d) return;
    e.preventDefault();
    spin.stop();
    st.yaw += d[0] * 0.12;
    st.pitch = Math.max(-1.4, Math.min(1.4, st.pitch + d[1] * 0.12));
    cv.redraw();
  });

  // A slow spin when the figure first comes into view: it stops by itself after 5 s of motion (WCAG 2.2.2),
  // and at once when the reader touches the canvas or moves keyboard focus into the figure.
  let spun = 0;
  const spin = OX.loop(host, (dt) => { spun += dt; st.yaw += dt * 0.25; cv.redraw(); return spun < 5; });
  body.addEventListener('focusin', () => spin.stop());
  if (!OX.reducedMotion()) spin.start();
  update();
});
