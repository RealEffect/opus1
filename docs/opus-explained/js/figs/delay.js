/* Figure 16: the delay budget = frame duration + CELT window overlap (look-ahead) + the encoder's delay compensation.
 * These are the libopus / opus1 encoder numbers: overlap Fs/400 = 2.5 ms; delay_compensation Fs/250 = 4 ms,
 * set by the application type alone and 0 for RestrictedLowDelay, RestrictedSilk and RestrictedCelt (src/lib.rs:2074, 3020;
 * opus_encoder.c:313, 1903-1906). The lookahead is 2.5 ms for RestrictedLowDelay and RestrictedCelt (src/lib.rs:3193-3203).
 * Frames under 10 ms are always CELT-only, but under Voip/Audio they still carry the 4 ms.
 */
OX.fig('delay', function (root, body) {
  const { el, segmented } = OX.ui;
  const OVERLAP = 2.5; // ms: CELT window overlap (120 samples at 48 kHz)
  const COMP = 4; // ms: encoder delay compensation, Fs/250 (0 for RestrictedLowDelay)
  const FRAMES = [2.5, 5, 10, 20, 40, 60];
  const MAX = 60 + OVERLAP + COMP; // the full track width
  let frame = 20, app = 'audio';

  OX.ui.css('delay', `
    .dl-bars { display: grid; gap: 10px; margin: 14px 0 6px; }
    .dl-row { display: grid; grid-template-columns: 7.5em minmax(0, 1fr); gap: 12px; align-items: center; font-size: var(--small); }
    :root[data-lang="en"] .dl-row { grid-template-columns: 8.5em minmax(0, 1fr); } /* the English row name is wider */
    .dl-row .nm { color: var(--ink-2); }
    .dl-track { position: relative; height: 22px; background: var(--sunken); border-radius: 5px; overflow: hidden; }
    .dl-seg { position: absolute; top: 0; bottom: 0; display: flex; align-items: center; justify-content: center; font: 500 0.7rem/1 var(--font-mono); color: var(--surface); white-space: nowrap; overflow: hidden; transition: left .25s ease, width .25s ease; }
    .dl-seg span { padding: 0 3px; }
    .dl-seg.f { background: var(--accent); }
    .dl-seg.o { background: var(--celt); }
    .dl-seg.c { background: var(--silk); }
    .dl-ros { display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: flex-end; }
    .dl-total { font: 600 1.3rem/1 var(--font-mono); font-variant-numeric: tabular-nums; }
    .dl-total small { font-size: 0.6em; color: var(--ink-2); margin-left: 4px; }
    .dl-note { margin: 10px 0 0; font-size: var(--small); color: var(--ink-2); line-height: 1.65; }
    @media (max-width: 520px) { .dl-row, :root[data-lang="en"] .dl-row { grid-template-columns: 1fr; gap: 4px; } }
    @media (prefers-reduced-motion: reduce) { .dl-seg { transition: none; } }
  `);

  const total = el('div', { class: 'dl-total' });
  const ahead = el('div', { class: 'dl-total' });
  const controls = el('div', { class: 'ox-controls' }, [
    segmented({ id: 'dl-frame', label: OX.t('帧长', 'Frame duration'), value: String(frame), options: FRAMES.map((f) => ({ value: String(f), label: f + ' ms' })), onChange: (v) => { frame = +v; render(); } }),
    segmented({ id: 'dl-app', label: OX.t('应用类型', 'Application'), value: app, options: [{ value: 'audio', label: 'Voip / Audio' }, { value: 'rld', label: OX.t('RestrictedLowDelay（受限低延迟）', 'RestrictedLowDelay') }], onChange: (v) => { app = v; render(); } }),
  ]);

  // One track with three segments that stay in place, so a change slides them instead of rebuilding the bar.
  const SEGS = [
    { cls: 'f', name: OX.t('一整帧', 'Full frame') },
    { cls: 'o', name: OX.t('CELT 窗口重叠', 'CELT window overlap') },
    { cls: 'c', name: OX.t('延迟补偿', 'Delay compensation') },
  ].map((s) => {
    const label = el('span');
    return { ...s, label, node: el('div', { class: 'dl-seg ' + s.cls, 'aria-hidden': 'true' }, [label]) };
  });
  const track = el('div', { class: 'dl-track', role: 'img' }, SEGS.map((s) => s.node));
  const note = el('p', { class: 'dl-note', 'aria-live': 'polite' });

  body.append(
    controls,
    el('div', { class: 'dl-bars' }, [el('div', { class: 'dl-row' }, [el('span', { class: 'nm', text: OX.t('编码 + 解码', 'Encode + decode') }), track])]),
    el('div', { class: 'ox-readouts dl-ros' }, [
      el('div', {}, [el('div', { class: 'ox-ro-label', text: OX.t('算法延迟合计', 'Total algorithmic delay') }), total]),
      el('div', {}, [el('div', { class: 'ox-ro-label', text: OX.t('其中前瞻（重叠 + 补偿）', 'Lookahead part (overlap + compensation)') }), ahead]),
    ]),
    note,
    OX.ui.legend([
      { label: OX.t('等一整帧到齐', 'Waiting for the whole frame to arrive'), color: 'var(--accent)', kind: 'box' },
      { label: OX.t('CELT 窗口重叠：2.5 毫秒前瞻', 'CELT window overlap: 2.5 ms lookahead'), color: 'var(--celt)', kind: 'box' },
      { label: OX.t('延迟补偿：CELT 输入再推迟 4 毫秒，与 SILK 对齐（也算前瞻）', 'Delay compensation: the CELT input is delayed by another 4 ms to line up with SILK (also counts as lookahead)'), color: 'var(--silk)', kind: 'box' },
    ]),
  );

  // Label a segment "N ms", or just "N", whichever fits its final width in pixels (not its share of the bar);
  // hide the label when neither fits. Runs again whenever the track is resized.
  function fitLabels() {
    const tw = track.clientWidth;
    for (const s of SEGS) {
      const px = (tw * s.v) / MAX;
      let fits = false;
      for (const t of [s.v + ' ms', String(s.v)]) {
        s.label.textContent = t;
        if (s.v > 0 && s.label.offsetWidth + 2 <= px) { fits = true; break; }
      }
      s.label.style.visibility = fits ? '' : 'hidden';
    }
  }

  function render() {
    const comp = app === 'rld' ? 0 : COMP;
    const look = OVERLAP + comp, sum = frame + look;
    let x = 0;
    [frame, OVERLAP, comp].forEach((v, i) => {
      const s = SEGS[i];
      s.v = v;
      s.node.style.left = (x / MAX) * 100 + '%';
      s.node.style.width = (v / MAX) * 100 + '%';
      s.node.title = `${s.name} ${v} ms`;
      x += v;
    });
    fitLabels();
    track.setAttribute('aria-label', OX.t(`算法延迟 ${sum} 毫秒：一整帧 ${frame} 毫秒，CELT 窗口重叠 ${OVERLAP} 毫秒，延迟补偿 ${comp} 毫秒`,
      `Algorithmic delay ${sum} ms: full frame ${frame} ms, CELT window overlap ${OVERLAP} ms, delay compensation ${comp} ms`));
    total.innerHTML = `${sum}<small>ms</small>`;
    ahead.innerHTML = `${look}<small>ms</small>`;
    const sumText = comp ? OX.t(`${frame} + ${OVERLAP} + ${comp} = ${sum} 毫秒`, `${frame} + ${OVERLAP} + ${comp} = ${sum} ms`)
      : OX.t(`${frame} + ${OVERLAP} = ${sum} 毫秒`, `${frame} + ${OVERLAP} = ${sum} ms`);
    if (app === 'rld') {
      note.textContent = OX.t(`受限低延迟（RestrictedLowDelay）只用 CELT，不需要和 SILK 对齐，所以没有 4 毫秒补偿：${sumText}。`,
        `RestrictedLowDelay uses only CELT and does not need to line up with SILK, so there is no 4 ms compensation: ${sumText}.`) +
        (frame === FRAMES[0] ? OX.t('这是 Opus 能做到的最低算法延迟。', ' This is the lowest algorithmic delay Opus can reach.') : '');
    } else {
      note.textContent = OX.t(`Voip 和 Audio 都会把 CELT 的输入推迟 4 毫秒，好和 SILK 对齐：${sumText}。`,
        `Voip and Audio both delay the CELT input by 4 ms to line up with SILK: ${sumText}.`) +
        (frame < 10 ? OX.t(`帧短于 10 毫秒时 SILK 编不了，只能用 CELT，但这 4 毫秒补偿仍然保留。`, ` SILK cannot code frames shorter than 10 ms, so only CELT is used, but the 4 ms compensation stays.`) : '');
    }
  }
  const ro = window.ResizeObserver ? new ResizeObserver(fitLabels) : null;
  if (ro) ro.observe(track);
  render();
  return () => { if (ro) ro.disconnect(); };
});
