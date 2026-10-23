/* Figure 11: range coding with the real Opus integer arithmetic.
 *
 * The encoder and decoder below follow src/range_coder.rs (and libopus celt/entenc.c, celt/entdec.c) step for step:
 * a 32-bit interval (val = low end, rng = width), renormalisation one byte at a time once rng <= 2^23, a one-byte
 * output buffer (rem) plus a count of pending 0xFF bytes (ext) so that a carry can still reach them, ec_enc_done,
 * and tell_frac (bits used, in 1/8 bit).
 *
 * The figure encodes a short sequence of the symbols A/B/C with a 3-entry ICDF table and shows it twice:
 *   - on the canvas, as nested intervals of [0, 1): each row stretches the previous row's chosen piece to full width;
 *   - in a table, as the coder's integer state after every symbol.
 * The absolute interval ends are tracked exactly with BigInt, so the decimals on the canvas are exact.
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* Range coder (RS: src/range_coder.rs; C: celt/entenc.c, entdec.c)    */
  /* ------------------------------------------------------------------ */
  const SYM_BITS = 8;
  const CODE_TOP = 0x80000000; // 2^31: initial rng, and the carry bit of val
  const CODE_BOT = 0x00800000; // 2^23: renormalise while rng <= this
  const CODE_SHIFT = 23; // val >>> 23 = carry bit + top 8 bits
  const CODE_EXTRA = 7; // bits of the first byte the decoder uses at init
  const CORRECTION = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
  const ilog = (x) => 32 - Math.clz32(x);

  /** ec_tell_frac: whole bits so far minus log2(rng), in 1/8 bit (RS range_coder.rs tell_frac). */
  function tellFrac(nbits, rng) {
    const l = ilog(rng);
    const r = rng >>> (l - 16);
    let b = (r >>> 12) - 8;
    if (r > CORRECTION[b]) b++;
    return nbits * 8 - l * 8 - b;
  }

  class Encoder {
    constructor() {
      this.rng = CODE_TOP; this.val = 0; this.rem = -1; this.ext = 0; this.nbits = 33;
      this.out = []; // bytes written so far (front of the buffer)
      this.events = null; // when an array, normalize() records what it did
    }
    /** Write rem (+carry) and the pending 0xFF bytes, or count one more 0xFF. */
    carryOut(c) {
      if (c !== 255) {
        const carry = c >>> SYM_BITS;
        if (this.rem >= 0) this.out.push((this.rem + carry) & 255);
        if (this.ext > 0) {
          const sym = (255 + carry) & 255;
          for (let j = 0; j < this.ext; j++) this.out.push(sym);
          this.ext = 0;
        }
        this.rem = c & 255;
      } else this.ext++;
    }
    normalize() {
      while (this.rng <= CODE_BOT) {
        const c = this.val >>> CODE_SHIFT;
        const n0 = this.out.length;
        if (this.events) this.events.push({ c, valBefore: this.val, rngBefore: this.rng });
        this.carryOut(c);
        if (this.events) this.events[this.events.length - 1].wrote = this.out.slice(n0);
        this.val = (this.val << SYM_BITS) & 0x7fffffff;
        this.rng = (this.rng << SYM_BITS) >>> 0;
        this.nbits += SYM_BITS;
      }
    }
    /** ec_encode: symbol 0 (fl = 0) also gets the rounding remainder rng - r*ft. */
    encode(fl, fh, ft) {
      const r = Math.floor(this.rng / ft);
      if (fl > 0) {
        this.val = (this.val + (this.rng - r * (ft - fl))) >>> 0;
        this.rng = (r * (fh - fl)) >>> 0;
      } else this.rng = (this.rng - r * (ft - fh)) >>> 0;
      this.normalize();
    }
    /** ec_enc_icdf: the same with ft = 2^ftb, so the division becomes a shift. */
    encodeIcdf(s, icdf, ftb) {
      const r = this.rng >>> ftb;
      if (s > 0) {
        this.val = (this.val + (this.rng - r * icdf[s - 1])) >>> 0;
        this.rng = (r * (icdf[s - 1] - icdf[s])) >>> 0;
      } else this.rng = (this.rng - r * icdf[s]) >>> 0;
      this.normalize();
    }
    tell() { return this.nbits - ilog(this.rng); }
    tellFrac() { return tellFrac(this.nbits, this.rng); }
    /** ec_enc_done without raw bits: pick the number in [val, val+rng) with the most trailing zeros, flush. */
    done() {
      let l = 32 - ilog(this.rng);
      let msk = 0x7fffffff >>> l;
      let end = (((this.val + msk) >>> 0) & ~msk) >>> 0;
      if (((end | msk) >>> 0) >= ((this.val + this.rng) >>> 0)) {
        l++;
        msk >>>= 1;
        end = (((this.val + msk) >>> 0) & ~msk) >>> 0;
      }
      const info = { end, l, before: this.out.length };
      while (l > 0) {
        this.carryOut(end >>> CODE_SHIFT);
        end = (end << SYM_BITS) & 0x7fffffff;
        l -= SYM_BITS;
      }
      if (this.rem >= 0 || this.ext > 0) this.carryOut(0);
      return info;
    }
  }

  class Decoder {
    constructor(bytes) {
      this.buf = bytes; this.offs = 0; this.ext = 0;
      this.nbits = 32 + 1 - Math.floor((32 - CODE_EXTRA) / SYM_BITS) * SYM_BITS; // 9
      this.rng = 1 << CODE_EXTRA;
      this.rem = this.readByte();
      this.val = (this.rng - 1 - (this.rem >>> (SYM_BITS - CODE_EXTRA))) >>> 0;
      this.normalize();
    }
    readByte() { return this.offs < this.buf.length ? this.buf[this.offs++] : 0; } // past the end it reads 0
    normalize() {
      while (this.rng <= CODE_BOT) {
        this.nbits += SYM_BITS;
        this.rng = (this.rng << SYM_BITS) >>> 0;
        const sym = this.rem;
        this.rem = this.readByte();
        const comb = ((sym << SYM_BITS) | this.rem) >>> (SYM_BITS - CODE_EXTRA);
        this.val = ((this.val << SYM_BITS) + (255 & ~comb)) & 0x7fffffff;
      }
    }
    /** ec_dec_icdf: walk the table with multiplications only. */
    decodeIcdf(icdf, ftb) {
      let s = this.rng, t = s, ret = 0;
      const d = this.val, r = s >>> ftb;
      do { t = s; s = r * icdf[ret++]; } while (d < s);
      this.val = (d - s) >>> 0;
      this.rng = (t - s) >>> 0;
      this.normalize();
      return ret - 1;
    }
    /** ec_decode + ec_dec_update, for a general total ft. */
    decode(ft) {
      this.ext = Math.floor(this.rng / ft);
      const s = Math.floor(this.val / this.ext);
      return ft - Math.min(s + 1, ft);
    }
    update(fl, fh, ft) {
      const s = this.ext * (ft - fh);
      this.val = (this.val - s) >>> 0;
      this.rng = fl > 0 ? (this.ext * (fh - fl)) >>> 0 : (this.rng - s) >>> 0;
      this.normalize();
    }
    tellFrac() { return tellFrac(this.nbits, this.rng); }
  }

  /** Probability counts of an ICDF table: f[k] = icdf[k-1] - icdf[k], with icdf[-1] = 2^ftb. */
  const freqs = (icdf, ftb) => icdf.map((v, k) => (k === 0 ? 1 << ftb : icdf[k - 1]) - v);

  /**
   * Encode `seq` with one ICDF table and record everything the figure shows. The absolute interval is
   * [L / 2^E, (L + rng) / 2^E): L grows by the same amount as val and gains 8 bits at every renormalisation.
   */
  function trace(seq, icdf, ftb) {
    const enc = new Encoder();
    const f = freqs(icdf, ftb);
    let L = 0n, E = 31;
    const snap = () => ({ val: enc.val, rng: enc.rng, rem: enc.rem, ext: enc.ext, out: enc.out.slice(), tf: enc.tellFrac(), L, E });
    const start = snap();
    const steps = [];
    let ideal = 0;
    for (const s of seq) {
      const rng0 = enc.rng, L0 = L, E0 = E;
      const r = rng0 >>> ftb;
      const cuts = [0];
      for (let k = 1; k < icdf.length; k++) cuts.push(rng0 - r * icdf[k - 1]);
      cuts.push(rng0);
      enc.events = [];
      enc.encodeIcdf(s, icdf, ftb);
      L = L0 + BigInt(cuts[s]);
      for (let i = 0; i < enc.events.length; i++) { L <<= 8n; E += 8; }
      const bits = -Math.log2(f[s] / (1 << ftb));
      ideal += bits;
      steps.push(Object.assign(snap(), { sym: s, r, rng0, L0, E0, cuts, events: enc.events, bits }));
      enc.events = null;
    }
    const end = snap();
    const doneInfo = enc.done();
    const bytes = enc.out.slice();
    const dec = new Decoder(bytes);
    const decoded = [], decRng = [];
    for (let i = 0; i < seq.length; i++) { decoded.push(dec.decodeIcdf(icdf, ftb)); decRng.push(dec.rng); }
    return { start, steps, end, doneInfo, bytes, decoded, decRng, ideal, tell: end.tf };
  }

  /* ------------------------------------------------------------------ */
  /* Exact decimals of N / 2^E (BigInt)                                  */
  /* ------------------------------------------------------------------ */
  const MAX_DIGITS = 14;
  function decStr(q, k) { // q / 10^k as a decimal string, trailing zeros removed
    if (k <= 0) return q.toString();
    const s = q.toString().padStart(k + 1, '0');
    const ip = s.slice(0, s.length - k), fp = s.slice(s.length - k).replace(/0+$/, '');
    return fp ? ip + '.' + fp : ip;
  }
  /** N / 2^E exactly when it has at most MAX_DIGITS decimals, else rounded to d decimals with "≈". */
  function frac(N, E, d) {
    if (N === 0n) return '0';
    let M = N, k = E;
    while (k > 0 && (M & 1n) === 0n) { M >>= 1n; k--; }
    if (k <= MAX_DIGITS) return decStr(M * 5n ** BigInt(k), k);
    const q = (N * 10n ** BigInt(d) * 2n + (1n << BigInt(E))) >> BigInt(E + 1);
    return '≈' + decStr(q, d);
  }
  /** Decimals needed to tell the two ends of an interval of width rng / 2^E apart, plus a margin. */
  function digitsFor(rng, E) { return Math.max(3, Math.min(MAX_DIGITS, Math.ceil(E * Math.log10(2) - Math.log10(rng)) + 3)); }

  const eq = (num) => (num[0] === '≈' ? '≈ ' + num.slice(1) : '= ' + num);
  const hex8 = (v) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(8, '0');
  const hex2 = (v) => v.toString(16).toUpperCase().padStart(2, '0');
  const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (n) => String(n).split('').map((c) => SUP[c]).join('');

  /**
   * Text of the table's last row: what ec_enc_done writes, in the order it writes it. The byte(s) still in the
   * buffer (rem and any pending 0xFF) go out first, possibly raised by a carry from `end`; then end's bytes.
   */
  function doneText(res) {
    const di = res.doneInfo, e = res.end;
    const written = res.bytes.slice(e.out.length);
    const bufOld = (e.rem >= 0 ? [e.rem] : []).concat(new Array(e.ext).fill(255));
    const bufNew = written.slice(0, bufOld.length), endBytes = written.slice(bufOld.length);
    const hx = (a) => `<span class="m">${a.map(hex2).join(' ')}</span>`;
    const carried = bufNew.some((b, i) => b !== bufOld[i]);
    const buf = hx(bufOld) + (carried ? OX.t(`（收尾时进位加 1，变成 ${hx(bufNew)}）`, ` (a carry at the end adds 1, making it ${hx(bufNew)})`) : '');
    if (di.l <= 0) return OX.t(`收尾：缓冲区里的字节后面全补 0，就已经落在最后的区间里。不用再写新的比特，只把缓冲区里的 ${buf} 写出。`, `Finish: the bytes in the buffer, followed by zeros, already lie in the final interval. No new bits are needed: only the buffered ${buf} is written out.`);
    const bits = OX.t(`第 30${di.l > 1 ? ' 到 ' + (31 - di.l) : ''} 位`, `bit${di.l > 1 ? 's' : ''} 30${di.l > 1 ? ' to ' + (31 - di.l) : ''}`); // the l bits just below the carry bit
    const nBits = `${di.l} bit${di.l === 1 ? '' : 's'}`, nBytes = `${endBytes.length} byte${endBytes.length === 1 ? '' : 's'}`;
    const head = OX.t(`收尾：在最后的区间里挑一个低位尽量多为 0 的数 <code>end = ${hex8(di.end)}</code>。它只有进位位下面的 ${di.l} 位（${bits}）要紧。`, `Finish: the encoder picks the number in the final interval with the most trailing zero bits, <code>end = ${hex8(di.end)}</code>. Only its ${nBits} just below the carry bit (${bits}) ${di.l === 1 ? 'matters' : 'matter'}. `);
    const tail = OX.t(`把这 ${di.l} 位补成 ${endBytes.length} 个字节 ${hx(endBytes)}`, `pads ${di.l === 1 ? 'this' : 'these'} ${nBits} to ${nBytes}, ${hx(endBytes)}, and writes the result`);
    return head + (bufOld.length ? OX.t(`先写出缓冲区里原有的 ${buf}，再${tail} 写在后面。`, `It first writes out what is already in the buffer, ${buf}. Then it ${tail} after it.`) : OX.t(`${tail} 写出。`, `It ${tail} out.`));
  }

  /* ------------------------------------------------------------------ */
  /* Figure                                                              */
  /* ------------------------------------------------------------------ */
  OX.fig('range-coder', function (root, body) {
    const { el, segmented, button, readout } = OX.ui;
    const NAMES = ['A', 'B', 'C'];
    const MODELS = {
      toy: { label: OX.t('示例表（16 份）', 'Example table (total 16)'), icdf: [6, 2, 0], ftb: 4 },
      tapset: { label: OX.t('CELT 抽头组表（4 份）', 'CELT tapset table (total 4)'), icdf: [2, 1, 0], ftb: 2 },
    };
    const MODEL_NOTE = {
      toy: OX.t('ICDF（反向累积分布）表 {6, 2, 0}，总份数 2⁴ = 16：A 占 16 − 6 = 10 份，B 占 6 − 2 = 4 份，C 占 2 − 0 = 2 份。', 'ICDF (inverse cumulative distribution function) table {6, 2, 0}, with a total count of 2⁴ = 16: A gets 16 − 6 = 10, B gets 6 − 2 = 4, and C gets 2 − 0 = 2.'),
      tapset: OX.t('Opus 里真实的表：ICDF {2, 1, 0}，总份数 2² = 4，A、B、C 分别占 2、1、1 份。CELT 用它编码基音预滤波器和后滤波器的抽头组（tapset）；比特很少时，粗能量也用这张表。', 'A real table from Opus: ICDF {2, 1, 0}, with a total count of 2² = 4: A, B, and C get 2, 1, and 1. CELT uses it to code the tap set (tapset) of the pitch pre-filter and post-filter. When very few bits are available, coarse energy uses this table too.'),
    };
    const EXAMPLE = [0, 1, 2, 2, 0]; // A B C C A
    const MAX = 10;
    const DUR = 0.9; // seconds per symbol animation
    const st = { model: 'toy', seq: EXAMPLE.slice() };
    let res = null; // trace of the current sequence
    let anim = null; // { t } while the newest row animates
    let play = null; // { i, wait } while the example plays
    let marker = null; // relative position of the output number on every row
    let labels = []; // per row: the interval ends as decimals (row n = the final interval)

    OX.ui.css('range-coder', `
      .rc-bar { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin: 12px 0 10px; }
      .rc-syms { display: flex; gap: 8px; flex-wrap: wrap; }
      .rc-sym { display: grid; grid-template-columns: auto auto; grid-template-rows: auto auto; column-gap: 10px; align-items: center;
        text-align: left; border: 1px solid var(--rule); background: var(--surface); color: var(--ink); border-radius: 8px;
        padding: 6px 12px 6px 10px; cursor: pointer; font: 500 var(--tiny)/1.35 var(--font-body); }
      .rc-sym:hover { border-color: var(--accent); }
      .rc-sym:hover b { color: var(--accent); }
      .rc-sym:disabled { opacity: 0.45; cursor: default; }
      .rc-sym:disabled:hover { border-color: var(--rule); }
      .rc-sym b { grid-row: 1 / 3; font: 600 1.3rem/1 var(--font-mono); }
      .rc-sym .p { font-family: var(--font-mono); color: var(--ink); white-space: nowrap; }
      .rc-sym .c { color: var(--ink-3); white-space: nowrap; }
      .rc-acts { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
      .rc-sym .p i { font-style: normal; }
      .rc-count { font: 500 var(--tiny)/1 var(--font-mono); color: var(--ink-3); white-space: nowrap; margin-left: 4px; }
      @media (max-width: 560px) {
        .rc-syms { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); width: 100%; gap: 6px; }
        .rc-sym { padding: 6px 8px; column-gap: 7px; }
        .rc-sym b { font-size: 1.1rem; }
        .rc-sym .p i { display: none; }
      }
      .rc-model-note { margin-top: 0; }
      .rc-ok { color: var(--good); }
      .rc-bad { color: var(--bad); }
      .rc-int { margin-top: 14px; border-top: 1px solid var(--rule); }
      .rc-int > summary { cursor: pointer; padding: 10px 0 6px; font: 600 var(--small)/1.4 var(--font-body); color: var(--ink); list-style: none; display: flex; gap: 8px; align-items: baseline; }
      .rc-int > summary::-webkit-details-marker { display: none; }
      .rc-int > summary::before { content: "+"; font: 500 0.95rem/1 var(--font-mono); color: var(--accent); width: 0.9em; }
      .rc-int[open] > summary::before { content: "−"; }
      .rc-int > summary span { font-weight: 400; color: var(--ink-3); font-size: var(--tiny); }
      .rc-tbl { border-collapse: collapse; font-size: var(--tiny); width: 100%; min-width: 640px; }
      .rc-tbl th, .rc-tbl td { text-align: left; padding: 6px 12px 6px 0; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }
      .rc-tbl th { font-weight: 600; color: var(--ink-2); }
      .rc-tbl th code { background: none; padding: 0; font-size: 1em; }
      .rc-tbl td { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--ink); }
      .rc-tbl td.t { font-family: var(--font-body); color: var(--ink-2); }
      .rc-tbl td.t.wrap { white-space: normal; min-width: 22em; }
      .rc-tbl td .sub { display: block; color: var(--ink-3); font-size: 0.92em; }
      .rc-tbl .m { font-family: var(--font-mono); }
      .rc-tbl td.sym { color: var(--accent); font-weight: 600; }
      .rc-tbl tr.edge td { color: var(--ink-2); }
      .rc-tbl .pend { color: var(--ink-3); border-bottom: 1px dashed var(--ink-3); }
      .rc-tbl tr.new td { animation: rc-flash 1.2s ease-out; }
      @keyframes rc-flash { from { background: var(--accent-soft); } to { background: transparent; } }
      @media (prefers-reduced-motion: reduce) { .rc-tbl tr.new td { animation: none; } }
      .rc-explain { margin: 10px 0 0; padding-left: 1.2em; font-size: var(--tiny); color: var(--ink-2); line-height: 1.7; }
      .rc-explain li + li { margin-top: 2px; }
      .rc-explain code { font-size: 1em; }
    `);

    // ---- controls ----
    const modelCtl = segmented({
      id: 'rc-model', label: OX.t('概率表', 'Probability table'), value: st.model,
      options: Object.keys(MODELS).map((k) => ({ value: k, label: MODELS[k].label })),
      onChange: (v) => { stopPlay(); st.model = v; anim = null; update(); },
    });
    const symBtns = NAMES.map((name, s) => {
      const b = el('button', { type: 'button', class: 'rc-sym', id: 'rc-sym-' + name });
      b.addEventListener('click', () => { stopPlay(); add(s, true); });
      return b;
    });
    const undoBtn = button({ label: OX.t('撤销', 'Undo'), id: 'rc-undo', onClick: () => { stopPlay(); st.seq.pop(); anim = null; update(); } });
    const clearBtn = button({ icon: 'reset', label: OX.t('清空', 'Clear'), id: 'rc-clear', onClick: () => { stopPlay(); st.seq = []; anim = null; update(); } });
    const playBtn = button({ icon: 'play', label: OX.t('播放示例 ABCCA', 'Play example ABCCA'), kind: 'primary', id: 'rc-play', onClick: () => (play ? stopPlay() : startPlay()) });
    const modelNote = el('p', { class: 'ox-note rc-model-note' });
    const count = el('span', { class: 'rc-count', 'aria-live': 'polite' });

    const host = el('div', { class: 'rc-stage' });
    const roBytes = readout(OX.t('输出字节（十六进制）', 'Output bytes (hex)'), '');
    const roIdeal = readout(OX.t('理想长度 Σ(−log₂p)', 'Ideal length Σ(−log₂p)'), '');
    const roTell = readout(OX.t('编码器计数（tell_frac ÷ 8）', 'Encoder count (tell_frac ÷ 8)'), '');
    const roLen = readout(OX.t('实际输出', 'Actual output'), '');
    const roDec = readout(OX.t('解码器读回', 'Decoder reads back'), '');
    const numNote = el('p', { class: 'ox-note' });

    const tblWrap = el('div', { class: 'ox-scroll' });
    const tblFoot = el('p', { class: 'ox-note' });
    const details = el('details', { class: 'rc-int' }, [
      el('summary', {}, [OX.t('整数运算过程（深入一点）', 'Integer arithmetic (going deeper)'), el('span', { text: OX.t('编码器每一步的真实状态', 'The real state of the encoder after each step') })]),
      tblWrap,
      tblFoot,
      el('ul', { class: 'rc-explain' }, [
        el('li', { html: OX.t('<b>十六进制</b>（hex）：每一位 0–F 代表 4 个比特，<code>0x80000000</code> 就是 2³¹。字节也用十六进制写，<code>8B</code> 是十进制的 139。', '<b>Hexadecimal</b> (hex): each digit 0–F stands for 4 bits, so <code>0x80000000</code> is 2³¹. Bytes are written in hex too: <code>8B</code> is 139 in decimal.') }),
        el('li', { html: OX.t('<b>val</b> 是区间的下界，<b>rng</b> 是区间的宽度，都是 32 位整数。开始时区间是 [0, 2³¹)，对应数轴上的 [0, 1)。', '<b>val</b> is the low end of the interval and <b>rng</b> is its width. Both are 32-bit integers. The interval starts as [0, 2³¹), which stands for [0, 1) on the number line.') }),
        el('li', { html: OX.t('<b>r</b> 是一份概率对应的宽度。B、C 各得 r × 份数；A 除了 r × 份数，还拿走除不尽的余数，所以几段宽度加起来正好是 rng。', '<b>r</b> is the width that corresponds to one count in the probability table. B and C each get r × their count. A gets r × its count plus the remainder of the division, so the pieces add up to exactly rng.') }),
        el('li', { html: OX.t('<b>重新归一化</b>：rng 不大于 2²³ 时，把 val 里进位位下面的 8 位（第 30 到 23 位）移出成一个字节，再把 val 和 rng 都乘 256。这样 rng 始终有 23 位以上的精度。', '<b>Renormalization</b>: when rng is at most 2²³, the 8 bits of val just below the carry bit (bits 30 to 23) are shifted out as one byte, and val and rng are both multiplied by 256. This way rng always keeps more than 23 bits of precision.') }),
        el('li', { html: OX.t('<b>虚线下的字节</b>还在缓冲区里。后面的加法可能向它进位，所以要等确定后才写出。val 的最高位就是进位位：val 达到 <code>0x80000000</code> 时，缓冲区里的字节要加 1。', '<b>Bytes with a dashed underline</b> are still in the buffer. A later addition can carry into them, so they are written out only once they are final. The top bit of val is the carry bit: when val reaches <code>0x80000000</code>, 1 is added to the byte in the buffer.') }),
        el('li', { html: OX.t('<b>tell_frac</b>：已用的比特数，单位是 1/8 比特。它只由写过的总位数和 rng 算出，解码器能算出同样的值。开始时就是 1 比特，这 1 比特留给结尾。', '<b>tell_frac</b>: the number of bits used so far, in units of 1/8 bit. It is computed only from the total number of bits written and from rng, so the decoder can compute the same value. It starts at 1 bit, and that bit is reserved for the end.') }),
      ]),
    ]);

    body.append(
      el('div', { class: 'ox-controls' }, [modelCtl]),
      modelNote,
      el('div', { class: 'rc-bar' }, [el('div', { class: 'rc-syms', role: 'group', 'aria-label': OX.t('加入符号', 'Add a symbol') }, symBtns), el('div', { class: 'rc-acts' }, [undoBtn, clearBtn, playBtn, count])]),
      host,
      OX.ui.legend([
        { label: OX.t('这一步编码的符号占的一段', 'Piece of the symbol coded in this step'), color: 'var(--accent)', kind: 'box' },
        { label: OX.t('输出字节代表的数：落在哪段，就读出哪个符号', 'Number given by the output bytes: the piece it falls in is the symbol read back'), color: 'var(--ink)' },
      ]),
      el('div', { class: 'ox-readouts' }, [roBytes, roIdeal, roTell, roLen, roDec]),
      numNote,
      details);

    // ---- state ----
    const model = () => MODELS[st.model];
    function retrace() {
      const m = model();
      res = trace(st.seq, m.icdf, m.ftb);
      const n = res.steps.length;
      labels = [];
      for (let i = 0; i <= n; i++) {
        const s = i < n ? res.steps[i] : null;
        const L = s ? s.L0 : res.end.L, E = s ? s.E0 : res.end.E, rng = s ? s.rng0 : res.end.rng;
        const d = digitsFor(rng, E);
        labels.push({ s, lo: frac(L, E, d), hi: frac(L + BigInt(rng), E, d), d });
      }
      marker = null;
      if (res.bytes.length && st.seq.length) {
        const m8 = BigInt(8 * res.bytes.length);
        let V = 0n;
        for (const b of res.bytes) V = (V << 8n) | BigInt(b);
        const rel = (L, E, rng) => {
          const num = V * (1n << BigInt(E)) - L * (1n << m8);
          const den = BigInt(rng) * (1n << m8);
          return Number((num * 1000000n) / den) / 1e6;
        };
        marker = { V, m: res.bytes.length, rows: res.steps.map((s) => rel(s.L0, s.E0, s.rng0)).concat([rel(res.end.L, res.end.E, res.end.rng)]) };
      }
    }
    retrace();

    function add(s, animate) {
      if (st.seq.length >= MAX) return;
      st.seq.push(s);
      anim = animate && !OX.reducedMotion() ? { t: 0 } : null;
      update(true);
      if (anim) loop.start();
    }
    function startPlay() {
      st.seq = []; anim = null;
      if (OX.reducedMotion()) { st.seq = EXAMPLE.slice(); update(); return; }
      play = { i: 0, wait: 0.25 };
      playBtn.setIcon('stop', OX.t('停止', 'Stop'));
      update();
      loop.start();
    }
    function stopPlay() {
      if (!play) return;
      play = null;
      playBtn.setIcon('play', OX.t('播放示例 ABCCA', 'Play example ABCCA'));
    }

    // ---- canvas ----
    const isNarrow = (w) => w < 600;
    const geom = (w) => {
      const narrow = isNarrow(w);
      return { narrow, head: 26, rowH: narrow ? 44 : 50, barH: narrow ? 20 : 22, foot: 26 };
    };
    const heightFor = (w) => { const g = geom(w); return g.head + (Math.max(st.seq.length, EXAMPLE.length) + 1) * g.rowH + g.foot - (g.rowH - g.barH) + 6; };

    const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.pow(1 - x, 3));
    const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

    function draw(ctx, w, h) {
      const T = OX.theme;
      const g = geom(w);
      const n = res.steps.length;
      const fs = g.narrow ? 10 : 10.5;
      const rows = labels;
      // columns
      ctx.font = OX.font(fs, 'mono');
      let decW = ctx.measureText('≈0.' + '0'.repeat(MAX_DIGITS)).width; // widest possible label: fixed, so bars never jump
      for (const r of rows) decW = Math.max(decW, ctx.measureText(r.lo).width, ctx.measureText(r.hi).width);
      decW = Math.ceil(decW) + 12;
      const leftW = g.narrow ? 42 : 52;
      const bitsW = g.narrow ? 50 : 96;
      let showDec = !g.narrow;
      let barL = leftW + (showDec ? decW : 6), barR = w - bitsW - (showDec ? decW : 8);
      if (showDec && barR - barL < w * 0.42) { showDec = false; barL = leftW + 6; barR = w - bitsW - 8; }
      const barW = barR - barL;
      const top = (i) => g.head + i * g.rowH;
      const X = (row, off) => barL + (off / row) * barW;

      // column titles
      ctx.font = OX.font(10.5, 'body', 500);
      ctx.fillStyle = T.ink3; ctx.textBaseline = 'alphabetic';
      ctx.textAlign = 'left'; ctx.fillText(OX.t('步骤', 'Step'), 0, 13);
      const longHead = OX.t('区间：每一行把上一行选中的那段放大到全宽，再按概率切开', 'Interval: each row stretches the piece chosen in the row above to full width, then splits it by probability');
      ctx.fillText(!g.narrow && ctx.measureText(longHead).width < barW - 8 ? longHead : OX.t('区间（每行放大上一行的选中段）', 'Interval (zoomed in row by row)'), barL, 13);
      if (showDec) { ctx.textAlign = 'right'; ctx.fillText(OX.t('下界', 'Low end'), barL - 8, 13); ctx.textAlign = 'left'; ctx.fillText(OX.t('上界', 'High end'), barR + 8, 13); }
      ctx.textAlign = 'right'; ctx.fillText(g.narrow ? OX.t('比特', 'Bits') : OX.t('这个符号的比特', 'Bits for this symbol'), w, 13);

      const a = anim ? anim.t : 1;
      const pSplit = anim ? ease(clamp01(a / 0.4)) : 1; // newest split row: highlight fades in
      const pZoom = anim ? ease(clamp01((a - 0.4) / 0.6)) : 1; // final row grows out of the chosen piece

      // where each row's bar spans (the final row animates out of the chosen piece above it)
      const span = (i) => {
        if (anim && i === n && n > 0) {
          const s = res.steps[n - 1];
          const x0 = X(s.rng0, s.cuts[s.sym]), x1 = X(s.rng0, s.cuts[s.sym + 1]);
          return [x0 + (barL - x0) * pZoom, x1 + (barR - x1) * pZoom];
        }
        return [barL, barR];
      };

      // zoom funnels between rows
      for (let i = 1; i <= n; i++) {
        const s = res.steps[i - 1];
        const alpha = anim && i === n ? pSplit : 1;
        if (alpha <= 0) continue;
        const y0 = top(i - 1) + g.barH, y1 = top(i) - 0.5;
        const x0 = X(s.rng0, s.cuts[s.sym]), x1 = X(s.rng0, s.cuts[s.sym + 1]);
        const [b0, b1] = span(i);
        const yy = anim && i === n ? y0 + (y1 - y0) * Math.max(pZoom, 0.001) : y1;
        ctx.fillStyle = OX.alpha(T.accent, (T.dark ? 0.1 : 0.06) * alpha);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.lineTo(b1, yy); ctx.lineTo(b0, yy); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = OX.alpha(T.accent, 0.5 * alpha); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(b0, yy); ctx.moveTo(x1, y0); ctx.lineTo(b1, yy); ctx.stroke();
      }

      // rows
      for (let i = 0; i <= n; i++) {
        const R = rows[i], y = top(i), s = R.s;
        const final = i === n;
        const rowAlpha = anim && final && n > 0 ? pZoom : 1;
        if (rowAlpha <= 0.001) continue;
        ctx.globalAlpha = rowAlpha;
        const yc = y + g.barH / 2;
        // left label
        ctx.textBaseline = 'middle';
        if (!final) {
          ctx.font = OX.font(10.5, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'left';
          ctx.fillText(String(i + 1), 0, yc);
          const cx = 16, cw = 20;
          ctx.fillStyle = T.accentSoft; roundRect(ctx, cx, yc - 10, cw, 20, 4); ctx.fill();
          ctx.fillStyle = T.accent; ctx.font = OX.font(12, 'mono', 600); ctx.textAlign = 'center';
          ctx.fillText(NAMES[s.sym], cx + cw / 2, yc + 0.5);
        } else {
          ctx.font = OX.font(11, 'body', 600); ctx.fillStyle = T.ink2; ctx.textAlign = 'left';
          ctx.fillText(n ? OX.t('结果', 'Result') : OX.t('开始', 'Start'), 0, yc);
        }
        // bar
        if (!final) {
          const hl = anim && i === n - 1 ? pSplit : 1;
          for (let k = 0; k < s.cuts.length - 1; k++) {
            const x0 = X(s.rng0, s.cuts[k]), x1 = X(s.rng0, s.cuts[k + 1]);
            const on = k === s.sym;
            ctx.fillStyle = T.sunken;
            ctx.fillRect(x0 + 0.5, y, Math.max(0.5, x1 - x0 - 1), g.barH);
            if (on && hl > 0) {
              ctx.fillStyle = OX.alpha(T.accent, (T.dark ? 0.3 : 0.18) * hl);
              ctx.fillRect(x0 + 0.5, y, Math.max(0.5, x1 - x0 - 1), g.barH);
              ctx.strokeStyle = OX.alpha(T.accent, hl); ctx.lineWidth = 1.25;
              ctx.strokeRect(x0 + 1.1, y + 0.6, Math.max(0.5, x1 - x0 - 2.2), g.barH - 1.2);
            }
            if (x1 - x0 >= 13) {
              ctx.font = OX.font(11, 'mono', on ? 600 : 500);
              ctx.fillStyle = on && hl > 0.5 ? T.accent : T.ink3;
              ctx.textAlign = 'center';
              ctx.fillText(NAMES[k], (x0 + x1) / 2, yc + 0.5);
            }
          }
        } else {
          const [b0, b1] = span(i);
          ctx.fillStyle = n ? OX.alpha(T.accent, T.dark ? 0.3 : 0.18) : T.sunken;
          ctx.fillRect(b0, y, b1 - b0, g.barH);
          ctx.strokeStyle = n ? T.accent : T.rule; ctx.lineWidth = 1.25;
          ctx.strokeRect(b0 + 0.6, y + 0.6, b1 - b0 - 1.2, g.barH - 1.2);
          if (!n) {
            ctx.font = OX.font(11, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'center';
            ctx.fillText(OX.t('点上面的 A / B / C，把符号加进来', 'Click A, B, or C above to add a symbol'), (b0 + b1) / 2, yc + 0.5);
          }
        }
        // interval ends
        if (showDec) {
          ctx.font = OX.font(fs, 'mono'); ctx.fillStyle = T.ink2;
          ctx.textAlign = 'right'; ctx.fillText(R.lo, barL - 8, yc);
          ctx.textAlign = 'left'; ctx.fillText(R.hi, barR + 8, yc);
        }
        // bits
        ctx.textAlign = 'right';
        if (!final) {
          ctx.font = OX.font(g.narrow ? 10.5 : 11.5, 'mono', 500); ctx.fillStyle = T.ink;
          const t = '+' + s.bits.toFixed(3);
          if (g.narrow) ctx.fillText(t, w, yc);
          else {
            ctx.font = OX.font(11, 'body'); ctx.fillStyle = T.ink3; ctx.fillText(OX.t('比特', 'bits'), w, yc);
            const uw = ctx.measureText(OX.t('比特', 'bits')).width + 4;
            ctx.font = OX.font(11.5, 'mono', 500); ctx.fillStyle = T.ink; ctx.fillText(t, w - uw, yc);
          }
        } else if (n) {
          ctx.font = OX.font(g.narrow ? 10.5 : 11.5, 'mono', 600); ctx.fillStyle = T.ink;
          const t = res.ideal.toFixed(3);
          if (g.narrow) ctx.fillText(t, w, yc);
          else {
            ctx.font = OX.font(11, 'body'); ctx.fillStyle = T.ink3; ctx.fillText(OX.t('比特', 'bits'), w, yc);
            let x = w - ctx.measureText(OX.t('比特', 'bits')).width - 4;
            ctx.font = OX.font(11.5, 'mono', 600); ctx.fillStyle = T.ink; ctx.fillText(t, x, yc);
            x -= ctx.measureText(t).width + 4;
            ctx.font = OX.font(11, 'body'); ctx.fillStyle = T.ink3; ctx.fillText(OX.t('合计', 'total'), x, yc);
          }
        }
        ctx.globalAlpha = 1;
      }

      // the output number on every row
      if (marker) {
        const ma = anim ? pZoom : 1;
        if (ma > 0) {
          ctx.globalAlpha = ma;
          ctx.strokeStyle = T.ink; ctx.fillStyle = T.ink; ctx.lineWidth = 1.25;
          for (let i = 0; i <= n; i++) {
            const y = top(i);
            const x = Math.round(barL + marker.rows[i] * barW) + 0.5;
            ctx.beginPath(); ctx.moveTo(x, y - 2); ctx.lineTo(x, y + g.barH + 2); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(x - 3.5, y - 6); ctx.lineTo(x + 3.5, y - 6); ctx.lineTo(x, y - 1.5); ctx.closePath(); ctx.fill();
          }
          // label under the final row
          const y = top(n) + g.barH + 5;
          const x = barL + marker.rows[n] * barW;
          const d = rows[n].d;
          const num = frac(marker.V, 8 * marker.m, d);
          const bytes = res.bytes.map(hex2).join(' ');
          ctx.font = OX.font(fs, 'mono');
          let label = OX.t(`输出 ${bytes} ${eq(num)}`, `Output ${bytes} ${eq(num)}`);
          if (ctx.measureText(label).width > barW) label = eq(num);
          const tw = ctx.measureText(label).width;
          ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = T.ink;
          ctx.fillText(label, Math.max(barL, Math.min(barR - tw, x - tw / 2)), y);
          ctx.globalAlpha = 1;
        }
      }
    }
    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }

    const cv = OX.view.canvas(host, { height: heightFor, label: OX.t('区间编码的数轴', 'Range coding on the number line') }, draw);

    const loop = OX.loop(host, (dt) => {
      if (anim) { anim.t += dt / DUR; if (anim.t >= 1) anim = null; }
      if (play && !anim) {
        play.wait -= dt;
        if (play.wait <= 0) {
          if (play.i < EXAMPLE.length) { add(EXAMPLE[play.i++], true); play.wait = 0.3; }
          else stopPlay();
        }
      }
      cv.redraw();
      return !!(anim || play);
    });

    // ---- table and readouts ----
    function bytesCell(out, rem, ext) {
      const parts = out.map(hex2).join(' ');
      const pend = (rem >= 0 ? [hex2(rem)] : []).concat(new Array(ext).fill('FF')).join(' ');
      if (!parts && !pend) return '—';
      return (parts ? parts : '') + (pend ? (parts ? ' ' : '') + OX.t(`<span class="pend" title="还在缓冲区，可能被进位改写">${pend}</span>`, `<span class="pend" title="Still in the buffer: a carry can still change it">${pend}</span>`) : '');
    }
    function normCell(ev) {
      if (!ev.length) return '—';
      return ev.map((e) => {
        const before = OX.t(`<span class="sub">原 rng <span class="m">${hex8(e.rngBefore)}</span></span><span class="sub">原 val <span class="m">${hex8(e.valBefore)}</span></span>`, `<span class="sub">rng before <span class="m">${hex8(e.rngBefore)}</span></span><span class="sub">val before <span class="m">${hex8(e.valBefore)}</span></span>`);
        if (e.c === 255) return OX.t(`移出 FF<span class="sub">先不写出，只记下个数（变量 ext）</span>${before}`, `Shift out FF<span class="sub">Not written yet, only counted (variable ext)</span>${before}`);
        const carry = e.c >>> 8;
        return OX.t(`移出 <span class="m">${hex2(e.c & 255)}</span>${carry ? '<span class="sub">并给前一个字节进位 +1</span>' : ''}${before}`, `Shift out <span class="m">${hex2(e.c & 255)}</span>${carry ? '<span class="sub">and carry +1 into the previous byte</span>' : ''}${before}`);
      }).join('');
    }
    function renderTable(isNew) {
      const m = model();
      const n = res.steps.length;
      const head = OX.t(`<tr><th>步骤</th><th>符号</th><th><code>r = rng &gt;&gt; ${m.ftb}</code></th><th><code>val</code>（下界）</th><th><code>rng</code>（宽度）</th><th>重新归一化</th><th>已写出的字节</th><th>已用比特</th></tr>`, `<tr><th>Step</th><th>Symbol</th><th><code>r = rng &gt;&gt; ${m.ftb}</code></th><th><code>val</code> (low end)</th><th><code>rng</code> (width)</th><th>Renormalization</th><th>Bytes written</th><th>Bits used</th></tr>`);
      const s0 = res.start;
      let rowsHtml = `<tr class="edge"><td class="t">${OX.t('开始', 'Start')}</td><td>—</td><td>—</td><td>${hex8(s0.val)}</td><td>${hex8(s0.rng)}</td><td>—</td><td>—</td><td>${(s0.tf / 8).toFixed(3)}</td></tr>`;
      res.steps.forEach((s, i) => {
        rowsHtml += `<tr${isNew && i === n - 1 ? ' class="new"' : ''}><td>${i + 1}</td><td class="sym">${NAMES[s.sym]}</td><td>${hex8(s.r)}</td><td>${hex8(s.val)}</td><td>${hex8(s.rng)}</td><td class="t">${normCell(s.events)}</td><td>${bytesCell(s.out, s.rem, s.ext)}</td><td>${(s.tf / 8).toFixed(3)}</td></tr>`;
      });
      if (n) {
        const txt = doneText(res);
        rowsHtml += `<tr class="edge"><td class="t">${OX.t('结束', 'End')}</td><td>—</td><td colspan="4" class="t wrap">${txt}</td><td>${bytesCell(res.bytes, -1, 0)}</td><td>${(res.tell / 8).toFixed(3)}</td></tr>`;
      }
      tblWrap.innerHTML = `<table class="rc-tbl">${head}${rowsHtml}</table>`;
      if (n) {
        const same = res.decRng[n - 1] === res.end.rng;
        tblFoot.textContent = OX.t(`解码器读完 ${n} 个符号后 rng = ${hex8(res.decRng[n - 1])}，${same ? '与编码器完全相同' : '与编码器不同（出错了）'}。两端的 val 含义不同，只有 rng 必须一致。`, `After decoding ${n} symbol${n === 1 ? '' : 's'}, the decoder has rng = ${hex8(res.decRng[n - 1])}, ${same ? 'exactly the same as the encoder' : 'different from the encoder (something went wrong)'}. On the two sides val means different things: only rng has to match.`);
      } else tblFoot.textContent = '';
    }

    function update(isNew) {
      const m = model();
      const f = freqs(m.icdf, m.ftb), ft = 1 << m.ftb;
      symBtns.forEach((b, s) => {
        const bits = -Math.log2(f[s] / ft);
        b.innerHTML = `<b>${NAMES[s]}</b><span class="p"><i>p = </i>${f[s]}/${ft}</span><span class="c">${bits.toFixed(3)} ${OX.t('比特', 'bits')}</span>`;
        b.setAttribute('aria-label', OX.t(`加入符号 ${NAMES[s]}，概率 ${f[s]}/${ft}，${bits.toFixed(3)} 比特`, `Add symbol ${NAMES[s]}, probability ${f[s]}/${ft}, ${bits.toFixed(3)} bits`));
        b.disabled = st.seq.length >= MAX;
      });
      undoBtn.disabled = clearBtn.disabled = st.seq.length === 0;
      count.textContent = st.seq.length >= MAX ? OX.t(`已满 ${MAX} 个符号`, `Full: ${MAX} symbols`) : OX.t(`${st.seq.length} / ${MAX} 个符号`, `${st.seq.length} / ${MAX} symbols`);
      modelNote.textContent = MODEL_NOTE[st.model];
      retrace();
      const n = st.seq.length;
      const seqText = st.seq.map((s) => NAMES[s]).join('');
      roBytes.set(n ? res.bytes.map(hex2).join(' ') : '—');
      roIdeal.set(n ? res.ideal.toFixed(3) + OX.t(' 比特', ' bits') : '—');
      roTell.set((res.tell / 8).toFixed(3) + OX.t(' 比特', ' bits'));
      roLen.set(n ? OX.t(`${res.bytes.length} 字节 = ${8 * res.bytes.length} 比特`, `${res.bytes.length} byte${res.bytes.length === 1 ? '' : 's'} = ${8 * res.bytes.length} bits`) : '—');
      const ok = res.decoded.length === n && res.decoded.every((v, i) => v === st.seq[i]);
      roDec.querySelector('.ox-ro-val').innerHTML = n
        ? `${res.decoded.map((s) => NAMES[s]).join('')} <span class="${ok ? 'rc-ok' : 'rc-bad'}">${ok ? '✓' : '✗'}</span>`
        : '—';
      if (n) {
        const d = digitsFor(res.end.rng, res.end.E);
        const lo = frac(res.end.L, res.end.E, d), hi = frac(res.end.L + BigInt(res.end.rng), res.end.E, d);
        const hexInt = '0x' + res.bytes.map(hex2).join('');
        const num = frac(marker ? marker.V : 0n, 8 * res.bytes.length, d);
        // tell_frac is an upper bound: the output never needs more whole bytes than it suggests, sometimes fewer
        const nb = res.bytes.length, byTell = Math.ceil(Math.ceil(res.tell / 8) / 8);
        const nbEn = `${nb} byte${nb === 1 ? '' : 's'}`;
        numNote.innerHTML = OX.t(`把输出字节依次接在小数点后面：${hexInt} ÷ 2${sup(8 * nb)} ${eq(num)}，它落在最后的区间 [${lo}, ${hi}) 里。`, `Read the output bytes, in order, as hex digits after the point: ${hexInt} ÷ 2${sup(8 * nb)} ${eq(num)}. This number lies in the final interval [${lo}, ${hi}). `) +
          OX.t('编码器计数比理想长度多出约 1 比特，那是留给结尾的。', 'The encoder count is about 1 bit more than the ideal length: that bit is reserved for the end. ') +
          (nb === byTell ? OX.t(`再取整到整字节，就是 ${nb} 字节。`, `Rounded up to whole bytes, that is ${nbEn}. `)
            : OX.t(`按计数取整要 ${byTell} 字节，但最后的区间里正好有一个低位全为 0 的数，${nb} 字节就能写下，所以只输出 ${nb} 字节。`, `Rounding the count up would take ${byTell} bytes, but the final interval happens to contain a number whose low bits are all 0, and it fits in ${nbEn}. So the output is only ${nbEn}. `)) +
          OX.t('真实的包里有几百个符号，这些零头就摊薄了。', 'A real packet holds hundreds of symbols, so these few extra bits are spread across all of them.');
      } else numNote.textContent = OX.t('还没有符号：区间是 [0, 1)，tell_frac 已经是 1 比特，这 1 比特留给结尾。', 'No symbols yet: the interval is [0, 1), and tell_frac is already 1 bit. That bit is reserved for the end.');
      cv.canvas.setAttribute('aria-label', n
        ? OX.t(`区间编码的数轴：依次编码 ${seqText}，每一行把上一行选中的那段放大到全宽。最后的区间宽度约 2 的 −${res.ideal.toFixed(2)} 次方，输出 ${res.bytes.length} 个字节。`, `Range coding on the number line: ${seqText} is encoded one symbol at a time, and each row stretches the piece chosen in the row above to full width. The final interval is about 2 to the power −${res.ideal.toFixed(2)} wide, and the output is ${res.bytes.length} byte${res.bytes.length === 1 ? '' : 's'}.`)
        : OX.t('区间编码的数轴：还没有符号，区间是 0 到 1。', 'Range coding on the number line: no symbols yet, so the interval is 0 to 1.'));
      renderTable(isNew);
      cv.redraw();
    }
    update(false);
  });
})();
