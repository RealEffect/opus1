/* Figure 9: how CELT shares one frame's bits among its 21 bands.
 * Top: the allocation of src/rate.rs (clt_compute_allocation + interp_bits2pulses), ported line by line for one
 * channel, 20 ms frames (LM = 3), all 21 bands. The reader drags the bitrate and the alloc trim and sees, per band,
 * the fine-energy bits and the bits left for the PVQ shape; bands with no PVQ bits are filled by folding.
 * Bottom: the real 64 kb/s music stream from data/opus-demo-data.js (pvq_bits per band and frame, as the decoder
 * recomputed them from the bitstream). Clicking a frame replays its allocation in the top chart, with the skip
 * decisions read from the bitstream as the decoder reads them; the port then reproduces every frame exactly.
 * Simplifications (slider mode only): no dynalloc boosts, no transient, the skip rule of a first frame at full
 * signal bandwidth, and the bits used before the allocation (header, coarse energy, tf, spread, dynalloc, trim)
 * are a constant: the median of the real stream.
 */
OX.fig('celt-alloc', function (root, body) {
  /**
   * CELT bit allocation, ported from src/rate.rs (clt_compute_allocation + interp_bits2pulses) for one channel
   * (C = 1), start band 0. All bit counts are in 1/8 bit (BITRES = 3), as in the Rust code.
   * Tables are copied from src/modes.rs.
   */
  function makeAllocator() {
    const BITRES = 3, FINE_OFFSET = 21, MAX_FINE_BITS = 8, NB = 21;
    // EBAND_5MS (modes.rs:90): band edges in units of 200 Hz (one bin of the 2.5 ms MDCT)
    const EB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];
    // BAND_ALLOCATION (modes.rs:6): 11 rows x 21 bands, in 1/32 bit per MDCT coefficient
    const ALLOC = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0,
      110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0, 0, 0, 0,
      118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4, 0, 0, 0, 0,
      126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12, 1, 0, 0,
      134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10, 1,
      144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
      152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1,
      162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1,
      172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45, 20,
      200, 200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153, 148, 129, 104,
    ];
    const NROWS = 11;
    // LOG_N_400 (modes.rs:165): log2 of the band width (in 200 Hz units), in 1/8 bit
    const LOG_N = [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36];
    // Pulse cache (modes.rs:20-61). bits[index[(LM+1)*21+band] + q] = cost of q "pseudo pulses" minus 1, in 1/8 bit.
    const CACHE_INDEX = [
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222, 0, 0, 0, 0,
      0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266, 283, 295, 41, 41, 41, 41, 41,
      41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305, 318, 328, 336, 123, 123, 123,
      123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318, 343, 351, 358, 364, 240,
      240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376, 382,
      387,
    ];
    const CACHE_BITS = [
      40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47,
      47, 49, 50, 51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70, 71,
      71, 40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92,
      94, 96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123, 124, 126, 128, 40, 23,
      39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107, 111, 115, 118, 121, 124, 126,
      129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169, 172, 174, 177, 179, 35,
      28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149, 153, 159, 165, 171, 176,
      180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251, 21, 33, 58, 79, 97,
      112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235, 243, 251, 17, 35,
      63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250, 25, 31, 55, 75,
      91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215, 222, 229, 235,
      240, 245, 255, 16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226, 234, 242, 250,
      11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, 9, 43, 79, 110, 138, 163, 186, 207,
      227, 246, 12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, 9, 44, 81, 113, 142,
      168, 192, 214, 235, 255, 7, 49, 90, 127, 160, 191, 220, 247, 6, 51, 95, 134, 170, 203, 234, 7,
      47, 87, 123, 155, 184, 212, 237, 6, 52, 97, 137, 174, 208, 240, 5, 57, 106, 151, 192, 231, 5,
      59, 111, 158, 202, 243, 5, 55, 103, 147, 187, 224, 5, 60, 113, 161, 206, 248, 4, 65, 122, 175,
      224, 4, 67, 127, 182, 234,
    ];
    const CACHE_CAPS = [
      224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168, 134,
      61, 37, 224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198,
      183, 144, 66, 40, 160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193,
      183, 183, 172, 138, 64, 38, 240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204,
      204, 204, 193, 193, 180, 143, 66, 40, 185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193,
      193, 193, 193, 193, 183, 183, 172, 138, 65, 39, 207, 207, 207, 207, 207, 207, 207, 207, 204,
      204, 204, 204, 201, 201, 201, 188, 188, 176, 141, 66, 40, 193, 193, 193, 193, 193, 193, 193,
      193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173, 139, 65, 39, 204, 204, 204, 204, 204,
      204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175, 140, 66, 40,
    ];

    /** Most bits a band may take (mono): cap[i] = ((caps[21*2LM + i] + 64) * N) >> 2, N = bins (celt.rs:5109). */
    function caps(LM) {
      const out = [];
      for (let i = 0; i < NB; i++) out.push(((CACHE_CAPS[NB * 2 * LM + i] + 64) * ((EB[i + 1] - EB[i]) << LM)) >> 2);
      return out;
    }
    /** get_pulses (rate.rs:16): pseudo-pulse index q -> pulse count K. */
    const getPulses = (q) => (q < 8 ? q : (8 + (q & 7)) << ((q >> 3) - 1));
    function cacheRow(band, LM) { const ix = CACHE_INDEX[(LM + 1) * NB + band]; return ix < 0 ? -1 : ix; }
    /** bits2pulses (rate.rs:55): the q whose cost is closest to `bits` (1/8 bit). */
    function bits2pulses(band, LM, bits) {
      const c = cacheRow(band, LM);
      if (c < 0) return 0;
      let lo = 0, hi = CACHE_BITS[c];
      bits--;
      for (let i = 0; i < 6; i++) {
        const mid = (lo + hi + 1) >> 1;
        if (CACHE_BITS[c + mid] >= bits) hi = mid; else lo = mid;
      }
      const loVal = lo === 0 ? -1 : CACHE_BITS[c + lo];
      return bits - loVal <= CACHE_BITS[c + hi] - bits ? lo : hi;
    }
    /** pulses2bits (rate.rs:176): cost of q pseudo pulses, in 1/8 bit. */
    function pulses2bits(band, LM, q) { if (q === 0) return 0; const c = cacheRow(band, LM); return c < 0 ? 0 : CACHE_BITS[c + q] + 1; }
    /** The largest band (in 1/8 bit) that is coded with one PVQ codeword; above it quant_partition splits it. */
    function splitThreshold(band, LM) { const c = cacheRow(band, LM); return c < 0 ? Infinity : CACHE_BITS[c + CACHE_BITS[c]] + 12; }

    /**
     * clt_compute_allocation for C = 1, start = 0.
     * o = { total, trim, LM, end, offsets?, keep?(j, codedBands, bandBits, info) }
     *   total: bits available for the allocation, in 1/8 bit (packet bits - bits already used - 1 - anti-collapse reserve)
     *   keep: the skip decision. Default: the opus1 encoder rule (rate.rs:730-737) for a first frame (prev = 0) at
     *         full signal bandwidth (signal_bandwidth = 20). Real frames pass the decision read from the bitstream.
     */
    function compute(o) {
      const LM = o.LM == null ? 3 : o.LM, end = o.end == null ? NB : o.end, start = 0, C = 1;
      const trim = o.trim == null ? 5 : o.trim;
      const offsets = o.offsets || new Array(NB).fill(0);
      const cap = o.cap || caps(LM);
      let total = Math.max(o.total, 0);
      let skipStart = start;
      const skipRsv = total >= 1 << BITRES ? 1 << BITRES : 0;
      total -= skipRsv;
      const thresh = new Array(NB).fill(0), trimOffset = new Array(NB).fill(0);
      for (let j = start; j < end; j++) {
        const n = EB[j + 1] - EB[j];
        thresh[j] = Math.max(C << BITRES, ((3 * n) << (LM + BITRES)) >> 4);
        trimOffset[j] = (C * n * (trim - 5 - LM) * (end - j - 1) * (1 << (LM + BITRES))) >> 6;
        if (n << LM === 1) trimOffset[j] -= C << BITRES;
      }
      // binary search for the highest table row that fits
      const rowBits = (row, j) => (C * (EB[j + 1] - EB[j]) * ALLOC[row * NB + j]) << LM >> 2;
      let lo = 1, hi = NROWS - 1;
      while (lo <= hi) {
        let done = false, psum = 0;
        const mid = (lo + hi) >> 1;
        for (let j = end - 1; j >= start; j--) {
          let b = rowBits(mid, j);
          if (b > 0) b = Math.max(0, b + trimOffset[j]);
          b += offsets[j];
          if (b >= thresh[j] || done) { done = true; psum += Math.min(b, cap[j]); }
          else if (b >= C << BITRES) psum += C << BITRES;
        }
        if (psum > total) hi = mid - 1; else lo = mid + 1;
      }
      const rowHi = lo, rowLo = lo - 1;
      const bits1 = new Array(NB).fill(0), bits2 = new Array(NB).fill(0);
      for (let j = start; j < end; j++) {
        let b1 = rowBits(rowLo, j);
        let b2 = rowHi >= NROWS ? cap[j] : rowBits(rowHi, j);
        if (b1 > 0) b1 = Math.max(0, b1 + trimOffset[j]);
        if (b2 > 0) b2 = Math.max(0, b2 + trimOffset[j]);
        if (rowLo > 0) b1 += offsets[j];
        b2 += offsets[j];
        if (offsets[j] > 0) skipStart = j;
        b2 = Math.max(0, b2 - b1);
        bits1[j] = b1; bits2[j] = b2;
      }
      return interp(o, LM, end, start, C, skipStart, bits1, bits2, thresh, cap, total, skipRsv, rowLo, rowHi);
    }

    function interp(o, LM, end, start, C, skipStart, bits1, bits2, thresh, cap, total, skipRsv, rowLo, rowHi) {
      const allocFloor = C << BITRES, stereo = 0, logM = LM << BITRES;
      let lo = 0, hi = 1 << 6, psum;
      for (let it = 0; it < 6; it++) {
        const mid = (lo + hi) >> 1;
        psum = 0;
        let done = false;
        for (let j = end - 1; j >= start; j--) {
          const t = bits1[j] + ((mid * bits2[j]) >> 6);
          if (t >= thresh[j] || done) { done = true; psum += Math.min(t, cap[j]); }
          else if (t >= allocFloor) psum += allocFloor;
        }
        if (psum > total) hi = mid; else lo = mid;
      }
      const frac = lo;
      const bits = new Array(NB).fill(0);
      psum = 0;
      let done = false;
      for (let j = end - 1; j >= start; j--) {
        let t = bits1[j] + ((lo * bits2[j]) >> 6);
        if (t < thresh[j] && !done) t = t >= allocFloor ? allocFloor : 0;
        else done = true;
        t = Math.min(t, cap[j]);
        bits[j] = t;
        psum += t;
      }
      // skip loop (encoder side): walk down from the top band; each band above its threshold costs one decision bit
      let coded = end, totalRsv = total, skipBits = 0;
      const keep = o.keep || encoderKeep;
      const why = new Array(NB).fill(''); // why a band ends up without PVQ bits: 'thresh' | 'skip' | 'fine'
      for (;;) {
        if (coded <= start) break;
        const j = coded - 1;
        if (j <= skipStart) { totalRsv += skipRsv; break; }
        const left = totalRsv - psum;
        const nbSamples = EB[coded] - EB[start];
        const percoeff = Math.trunc(left / nbSamples);
        const leftRem = left - nbSamples * percoeff;
        const rem = Math.max(leftRem - (EB[j] - EB[start]), 0);
        const bandWidth = EB[coded] - EB[j];
        let bandBits = bits[j] + percoeff * bandWidth + rem;
        if (bandBits >= Math.max(thresh[j], allocFloor + (1 << BITRES))) {
          skipBits++;
          if (keep(j, coded, bandBits, { LM, start, bandWidth })) break;
          psum += 1 << BITRES;
          bandBits -= 1 << BITRES;
          why[j] = 'skip';
        } else why[j] = 'thresh';
        psum -= bits[j];
        if (bandBits >= allocFloor) { psum += allocFloor; bits[j] = allocFloor; } else bits[j] = 0;
        coded--;
      }
      // spread what is left evenly per coefficient, then the remainder from the bottom band up
      let left = totalRsv - psum;
      const nbSamples = EB[coded] - EB[start];
      const percoeff = Math.trunc(left / nbSamples);
      left -= nbSamples * percoeff;
      for (let j = start; j < coded; j++) bits[j] += percoeff * (EB[j + 1] - EB[j]);
      for (let j = start; j < coded; j++) { const t = Math.min(left, EB[j + 1] - EB[j]); bits[j] += t; left -= t; }
      // split each band into fine-energy bits (ebits, whole bits) and PVQ bits (pulses, 1/8 bit)
      const ebits = new Array(NB).fill(0), pulses = new Array(NB).fill(0), prio = new Array(NB).fill(0);
      let balance = 0;
      for (let j = start; j < coded; j++) {
        const n0 = EB[j + 1] - EB[j], n = n0 << LM;
        const bit = bits[j] + balance;
        let excess;
        if (n > 1) {
          excess = Math.max(bit - cap[j], 0);
          bits[j] = bit - excess;
          const den = C * n;
          const ncLogN = den * (LOG_N[j] + logM);
          let offset = (ncLogN >> 1) - den * FINE_OFFSET;
          if (n === 2) offset += (den << BITRES) >> 2;
          if (bits[j] + offset < (den * 2) << BITRES) offset += ncLogN >> 2;
          else if (bits[j] + offset < (den * 3) << BITRES) offset += ncLogN >> 3;
          ebits[j] = Math.max(0, bits[j] + offset + (den << (BITRES - 1)));
          ebits[j] = Math.floor(ebits[j] / den) >> BITRES;
          if (C * ebits[j] > bits[j] >> BITRES) ebits[j] = bits[j] >> stereo >> BITRES;
          ebits[j] = Math.min(ebits[j], MAX_FINE_BITS);
          prio[j] = ebits[j] * (den << BITRES) >= bits[j] + offset ? 1 : 0;
          bits[j] -= (C * ebits[j]) << BITRES;
        } else {
          excess = Math.max(0, bit - (C << BITRES));
          bits[j] = bit - excess;
          ebits[j] = 0;
          prio[j] = 1;
        }
        if (excess > 0) {
          const extraFine = Math.min(excess >> (stereo + BITRES), MAX_FINE_BITS - ebits[j]);
          ebits[j] += extraFine;
          const extraBits = (extraFine * C) << BITRES;
          prio[j] = extraBits >= excess - balance ? 1 : 0;
          excess -= extraBits;
        }
        balance = excess;
        pulses[j] = bits[j];
        if (pulses[j] <= 0) why[j] = 'fine';
      }
      for (let j = coded; j < end; j++) {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        prio[j] = ebits[j] < 1 ? 1 : 0;
        pulses[j] = 0;
      }
      return { rowLo, rowHi, frac, coded, pulses, ebits, prio, balance, thresh, cap, skipBits, why };
    }

    /** The opus1 encoder's skip decision (rate.rs:730-737) with prev = 0 and signal_bandwidth = 20 (no earlier frame, full band). */
    function encoderKeep(j, coded, bandBits, info) {
      const depth = coded > 17 ? 9 : 0; // j < prev never holds with prev = 0, so 9 rather than 7
      return coded <= info.start + 2 || ((bandBits > ((depth * info.bandWidth) << info.LM << BITRES) >> 4) && j <= 20);
    }

    return { EB, ALLOC, NROWS, LOG_N, caps, compute, bits2pulses, pulses2bits, getPulses, splitThreshold, BITRES };
  }

  const { el, slider, segmented, readout, button } = OX.ui;
  const A = makeAllocator();
  const LM = 3, NB = 21, EB = A.EB, BITRES = A.BITRES;
  const nbins = (j) => (EB[j + 1] - EB[j]) << LM; // MDCT coefficients in band j of a 20 ms frame
  const khz = (e) => { const f = EB[e] * 200; return f >= 1000 ? (f / 1000).toFixed(f % 1000 ? 1 : 0) : String(f); };
  const bandHz = (j) => { const a = EB[j] * 200, b = EB[j + 1] * 200; return b < 1000 ? `${a}–${b} Hz` : `${(a / 1000).toFixed(1)}–${(b / 1000).toFixed(1)} kHz`; };
  const fmtBits = (q3) => { const b = q3 / 8; return Number.isInteger(b) ? String(b) : b.toFixed(b < 10 ? 2 : 1).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, ''); };
  const bitsEn = (v) => `${v} ${String(v) === '1' ? 'bit' : 'bits'}`; // English only: "1 bit", "0.5 bits", "12 bits"

  // ---- real data: the 64 kb/s CELT music stream, frame by frame ----
  const raw = OX.data.raw;
  const frames = (raw && raw.music && raw.music.celt_frames && Array.isArray(raw.music.celt_frames.frames))
    ? raw.music.celt_frames.frames.filter((f) => f && !f.silence && f.lm === LM && f.end_band === NB && f.bits) : null;
  const streamKbps = raw && raw.music && raw.music.celt_frames ? raw.music.celt_frames.bitrate / 1000 : 64;
  /** Bits the range coder has used before the allocation: header, coarse energy, tf, spread, dynalloc, trim (1/8 bit). */
  const usedBefore = (f) => { const b = f.bits; return Math.round(8 * (b.header + b.coarse_energy + b.tf + b.spread + b.dynalloc + b.trim)); };
  // For the slider: those bits are estimated by the median of the real stream's non-transient frames.
  const HEAD_Q3 = (() => {
    if (!frames) return 53 * 8;
    const v = frames.filter((f) => !f.transient).map(usedBefore).sort((a, b) => a - b);
    return v.length ? v[v.length >> 1] : 53 * 8;
  })();

  /**
   * Inputs of the allocation for one real frame, exactly as the decoder sees them (celt.rs:8179-8227). The skip
   * decisions are read from the bitstream (coded_bands), as the decoder reads them. The encoder's rule also depends
   * on the bands coded in earlier frames and on the bandwidth found by the analysis (celt.rs:5337-5377), and the data
   * does not hold that bandwidth.
   */
  function frameInputs(f) {
    const head = usedBefore(f);
    let total = f.bits.payload * 8 - head - 1;
    const acr = f.transient && total >= (LM + 2) << BITRES ? 1 << BITRES : 0; // anti-collapse reserve
    total -= acr;
    return { bytes: f.bytes, head, acr, total, trim: f.alloc_trim, offsets: f.boost_bits.map((v) => Math.round(v * 8)), keep: (j, coded) => coded <= f.coded_bands, transient: f.transient, f };
  }
  // Re-run the allocation for every real frame and compare with what the decoder got from the bitstream.
  const check = frames ? frames.map((f) => {
    const fi = frameInputs(f);
    const r = A.compute({ total: fi.total, trim: fi.trim, LM, offsets: fi.offsets, keep: fi.keep });
    return r.coded === f.coded_bands && r.pulses.every((v, i) => v === Math.round(f.pvq_bits[i] * 8)) && r.ebits.every((v, i) => v === f.fine_bits[i]);
  }) : [];
  const nMatch = check.filter(Boolean).length;
  const clipS = raw && raw.music && raw.music.dur_s ? raw.music.dur_s : frames ? frames.length * 0.02 : 0;
  // Packets that start after the clip ends: the encoder only saw the silence appended to flush its delay.
  const clipEnd = raw && raw.music && raw.music.n_samples ? raw.music.n_samples : Infinity;
  const isPad = (f) => f.start_sample != null && f.start_sample >= clipEnd;
  const nPad = frames ? frames.filter(isPad).length : 0;

  // ---- state ----
  const BYTES_MIN = 15, BYTES_MAX = 640, POS_MAX = 400; // 6 kb/s .. 256 kb/s, one packet per 20 ms
  const posToBytes = (p) => Math.round(BYTES_MIN * Math.pow(BYTES_MAX / BYTES_MIN, p / POS_MAX));
  const bytesToPos = (b) => Math.round((POS_MAX * Math.log(b / BYTES_MIN)) / Math.log(BYTES_MAX / BYTES_MIN));
  const kbps = (bytes) => bytes * 0.4; // bytes per 20 ms -> kb/s
  const fmtKbps = (bytes) => { const k = kbps(bytes); return (k >= 100 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)) + ' kb/s'; };
  const st = { pos: bytesToPos(80), trim: 5, unit: 'bits', frame: -1, hoverBand: -1, hoverFrame: -1, hoverRow: -1, sweeping: false };
  let inp = null, res = null; // current inputs and allocation
  const disp = { fine: new Float64Array(NB), pvq: new Float64Array(NB), ymax: 60, ready: false };
  const target = { fine: new Float64Array(NB), pvq: new Float64Array(NB), ymax: 60 };
  let heatPattern = null, heatPatternKey = '', heatLut = null, heatLutKey = '';

  function inputs() {
    if (st.frame >= 0 && frames) return frameInputs(frames[st.frame]);
    const bytes = posToBytes(st.pos);
    return { bytes, head: HEAD_Q3, acr: 0, total: (bytes - 1) * 64 - HEAD_Q3 - 1, trim: st.trim, offsets: null, transient: false, f: null };
  }

  // ---- DOM ----
  OX.ui.css('celt-alloc', `
    .ca-controls .ox-slider { grid-template-columns: auto minmax(110px, 210px) 5.6em; }
    .ca-mode { display: none; align-items: center; gap: 10px 14px; flex-wrap: wrap; margin: 8px 0 0; padding: 7px 10px; border-radius: 8px; background: var(--accent-soft); color: var(--ink); font-size: var(--small); }
    .ca-mode.on { display: flex; }
    .ca-mode .what { flex: 1 1 100%; }
    .ca-mode .ok { color: var(--good); font-weight: 600; }
    .ca-mode .no { color: var(--bad); font-weight: 600; }
    .ca-budget { margin: 12px 0 4px; }
    .ca-budget-cap { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 14px; font-size: var(--tiny); color: var(--ink-2); margin-bottom: 5px; }
    .ca-budget-cap b { font: 500 var(--tiny)/1.3 var(--font-mono); color: var(--ink); font-weight: 500; }
    .ca-track { display: flex; height: 20px; background: var(--sunken); border-radius: 5px; overflow: hidden; }
    .ca-seg { flex: none; height: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; white-space: nowrap; font: 500 0.68rem/1 var(--font-mono); transition: width .18s ease-out; }
    .ca-seg + .ca-seg { box-shadow: inset 2px 0 0 var(--surface); }
    .ca-seg.toc { background: var(--ink-3); color: var(--surface); }
    .ca-seg.head { background: color-mix(in srgb, var(--ink-3) 38%, var(--surface)); color: var(--ink); }
    .ca-seg.fine { background: color-mix(in srgb, var(--celt) 38%, var(--surface)); color: var(--ink); }
    .ca-seg.pvq { background: var(--celt); color: var(--surface); }
    @media (prefers-reduced-motion: reduce) { .ca-seg { transition: none; } }
    .ca-detail { min-height: 3.2em; margin: 6px 0 0; font-size: var(--small); color: var(--ink-2); line-height: 1.6; }
    .ca-detail b { color: var(--ink); font-weight: 600; }
    .ca-detail .m { font-family: var(--font-mono); font-size: 0.92em; color: var(--ink); }
    .ca-sw-fold { width: 12px; height: 10px; border: 1px dashed var(--ink-3); border-radius: 2px; display: inline-block;
      background: repeating-linear-gradient(45deg, transparent 0 3px, color-mix(in srgb, var(--ink-3) 45%, transparent) 3px 4px); }
    .ca-sw-boost { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid var(--accent); display: inline-block; }
    .ca-sec { margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--rule); }
    .ca-sec h4 { margin: 0 0 2px; font: 600 0.98rem/1.4 var(--font-body); }
    .ca-sec .sub { margin: 0 0 8px; font-size: var(--tiny); color: var(--ink-3); }
    .ca-ro-sub { display: block; margin-top: 1px; font: 400 var(--tiny)/1.4 var(--font-body); color: var(--ink-3); }
    .ca-heat canvas { cursor: pointer; }
    .ca-main canvas:focus-visible, .ca-heat canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  `);

  const sl = slider({ id: 'ca-rate', label: OX.t('码率', 'Bitrate'), min: 0, max: POS_MAX, step: 1, value: st.pos,
    format: (p) => fmtKbps(st.frame >= 0 && frames ? frames[st.frame].bytes : posToBytes(p)),
    onInput: (p) => { stopSweep(); st.pos = p; if (st.frame >= 0) { st.frame = -1; sl.set(p); } update(); } });
  const trimSl = slider({ id: 'ca-trim', label: OX.t('分配倾斜（trim）', 'Allocation trim'), min: 0, max: 10, step: 1, value: st.trim, format: (v) => String(v),
    onInput: (v) => {
      st.trim = v;
      if (st.frame >= 0) { st.pos = bytesToPos(Math.min(BYTES_MAX, Math.max(BYTES_MIN, frames[st.frame].bytes))); st.frame = -1; sl.set(st.pos); }
      update();
    } });
  const unitSeg = segmented({ id: 'ca-unit', label: OX.t('纵轴', 'Y axis'), value: st.unit, options: [{ value: 'bits', label: OX.t('每个频带的比特', 'Bits per band') }, { value: 'coef', label: OX.t('每个系数的比特', 'Bits per coefficient') }],
    onChange: (v) => { st.unit = v; computeTarget(); snap(); main.redraw(); } });
  const sweepBtn = button({ id: 'ca-sweep', icon: 'play', label: OX.t('从低到高扫一遍', 'Sweep from low to high'), kind: 'ghost', onClick: () => (st.sweeping ? stopSweep() : startSweep()) });

  const modeBar = el('div', { class: 'ca-mode', role: 'status' });
  const capL = el('span'), capR = el('span');
  const track = el('div', { class: 'ca-track', role: 'img' });
  const segs = ['toc', 'head', 'fine', 'pvq'].map((c) => { const s = el('div', { class: 'ca-seg ' + c }); track.appendChild(s); return s; });
  const host = el('div', { class: 'ca-main' });
  const detail = el('p', { class: 'ca-detail', 'aria-live': 'polite' });
  const roRows = readout(OX.t('分配表插值', 'Allocation table interpolation'), '');
  const roCoded = readout(OX.t('编码形状的频带', 'Bands with a coded shape'), '');
  const roUsed = readout(OX.t('分给频带（精细能量 + PVQ）', 'Allocated to bands (fine energy + PVQ)'), '');
  const roUsedVal = roUsed.querySelector('.ox-ro-val');
  const roLeft = readout(OX.t('到达上限，没用上', 'Unused: bands at their cap'), '');

  body.append(
    el('div', { class: 'ox-controls ca-controls' }, [sl, trimSl, unitSeg, sweepBtn]),
    el('p', { class: 'ox-note', text: OX.t('分配倾斜（trim）越大，比特越偏向低频；越小，越偏向高频。20 毫秒的帧里 trim = 8 时不倾斜。编码器的起点是 5（码率低于 80 kb/s 时降到 4 到 5 之间），再按频谱的走势和瞬态调整。编码器分析输入信号时（本页的音乐码流就是这样编码的），还会看各频带的音调性（有多像纯音）怎样随频率变化：高频比低频越像纯音，trim 越小。', 'A larger allocation trim shifts bits toward low frequencies; a smaller one shifts them toward high frequencies. In a 20 ms frame, trim = 8 means no tilt. The encoder starts from 5 (lowered to between 4 and 5 below 80 kb/s), then adjusts it to the slope of the spectrum and to transients. When the encoder analyzes the input signal (as it did for the music stream on this page), it also looks at how the tonality of the bands (how much they sound like pure tones) changes with frequency: the more tonal the high bands are compared with the low bands, the smaller the trim.') }),
    modeBar,
    el('div', { class: 'ca-budget' }, [el('div', { class: 'ca-budget-cap' }, [capL, capR]), track]),
    host,
    detail,
    el('div', { class: 'ox-readouts' }, [roRows, roCoded, roUsed, roLeft]),
    (() => {
      const lg = OX.ui.legend([
        { label: OX.t('TOC 字节（包头，见第 8 节）', 'TOC byte (packet header, see Section 8)'), color: 'var(--ink-3)', kind: 'box' },
        { label: OX.t('头部、粗能量等', 'Header, coarse energy, etc.'), color: 'color-mix(in srgb, var(--ink-3) 38%, var(--surface))', kind: 'box' },
        { label: OX.t('精细能量', 'Fine energy'), color: 'color-mix(in srgb, var(--celt) 38%, var(--surface))', kind: 'box' },
        { label: OX.t('PVQ（形状）', 'PVQ (shape)'), color: 'var(--celt)', kind: 'box' },
      ]);
      lg.appendChild(el('span', { class: 'ox-legend-item' }, [el('span', { class: 'ca-sw-fold' }), OX.t('放不下 PVQ 脉冲，靠折叠填充', 'No room for PVQ pulses, filled by folding')]));
      lg.appendChild(el('span', { class: 'ox-legend-item' }, [el('span', { class: 'ca-sw-boost' }), OX.t('动态分配（dynalloc）加了比特', 'Bits added by dynamic allocation (dynalloc)')]));
      return lg;
    })(),
    el('p', { class: 'ox-note', text: OX.t(`上图的分配算法从 src/rate.rs 逐行移植，取单声道、20 毫秒一帧、全带。拖动码率时，假设没有动态分配，头部和粗能量等按真实码流的中位数估为 ${fmtBits(HEAD_Q3)} 比特（实际随信号变化）。跳过哪些频带，按 opus1 编码器的规则决定。编码器还会参考前几帧编码了多少个频带，以及分析输入信号时测出的信号带宽；图里当作没有前一帧，信号占满 20 kHz。实际的 Opus 在很低的码率下会改用 SILK 或更窄的带宽，这里只演示分配算法本身。`,
      `The allocation algorithm in the chart above is ported line by line from src/rate.rs, for mono, 20 ms frames, and fullband. When you drag the bitrate, the chart assumes no dynamic allocation and estimates the header, coarse energy, and the rest at ${bitsEn(fmtBits(HEAD_Q3))}, the median of the real stream (the real value varies with the signal). Which bands are skipped follows the rule of the opus1 encoder. The encoder also takes into account how many bands the previous frames coded and the signal bandwidth found by its analysis of the input signal. The chart assumes that there is no previous frame and that the signal fills the full 20 kHz. At very low bitrates, real Opus switches to SILK or to a narrower bandwidth; this chart shows only the allocation algorithm itself.`) }),
  );

  // ---- second panel: the real stream ----
  const heatHost = el('div', { class: 'ca-heat' });
  const heatDetail = el('p', { class: 'ca-detail', 'aria-live': 'polite' });
  body.append(el('div', { class: 'ca-sec' }, [
    el('h4', { text: OX.t('真实码流：每一帧各频带分到的 PVQ 比特', 'Real stream: the PVQ bits of each band in every frame') }),
    el('p', { class: 'sub', text: frames ? OX.t(`opus1 编码的 ${streamKbps} kb/s 可变码率码流 · 单声道 · 20 毫秒一帧 · ${frames.length} 帧（${clipS} 秒合成音乐${nPad ? `；最后 ${nPad} 帧只编码了末尾补上的静音` : ''}）`,
      `${streamKbps} kb/s variable-bitrate stream encoded by opus1 · mono · 20 ms frames · ${frames.length} frames (${clipS} s of synthesized music${nPad ? `; the last ${nPad === 1 ? 'frame codes' : `${nPad} frames code`} only the silence padded at the end` : ''})`) : '' }),
    heatHost, heatDetail,
    el('p', { class: 'ox-note', text: frames
      ? OX.t(`格子里的值是解码器从真实码流里解出的。本页用上面的 JavaScript 版分配算法，按每帧的预算、trim 和动态分配重算了一遍。跳过哪些频带，和解码器一样从码流里读出。${nMatch === frames.length ? `全部 ${frames.length} 帧里，每个频带的 PVQ 比特和精细能量比特都与码流一致。` : `${frames.length} 帧里有 ${nMatch} 帧一致（每个频带的 PVQ 比特和精细能量比特都相同）。`}`,
        `The values in the cells are what the decoder read from the real bitstream. This page reran the JavaScript allocation above with each frame's budget, trim, and dynamic allocation. Which bands are skipped is read from the bitstream, as the decoder does. ${nMatch === frames.length ? `In all ${frames.length} frames, the PVQ bits and fine-energy bits of every band match the bitstream.` : `${nMatch} of the ${frames.length} frames match (the same PVQ bits and fine-energy bits in every band).`}`)
      : OX.t('缺少数据文件 data/opus-demo-data.js，无法显示真实码流。', 'The data file data/opus-demo-data.js is missing, so the real stream cannot be shown.') }),
  ]));


  // ---- computing ----
  function computeTarget() {
    inp = inputs();
    res = A.compute({ total: inp.total, trim: inp.trim, LM, offsets: inp.offsets, keep: inp.keep });
    // A band folds when it has no PVQ bits, or (long blocks) when its bits round to zero pulses in bits2pulses.
    // The bits carried over from earlier bands at coding time (balance) are ignored here.
    res.fold = res.pulses.map((p, j) => p <= 0 || (!inp.transient && p <= A.splitThreshold(j, LM) && A.bits2pulses(j, LM, p) === 0));
    let mx = 0;
    for (let j = 0; j < NB; j++) {
      const k = st.unit === 'coef' ? 1 / nbins(j) : 1;
      target.fine[j] = res.ebits[j] * k;
      target.pvq[j] = (res.pulses[j] / 8) * k;
      mx = Math.max(mx, target.fine[j] + target.pvq[j]);
    }
    target.ymax = st.unit === 'coef' ? 8.5 : Math.max(10, mx * 1.12);
  }
  function snap() { disp.fine.set(target.fine); disp.pvq.set(target.pvq); disp.ymax = target.ymax; disp.ready = true; }

  const anim = OX.loop(root, (dt) => {
    if (st.sweeping) {
      st.pos = Math.min(POS_MAX, st.pos + (dt * POS_MAX) / 8);
      sl.set(Math.round(st.pos));
      computeTarget(); if (OX.reducedMotion()) snap();
      renderText();
      if (st.pos >= POS_MAX) stopSweep();
    }
    const a = 1 - Math.exp(-dt * 12);
    let moving = false;
    const ease = (cur, tgt, eps) => { const d = tgt - cur; if (Math.abs(d) > eps) moving = true; return Math.abs(d) > eps ? cur + d * a : tgt; };
    for (let j = 0; j < NB; j++) { disp.fine[j] = ease(disp.fine[j], target.fine[j], 0.01); disp.pvq[j] = ease(disp.pvq[j], target.pvq[j], 0.01); }
    disp.ymax = ease(disp.ymax, target.ymax, 0.05);
    main.redraw();
    return st.sweeping || moving;
  });

  function update() {
    computeTarget();
    if (!disp.ready || OX.reducedMotion()) { snap(); main.redraw(); } else anim.start();
    renderText();
    heat.redraw();
    renderHeatDetail();
    heatAria();
  }
  function startSweep() {
    st.frame = -1; st.sweeping = true; st.pos = 0; sl.set(0);
    sweepBtn.setIcon('pause', OX.t('暂停', 'Pause'));
    update(); anim.start();
  }
  function stopSweep() { if (!st.sweeping) return; st.sweeping = false; st.pos = Math.round(st.pos); sweepBtn.setIcon('play', OX.t('从低到高扫一遍', 'Sweep from low to high')); }

  function setFrame(i) {
    if (!frames) return;
    stopSweep();
    st.frame = Math.max(0, Math.min(frames.length - 1, i));
    const f = frames[st.frame];
    st.trim = f.alloc_trim; trimSl.set(st.trim);
    st.pos = bytesToPos(Math.min(BYTES_MAX, Math.max(BYTES_MIN, f.bytes))); sl.set(st.pos);
    update();
  }
  function leaveFrame() { if (st.frame < 0) return; st.frame = -1; sl.set(st.pos); update(); }

  // ---- text: budget bar, readouts, detail ----
  function renderText() {
    const totalBits = inp.bytes * 8;
    const fineB = res.ebits.reduce((s, v) => s + v, 0);
    const pvqQ3 = res.pulses.reduce((s, v) => s + v, 0);
    const parts = [64, inp.head, fineB * 8, pvqQ3]; // 1/8 bit
    const w = track.clientWidth || 600;
    const names = ['TOC', OX.t('头部等', 'Header etc.'), OX.t('精细能量', 'Fine energy'), 'PVQ'];
    const textW = (t) => [...t].reduce((a, c) => a + (c.charCodeAt(0) > 0x2e7f ? 11 : 6.6), 0); // ~0.68rem
    parts.forEach((v, i) => {
      const frac = v / 8 / totalBits;
      segs[i].style.width = (frac * 100).toFixed(3) + '%';
      const num = fmtBits(v), label = i === 0 ? num : `${names[i]} ${num}`;
      const px = frac * w;
      segs[i].textContent = px > textW(label) + 10 ? label : px > textW(num) + 8 ? num : '';
    });
    track.setAttribute('aria-label', OX.t(`一帧 ${totalBits} 比特：TOC 8 比特，头部和粗能量等 ${fmtBits(inp.head)} 比特，精细能量 ${fineB} 比特，PVQ ${fmtBits(pvqQ3)} 比特`,
      `One frame of ${totalBits} bits: TOC byte 8 bits; header, coarse energy, etc. ${bitsEn(fmtBits(inp.head))}; fine energy ${bitsEn(fineB)}; PVQ ${bitsEn(fmtBits(pvqQ3))}`));
    capL.innerHTML = OX.t(`一帧 <b>${inp.bytes} 字节 = ${totalBits} 比特</b>（20 毫秒，相当于 ${fmtKbps(inp.bytes)}）`,
      `One frame: <b>${inp.bytes} bytes = ${totalBits} bits</b> (20 ms, equivalent to ${fmtKbps(inp.bytes)})`);
    capR.innerHTML = st.frame >= 0
      ? OX.t(`头部、粗能量等（码流实测）<b>${fmtBits(inp.head)} 比特</b>`, `Header, coarse energy, etc. (measured in the stream): <b>${bitsEn(fmtBits(inp.head))}</b>`)
      : OX.t(`头部、粗能量等（估计）<b>≈ ${fmtBits(inp.head)} 比特</b>`, `Header, coarse energy, etc. (estimated): <b>≈ ${bitsEn(fmtBits(inp.head))}</b>`);

    roRows.set(res.rowHi >= A.NROWS ? OX.t(`第 10 行 → 上限的 ${res.frac}/64 处`, `Row 10 → ${res.frac}/64 of the way to the cap`) : OX.t(`第 ${res.rowLo} 行 → 第 ${res.rowHi} 行的 ${res.frac}/64 处`, `Row ${res.rowLo} → ${res.frac}/64 of the way to row ${res.rowHi}`));
    const nFold = res.fold.filter(Boolean).length;
    roCoded.set(nFold ? OX.t(`${NB - nFold} 个（另有 ${nFold} 个靠折叠）`, `${NB - nFold} (another ${nFold} filled by folding)`) : OX.t(`全部 ${NB} 个`, `All ${NB}`));
    const usedQ3 = fineB * 8 + pvqQ3;
    roUsedVal.innerHTML = OX.t(`${fmtBits(usedQ3)} 比特`, `${bitsEn(fmtBits(usedQ3))}`) + (res.skipBits ? OX.t(`<span class="ca-ro-sub">另有 ${res.skipBits} 比特用来标记跳过哪些高频频带</span>`, `<span class="ca-ro-sub">plus ${bitsEn(res.skipBits)} to mark which high bands are skipped</span>`) : '');
    const left = inp.total - usedQ3 - res.skipBits * 8;
    roLeft.style.display = left >= 8 ? '' : 'none';
    roLeft.set(OX.t(`${fmtBits(left)} 比特`, `${bitsEn(fmtBits(left))}`));

    if (st.frame >= 0) {
      const f = frames[st.frame], ok = check[st.frame];
      modeBar.classList.add('on');
      modeBar.innerHTML = '';
      const tags = [f.transient ? OX.t('瞬态帧', 'transient frame') : OX.t('非瞬态帧', 'non-transient frame'), `trim ${f.alloc_trim}`];
      if (f.boost_bits.some((v) => v > 0)) tags.push(OX.t('有动态分配', 'with dynamic allocation'));
      if (isPad(f)) tags.push(OX.t('只含末尾补上的静音', 'codes only the silence padded at the end'));
      const acr = inp.acr ? OX.t('预算里先留出 1 比特给防坍缩标志（瞬态帧里某个短块分不到脉冲时，解码器往里补噪声）。', 'The budget first sets aside 1 bit for the anti-collapse flag (when a short block in a transient frame gets no pulses, the decoder fills it with noise).') : '';
      modeBar.append(
        el('span', { class: 'what', html: OX.t(`正在看真实码流的<b>第 ${f.index} 帧</b>（${(f.index * 0.02).toFixed(2)} 秒处，${tags.join('，')}）。${acr}`,
          `Showing <b>frame ${f.index}</b> of the real stream (at ${(f.index * 0.02).toFixed(2)} s, ${tags.join(', ')}). ${acr}`) }),
        el('span', { class: ok ? 'ok' : 'no', text: ok ? OX.t('✓ 按码流里的预算重算，与解码器得到的分配完全一致', '✓ Recomputed from the budget in the bitstream: identical to the allocation the decoder got') : OX.t('✗ 重算结果与码流不一致', '✗ The recomputed allocation does not match the bitstream') }),
        button({ label: OX.t('回到码率滑块', 'Back to the bitrate slider'), kind: 'ghost', onClick: () => { leaveFrame(); sl.input.focus(); } }), // the button is removed with modeBar's contents
      );
    } else { modeBar.classList.remove('on'); modeBar.innerHTML = ''; }
    renderDetail();
  }

  function renderDetail() {
    const j = st.hoverBand;
    mainAria();
    if (j < 0) {
      const nf = res.fold.filter(Boolean).length;
      detail.innerHTML = OX.t('把指针移到一个频带上，或点一下，看它分到多少比特、能放几个脉冲。', 'Hover over a band or tap it to see how many bits it gets and how many pulses fit.') + (nf
        ? OX.t(`现在有 <b>${nf}</b> 个频带放不下 PVQ 脉冲，解码时用折叠填充。`, ` Right now <b>${nf}</b> ${nf === 1 ? 'band has' : 'bands have'} no room for PVQ pulses; the decoder fills ${nf === 1 ? 'it' : 'them'} by folding.`)
        : OX.t('现在每个频带都能放下 PVQ 脉冲，不需要折叠。', ' Right now every band has room for PVQ pulses, so no folding is needed.'));
      return;
    }
    const n = nbins(j), p = res.pulses[j], fine = res.ebits[j];
    let s = OX.t(`<b>频带 ${j}</b> · ${bandHz(j)} · ${n} 个系数 —— 精细能量 <span class="m">${fine} 比特</span>，PVQ <span class="m">${fmtBits(p)} 比特</span>（<span class="m">${(p / 8 / n).toFixed(2)}</span> 比特/系数）。`,
      `<b>Band ${j}</b> · ${bandHz(j)} · ${n} coefficients — fine energy <span class="m">${bitsEn(fine)}</span>, PVQ <span class="m">${bitsEn(fmtBits(p))}</span> (<span class="m">${(p / 8 / n).toFixed(2)}</span> bits/coefficient).`);
    if (inp.offsets && inp.offsets[j] > 0) s += OX.t(`其中动态分配加了 <span class="m">${fmtBits(inp.offsets[j])} 比特</span>。`, ` Dynamic allocation added <span class="m">${bitsEn(fmtBits(inp.offsets[j]))}</span> of these.`);
    if (p > 0 && res.fold[j]) {
      s += OX.t(`不够放 1 个脉冲（K = 1 要 ${fmtBits(A.pulses2bits(j, LM, 1))} 比特），编码时多半也会折叠；前面频带没用完的比特有时会顺延过来。`,
        ` Not enough for 1 pulse (K = 1 needs ${bitsEn(fmtBits(A.pulses2bits(j, LM, 1)))}), so this band will most likely be folded too; bits left unused by lower bands sometimes carry over to it.`);
    } else if (p <= 0) {
      const why = res.why[j];
      s += why === 'skip' ? OX.t('编码器选择跳过：8 kHz 以上的频带，每个系数分不到 9/16 比特就不编码形状（前几帧编码过的频带，门限是 7/16），省下的比特分给更低的频带。', ' The encoder chose to skip it: above 8 kHz, a band that gets less than 9/16 bit per coefficient (7/16 for a band that earlier frames coded) has no shape coded, and the bits saved go to lower bands.')
        : why === 'fine' ? OX.t('比特全部用在精细能量上，没有剩给形状。', ' All its bits went to fine energy, with none left for the shape.')
        : OX.t('插值后不到这个频带的门限（每个系数 3/16 比特），不编码形状。', ' After interpolation it is below the threshold of this band (3/16 bit per coefficient), so no shape is coded.');
      s += OX.t('解码器把低频已解码的形状复制过来（折叠），按这个频带的能量缩放。', ' The decoder copies an already decoded shape from lower frequencies (folding) and scales it to the energy of this band.');
    } else if (inp.transient) {
      s += OX.t('瞬态帧按 8 个短块编码，这里不估算脉冲数。', ' A transient frame is coded as 8 short blocks, so the number of pulses is not estimated here.');
    } else if (p > A.splitThreshold(j, LM)) {
      s += OX.t(`超过 ${fmtBits(A.splitThreshold(j, LM))} 比特，一个 PVQ 码本装不下：先对半切开、编码两半的能量比例，再分别做 PVQ。`,
        ` More than ${bitsEn(fmtBits(A.splitThreshold(j, LM)))} is too much for one PVQ codebook: the band is split in half, the energy ratio of the two halves is coded, and then each half is coded with PVQ.`);
    } else {
      const q = A.bits2pulses(j, LM, p);
      s += OX.t(`大约能放 <span class="m">K = ${A.getPulses(q)}</span> 个脉冲（码本序号约 ${fmtBits(A.pulses2bits(j, LM, q))} 比特）。`,
        ` Pulses that fit: about <span class="m">K = ${A.getPulses(q)}</span> (the codebook index takes about ${bitsEn(fmtBits(A.pulses2bits(j, LM, q)))}).`);
    }
    detail.innerHTML = s;
  }

  // ---- drawing helpers ----
  function rgb(c) {
    c = (c || '').trim();
    if (c[0] === '#') { let h = c.slice(1); if (h.length === 3) h = h.split('').map((x) => x + x).join(''); const v = parseInt(h, 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
    const m = c.match(/[\d.]+/g);
    return m && m.length >= 3 ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }
  const mixRgb = (a, b, t) => { const x = rgb(a), y = rgb(b); return [0, 1, 2].map((i) => Math.round(x[i] + (y[i] - x[i]) * t)); };
  const css = (v) => `rgb(${v[0]},${v[1]},${v[2]})`;
  function niceStep(max, n) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  }
  function hatch(ctx, x, y, w, h, color, gap) {
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let d = -h; d < w; d += gap) { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); }
    ctx.stroke();
    ctx.restore();
  }
  const FREQ_EDGES = [0, 5, 9, 13, 17, 19, 21]; // 0, 1, 2, 4, 8, 12, 20 kHz

  function drawMain(ctx, w, h) {
    const T = OX.theme, narrow = w < 560;
    if (!res) return;
    const L = narrow ? 40 : 50, R = 6, top = 24, bottom = h - 42;
    const pw = w - L - R, slot = pw / NB;
    const coef = st.unit === 'coef';
    const ymax = disp.ymax;
    const ys = OX.view.linear(0, ymax, bottom, top);
    const sx = (j) => L + j * slot;

    // y grid and ticks
    const step = coef ? 2 : niceStep(ymax, narrow ? 4 : 5);
    const ticks = [];
    for (let v = 0; v <= ymax + 1e-9; v += step) ticks.push({ v, label: coef ? String(v) : v >= 1000 ? (v / 1000) + 'k' : String(+v.toFixed(3)) });
    OX.view.axis(ctx, ys, { side: 'left', at: L, ticks, grid: pw });
    ctx.fillStyle = T.ink3; ctx.font = OX.font(10.5, 'mono'); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.font = OX.font(10.5, 'body');
    ctx.fillText(coef ? OX.t('比特/系数', 'Bits/coefficient') : OX.t('比特/频带', 'Bits/band'), 4, 12);

    // hovered band
    if (st.hoverBand >= 0) { ctx.fillStyle = OX.alpha(T.ink3, T.dark ? 0.12 : 0.08); ctx.fillRect(sx(st.hoverBand), top - 6, slot, bottom - top + 6); }

    // the 11 rows of the allocation table (per-coefficient view)
    if (coef) {
      const rowY = (r, j) => ys(A.ALLOC[r * NB + j] / 32);
      const capY = (j) => ys(res.cap[j] / 8 / nbins(j));
      const stepLine = (yOf) => { ctx.beginPath(); for (let j = 0; j < NB; j++) { const y = Math.round(yOf(j)) + 0.5; if (j === 0) ctx.moveTo(sx(j), y); else ctx.lineTo(sx(j), y); ctx.lineTo(sx(j + 1), y); } ctx.stroke(); };
      ctx.lineWidth = 1;
      for (let r = 1; r < A.NROWS; r++) {
        if (r === res.rowLo || r === res.rowHi) continue;
        ctx.strokeStyle = OX.alpha(T.ink3, 0.32); stepLine((j) => rowY(r, j));
      }
      ctx.lineWidth = 1.5;
      const hl = [res.rowLo, res.rowHi];
      hl.forEach((r, k) => {
        if (r === 0) return;
        ctx.strokeStyle = T.accent; ctx.setLineDash(k === 0 ? [4, 3] : []);
        stepLine((j) => (r >= A.NROWS ? capY(j) : rowY(r, j)));
        ctx.setLineDash([]);
      });
      // labels for the two rows, at band 1, one above the upper line and one below the lower one
      ctx.font = OX.font(10.5, 'mono', 500); ctx.textAlign = 'left';
      const yHi = res.rowHi >= A.NROWS ? capY(1) : rowY(res.rowHi, 1), yLo = rowY(res.rowLo, 1);
      const tag = (text, x, y) => { // label on a surface-colored chip so lines and bars do not run through it
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = OX.alpha(T.surface, 0.88); ctx.fillRect(x - 2, y - 7, tw + 4, 14);
        ctx.fillStyle = T.accent; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y + 0.5);
      };
      tag(res.rowHi >= A.NROWS ? OX.t('上限', 'Cap') : OX.t(`第 ${res.rowHi} 行`, `Row ${res.rowHi}`), sx(1) + 3, yHi - 9);
      if (res.rowLo > 0) tag(OX.t(`第 ${res.rowLo} 行`, `Row ${res.rowLo}`), sx(1) + 3, yLo + 9);
      ctx.font = OX.font(10.5, 'body'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(narrow ? OX.t('折线：分配表的 10 行', 'Lines: the 10 table rows') : OX.t('折线：分配表第 1–10 行（每个系数的比特，未加分配倾斜）；蓝色：正在插值的两行', 'Lines: allocation table rows 1–10 (bits per coefficient, before trim); blue: the two rows being interpolated'), w - R, 12);
    }

    // bars: fine energy (light) under PVQ (solid), with a 2px surface gap
    const gap = Math.max(1, Math.min(4, slot * 0.14));
    const fineCol = css(mixRgb(T.surface, T.celt, 0.38));
    const folded = [];
    for (let j = 0; j < NB; j++) {
      const x = sx(j) + gap, bw = slot - 2 * gap;
      const fH = Math.max(0, bottom - ys(disp.fine[j]));
      const pH = Math.max(0, ys(disp.fine[j]) - ys(disp.fine[j] + disp.pvq[j]));
      const isFold = res.fold[j];
      folded.push(isFold);
      if (fH > 0.3) { ctx.fillStyle = fineCol; ctx.fillRect(x, bottom - fH, bw, fH); }
      if (pH > 0.3) {
        const g = fH > 0.3 && pH > 3 ? 2 : 0;
        ctx.fillStyle = T.celt;
        const y0 = bottom - fH - pH, hh = pH - g;
        ctx.beginPath();
        const rr = Math.max(0, Math.min(3, bw / 3, hh));
        if (ctx.roundRect && bw > 0 && hh > 0) ctx.roundRect(x, y0, bw, hh, [rr, rr, 0, 0]); else ctx.rect(x, y0, Math.max(0, bw), Math.max(0, hh));
        ctx.fill();
      }
      if (isFold) {
        const bh = Math.max(18, fH + 4);
        ctx.strokeStyle = OX.alpha(T.ink3, 0.9); ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
        ctx.strokeRect(Math.round(x) + 0.5, Math.round(bottom - bh) + 0.5, Math.round(bw) - 1, Math.round(bh) - 1);
        ctx.setLineDash([]);
        hatch(ctx, x + 1, bottom - bh + 1, bw - 2, bh - 2, OX.alpha(T.ink3, 0.45), 5);
      }
      if (inp.offsets && inp.offsets[j] > 0) {
        const yb = bottom - fH - pH - 5, cx = sx(j) + slot / 2;
        ctx.fillStyle = T.accent; ctx.beginPath(); ctx.moveTo(cx - 4, yb - 6); ctx.lineTo(cx + 4, yb - 6); ctx.lineTo(cx, yb); ctx.closePath(); ctx.fill();
      }
    }
    // a bracket over each run of folded bands
    for (let j = 0; j < NB; j++) {
      if (!folded[j] || (j > 0 && folded[j - 1])) continue;
      let k = j; while (k + 1 < NB && folded[k + 1]) k++;
      const x0 = sx(j) + gap, x1 = sx(k + 1) - gap, yb = bottom - 26;
      ctx.strokeStyle = T.ink3; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0 + 0.5, yb + 4); ctx.lineTo(x0 + 0.5, yb + 0.5); ctx.lineTo(x1 - 0.5, yb + 0.5); ctx.lineTo(x1 - 0.5, yb + 4); ctx.stroke();
      ctx.fillStyle = T.ink2; ctx.font = OX.font(11, 'body', 500); ctx.textBaseline = 'bottom';
      const long = OX.t('折叠：复制低频的形状', 'Folding: shape copied from lower bands'), short = OX.t('折叠', 'Folding');
      const text = ctx.measureText(long).width + 6 < x1 - x0 ? long : short;
      const tw = ctx.measureText(text).width;
      // keep the label inside the canvas when the run is narrower than the text
      let cx = (x0 + x1) / 2; cx = Math.min(w - R - tw / 2, Math.max(L + tw / 2, cx));
      ctx.textAlign = 'center'; ctx.fillText(text, cx, yb - 3);
    }

    // baseline
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, Math.round(bottom) + 0.5); ctx.lineTo(L + pw, Math.round(bottom) + 0.5); ctx.stroke();
    // x axis: band numbers, then band edges in kHz
    ctx.font = OX.font(10, 'mono'); ctx.textBaseline = 'top';
    for (let j = 0; j < NB; j++) {
      if (slot < 17 && j % 5 !== 0 && j !== NB - 1) continue;
      ctx.fillStyle = j === st.hoverBand ? T.ink : T.ink3; ctx.textAlign = 'center';
      ctx.fillText(String(j), sx(j) + slot / 2, bottom + 5);
    }
    const yF = bottom + 22;
    for (const e of FREQ_EDGES) {
      const x = Math.round(sx(e)) + 0.5;
      ctx.strokeStyle = T.rule; ctx.beginPath(); ctx.moveTo(x, yF - 3); ctx.lineTo(x, yF + 1); ctx.stroke();
      ctx.fillStyle = T.ink3; ctx.textAlign = e === 0 ? 'left' : e === NB ? 'right' : 'center';
      ctx.fillText(e === 0 ? '0' : khz(e), e === 0 ? x - 0.5 : x, yF + 3);
    }
    ctx.fillStyle = T.ink3; ctx.textAlign = 'right';
    ctx.fillText(OX.t('频带', 'Band'), L - 6, bottom + 5);
    ctx.fillText('kHz', L - 6, yF + 3);
  }

  function heatColors(T) {
    const key = T.surface + T.celt + T.ink + T.dark;
    if (heatLutKey === key) return heatLut;
    const lo = mixRgb(T.surface, T.celt, T.dark ? 0.2 : 0.1), mid = rgb(T.celt), hi = T.dark ? mixRgb(T.celt, '#ffffff', 0.6) : mixRgb(T.celt, T.ink, 0.6);
    heatLut = [];
    for (let i = 0; i < 64; i++) {
      const t = i / 63;
      const c = t < 0.5 ? mixRgb(css(lo), css(mid), t / 0.5) : mixRgb(css(mid), css(hi), (t - 0.5) / 0.5);
      heatLut.push(css(c));
    }
    heatLutKey = key;
    return heatLut;
  }
  function zeroPattern(ctx, T) {
    const key = T.ink3 + T.sunken;
    if (heatPattern && heatPatternKey === key) return heatPattern;
    const c = document.createElement('canvas'); c.width = 6; c.height = 6;
    const g = c.getContext('2d');
    g.fillStyle = T.sunken; g.fillRect(0, 0, 6, 6);
    g.strokeStyle = OX.alpha(T.ink3, 0.6); g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, 6); g.lineTo(6, 0); g.moveTo(-3, 3); g.lineTo(3, -3); g.moveTo(3, 9); g.lineTo(9, 3); g.stroke();
    heatPattern = ctx.createPattern(c, 'repeat'); heatPatternKey = key;
    return heatPattern;
  }
  const HEAT_MAX = 4; // bit per coefficient at the top of the color scale
  let hGeo = null;

  function drawHeat(ctx, w, h) {
    const T = OX.theme, narrow = w < 560;
    if (!frames || !frames.length) {
      ctx.fillStyle = T.ink3; ctx.font = OX.font(13, 'body'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(OX.t('缺少数据文件', 'Data file missing'), w / 2, h / 2);
      return;
    }
    const nF = frames.length;
    const L = narrow ? 40 : 50, R = 6;
    const sTop = 16, sH = narrow ? 22 : 28;
    const hTop = sTop + sH + 6, rowH = narrow ? 7 : 8, hBot = hTop + rowH * NB;
    const cw = (w - L - R) / nF;
    hGeo = { L, R, cw, hTop, hBot, rowH, sTop, nF };
    const x = (i) => L + i * cw;

    // strip: bytes per frame; transient frames in the accent color
    const maxB = Math.max(...frames.map((f) => f.bytes));
    const yb = (b) => sTop + sH - (b / maxB) * sH;
    for (let i = 0; i < nF; i++) {
      const f = frames[i];
      ctx.fillStyle = f.transient ? T.accent : OX.alpha(T.ink3, 0.55);
      ctx.fillRect(x(i) + (cw > 3 ? 0.5 : 0), yb(f.bytes), Math.max(1, cw - (cw > 3 ? 1 : 0)), sTop + sH - yb(f.bytes));
    }
    const cbr = streamKbps * 1000 / 400; // bytes of a constant-rate packet
    ctx.strokeStyle = T.ink2; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, Math.round(yb(cbr)) + 0.5); ctx.lineTo(w - R, Math.round(yb(cbr)) + 0.5); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    ctx.fillText(OX.t('每帧字节数', 'Bytes/frame'), 4, 11);
    ctx.textAlign = 'right';
    ctx.fillText(narrow ? OX.t(`虚线 = ${streamKbps} kb/s · 蓝 = 瞬态帧`, `Dashed = ${streamKbps} kb/s · blue = transient`) : OX.t(`虚线 ${cbr} 字节 = ${streamKbps} kb/s · 蓝色 = 瞬态帧`, `Dashed line: ${cbr} bytes = ${streamKbps} kb/s · blue: transient frames`), w - R, 11);
    ctx.textBaseline = 'middle'; ctx.fillText(String(maxB), L - 6, sTop + 3);

    // heat cells: PVQ bits per coefficient
    const lut = heatColors(T), pat = zeroPattern(ctx, T);
    for (let i = 0; i < nF; i++) {
      const f = frames[i];
      for (let j = 0; j < NB; j++) {
        const v = f.pvq_bits[j] / nbins(j);
        ctx.fillStyle = v <= 0 ? pat : lut[Math.min(63, Math.round((v / HEAT_MAX) * 63))];
        const y = hBot - (j + 1) * rowH;
        ctx.fillRect(x(i), y, cw + 0.6, rowH + 0.6);
      }
    }
    // selected and hovered frames
    const outline = (i, color, lw) => { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.strokeRect(x(i) - lw / 2, sTop - 1, cw + lw, hBot - sTop + 1 + lw / 2); };
    if (st.hoverFrame >= 0 && st.hoverFrame !== st.frame) outline(st.hoverFrame, T.ink2, 1);
    if (st.frame >= 0) outline(st.frame, T.accent, 1.5);
    if (st.hoverFrame >= 0 && st.hoverRow >= 0) {
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1;
      ctx.strokeRect(x(st.hoverFrame) - 0.5, hBot - (st.hoverRow + 1) * rowH - 0.5, cw + 1, rowH + 1);
    }

    // left axis: band edges in kHz
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const e of FREQ_EDGES) {
      const y = hBot - e * rowH;
      ctx.strokeStyle = T.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L - 4, Math.round(y) + 0.5); ctx.lineTo(L, Math.round(y) + 0.5); ctx.stroke();
      ctx.fillText(e === 0 ? '0' : khz(e) + 'k', L - 6, Math.min(hBot - 4, Math.max(hTop + 4, y)));
    }
    ctx.save(); ctx.translate(10, (hTop + hBot) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(OX.t('频率 Hz', 'Frequency (Hz)'), 0, 0); ctx.restore();
    // bottom axis: time
    const xs = OX.view.linear(0, nF * 0.02, L, L + nF * cw);
    const tt = []; for (let t = 0; t <= nF * 0.02 + 1e-9; t += 0.5) tt.push({ v: t, label: t === 0 ? '0 s' : narrow && xs(t) + 18 <= w ? t.toFixed(1) : t.toFixed(1) + ' s', align: t === 0 ? 'left' : xs(t) + 18 > w ? 'right' : 'center' });
    OX.view.axis(ctx, xs, { side: 'bottom', at: hBot, ticks: tt });

    // color legend
    const ly = hBot + 30, lw = narrow ? 110 : 150, lx = w - R - lw;
    for (let i = 0; i < 64; i++) { ctx.fillStyle = lut[i]; ctx.fillRect(lx + (i / 64) * lw, ly, lw / 64 + 0.6, 8); }
    ctx.fillStyle = pat; ctx.fillRect(lx - 34, ly, 18, 8);
    ctx.strokeStyle = T.rule; ctx.lineWidth = 1; ctx.strokeRect(lx - 34 + 0.5, ly + 0.5, 17, 7);
    ctx.font = OX.font(10, 'mono'); ctx.fillStyle = T.ink3; ctx.textBaseline = 'top';
    ctx.textAlign = 'center'; ctx.font = OX.font(10, 'body'); ctx.fillText(OX.t('折叠', 'Folding'), lx - 25, ly + 11); ctx.font = OX.font(10, 'mono');
    for (let v = 0; v <= HEAT_MAX; v++) { ctx.textAlign = v === 0 ? 'left' : v === HEAT_MAX ? 'right' : 'center'; ctx.fillText(v === HEAT_MAX ? '≥4' : String(v), lx + (v / HEAT_MAX) * lw, ly + 11); }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.font = OX.font(10.5, 'body');
    ctx.fillText(narrow ? OX.t('PVQ 比特/系数', 'PVQ bits/coefficient') : OX.t('颜色：每个系数的 PVQ 比特', 'Color: PVQ bits per coefficient'), lx - 42, ly + 4);
  }

  function renderHeatDetail() {
    if (!frames) { heatDetail.textContent = ''; return; }
    const i = st.hoverFrame >= 0 ? st.hoverFrame : st.frame;
    if (i < 0) {
      heatDetail.innerHTML = OX.t('把指针移到热图上，或点一下，看每一帧的数值；点一帧，上图会按这一帧的真实预算重算。', 'Hover over the heat map or tap it to see the values of each frame. Selecting a frame recomputes the chart above with the real budget of that frame. ')
        + OX.t('瞬态帧落在拨弦、敲击，以及片段开头和结尾的位置。可变码率给它们更多字节，最上面的频带大多也分到了 PVQ 比特。', 'Transient frames fall on plucks and hits, and at the start and end of the clip. Variable bitrate gives them more bytes, and their top bands mostly get PVQ bits too. ')
        + OX.t('其余的帧里，最上面的频带（15.6–20 kHz）大多分不到 PVQ 比特，靠折叠填充。', 'In most of the other frames, the top band (15.6–20 kHz) gets no PVQ bits and is filled by folding.');
      return;
    }
    const f = frames[i];
    let s = OX.t(`<b>第 ${f.index} 帧</b> · ${(f.index * 0.02).toFixed(2)} 秒处 · <span class="m">${f.bytes}</span> 字节 · ${f.transient ? '瞬态' : '非瞬态'} · trim ${f.alloc_trim} · 编码形状的频带 ${f.pvq_bits.filter((v) => v > 0).length} 个`,
      `<b>Frame ${f.index}</b> · at ${(f.index * 0.02).toFixed(2)} s · <span class="m">${f.bytes}</span> bytes · ${f.transient ? 'transient' : 'non-transient'} · trim ${f.alloc_trim} · bands with a coded shape: ${f.pvq_bits.filter((v) => v > 0).length}`);
    const j = st.hoverFrame >= 0 ? st.hoverRow : -1;
    if (j >= 0) s += OX.t(` —— 频带 ${j}（${bandHz(j)}）：PVQ <span class="m">${fmtBits(Math.round(f.pvq_bits[j] * 8))} 比特</span>，精细能量 <span class="m">${f.fine_bits[j]} 比特</span>`,
      ` — band ${j} (${bandHz(j)}): PVQ <span class="m">${bitsEn(fmtBits(Math.round(f.pvq_bits[j] * 8)))}</span>, fine energy <span class="m">${bitsEn(f.fine_bits[j])}</span>`);
    s += OX.t('。', '.');
    if (isPad(f)) s += OX.t(`片段在 ${clipS} 秒已经结束，这一帧只编码了末尾补上的静音。`, ` The clip ends at ${clipS} s; this frame codes only the silence padded at the end.`);
    if (st.frame !== i) s += OX.t('点一下，在上图查看这一帧。', ' Click or tap to show this frame in the chart above.');
    heatDetail.innerHTML = s;
  }

  // Everything draw() reads is declared above this line (OX.view.canvas draws immediately).
  const main = OX.view.canvas(host, { height: (w) => (w < 560 ? 250 : 300), label: OX.t('CELT 21 个频带分到的比特：浅色是精细能量，深色是 PVQ 形状', 'Bits given to the 21 CELT bands: light is fine energy, dark is the PVQ shape') }, drawMain);
  const heat = OX.view.canvas(heatHost, { height: (w) => (w < 560 ? 262 : 292), label: OX.t('真实码流每帧每个频带的 PVQ 比特热图，横轴是时间，纵轴是频带', 'Heat map of the PVQ bits of each band in each frame of the real stream; time runs across, bands run up') }, drawHeat);
  heat.canvas.tabIndex = 0;

  // ---- pointer and keyboard ----
  function bandAt(ev) {
    const r = main.canvas.getBoundingClientRect();
    const w = r.width, L = w < 560 ? 40 : 50, slot = (w - L - 6) / NB;
    const j = Math.floor((ev.clientX - r.left - L) / slot);
    return j >= 0 && j < NB ? j : -1;
  }
  main.canvas.addEventListener('pointermove', (ev) => { const j = bandAt(ev); if (j !== st.hoverBand) setBand(j); });
  main.canvas.addEventListener('pointerdown', (ev) => setBand(bandAt(ev)));
  main.canvas.addEventListener('pointerleave', () => { if (st.hoverBand >= 0) setBand(-1); });

  function cellAt(ev) {
    if (!hGeo) return { i: -1, j: -1 };
    const r = heat.canvas.getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const i = Math.floor((px - hGeo.L) / hGeo.cw);
    if (i < 0 || i >= hGeo.nF || py < hGeo.sTop - 4 || py > hGeo.hBot + 4) return { i: -1, j: -1 };
    const j = Math.floor((hGeo.hBot - py) / hGeo.rowH);
    return { i, j: j >= 0 && j < NB ? j : -1 };
  }
  heat.canvas.addEventListener('pointermove', (ev) => { const c = cellAt(ev); if (c.i !== st.hoverFrame || c.j !== st.hoverRow) { st.hoverFrame = c.i; st.hoverRow = c.j; renderHeatDetail(); heat.redraw(); } });
  heat.canvas.addEventListener('pointerleave', () => { st.hoverFrame = -1; st.hoverRow = -1; renderHeatDetail(); heat.redraw(); });
  heat.canvas.addEventListener('click', (ev) => { const c = cellAt(ev); if (c.i >= 0) setFrame(c.i); });
  heat.canvas.addEventListener('keydown', (ev) => {
    if (!frames) return;
    const k = ev.key, last = frames.length - 1;
    let i = null;
    if (k === 'ArrowRight' || k === 'ArrowUp') i = st.frame < 0 ? 0 : st.frame + 1; // the first press selects frame 0
    else if (k === 'ArrowLeft' || k === 'ArrowDown') i = st.frame < 0 ? 0 : st.frame - 1;
    else if (k === 'Home') i = 0;
    else if (k === 'End') i = last;
    else if (k === 'Escape' && st.frame >= 0) { ev.preventDefault(); leaveFrame(); return; }
    else return;
    ev.preventDefault();
    setFrame(i);
  });
  if (frames) {
    // a focusable picker: expose it as a slider over the frames (as silk-frame.js does)
    heat.canvas.setAttribute('role', 'slider');
    heat.canvas.setAttribute('aria-valuemin', '0');
    heat.canvas.setAttribute('aria-valuemax', String(frames.length - 1));
    heat.canvas.setAttribute('aria-label', OX.t('真实码流每帧每个频带的 PVQ 比特热图，横轴是时间，纵轴是频带。按左右方向键逐帧查看，上图会显示这一帧的分配；按 Esc 回到码率滑块。', 'Heat map of the PVQ bits of each band in each frame of the real stream; time runs across, bands run up. Use the left and right arrow keys to step through the frames; the chart above shows the allocation of the current frame. Press Esc to go back to the bitrate slider.'));
  }
  function heatAria() {
    if (!frames) return;
    if (st.frame >= 0) {
      const f = frames[st.frame];
      heat.canvas.setAttribute('aria-valuenow', String(st.frame));
      heat.canvas.setAttribute('aria-valuetext', OX.t(`第 ${f.index} 帧，${f.bytes} 字节，${f.transient ? '瞬态帧' : '非瞬态帧'}`, `Frame ${f.index}, ${f.bytes} bytes, ${f.transient ? 'transient frame' : 'non-transient frame'}`));
    } else {
      heat.canvas.removeAttribute('aria-valuenow');
      heat.canvas.setAttribute('aria-valuetext', OX.t('未选帧，上图跟随码率滑块', 'No frame selected; the chart above follows the bitrate slider'));
    }
  }

  // the band chart: arrow keys walk the bands, like the pointer does
  main.canvas.tabIndex = 0;
  main.canvas.setAttribute('role', 'slider');
  main.canvas.setAttribute('aria-valuemin', '0');
  main.canvas.setAttribute('aria-valuemax', String(NB - 1));
  main.canvas.setAttribute('aria-label', OX.t('CELT 21 个频带分到的比特：浅色是精细能量，深色是 PVQ 形状。按左右方向键逐个查看频带。', 'Bits given to the 21 CELT bands: light is fine energy, dark is the PVQ shape. Use the left and right arrow keys to go through the bands one by one.'));
  function mainAria() {
    const j = st.hoverBand;
    if (j >= 0 && res) {
      main.canvas.setAttribute('aria-valuenow', String(j));
      main.canvas.setAttribute('aria-valuetext', OX.t(`频带 ${j}，${bandHz(j)}，精细能量 ${res.ebits[j]} 比特，PVQ ${fmtBits(res.pulses[j])} 比特${res.fold[j] ? '，靠折叠填充' : ''}`,
        `Band ${j}, ${bandHz(j)}, fine energy ${bitsEn(res.ebits[j])}, PVQ ${bitsEn(fmtBits(res.pulses[j]))}${res.fold[j] ? ', filled by folding' : ''}`));
    } else {
      main.canvas.removeAttribute('aria-valuenow');
      main.canvas.setAttribute('aria-valuetext', OX.t('未选频带，按左右方向键选择', 'No band selected; use the left and right arrow keys to choose one'));
    }
  }
  function setBand(j) { st.hoverBand = j; renderDetail(); main.redraw(); }
  main.canvas.addEventListener('keydown', (ev) => {
    const k = ev.key, j = st.hoverBand;
    let n = null;
    if (k === 'ArrowRight' || k === 'ArrowUp') n = j < 0 ? 0 : Math.min(NB - 1, j + 1);
    else if (k === 'ArrowLeft' || k === 'ArrowDown') n = j < 0 ? 0 : Math.max(0, j - 1);
    else if (k === 'Home') n = 0;
    else if (k === 'End') n = NB - 1;
    else if (k === 'Escape' && j >= 0) n = -1;
    else return;
    ev.preventDefault();
    setBand(n);
  });
  main.canvas.addEventListener('blur', () => { if (st.hoverBand >= 0) setBand(-1); });

  update();
});
