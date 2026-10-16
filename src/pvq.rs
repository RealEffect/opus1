use crate::mathops::mac;
use crate::range_coder::RangeCoder;
use core::mem::MaybeUninit;

#[cfg(not(feature = "std"))]
use crate::compat::Math;

pub const CELT_PVQ_U_DATA: [u32; 1272] = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97,
    99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135,
    137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173,
    175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211,
    213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249,
    251, 253, 255, 257, 259, 261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285, 287,
    289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325,
    327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351, 13, 25, 41, 61, 85, 113, 145,
    181, 221, 265, 313, 365, 421, 481, 545, 613, 685, 761, 841, 925, 1013, 1105, 1201, 1301, 1405,
    1513, 1625, 1741, 1861, 1985, 2113, 2245, 2381, 2521, 2665, 2813, 2965, 3121, 3281, 3445, 3613,
    3785, 3961, 4141, 4325, 4513, 4705, 4901, 5101, 5305, 5513, 5725, 5941, 6161, 6385, 6613, 6845,
    7081, 7321, 7565, 7813, 8065, 8321, 8581, 8845, 9113, 9385, 9661, 9941, 10225, 10513, 10805,
    11101, 11401, 11705, 12013, 12325, 12641, 12961, 13285, 13613, 13945, 14281, 14621, 14965,
    15313, 15665, 16021, 16381, 16745, 17113, 17485, 17861, 18241, 18625, 19013, 19405, 19801,
    20201, 20605, 21013, 21425, 21841, 22261, 22685, 23113, 23545, 23981, 24421, 24865, 25313,
    25765, 26221, 26681, 27145, 27613, 28085, 28561, 29041, 29525, 30013, 30505, 31001, 31501,
    32005, 32513, 33025, 33541, 34061, 34585, 35113, 35645, 36181, 36721, 37265, 37813, 38365,
    38921, 39481, 40045, 40613, 41185, 41761, 42341, 42925, 43513, 44105, 44701, 45301, 45905,
    46513, 47125, 47741, 48361, 48985, 49613, 50245, 50881, 51521, 52165, 52813, 53465, 54121,
    54781, 55445, 56113, 56785, 57461, 58141, 58825, 59513, 60205, 60901, 61601, 63, 129, 231, 377,
    575, 833, 1159, 1561, 2047, 2625, 3303, 4089, 4991, 6017, 7175, 8473, 9919, 11521, 13287,
    15225, 17343, 19649, 22151, 24857, 27775, 30913, 34279, 37881, 41727, 45825, 50183, 54809,
    59711, 64897, 70375, 76153, 82239, 88641, 95367, 102425, 109823, 117569, 125671, 134137,
    142975, 152193, 161799, 171801, 182207, 193025, 204263, 215929, 228031, 240577, 253575, 267033,
    280959, 295361, 310247, 325625, 341503, 357889, 374791, 392217, 410175, 428673, 447719, 467321,
    487487, 508225, 529543, 551449, 573951, 597057, 620775, 645113, 670079, 695681, 721927, 748825,
    776383, 804609, 833511, 863097, 893375, 924353, 956039, 988441, 1021567, 1055425, 1090023,
    1125369, 1161471, 1198337, 1235975, 1274393, 1313599, 1353601, 1394407, 1436025, 1478463,
    1521729, 1565831, 1610777, 1656575, 1703233, 1750759, 1799161, 1848447, 1898625, 1949703,
    2001689, 2054591, 2108417, 2163175, 2218873, 2275519, 2333121, 2391687, 2451225, 2511743,
    2573249, 2635751, 2699257, 2763775, 2829313, 2895879, 2963481, 3032127, 3101825, 3172583,
    3244409, 3317311, 3391297, 3466375, 3542553, 3619839, 3698241, 3777767, 3858425, 3940223,
    4023169, 4107271, 4192537, 4278975, 4366593, 4455399, 4545401, 4636607, 4729025, 4822663,
    4917529, 5013631, 5110977, 5209575, 5309433, 5410559, 5512961, 5616647, 5721625, 5827903,
    5935489, 6044391, 6154617, 6266175, 6379073, 6493319, 6608921, 6725887, 6844225, 6963943,
    7085049, 7207551, 321, 681, 1289, 2241, 3649, 5641, 8361, 11969, 16641, 22569, 29961, 39041,
    50049, 63241, 78889, 97281, 118721, 143529, 172041, 204609, 241601, 283401, 330409, 383041,
    441729, 506921, 579081, 658689, 746241, 842249, 947241, 1061761, 1186369, 1321641, 1468169,
    1626561, 1797441, 1981449, 2179241, 2391489, 2618881, 2862121, 3121929, 3399041, 3694209,
    4008201, 4341801, 4695809, 5071041, 5468329, 5888521, 6332481, 6801089, 7295241, 7815849,
    8363841, 8940161, 9545769, 10181641, 10848769, 11548161, 12280841, 13047849, 13850241,
    14689089, 15565481, 16480521, 17435329, 18431041, 19468809, 20549801, 21675201, 22846209,
    24064041, 25329929, 26645121, 28010881, 29428489, 30899241, 32424449, 34005441, 35643561,
    37340169, 39096641, 40914369, 42794761, 44739241, 46749249, 48826241, 50971689, 53187081,
    55473921, 57833729, 60268041, 62778409, 65366401, 68033601, 70781609, 73612041, 76526529,
    79526721, 82614281, 85790889, 89058241, 92418049, 95872041, 99421961, 103069569, 106816641,
    110664969, 114616361, 118672641, 122835649, 127107241, 131489289, 135983681, 140592321,
    145317129, 150160041, 155123009, 160208001, 165417001, 170752009, 176215041, 181808129,
    187533321, 193392681, 199388289, 205522241, 211796649, 218213641, 224775361, 231483969,
    238341641, 245350569, 252512961, 259831041, 267307049, 274943241, 282741889, 290705281,
    298835721, 307135529, 315607041, 324252609, 333074601, 342075401, 351257409, 360623041,
    370174729, 379914921, 389846081, 399970689, 410291241, 420810249, 431530241, 442453761,
    453583369, 464921641, 476471169, 488234561, 500214441, 512413449, 524834241, 537479489,
    550351881, 563454121, 576788929, 590359041, 604167209, 618216201, 632508801, 1683, 3653, 7183,
    13073, 22363, 36365, 56695, 85305, 124515, 177045, 246047, 335137, 448427, 590557, 766727,
    982729, 1244979, 1560549, 1937199, 2383409, 2908411, 3522221, 4235671, 5060441, 6009091,
    7095093, 8332863, 9737793, 11326283, 13115773, 15124775, 17372905, 19880915, 22670725,
    25765455, 29189457, 32968347, 37129037, 41699767, 46710137, 52191139, 58175189, 64696159,
    71789409, 79491819, 87841821, 96879431, 106646281, 117185651, 128542501, 140763503, 153897073,
    167993403, 183104493, 199284183, 216588185, 235074115, 254801525, 275831935, 298228865,
    322057867, 347386557, 374284647, 402823977, 433078547, 465124549, 499040399, 534906769,
    572806619, 612825229, 655050231, 699571641, 746481891, 795875861, 847850911, 902506913,
    959946283, 1020274013, 1083597703, 1150027593, 1219676595, 1292660325, 1369097135, 1449108145,
    1532817275, 1620351277, 1711839767, 1807415257, 1907213187, 2011371957, 2120032959, 8989,
    19825, 40081, 75517, 134245, 227305, 369305, 579125, 880685, 1303777, 1884961, 2668525,
    3707509, 5064793, 6814249, 9041957, 11847485, 15345233, 19665841, 24957661, 31388293, 39146185,
    48442297, 59511829, 72616013, 88043969, 106114625, 127178701, 151620757, 179861305, 212358985,
    249612805, 292164445, 340600625, 395555537, 457713341, 527810725, 606639529, 695049433,
    793950709, 904317037, 1027188385, 1163673953, 1314955181, 1482288821, 1667010073, 1870535785,
    2094367717, 48639, 108545, 224143, 433905, 795455, 1392065, 2340495, 3800305, 5984767, 9173505,
    13726991, 20103025, 28875327, 40754369, 56610575, 77500017, 104692735, 139703809, 184327311,
    240673265, 311207743, 398796225, 506750351, 638878193, 799538175, 993696769, 1226990095,
    1505789553, 1837271615, 2229491905, 265729, 598417, 1256465, 2485825, 4673345, 8405905,
    14546705, 24331777, 39490049, 62390545, 96220561, 145198913, 214828609, 312193553, 446304145,
    628496897, 872893441, 1196924561, 1621925137, 2173806145, 1462563, 3317445, 7059735, 14218905,
    27298155, 50250765, 89129247, 152951073, 254831667, 413442773, 654862247, 1014889769,
    1541911931, 2300409629, 3375210671, 8097453, 18474633, 39753273, 81270333, 158819253,
    298199265, 540279585, 948062325, 1616336765, 45046719, 103274625, 224298231, 464387817,
    921406335, 1759885185, 3248227095, 251595969, 579168825, 1267854873, 2653649025, 1409933619,
];

const CELT_PVQ_U_ROW: [u32; 15] = [
    0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257,
];

/// Largest column (max(n, k)) covered by each row of `CELT_PVQ_U_DATA`. Row `r`
/// holds `U(min, max)` for `max` in `[r, CMAX[r]]`; a lookup with a larger
/// `max` would alias the next row's block, so it must be computed instead.
const CELT_PVQ_U_CMAX: [usize; 15] = [
    176, 176, 176, 176, 176, 176, 96, 54, 37, 28, 24, 19, 18, 16, 14,
];

// The table reads without range checks below rely on these two facts.
const _: () = {
    let mut r = 0;
    while r < CELT_PVQ_U_ROW.len() {
        // The last entry of row `r` is inside the data.
        assert!(CELT_PVQ_U_ROW[r] as usize + CELT_PVQ_U_CMAX[r] < CELT_PVQ_U_DATA.len());
        // A row is never longer than the row before it.
        assert!(r == 0 || CELT_PVQ_U_CMAX[r] <= CELT_PVQ_U_CMAX[r - 1]);
        r += 1;
    }
};

/// True when `U(n, k + 1)` is stored in `CELT_PVQ_U_DATA`.
///
/// A row is never longer than the row before it. So when this entry exists,
/// every `U(a, b)` with `min(a, b) <= min(n, k + 1)` and
/// `max(a, b) <= max(n, k + 1)` exists as well. These are all the entries
/// that `cwrsi`, `icwrs` and `celt_pvq_v` read for `n` dimensions and `k`
/// pulses.
#[inline(always)]
fn pvq_table_covers(n: u32, k: u32) -> bool {
    let k1 = k as u64 + 1;
    let r = (n as u64).min(k1);
    let c = (n as u64).max(k1);
    r < CELT_PVQ_U_ROW.len() as u64 && c <= CELT_PVQ_U_CMAX[r as usize] as u64
}

/// `U(r, c)` for `r <= c`, read from the table without range checks.
///
/// # Safety
/// `r < 15` and `c <= CELT_PVQ_U_CMAX[r]`. `pvq_table_covers` proves this
/// for all reads of one call.
#[inline(always)]
unsafe fn pvq_u_unchecked(r: usize, c: usize) -> u32 {
    debug_assert!(r < CELT_PVQ_U_ROW.len() && r <= c && c <= CELT_PVQ_U_CMAX[r]);
    // SAFETY: `r` is a valid row. The index is at most
    // `CELT_PVQ_U_ROW[r] + CELT_PVQ_U_CMAX[r]`, which the check above
    // `pvq_table_covers` shows to be inside the data.
    unsafe { *CELT_PVQ_U_DATA.get_unchecked(*CELT_PVQ_U_ROW.get_unchecked(r) as usize + c) }
}

/// Maximum number of pulses per band (`CELT_MAX_PULSES` in libopus).
pub const MAX_PVQ_K: usize = 128;
const MAX_PVQ_U: usize = MAX_PVQ_K + 2;
pub const MAX_PVQ_N: usize = 352;

/// `saturating_add(a, b, c)`, clamped to `u32::MAX` so an out-of-domain
/// `(n, k)` can never silently wrap to a plausible-but-wrong codebook size.
#[inline(always)]
fn saturating_add3(a: u32, b: u32, c: u32) -> u32 {
    let s = a as u64 + b as u64 + c as u64;
    s.min(u32::MAX as u64) as u32
}

/// `U(n, k)` (`CELT_PVQ_U` in libopus), read from the table when it holds
/// the value and computed otherwise. Saturates at `u32::MAX`.
#[inline(always)]
pub fn celt_pvq_u(n: u32, k: u32) -> u32 {
    let r = n.min(k) as usize;
    let c = n.max(k) as usize;

    if r >= CELT_PVQ_U_ROW.len() {
        return compute_u(n, k);
    }
    if c > CELT_PVQ_U_CMAX[r] {
        return compute_u(n, k);
    }
    // SAFETY: `r < 15` and `r <= c <= CELT_PVQ_U_CMAX[r]` were checked above.
    unsafe { pvq_u_unchecked(r, c) }
}

fn compute_u(n: u32, k: u32) -> u32 {
    if n == 0 {
        return if k == 0 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    if k > MAX_PVQ_K as u32 {
        return u32::MAX;
    }
    let mut u = [0u32; MAX_PVQ_U];
    u[0] = 0;
    u[1] = 1;
    for ki in 2..=(k + 1) as usize {
        u[ki] = ((ki as u32 * 2) as u64).saturating_sub(1) as u32;
    }
    let mut curr_n = n;
    while curr_n > 2 {
        unext(&mut u[1..], (k + 1) as usize, 1);
        curr_n -= 1;
    }
    u[k as usize]
}

/// `V(n, k) = U(n, k) + U(n, k + 1)` (`CELT_PVQ_V` in libopus), the number
/// of pulse vectors of `n` values and `k` pulses. Saturates at `u32::MAX`.
#[inline(always)]
pub fn celt_pvq_v(n: u32, k: u32) -> u32 {
    celt_pvq_u(n, k).saturating_add(celt_pvq_u(n, k + 1))
}

fn unext(u: &mut [u32], len: usize, mut u0: u32) {
    let mut j = 1;
    while j < len {
        let u1 = saturating_add3(u[j], u[j - 1], u0);
        u[j - 1] = u0;
        u0 = u1;
        j += 1;
    }
    u[j - 1] = u0;
}

/// The index of the pulse vector `y[..n]` among the vectors with `n`
/// dimensions and as many pulses as `y` has (`icwrs` in libopus
/// `celt/cwrs.c`).
///
/// `k` is the expected number of pulses. It selects the fast path only; the
/// result does not depend on it.
#[inline(always)]
pub fn icwrs(n: u32, k: u32, y: &[i32]) -> u32 {
    if n == 1 {
        return if y[0] < 0 { 1 } else { 0 };
    }
    debug_assert!(n >= 2, "icwrs: n must be >= 2");
    if n >= 2 && y.len() >= n as usize && pvq_table_covers(n, k) {
        // SAFETY: the table covers `(n, k)`.
        if let Some(i) = unsafe { icwrs_table(k, &y[..n as usize]) } {
            return i;
        }
    }
    icwrs_checked(n, y)
}

/// `icwrs` with table reads that are not range checked. Returns `None` when
/// `y` has more than `k_max` pulses; the caller then uses `icwrs_checked`.
///
/// # Safety
/// `pvq_table_covers(y.len(), k_max)` must hold and `y` must have at least
/// two entries. Every read is `U(m, k)` or `U(m, k + 1)` with `m <= y.len()`
/// and `k <= k_max`, which `pvq_table_covers` names.
#[inline(always)]
unsafe fn icwrs_table(k_max: u32, y: &[i32]) -> Option<u32> {
    debug_assert!(y.len() >= 2 && pvq_table_covers(y.len() as u32, k_max));
    let (last, head) = y.split_last()?;
    let mut i = (*last < 0) as u32;
    let mut k = last.unsigned_abs();
    if k > k_max {
        return None;
    }
    // `m` counts the dimensions from the current entry to the end.
    for (&yj, m) in head.iter().rev().zip(2usize..) {
        let ku = k as usize;
        // SAFETY: `m <= y.len()` and `k <= k_max`.
        i = i.wrapping_add(unsafe { pvq_u_unchecked(m.min(ku), m.max(ku)) });
        // No overflow: `k <= k_max <= 175` and the magnitude is at most 2^31.
        k += yj.unsigned_abs();
        if k > k_max {
            return None;
        }
        let ku = k as usize + 1;
        // SAFETY: `m <= y.len()` and `k <= k_max`.
        let u = unsafe { pvq_u_unchecked(m.min(ku), m.max(ku)) };
        // The second term counts only for a negative entry.
        i = i.wrapping_add(((yj >> 31) as u32) & u);
    }
    Some(i)
}

/// `icwrs` with a range check on every table lookup. It runs when the vector
/// may need values from outside the table.
#[cold]
#[inline(never)]
fn icwrs_checked(n: u32, y: &[i32]) -> u32 {
    let mut j = (n - 1) as usize;

    let mut i: u32 = if y[j] < 0 { 1 } else { 0 };
    let mut k = y[j].unsigned_abs();

    while j > 0 {
        j -= 1;
        let yj = y[j];
        let m = n - j as u32;
        i = i.wrapping_add(celt_pvq_u(m, k));
        k += yj.unsigned_abs();

        let sign_mask = yj >> 31;
        let lookup = (sign_mask as u32) & celt_pvq_u(m, k + 1);
        i = i.wrapping_add(lookup);
    }
    i
}

/// Decodes the pulse vector with index `i` among the vectors of `n`
/// dimensions and `k` pulses (`cwrsi` in libopus `celt/cwrs.c`).
#[inline(always)]
pub fn cwrsi(n: u32, k: u32, i: u32, y: &mut [i32]) {
    debug_assert!(k > 0, "cwrsi: k must be > 0");

    if n == 1 {
        let s = -(i as i32);
        y[0] = ((k as i32) + s) ^ s;
        return;
    }
    if n >= 2 && y.len() >= n as usize && pvq_table_covers(n, k) {
        // SAFETY: the table covers `(n, k)`.
        let v = unsafe { pvq_v_unchecked(n, k) };
        if (i as u64) < v {
            // SAFETY: the table covers `(n, k)` and `i` is a valid index.
            unsafe { cwrsi_table(n as usize, k as usize, i, &mut y[..n as usize]) };
            return;
        }
    }
    cwrsi_checked(n, k, i, y);
}

/// `V(n, k) = U(n, k) + U(n, k + 1)`, the number of pulse vectors, without
/// saturation and without range checks.
///
/// # Safety
/// `pvq_table_covers(n, k)` must hold.
#[inline(always)]
unsafe fn pvq_v_unchecked(n: u32, k: u32) -> u64 {
    // SAFETY: both entries are among those named at `pvq_table_covers`.
    unsafe {
        pvq_u_unchecked(n.min(k) as usize, n.max(k) as usize) as u64
            + pvq_u_unchecked(n.min(k + 1) as usize, n.max(k + 1) as usize) as u64
    }
}

/// `cwrsi` as a port of the libopus loop: one table read and one compare per
/// search step.
///
/// # Safety
/// `pvq_table_covers(n, k)` must hold and `i` must be below `V(n, k)`, the
/// number of pulse vectors. Then `k` never drops below 0 and every read is
/// one of the entries named at `pvq_table_covers`.
#[inline(always)]
unsafe fn cwrsi_table(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) {
    use core::hint::select_unpredictable as select;

    debug_assert!(n >= 2 && y.len() == n);
    debug_assert!(pvq_table_covers(n as u32, k as u32));
    let (head, tail) = y.split_at_mut(n - 2);
    // SAFETY: see the function comment. `U(0, n) = 0 <= i` ends every search
    // at `k = 0` at the latest.
    unsafe {
        for yj in head {
            let s;
            let k0 = k;
            let p;
            if k >= n {
                // Many pulses: the values are in row `n`.
                let q = pvq_u_unchecked(n, k + 1);
                // Are the pulses in this dimension negative?
                s = -((i >= q) as i32);
                i -= q & (s as u32);
                // Count how many pulses were placed in this dimension: the
                // search goes down from `k` to the first value that is not
                // above `i`.
                p = 'search: {
                    // With many pulses the number of steps is hard to
                    // predict, and a wrong guess costs more than the search.
                    // So take up to three steps at once without a branch:
                    // the values grow with `k`, so the number of values
                    // above `i` among four neighbours is the number of
                    // steps. All four above `i`: go on four values lower.
                    while k >= n + 3 {
                        let u0 = pvq_u_unchecked(n, k);
                        let u1 = pvq_u_unchecked(n, k - 1);
                        let u2 = pvq_u_unchecked(n, k - 2);
                        let u3 = pvq_u_unchecked(n, k - 3);
                        let (c0, c1, c2, c3) = (u0 > i, u1 > i, u2 > i, u3 > i);
                        let steps = c0 as usize + c1 as usize + c2 as usize + c3 as usize;
                        if steps < 4 {
                            k -= steps;
                            break 'search select(c0, select(c1, select(c2, u3, u2), u1), u0);
                        }
                        k -= 4;
                    }
                    // Fewer than four values are left in row `n`. Here `k`
                    // can be `n - 1`; then `U(n, n)` was above `i` and the
                    // first branch below is taken.
                    if pvq_u_unchecked(n, n) > i {
                        k = n;
                        loop {
                            k -= 1;
                            let p = pvq_u_unchecked(k, n);
                            if p <= i {
                                break p;
                            }
                        }
                    } else {
                        let mut pk = pvq_u_unchecked(n, k);
                        while pk > i {
                            k -= 1;
                            pk = pvq_u_unchecked(n, k);
                        }
                        pk
                    }
                };
            } else {
                // Many dimensions: the values are in column `n`.
                let p0 = pvq_u_unchecked(k, n);
                let q = pvq_u_unchecked(k + 1, n);
                if p0 <= i && i < q {
                    // No pulse in this dimension.
                    s = 0;
                    p = p0;
                } else {
                    s = -((i >= q) as i32);
                    i -= q & (s as u32);
                    p = loop {
                        k -= 1;
                        let p = pvq_u_unchecked(k, n);
                        if p <= i {
                            break p;
                        }
                    };
                }
            }
            i -= p;
            *yj = ((k0 - k) as i32 + s) ^ s;
            n -= 1;
        }
    }
    // n == 2
    let p = 2 * k as u32 + 1;
    let s = -((i >= p) as i32);
    i -= p & (s as u32);
    let k0 = k;
    k = ((i + 1) >> 1) as usize;
    if k != 0 {
        i -= 2 * k as u32 - 1;
    }
    tail[0] = ((k0 - k) as i32 + s) ^ s;
    // n == 1
    let s = -(i as i32);
    tail[1] = (k as i32 + s) ^ s;
}

/// `cwrsi` with a range check on every table lookup. It runs when `(n, k)`
/// is outside the table or `i` is not a valid index.
#[cold]
#[inline(never)]
fn cwrsi_checked(n: u32, k: u32, mut i: u32, y: &mut [i32]) {
    let mut curr_n = n;

    let mut curr_k = k as i32;
    let mut j = 0usize;

    while curr_n > 2 {
        if curr_k >= curr_n as i32 {
            let p_kp1 = celt_pvq_u(curr_n, (curr_k + 1) as u32);
            let s: i32 = if i >= p_kp1 {
                i -= p_kp1;
                -1
            } else {
                0
            };
            let k0 = curr_k;
            let q = celt_pvq_u(curr_n, curr_n);
            let mut p;
            if q > i {
                curr_k = curr_n as i32;
                loop {
                    curr_k -= 1;
                    p = celt_pvq_u(curr_n, curr_k.max(0) as u32);
                    if p <= i || curr_k <= 0 {
                        break;
                    }
                }
            } else {
                p = celt_pvq_u(curr_n, curr_k as u32);
                while p > i && curr_k > 0 {
                    curr_k -= 1;
                    p = celt_pvq_u(curr_n, curr_k as u32);
                }
            }
            i -= p;
            let val = k0 - curr_k;
            y[j] = (val + s) ^ s;
        } else {
            let p_k = celt_pvq_u(curr_k as u32, curr_n);
            let p_kp1 = celt_pvq_u((curr_k + 1) as u32, curr_n);
            if p_k <= i && i < p_kp1 {
                i -= p_k;
                y[j] = 0;
                j += 1;
                curr_n -= 1;
                continue;
            }
            let s: i32 = if i >= p_kp1 {
                i -= p_kp1;
                -1
            } else {
                0
            };
            let k0 = curr_k;

            let mut p;
            loop {
                curr_k -= 1;
                p = celt_pvq_u(curr_k.max(0) as u32, curr_n);
                if p <= i || curr_k <= 0 {
                    break;
                }
            }
            i -= p;
            let val = k0 - curr_k;
            y[j] = (val + s) ^ s;
        }
        j += 1;
        curr_n -= 1;
    }

    let p2 = (2u32).wrapping_mul(curr_k as u32).wrapping_add(1);
    let s2: i32 = if i >= p2 {
        i -= p2;
        -1
    } else {
        0
    };
    let k0 = curr_k;
    curr_k = ((i + 1) >> 1) as i32;
    if curr_k > 0 {
        i -= 2 * curr_k as u32 - 1;
    }
    y[j] = ((k0 - curr_k) + s2) ^ s2;
    j += 1;

    let s1 = -(i as i32);
    y[j] = (curr_k + s1) ^ s1;
}

#[inline(always)]
pub fn encode_pulses(y: &[i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        return;
    }
    if n >= 2 && y.len() >= n as usize && pvq_table_covers(n, k) {
        // SAFETY: the table covers `(n, k)`.
        if let Some(fl) = unsafe { icwrs_table(k, &y[..n as usize]) } {
            // The same value as `celt_pvq_v(n, k)`.
            // SAFETY: the table covers `(n, k)`.
            let ft = unsafe { pvq_v_unchecked(n, k) }.min(u32::MAX as u64) as u32;
            debug_assert!(fl < ft, "encode_pulses: fl={fl} >= ft={ft}, n={n}, k={k}");
            rc.enc_uint(fl, ft);
            return;
        }
    }
    let fl = icwrs(n, k, y);
    let ft = celt_pvq_v(n, k);
    debug_assert!(fl < ft, "encode_pulses: fl={fl} >= ft={ft}, n={n}, k={k}");
    rc.enc_uint(fl, ft);
}

#[inline(always)]
pub fn decode_pulses(y: &mut [i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        for i in 0..n as usize {
            y[i] = 0;
        }
        return;
    }
    if n >= 2 && y.len() >= n as usize && pvq_table_covers(n, k) {
        // The same value as `celt_pvq_v(n, k)`. It is at least 2 here.
        // SAFETY: the table covers `(n, k)`.
        let ft = unsafe { pvq_v_unchecked(n, k) }.min(u32::MAX as u64) as u32;
        let fl = rc.dec_uint(ft).min(ft - 1);
        // SAFETY: the table covers `(n, k)` and `fl < ft <= V(n, k)`.
        unsafe { cwrsi_table(n as usize, k as usize, fl, &mut y[..n as usize]) };
        return;
    }
    let ft = celt_pvq_v(n, k);
    let fl = rc.dec_uint(ft).min(ft.saturating_sub(1));
    cwrsi(n, k, fl, y);
}

/// Pulse search on x86-64 for bands of four values.
#[cfg(target_arch = "x86_64")]
#[inline]
fn pvq_search_n4(x: &[f32], y: &mut [i32], k: i32) {
    // The x86 code below loads `x[..4]` and stores `y[..4]` through
    // pointers, so the lengths are checked in release builds too.
    assert!(x.len() >= 4 && y.len() >= 4);

    if k == 0 {
        y[0] = 0;
        y[1] = 0;
        y[2] = 0;
        y[3] = 0;
        return;
    }

    // SAFETY: SSE2 is part of the baseline of x86_64. The one load reads
    // `x[..4]` and the one store writes `y[..4]`; both slices hold at least
    // four values (checked at entry).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
        let vx = _mm_loadu_ps(x.as_ptr());
        let vabs = _mm_and_ps(vx, sign_mask);

        let vzero_f = _mm_setzero_ps();

        let vneg_mask = _mm_cmplt_ps(vx, vzero_f);

        let vsigns = _mm_and_si128(_mm_castps_si128(vneg_mask), _mm_set1_epi32(1));

        let vabs_x = vabs;
        let mut vy2f = _mm_setzero_ps();
        let mut vy = _mm_setzero_si128();
        let mut xy = 0.0f32;
        let mut yy = 0.0f32;

        let vtwo = _mm_set1_ps(2.0);

        let vone_i = _mm_set1_epi32(1);

        let mut pulses_left = k;

        // Pre-search by projecting on the pyramid (C vq_sse2.c: K > N>>1 branch).
        if k > 2 {
            // sum of |x|
            let mut t = _mm_add_ps(vabs, _mm_shuffle_ps(vabs, vabs, 0b01001110));
            t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 0b10110001));
            let sum = _mm_cvtss_f32(t);
            // C: if X too small, just replace it with a pulse at 0
            let sum = if !(sum > 1e-15 && sum < 64.0) {
                1.0
            } else {
                sum
            };
            // C uses the _mm_rcp_ps approximation (not exact division)
            let vrcp = _mm_mul_ps(_mm_set1_ps(k as f32 + 0.8), _mm_rcp_ps(_mm_set1_ps(sum)));
            let vyi = _mm_cvttps_epi32(_mm_mul_ps(vabs_x, vrcp));
            vy = vyi;
            let vyi_f = _mm_cvtepi32_ps(vyi);
            let vy2 = _mm_add_ps(vyi_f, vyi_f);
            vy2f = vy2;
            // xy = dot(x, iy), yy = dot(iy, iy)
            let vxy4 = _mm_mul_ps(vabs_x, vyi_f);
            let vyy4 = _mm_mul_ps(vyi_f, vyi_f);
            let mut t = _mm_add_ps(vxy4, _mm_shuffle_ps(vxy4, vxy4, 0b01001110));
            t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 0b10110001));
            xy = _mm_cvtss_f32(t);
            let mut t = _mm_add_ps(vyy4, _mm_shuffle_ps(vyy4, vyy4, 0b01001110));
            t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 0b10110001));
            yy = _mm_cvtss_f32(t);
            // pulses_left -= sum(iy)
            let mut ps = _mm_add_epi32(vyi, _mm_shuffle_epi32(vyi, 0b01001110));
            ps = _mm_add_epi32(ps, _mm_shuffle_epi32(ps, 0b10110001));
            pulses_left -= _mm_cvtsi128_si32(ps);
        }

        for _ in 0..pulses_left {
            let vxy = _mm_set1_ps(xy);
            let vrxy = _mm_add_ps(vabs_x, vxy);
            let vyy1 = _mm_add_ps(vy2f, _mm_set1_ps(yy + 1.0));

            let vscore = _mm_mul_ps(vrxy, _mm_rsqrt_ps(vyy1));

            let s0 = _mm_cvtss_f32(vscore);
            let s1 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b01_01_01_01));
            let s2 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b10_10_10_10));
            let s3 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b11_11_11_11));
            let mut best_score = s0;
            let mut best_i: u32 = 0;
            // C SSE2 keeps the HIGHEST index on ties (pos = max of matching lanes).
            if s1 >= best_score {
                best_score = s1;
                best_i = 1;
            }
            if s2 >= best_score {
                best_score = s2;
                best_i = 2;
            }
            if s3 >= best_score {
                best_i = 3;
            }
            let _ = best_score;

            let vbest = _mm_set1_epi32(best_i as i32);
            let vlane = _mm_setr_epi32(0, 1, 2, 3);
            let vmask = _mm_castsi128_ps(_mm_cmpeq_epi32(vlane, vbest));

            let vpick_ax = _mm_and_ps(vabs_x, vmask);

            let vpick_ax_hi = _mm_movehl_ps(vpick_ax, vpick_ax);
            let vpick_ax2 = _mm_add_ps(vpick_ax, vpick_ax_hi);
            let vpick_ax3 = _mm_add_ss(vpick_ax2, _mm_shuffle_ps(vpick_ax2, vpick_ax2, 1));
            xy += _mm_cvtss_f32(vpick_ax3);

            let vpick_ryy = _mm_and_ps(vyy1, vmask);
            let vpick_ryy_hi = _mm_movehl_ps(vpick_ryy, vpick_ryy);
            let vpick_ryy2 = _mm_add_ps(vpick_ryy, vpick_ryy_hi);
            let vpick_ryy3 = _mm_add_ss(vpick_ryy2, _mm_shuffle_ps(vpick_ryy2, vpick_ryy2, 1));
            yy = _mm_cvtss_f32(vpick_ryy3);

            let vadd2 = _mm_and_ps(vtwo, vmask);
            vy2f = _mm_add_ps(vy2f, vadd2);

            let vadd1 = _mm_and_si128(vone_i, _mm_castps_si128(vmask));
            vy = _mm_add_epi32(vy, vadd1);
        }

        let vneg_s = _mm_sub_epi32(_mm_setzero_si128(), vsigns);
        let vy_xor = _mm_xor_si128(vy, vneg_s);
        let vy_out = _mm_add_epi32(vy_xor, vsigns);
        _mm_storeu_si128(y.as_mut_ptr() as *mut __m128i, vy_out);
    }
}

#[inline(always)]
pub fn pvq_search(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    // `n` must fit the scratch buffers, and `x` and `y` must hold `n` values.
    // Checked in release builds too: the vector kernels read and write
    // through pointers.
    assert!(
        n <= MAX_PVQ_N && x.len() >= n && y.len() >= n,
        "pvq_search: n={} exceeds MAX_PVQ_N={} or slices too short",
        n,
        MAX_PVQ_N
    );
    // Without a dimension there is nowhere to put a pulse.
    if n == 0 {
        return;
    }
    // libopus runs `op_pvq_search_c` on every target but x86, and so does
    // opus1. The kernels below are for x86-64, where libopus runs its own
    // SSE2 search.
    #[cfg(not(target_arch = "x86_64"))]
    {
        pvq_search_libopus(x, y, k, n);
    }
    #[cfg(target_arch = "x86_64")]
    pvq_search_x86(x, y, k, n);
}

/// The pulse search on x86-64.
#[cfg(target_arch = "x86_64")]
#[inline]
fn pvq_search_x86(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    if k == 1 {
        let mut best_i = 0;
        let mut best_abs = x[0].abs();
        for i in 1..n {
            let abs_xi = x[i].abs();
            if abs_xi > best_abs {
                best_abs = abs_xi;
                best_i = i;
            }
        }
        for j in 0..n {
            y[j] = 0;
        }
        let sign: i32 = if x[best_i] >= 0.0 { 1 } else { -1 };
        y[best_i] = sign;
        return;
    }

    if n == 2 {
        // libopus has no special case for two values: the projection rounds
        // down and the search places the pulses that are left (celt/vq.c,
        // op_pvq_search_c).
        pvq_search_libopus(x, y, k, 2);
        return;
    }

    if n == 4 {
        pvq_search_n4(x, y, k);
        return;
    }

    if n >= 32 {
        pvq_search_fast_select(x, y, k, n);
        return;
    }

    if k > 4 && crate::compat::x86_has_avx2_fma() {
        // SAFETY: `n < 32` here (larger bands returned above), `x` and `y`
        // hold at least `n` values (checked at entry), and the CPU has
        // AVX2 and FMA.
        unsafe {
            pvq_search_avx2(x, y, k, n);
        }
        return;
    }

    pvq_search_scalar(x, y, k, n);
}

/// Sum of the magnitudes in `abs_x`, for the projection step of
/// `pvq_search_fast_select`.
///
/// The fast search runs on x86-64 only. The unit tests run it on every
/// target, so it is compiled for them too.
///
/// The order of the additions is part of the result, and each target keeps
/// the order it has always had:
/// - aarch64: four running sums that take the values 4m, 4m+1, 4m+2 and
///   4m+3 of the largest multiple of four values. They are added as
///   (s0 + s1) + (s2 + s3), and the last values are added one by one. The
///   compiler turns the first loop into vector additions.
/// - every other target: one value after the other.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
fn pvq_magnitude_sum(abs_x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let (quads, tail) = abs_x.as_chunks::<4>();
        let mut lanes = [0.0f32; 4];
        for q in quads {
            for l in 0..4 {
                lanes[l] += q[l];
            }
        }
        let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for &v in tail {
            sum += v;
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for &v in abs_x {
            sum += v;
        }
        sum
    }
}

/// Position of the next pulse in `pvq_search_fast_select` while it places
/// one pulse at a time: the exact comparison of libopus, see
/// `pvq_best_position_scalar` for the meaning of the arguments.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
fn pvq_fast_select_position(abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32) -> usize {
    pvq_best_position::<true>(abs_x, y2f, xy, yy)
}

/// Pulse search for bands of 32 and more values: finds `y[..n]` with `k`
/// pulses that is close to `x[..n]`, and returns the energy of `y`.
///
/// Panics when `x` or `y` holds fewer than `n` values or `n` exceeds
/// `MAX_PVQ_N`.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline]
pub fn pvq_search_fast_select(x: &[f32], y: &mut [i32], k: i32, n: usize) -> f32 {
    pvq_search_fast_select_with(x, y, k, n, pvq_fast_select_position)
}

/// `pvq_search_fast_select` with the function that picks the position of
/// one pulse as an argument, so that the tests can run the portable choice
/// on every target.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
fn pvq_search_fast_select_with(
    x: &[f32],
    y: &mut [i32],
    k: i32,
    n: usize,
    position: impl Fn(&[f32], &[f32], f32, f32) -> usize,
) -> f32 {
    // The scratch arrays below hold MAX_PVQ_N values, so the sizes are
    // checked here, in release builds too. `pvq_search` has checked them
    // already, and the compiler then drops this check.
    assert!(
        n <= MAX_PVQ_N && x.len() >= n && y.len() >= n,
        "pvq_search_fast_select: n={n} exceeds MAX_PVQ_N or slices too short"
    );
    let x = &x[..n];
    let y = &mut y[..n];

    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y.fill(0);

    // Without a dimension there is nowhere to put a pulse. The code below
    // relies on `n >= 1`: it always adds a pulse at some position.
    if k <= 0 || n == 0 {
        return 0.0;
    }

    // The magnitudes of `x`. The compiler turns this loop into vector code.
    let mut abs_x_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    for (a, &xi) in abs_x_mu.iter_mut().zip(x) {
        a.write(xi.abs());
    }
    // SAFETY: the loop above wrote the first `n` values: `x` holds `n`
    // values and `n <= MAX_PVQ_N` (checked at entry).
    let abs_x = unsafe { core::slice::from_raw_parts(abs_x_mu.as_ptr() as *const f32, n) };

    // The projection: each position first gets its share of the pulses,
    // rounded down. It needs more pulses than half the band size. A band of
    // 32 or more values never has that many in the codec (its index would
    // not fit 32 bits), so only direct callers get here, and only this step
    // needs the sum of the magnitudes.
    let mut projected = false;
    if k > (n >> 1) as i32 {
        let sum = pvq_magnitude_sum(abs_x);
        if sum > 1e-15 {
            projected = true;
            let rcp = (k as f32 + 0.8) / sum;
            for (yi_out, &ax) in y.iter_mut().zip(abs_x) {
                let yi = (ax * rcp) as i32;
                *yi_out = yi;
                let yf = yi as f32;
                yy += yf * yf;
                xy += yf * ax;
                k -= yi;
            }

            if k > n as i32 + 3 {
                let tmp = k as f32;
                yy += tmp * tmp + tmp * y[0] as f32;
                y[0] += k;
                k = 0;
            }
        }
    }

    // Twice the number of pulses at each position so far. With `yy` it gives
    // the energy of `y` after one more pulse at that position.
    let mut y2f_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    if projected {
        for (d, &yi) in y2f_mu.iter_mut().zip(y.iter()) {
            d.write(2.0 * yi as f32);
        }
    } else {
        // No pulse yet.
        for d in y2f_mu[..n].iter_mut() {
            d.write(0.0);
        }
    }
    // SAFETY: either loop above wrote the first `n` values (`y` holds `n`).
    let y2f = unsafe { core::slice::from_raw_parts_mut(y2f_mu.as_mut_ptr() as *mut f32, n) };

    const BATCH_SIZE: i32 = 4;

    if k < BATCH_SIZE * 2 || n < 16 {
        // One pulse at a time.
        while k > 0 {
            yy += 1.0;
            let best_id = position(abs_x, y2f, xy, yy);
            xy += abs_x[best_id];
            yy += y2f[best_id];
            y2f[best_id] += 2.0;
            y[best_id] += 1;
            k -= 1;
        }
    } else {
        // Up to four pulses per pass: one each at the positions with the
        // best scores. `n >= 16` here, so position `batch` exists.
        let mut scores_mu = [MaybeUninit::<(f32, usize)>::uninit(); MAX_PVQ_N];
        while k > 0 {
            let batch = BATCH_SIZE.min(k) as usize;

            for (i, (s, (&ax, &ay))) in scores_mu.iter_mut().zip(abs_x.iter().zip(y2f.iter())).enumerate() {
                let rxy = xy + ax;
                let ryy = yy + ay + 1.0;
                let score = rxy * rxy / ryy;
                s.write((score, i));
            }
            // SAFETY: the loop above wrote the first `n` entries
            // (`abs_x` and `y2f` hold `n` values each).
            let scores = unsafe {
                core::slice::from_raw_parts_mut(scores_mu.as_mut_ptr() as *mut (f32, usize), n)
            };

            scores.select_nth_unstable_by(batch, |a, b| {
                if a.0 > b.0 {
                    core::cmp::Ordering::Less
                } else if a.0 < b.0 {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Equal
                }
            });

            for &(_, idx) in &scores[..batch] {
                xy += abs_x[idx];
                yy += y2f[idx] + 1.0;
                y2f[idx] += 2.0;
                y[idx] += 1;
            }

            k -= batch as i32;
        }
    }

    // A pulse gets the sign of its coefficient. Written without a branch,
    // so that the compiler makes vector code of it: `m` is all ones for a
    // negative coefficient, and `(y ^ m) - m` is then `-y`.
    for (yi, &xi) in y.iter_mut().zip(x) {
        let m = -((xi < 0.0) as i32);
        *yi = (*yi ^ m).wrapping_sub(m);
    }

    yy
}

/// Marks the code path that calls it as rarely taken. The compiler then
/// keeps the test in front of it as a branch instead of a conditional select.
#[cold]
#[inline]
fn rarely_taken() {}

/// One step of the pulse search: the position where the next pulse goes.
///
/// Position `i` has the numerator `xy + abs_x[i]` (squared when `SQUARE` is
/// set) and the denominator `yy + y2f[i]`. The positions are tried in order.
/// A position replaces the best one so far when its ratio is larger, tested
/// as `best_den * num > den * best_num`. `abs_x` must not be empty and `y2f`
/// must be at least as long.
///
/// This form is portable. It is the reference for `pvq_best_position_neon`.
/// A new best position is rare, so the test is kept as a branch, as in
/// libopus (`opus_unlikely` in `op_pvq_search_c`). As conditional selects
/// the loop would wait at every position for the one before it.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn pvq_best_position_scalar<const SQUARE: bool>(
    abs_x: &[f32],
    y2f: &[f32],
    xy: f32,
    yy: f32,
) -> usize {
    let y2f = &y2f[..abs_x.len()];
    let rxy = xy + abs_x[0];
    let mut best_id = 0;
    let mut best_num = if SQUARE { rxy * rxy } else { rxy };
    let mut best_den = yy + y2f[0];
    for (i, (&ax, &ay)) in abs_x.iter().zip(y2f.iter()).enumerate().skip(1) {
        let rxy = xy + ax;
        let num = if SQUARE { rxy * rxy } else { rxy };
        let den = yy + ay;
        if best_den * num > den * best_num {
            rarely_taken();
            best_num = num;
            best_den = den;
            best_id = i;
        }
    }
    best_id
}

/// `pvq_best_position_scalar` for aarch64, four positions at a time.
///
/// The four positions of a group are tested against the best position known
/// when the group starts. If none of them is better, the scalar loop would
/// not have changed the best position either, and the group is done. If one
/// is better, the group is redone in order with the scalar test. So the
/// result is always the one of the scalar loop.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn pvq_best_position_neon<const SQUARE: bool>(
    abs_x: &[f32],
    y2f: &[f32],
    xy: f32,
    yy: f32,
) -> usize {
    use core::arch::aarch64::*;

    let y2f = &y2f[..abs_x.len()];
    let rxy = xy + abs_x[0];
    let mut best_id = 0;
    let mut best_num = if SQUARE { rxy * rxy } else { rxy };
    let mut best_den = yy + y2f[0];

    let vxy = vdupq_n_f32(xy);
    let vyy = vdupq_n_f32(yy);
    let (x4, x_tail) = abs_x.as_chunks::<4>();
    let (y4, y_tail) = y2f.as_chunks::<4>();
    for (g, (xa, ya)) in x4.iter().zip(y4.iter()).enumerate() {
        // SAFETY: `xa` and `ya` are references to four values each.
        let (vx, vy) = unsafe { (vld1q_f32(xa.as_ptr()), vld1q_f32(ya.as_ptr())) };
        let vrxy = vaddq_f32(vxy, vx);
        let vnum = if SQUARE { vmulq_f32(vrxy, vrxy) } else { vrxy };
        let vden = vaddq_f32(vyy, vy);
        let better = vcgtq_f32(vmulq_n_f32(vnum, best_den), vmulq_n_f32(vden, best_num));
        if vmaxvq_u32(better) != 0 {
            rarely_taken();
            // Position 0 is the starting point; tested against itself it is
            // not better, so it needs no special case.
            for l in 0..4 {
                let rxy = xy + xa[l];
                let num = if SQUARE { rxy * rxy } else { rxy };
                let den = yy + ya[l];
                if best_den * num > den * best_num {
                    best_num = num;
                    best_den = den;
                    best_id = 4 * g + l;
                }
            }
        }
    }
    let done = 4 * x4.len();
    for (l, (&ax, &ay)) in x_tail.iter().zip(y_tail.iter()).enumerate() {
        let rxy = xy + ax;
        let num = if SQUARE { rxy * rxy } else { rxy };
        let den = yy + ay;
        if best_den * num > den * best_num {
            rarely_taken();
            best_num = num;
            best_den = den;
            best_id = done + l;
        }
    }
    best_id
}

/// True when the scan of libopus (`pvq_best_position_scalar::<true>`)
/// returns the candidate `m`, as far as this check can tell.
///
/// The scan returns `m` when `m` beats every position before it and no
/// position after it beats `m`, with the comparison of libopus
/// (`best_den * num > den * best_num`, both products rounded). Whichever
/// position is the best one when the scan reaches `m`, `m` beats it, and no
/// later position beats `m`. A false result does not mean that the scan
/// picks another position: a tie within the rounding, or a value that is
/// not a number, also fails the check.
///
/// Every group of four goes through the same instructions, so the loop has
/// no branch that depends on where `m` is.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn pvq_candidate_is_scan_result(abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32, m: usize) -> bool {
    use core::arch::aarch64::*;

    let n = abs_x.len();
    let y2f = &y2f[..n];
    let (x4, x_tail) = abs_x.as_chunks::<4>();
    let (y4, y_tail) = y2f.as_chunks::<4>();
    let vxy = vdupq_n_f32(xy);
    let vyy = vdupq_n_f32(yy);
    let lanes = [0u32, 1, 2, 3];
    // SAFETY: `lanes` holds four values.
    let lane_idx = unsafe { vld1q_u32(lanes.as_ptr()) };
    let four = vdupq_n_u32(4);

    let rm = xy + abs_x[m];
    let num_m = rm * rm;
    let den_m = yy + y2f[m];
    let vm = vdupq_n_u32(m as u32);
    let mut vidx = lane_idx;
    let mut vok = vdupq_n_u32(u32::MAX);
    for (xa, ya) in x4.iter().zip(y4.iter()) {
        // SAFETY: `xa` and `ya` are references to four values each.
        let (vx, vy) = unsafe { (vld1q_f32(xa.as_ptr()), vld1q_f32(ya.as_ptr())) };
        let vr = vaddq_f32(vxy, vx);
        let vnum = vmulq_f32(vr, vr);
        let vden = vaddq_f32(vyy, vy);
        vok = vandq_u32(vok, pvq_check_group(vnum, vden, num_m, den_m, vidx, vm));
        vidx = vaddq_u32(vidx, four);
    }
    vminvq_u32(vok) != 0 && pvq_check_tail(x_tail, y_tail, 4 * x4.len(), xy, yy, num_m, den_m, m)
}

/// The check of `pvq_candidate_is_scan_result` for four positions `vidx`
/// with numerators `vnum` and denominators `vden`, against the candidate
/// `vm` with `num_m` and `den_m`. All lanes are set where the check holds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn pvq_check_group(
    vnum: core::arch::aarch64::float32x4_t,
    vden: core::arch::aarch64::float32x4_t,
    num_m: f32,
    den_m: f32,
    vidx: core::arch::aarch64::uint32x4_t,
    vm: core::arch::aarch64::uint32x4_t,
) -> core::arch::aarch64::uint32x4_t {
    use core::arch::aarch64::*;
    // `m` as the candidate against `j` as the best: den_j * num_m.
    let p = vmulq_n_f32(vden, num_m);
    // `j` as the candidate against `m` as the best: den_m * num_j.
    let q = vmulq_n_f32(vnum, den_m);
    // Before `m`: `m` beats `j` (p > q). After `m`: `j` does not beat `m`
    // (q <= p, false for a NaN). At `m`: nothing to test.
    let ok_before = vandq_u32(vcltq_u32(vidx, vm), vcgtq_f32(p, q));
    let ok_after = vandq_u32(vcgtq_u32(vidx, vm), vcleq_f32(q, p));
    vorrq_u32(vorrq_u32(ok_before, ok_after), vceqq_u32(vidx, vm))
}

/// The check of `pvq_candidate_is_scan_result` for the last positions,
/// fewer than four, from position `done` on.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pvq_check_tail(
    x_tail: &[f32],
    y_tail: &[f32],
    done: usize,
    xy: f32,
    yy: f32,
    num_m: f32,
    den_m: f32,
    m: usize,
) -> bool {
    let mut ok = true;
    for (l, (&ax, &ay)) in x_tail.iter().zip(y_tail.iter()).enumerate() {
        let j = done + l;
        let r = xy + ax;
        let (num, den) = (r * r, yy + ay);
        let (p, q) = (den * num_m, den_m * num);
        ok &= if j < m {
            p > q
        } else if j > m {
            q <= p
        } else {
            true
        };
    }
    ok
}

/// The first pass for one pulse: the first position with the largest
/// quotient `(xy + abs_x[j])^2 / (yy + y2f[j])`, computed with a division.
/// It is only a candidate for `pvq_candidate_is_scan_result`.
///
/// Each lane keeps its largest quotient and the first position that has
/// it. A larger quotient replaces the kept one only when it is strictly
/// larger, so the first of equal quotients stays. A NaN quotient is never
/// larger.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn pvq_candidate(abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32) -> usize {
    use core::arch::aarch64::*;

    let n = abs_x.len();
    let y2f = &y2f[..n];
    let (x4, x_tail) = abs_x.as_chunks::<4>();
    let (y4, y_tail) = y2f.as_chunks::<4>();
    let vxy = vdupq_n_f32(xy);
    let vyy = vdupq_n_f32(yy);
    let lanes = [0u32, 1, 2, 3];
    // SAFETY: `lanes` holds four values.
    let lane_idx = unsafe { vld1q_u32(lanes.as_ptr()) };
    let four = vdupq_n_u32(4);
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vbest = lane_idx;
    let mut vidx = lane_idx;
    for (xa, ya) in x4.iter().zip(y4.iter()) {
        // SAFETY: `xa` and `ya` are references to four values each.
        let (vx, vy) = unsafe { (vld1q_f32(xa.as_ptr()), vld1q_f32(ya.as_ptr())) };
        let vr = vaddq_f32(vxy, vx);
        let vq = vdivq_f32(vmulq_f32(vr, vr), vaddq_f32(vyy, vy));
        let larger = vcgtq_f32(vq, vmax);
        vmax = vbslq_f32(larger, vq, vmax);
        vbest = vbslq_u32(larger, vidx, vbest);
        vidx = vaddq_u32(vidx, four);
    }
    pvq_candidate_finish(vmax, vbest, x_tail, y_tail, 4 * x4.len(), xy, yy)
}

/// The end of the first pass: the first position over all lanes that has
/// the largest quotient, then the last positions (fewer than four), one
/// after the other. `done` is the position of the first of them.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn pvq_candidate_finish(
    vmax: core::arch::aarch64::float32x4_t,
    vbest: core::arch::aarch64::uint32x4_t,
    x_tail: &[f32],
    y_tail: &[f32],
    done: usize,
    xy: f32,
    yy: f32,
) -> usize {
    use core::arch::aarch64::*;
    let mut max = vmaxvq_f32(vmax);
    let top = vceqq_f32(vmax, vdupq_n_f32(max));
    let mut m = vminvq_u32(vbslq_u32(top, vbest, vdupq_n_u32(u32::MAX))) as usize;
    for (l, (&ax, &ay)) in x_tail.iter().zip(y_tail.iter()).enumerate() {
        let r = xy + ax;
        let q = r * r / (yy + ay);
        if q > max {
            max = q;
            m = done + l;
        }
    }
    m
}

/// Places `pulses` pulses one after the other, for bands of 8 to 64 values:
/// the loop of libopus `op_pvq_search_c` after the projection. Returns the
/// new `yy`.
///
/// Each pulse goes to the position the scan of libopus picks. A candidate
/// from the first pass (`pvq_candidate`) is taken when it passes the check
/// of `pvq_candidate_is_scan_result`. Otherwise the exact scan
/// (`pvq_best_position_neon`) picks the position.
///
/// The check of one pulse and the first pass of the next pulse read the
/// same values, so they run in one loop. The first pass assumes that the
/// check passes: it uses `xy`, `yy` and `y2f` after a pulse at the
/// candidate. When the check fails, its result is not used. Compared with
/// two loops per pulse (first pass, then check), this halves the loads and
/// the loop overhead, and its two independent halves keep an in-order core
/// busy while each waits for its multiplies.
///
/// Measured on the pulse searches recorded from 2 s of four harness
/// scenarios (48 kHz CELT, mono and stereo, complexity 9 and 10, 20 ms and
/// 2.5 ms frames), against the search with two loops per pulse: 6 % to
/// 19 % less time on a Cortex-A55, 9 % to 15 % on a Cortex-A76, 4 % to 9 %
/// on an Apple M4. A first pass that multiplies by the reciprocal estimate
/// (`frecpe`) instead of dividing made encoding 0.3 % to 1.0 % faster on
/// the Cortex-A55 and 0.6 % to 1.3 % slower on the Cortex-A76.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
fn pvq_place_pulses(
    abs_x: &[f32],
    y2f: &mut [f32],
    y: &mut [i32],
    mut xy: f32,
    mut yy: f32,
    pulses: i32,
) -> f32 {
    use core::arch::aarch64::*;

    let n = abs_x.len();
    let (y2f, y) = (&mut y2f[..n], &mut y[..n]);
    let lanes = [0u32, 1, 2, 3];
    // SAFETY: `lanes` holds four values.
    let lane_idx = unsafe { vld1q_u32(lanes.as_ptr()) };
    let four = vdupq_n_u32(4);
    let two = vreinterpretq_u32_f32(vdupq_n_f32(2.0));
    let (x4, x_tail) = abs_x.as_chunks::<4>();
    let done = 4 * x4.len();

    // "The squared magnitude term gets added anyway, so we might as well
    // add it outside the loop."
    yy += 1.0;
    let mut m = pvq_candidate(abs_x, y2f, xy, yy);
    for left in (0..pulses).rev() {
        if left == 0 {
            // The last pulse needs the check alone.
            if !pvq_candidate_is_scan_result(abs_x, y2f, xy, yy, m) {
                rarely_taken();
                m = pvq_best_position_neon::<true>(abs_x, y2f, xy, yy);
            }
            yy += y2f[m];
            y2f[m] += 2.0;
            y[m] += 1;
            break;
        }
        // The candidate `m` in the state of this pulse.
        let rm = xy + abs_x[m];
        let (num_m, den_m) = (rm * rm, yy + y2f[m]);
        // The state of the next pulse if `m` passes: the update below, then
        // `yy += 1`.
        let (xy_next, yy_next) = (xy + abs_x[m], (yy + y2f[m]) + 1.0);
        let (vxy, vyy) = (vdupq_n_f32(xy), vdupq_n_f32(yy));
        let (vxy_next, vyy_next) = (vdupq_n_f32(xy_next), vdupq_n_f32(yy_next));
        let vm = vdupq_n_u32(m as u32);
        let (y4, y_tail) = y2f.as_chunks::<4>();
        let mut vok = vdupq_n_u32(u32::MAX);
        let mut vidx = lane_idx;
        let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
        let mut vbest = lane_idx;
        for (xa, ya) in x4.iter().zip(y4.iter()) {
            // SAFETY: `xa` and `ya` are references to four values each.
            let (vx, vy) = unsafe { (vld1q_f32(xa.as_ptr()), vld1q_f32(ya.as_ptr())) };
            // The check of this pulse.
            let vr = vaddq_f32(vxy, vx);
            let (vnum, vden) = (vmulq_f32(vr, vr), vaddq_f32(vyy, vy));
            vok = vandq_u32(vok, pvq_check_group(vnum, vden, num_m, den_m, vidx, vm));
            // The first pass of the next pulse, with two more in `y2f` at
            // `m`.
            let vr = vaddq_f32(vxy_next, vx);
            let at_m = vandq_u32(vceqq_u32(vidx, vm), two);
            let vy_next = vaddq_f32(vy, vreinterpretq_f32_u32(at_m));
            let vq = vdivq_f32(vmulq_f32(vr, vr), vaddq_f32(vyy_next, vy_next));
            let larger = vcgtq_f32(vq, vmax);
            vmax = vbslq_f32(larger, vq, vmax);
            vbest = vbslq_u32(larger, vidx, vbest);
            vidx = vaddq_u32(vidx, four);
        }
        let ok =
            vminvq_u32(vok) != 0 && pvq_check_tail(x_tail, y_tail, done, xy, yy, num_m, den_m, m);
        if !ok {
            rarely_taken();
            m = pvq_best_position_neon::<true>(abs_x, y2f, xy, yy);
        }
        xy += abs_x[m];
        yy += y2f[m];
        y2f[m] += 2.0;
        y[m] += 1;
        yy += 1.0;
        m = if ok {
            // `xy` and `yy` are now `xy_next` and `yy_next`, and `y2f` has
            // the pulse at `m`, as the first pass assumed. Its last
            // positions are done here, with that pulse.
            pvq_candidate_finish(vmax, vbest, x_tail, &y2f[done..], done, xy, yy)
        } else {
            pvq_candidate(abs_x, y2f, xy, yy)
        };
    }
    yy
}

/// The position where the next pulse goes. See `pvq_best_position_scalar`.
#[inline(always)]
fn pvq_best_position<const SQUARE: bool>(abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32) -> usize {
    // SAFETY: NEON is part of the baseline of every aarch64 target.
    // On aarch64 the search places the pulses of bands of 8 to 64 values
    // with `pvq_place_pulses`; the other bands come here.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        pvq_best_position_neon::<SQUARE>(abs_x, y2f, xy, yy)
    }
    #[cfg(not(target_arch = "aarch64"))]
    pvq_best_position_scalar::<SQUARE>(abs_x, y2f, xy, yy)
}

/// The pulse search of libopus: a port of `op_pvq_search_c` (celt/vq.c,
/// float build), line by line, for every band size. Every target but
/// x86-64 runs it (`pvq_search`). On 64-bit ARM libopus runs this C code
/// too, built by clang; on x86-64 it runs its SSE2 search instead.
///
/// It finds the vector `y[..n]` of `k` pulses that is close to `x[..n]`:
/// 1. With more pulses than half the band size, each position first gets
///    its share of the pulses, rounded down (the projection onto the
///    pyramid, vq.c lines 240 to 282).
/// 2. The remaining pulses are placed one at a time, each where it gives
///    the largest `Rxy * Rxy / Ryy` (lines 299 to 360). `Rxy` is the
///    correlation of the pulse vector with the input and `Ryy` its energy.
///    The comparison is exact and the first of equal positions wins, so
///    the kernels can test four positions at a time.
///
/// On aarch64, bands of 8 to 64 values place their pulses with
/// `pvq_place_pulses`, the others with `pvq_best_position`.
///
/// Returns `Ryy`, the energy of `y`, as libopus does.
///
/// Panics when `n` is 0 or above `MAX_PVQ_N`, or when `x` or `y` holds
/// fewer than `n` values.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
fn pvq_search_libopus(x: &[f32], y: &mut [i32], k: i32, n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        pvq_search_libopus_place(x, y, k, n, |abs_x, y2f, y, xy, yy, pulses| {
            if (8..=64).contains(&abs_x.len()) {
                // SAFETY: NEON is part of the baseline of every aarch64
                // target.
                unsafe { pvq_place_pulses(abs_x, y2f, y, xy, yy, pulses) }
            } else {
                place_pulses_one_by_one(abs_x, y2f, y, xy, yy, pulses, pvq_best_position::<true>)
            }
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    pvq_search_libopus_with(x, y, k, n, pvq_best_position::<true>)
}

/// The loop of libopus that places the pulses left after the projection,
/// one at a time, each at `position`. Returns the new `yy`.
#[inline(always)]
fn place_pulses_one_by_one(
    abs_x: &[f32],
    y2f: &mut [f32],
    y: &mut [i32],
    mut xy: f32,
    mut yy: f32,
    pulses: i32,
    position: impl Fn(&[f32], &[f32], f32, f32) -> usize,
) -> f32 {
    for _ in 0..pulses {
        // "The squared magnitude term gets added anyway, so we might as well
        // add it outside the loop."
        yy += 1.0;
        let best_id = position(abs_x, y2f, xy, yy);
        xy += abs_x[best_id];
        yy += y2f[best_id];
        y2f[best_id] += 2.0;
        y[best_id] += 1;
    }
    yy
}

/// `pvq_search_libopus` with the function that picks the position of each
/// pulse as an argument, so that the tests can run every form of it. On
/// x86-64 and aarch64 only the tests use it.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn pvq_search_libopus_with(
    x: &[f32],
    y: &mut [i32],
    k: i32,
    n: usize,
    position: impl Fn(&[f32], &[f32], f32, f32) -> usize,
) -> f32 {
    pvq_search_libopus_place(x, y, k, n, |abs_x, y2f, y, xy, yy, pulses| {
        place_pulses_one_by_one(abs_x, y2f, y, xy, yy, pulses, &position)
    })
}

/// `pvq_search_libopus` with the loop that places the pulses left after
/// the projection as an argument (`place_pulses_one_by_one` or
/// `pvq_place_pulses`). It gets `abs_x`, `y2f`, `y`, `xy`, `yy` and the
/// number of pulses, and returns the new `yy`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline(always)]
fn pvq_search_libopus_place(
    x: &[f32],
    y: &mut [i32],
    k: i32,
    n: usize,
    place: impl Fn(&[f32], &mut [f32], &mut [i32], f32, f32, i32) -> f32,
) -> f32 {
    assert!((1..=MAX_PVQ_N).contains(&n) && x.len() >= n && y.len() >= n);
    let x = &x[..n];
    let y = &mut y[..n];

    // One pulse and two or more values: no projection (1 <= n / 2), and the
    // loop below runs once with `xy = 0` and `yy = 1`. Each position then
    // has `Rxy = |x|`, `Ryy = 1`, and the comparison `1 * Rxy^2 > 1 *
    // best_num` is `|x|^2 > best_num`: the first largest square wins. NaN
    // never wins, as in the comparison of libopus.
    if k == 1 && n >= 2 {
        let mut best_id = 0;
        let mut best = x[0] * x[0];
        for (i, &xi) in x.iter().enumerate().skip(1) {
            let sq = xi * xi;
            if sq > best {
                best = sq;
                best_id = i;
            }
        }
        y.fill(0);
        y[best_id] = if x[best_id] < 0.0 { -1 } else { 1 };
        return 1.0;
    }

    // "Get rid of the sign." The signs are taken from `x` again at the end.
    let mut abs_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    for (a, &xi) in abs_mu.iter_mut().zip(x) {
        a.write(xi.abs());
    }
    // SAFETY: the loop above wrote the first `n` values (`x` holds `n`).
    let abs_x = unsafe { core::slice::from_raw_parts_mut(abs_mu.as_mut_ptr() as *mut f32, n) };
    // Twice the number of pulses at each position (`y` in libopus).
    let mut y2f_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    for d in y2f_mu[..n].iter_mut() {
        d.write(0.0);
    }
    // SAFETY: the loop above wrote the first `n` values.
    let y2f = unsafe { core::slice::from_raw_parts_mut(y2f_mu.as_mut_ptr() as *mut f32, n) };
    y.fill(0);

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k;

    // "Do a pre-search by projecting on the pyramid."
    if k > (n >> 1) as i32 {
        // One value after the other, as the C loop adds them.
        let mut sum = 0.0f32;
        for &a in abs_x.iter() {
            sum += a;
        }
        // "If X is too small, just replace it with a pulse at 0." The test
        // also catches a sum that is not a number, and one of 64 or more
        // ("an approximation of infinity").
        if !(sum > 1e-15 && sum < 64.0) {
            abs_x.fill(0.0);
            abs_x[0] = 1.0;
            sum = 1.0;
        }
        // "Using K+e with e < 1 guarantees we cannot get more than K
        // pulses." C: `MULT16_32_Q16(K+0.8f, celt_rcp(sum))`, the product
        // with the reciprocal, not a division.
        let rcp = (k as f32 + 0.8) * (1.0 / sum);
        for ((yi, y2), &a) in y.iter_mut().zip(y2f.iter_mut()).zip(abs_x.iter()) {
            // C: `floor(rcp*X[j])`. The value is not negative, so the cast
            // rounds down like floor.
            let pulses = (rcp * a) as i32;
            *yi = pulses;
            let yf = pulses as f32;
            // C: `yy = MAC16_16(yy, y[j], y[j])` and
            // `xy = MAC16_16(xy, X[j], y[j])`: fused on aarch64 (see `mac`).
            // The last bit of `xy` decides ties in the search below. `yy`
            // is a whole number either way.
            yy = mac(yy, yf, yf);
            xy = mac(xy, a, yf);
            *y2 = 2.0 * yf;
            pulses_left -= pulses;
        }
    }

    // "This should never happen, but just in case it does (e.g. on silence)
    // we fill the first bin with pulses."
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy = mac(yy, tmp, tmp);
        yy = mac(yy, tmp, y2f[0]);
        y[0] += pulses_left;
        pulses_left = 0;
    }

    if pulses_left > 0 {
        yy = place(abs_x, y2f, y, xy, yy, pulses_left);
    }

    // "Put the original sign back."
    for (yi, &xi) in y.iter_mut().zip(x) {
        let sg = (xi < 0.0) as i32;
        *yi = (*yi ^ -sg) + sg;
    }
    yy
}

/// Pulse search on x86-64 for bands of up to 32 values without AVX2, with
/// the exact comparison of libopus (`op_pvq_search_c`).
///
/// Panics when `x` or `y` holds fewer than `n` values or `n` exceeds 32.
#[cfg(target_arch = "x86_64")]
#[inline]
fn pvq_search_scalar(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    // The scratch arrays hold 32 values and the start-up kernels read
    // `x[..n]` through pointers, so the sizes are checked in release builds
    // too. `pvq_search` passes at most 31 values and has checked the slices;
    // the compiler then drops this check.
    assert!(n <= 32 && x.len() >= n && y.len() >= n);
    let x = &x[..n];
    let y = &mut y[..n];

    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y.fill(0);

    if k <= 0 {
        return;
    }

    if k <= 4 {
        pvq_search_libopus(x, y, k, n);
        return;
    }

    let mut abs_x = [0.0f32; 32];
    let mut y2f = [0.0f32; 32];
    let mut sign_x = [0i32; 32];

    // SAFETY: `x` holds exactly `n` values and `n <= 32` (checked at entry).
    // The AVX2 kernel runs only when the CPU has AVX2.
    let sum = unsafe {
        if crate::compat::x86_has_avx2() {
            pvq_search_scalar_init_avx2(x, n, &mut abs_x, &mut sign_x)
        } else {
            let mut s = 0.0f32;
            for i in 0..n {
                let xi = x[i];
                let abs_xi = xi.abs();
                abs_x[i] = abs_xi;
                s += abs_xi;
                sign_x[i] = (xi < 0.0) as i32;
            }
            s
        }
    };

    let abs_x = &abs_x[..n];
    let y2f = &mut y2f[..n];
    let sign_x = &sign_x[..n];

    if k > (n >> 1) as i32 && sum > 1e-15 {
        let rcp = (k as f32 + 0.8) / sum;

        for ((yi_out, y2), &ax) in y.iter_mut().zip(y2f.iter_mut()).zip(abs_x) {
            let yi = (ax * rcp) as i32;
            *yi_out = yi;
            let yf = yi as f32;
            yy += yf * yf;
            xy += yf * ax;
            *y2 = 2.0 * yf;
            k -= yi;
        }

        if k > n as i32 + 3 {
            let tmp = k as f32;
            yy += tmp * tmp;
            yy += tmp * y[0] as f32;
            y[0] += k;
            y2f[0] = 2.0 * y[0] as f32;
            k = 0;
        }
    }

    while k > 0 {
        yy += 1.0;
        let best_id = pvq_best_position::<true>(abs_x, y2f, xy, yy);
        xy += abs_x[best_id];
        yy += y2f[best_id];
        y2f[best_id] += 2.0;
        y[best_id] += 1;
        k -= 1;
    }

    for (yi, &s) in y.iter_mut().zip(sign_x) {
        *yi = (*yi ^ -s) + s;
    }
}

/// # Safety
///
/// `n` must be at most 32, `x` and `y` must hold at least `n` values each,
/// and the CPU must have AVX2 and FMA.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_search_avx2(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    debug_assert!(n <= 31);
    debug_assert!(k > 4);

    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y[..n].fill(0);

    let mut abs_x = [0.0f32; 32];
    let mut y2f = [0.0f32; 32];
    let mut sign_x = [0i32; 32];

    let sign_mask = _mm256_set1_ps(-0.0f32);
    let vzero_ps = _mm256_setzero_ps();
    let vone_i = _mm256_set1_epi32(1);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let a = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(abs_x.as_mut_ptr().add(i), a);
        acc = _mm256_add_ps(acc, a);

        let neg_mask = _mm256_cmp_ps(v, vzero_ps, _CMP_LT_OS);
        let sign_i = _mm256_and_si256(_mm256_castps_si256(neg_mask), vone_i);
        _mm256_storeu_si256(sign_x.as_mut_ptr().add(i) as *mut __m256i, sign_i);
        i += 8;
    }

    let lo4 = _mm256_castps256_ps128(acc);
    let hi4 = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo4, hi4);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    for j in i..n {
        let xi = x[j];
        let a = xi.abs();
        abs_x[j] = a;
        sum += a;
        sign_x[j] = (xi < 0.0) as i32;
    }

    if k > (n >> 1) as i32 && sum > 1e-15 {
        let rcp = (k as f32 + 0.8) / sum;
        let vrcp = _mm256_set1_ps(rcp);
        let mut vyy_acc = _mm256_setzero_ps();
        let mut vxy_acc = _mm256_setzero_ps();
        let mut vk_acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let vabs = _mm256_loadu_ps(abs_x.as_ptr().add(i));
            let vyi_f32 = _mm256_mul_ps(vabs, vrcp);

            let vyi_i = _mm256_cvttps_epi32(vyi_f32);
            _mm256_storeu_si256(y.as_mut_ptr().add(i) as *mut __m256i, vyi_i);
            let vyi_f = _mm256_cvtepi32_ps(vyi_i);

            vyy_acc = _mm256_fmadd_ps(vyi_f, vyi_f, vyy_acc);
            vxy_acc = _mm256_fmadd_ps(vyi_f, vabs, vxy_acc);
            vk_acc = _mm256_add_ps(vk_acc, vyi_f);

            let vy2f = _mm256_add_ps(vyi_f, vyi_f);
            _mm256_storeu_ps(y2f.as_mut_ptr().add(i), vy2f);
            i += 8;
        }

        let hsum = |v: __m256| -> f32 {
            let lo = _mm256_castps256_ps128(v);
            let hi = _mm256_extractf128_ps(v, 1);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            _mm_cvtss_f32(s1)
        };
        yy += hsum(vyy_acc);
        xy += hsum(vxy_acc);
        k -= hsum(vk_acc) as i32;

        while i < n {
            let yi = (abs_x[i] * rcp) as i32;
            y[i] = yi;
            let yf = yi as f32;
            yy += yf * yf;
            xy += yf * abs_x[i];
            y2f[i] = 2.0 * yf;
            k -= yi;
            i += 1;
        }
        if k > n as i32 + 3 {
            let tmp = k as f32;
            yy += tmp * tmp + tmp * y[0] as f32;
            y[0] += k;
            y2f[0] = 2.0 * y[0] as f32;
            k = 0;
        }
    }

    let abs_x_ptr = abs_x.as_ptr();
    let y2f_ptr = y2f.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    let n8 = n & !7;
    let n_ceil8 = (n + 7) & !7;
    let mut scores = [0.0f32; 32];

    while k > 0 {
        yy += 1.0;
        let vxy = _mm256_set1_ps(xy);
        let vyy = _mm256_set1_ps(yy);

        let mut vmax = _mm256_setzero_ps();
        let mut j = 0;
        while j < n8 {
            let vabs = _mm256_loadu_ps(abs_x_ptr.add(j));
            let vy2f = _mm256_loadu_ps(y2f_ptr.add(j));
            let rxy = _mm256_add_ps(vabs, vxy);
            let ryy = _mm256_add_ps(vy2f, vyy);
            let score = _mm256_mul_ps(rxy, _mm256_rsqrt_ps(ryy));
            _mm256_storeu_ps(scores.as_mut_ptr().add(j), score);
            vmax = _mm256_max_ps(vmax, score);
            j += 8;
        }

        while j < n {
            let rxy = xy + *abs_x_ptr.add(j);
            let ryy = yy + *y2f_ptr.add(j);
            scores[j] = rxy * (1.0 / ryy.sqrt());
            j += 1;
        }

        let global_max = {
            let hi = _mm256_extractf128_ps(vmax, 1);
            let lo = _mm256_castps256_ps128(vmax);
            let m4 = _mm_max_ps(lo, hi);
            let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
            _mm_cvtss_f32(m1)
        };

        let mut gmax = global_max;
        for j in n8..n {
            if scores[j] > gmax {
                gmax = scores[j];
            }
        }

        let vgmax = _mm256_set1_ps(gmax);
        let mut best_id: usize = 0;
        let mut j = 0;
        while j < n_ceil8 {
            let vs = _mm256_loadu_ps(scores.as_ptr().add(j));
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(vs, vgmax, _CMP_EQ_OQ)) as u32;
            if mask != 0 {
                // C SSE2 keeps the HIGHEST index on ties (pos = max of lanes).
                best_id = j + (31 - mask.leading_zeros()) as usize;
            }
            j += 8;
        }

        xy += *abs_x_ptr.add(best_id);
        yy += *y2f_ptr.add(best_id);
        *y2f_ptr.add(best_id) += 2.0;
        *y_ptr.add(best_id) += 1;
        k -= 1;
    }

    for i in 0..n {
        let s = sign_x[i];
        y[i] = (y[i] ^ -s) + s;
    }
}

/// One step of the spreading rotation: the new values of `x1` (`x[i]`) and
/// `x2` (`x[i + stride]`).
///
/// libopus (celt/vq.c, `exp_rotation1`) writes
/// `MAC16_16(MULT16_16(c, x2), s, x1)` and `MAC16_16(MULT16_16(c, x1), ms,
/// x2)`. clang rounds `s * x1` and `ms * x2` and fuses the products with `c`
/// into the adds (the arm64 objects of libopus 1.6.1: `fmul` then `fmadd`).
/// Without fusing the two forms give the same values.
#[inline(always)]
fn rotate_pair(x1: f32, x2: f32, c: f32, s: f32) -> (f32, f32) {
    let ms = -s;
    (mac(ms * x2, c, x1), mac(s * x1, c, x2))
}

/// One pass of the spreading rotation over `x[..len]` (`exp_rotation1` in
/// libopus `celt/vq.c`). A forward sweep and then a backward sweep rotate the
/// pairs `(x[i], x[i + stride])` by the angle given through `c` and `s`.
///
/// `stride` must be at least 1. The work is done by one of three functions,
/// chosen here, once per pass. All three compute each value with
/// `rotate_pair`.
#[inline]
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    debug_assert!(stride >= 1);
    match stride {
        1 => exp_rotation1_carry::<1>(x, len, c, s),
        2 => exp_rotation1_carry::<2>(x, len, c, s),
        3 => exp_rotation1_carry::<3>(x, len, c, s),
        _ => {
            // SAFETY: NEON is part of the baseline of every aarch64 target.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                exp_rotation1_neon(x, len, stride, c, s);
            }
            #[cfg(not(target_arch = "aarch64"))]
            exp_rotation1_scalar(x, len, stride, c, s);
        }
    }
}

/// The plain form of the rotation pass. It is the reference for the other
/// two forms, and it runs for strides of 4 and more on targets other than
/// aarch64. Like the C code it does nothing when `len <= stride`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn exp_rotation1_scalar(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let x = &mut x[..len];
    if len <= stride {
        return;
    }
    for i in 0..(len - stride) {
        (x[i], x[i + stride]) = rotate_pair(x[i], x[i + stride], c, s);
    }
    if len >= 2 * stride {
        for i in (0..(len - 2 * stride)).rev() {
            (x[i], x[i + stride]) = rotate_pair(x[i], x[i + stride], c, s);
        }
    }
}

/// The rotation pass for a small stride `S` (1, 2 or 3), on every target.
///
/// In the plain form each step reads a value that an earlier step stored
/// `S` positions back, so every step waits for a store and a reload. Here
/// those `S` values stay in local variables. Each element is computed with
/// the same operations in the same order as in `exp_rotation1_scalar`, so
/// the results are identical.
#[inline(always)]
fn exp_rotation1_carry<const S: usize>(x: &mut [f32], len: usize, c: f32, s: f32) {
    debug_assert!(S >= 1);
    let x = &mut x[..len];
    if len <= S {
        return;
    }
    // Cells allow reading `x[i + S]` and writing `x[i]` through two slices
    // that overlap, without index checks in the loops.
    let x = core::cell::Cell::from_mut(x).as_slice_of_cells();

    // Forward sweep: step `i` rotates `(x[i], x[i + S])` for `i` in
    // `0..steps`. `carry[j]` holds the current value of the next element
    // that a step with `i % S == j` reads as `x[i]`.
    let steps = len - S;
    let mut carry = [0.0f32; S];
    for (v, cell) in carry.iter_mut().zip(x.iter()) {
        *v = cell.get();
    }
    let (lo, lo_rest) = x[..steps].as_chunks::<S>();
    let (hi, hi_rest) = x[S..].as_chunks::<S>();
    for (lo, hi) in lo.iter().zip(hi.iter()) {
        for j in 0..S {
            let (y1, y2) = rotate_pair(carry[j], hi[j].get(), c, s);
            carry[j] = y2;
            lo[j].set(y1);
        }
    }
    // Fewer than `S` steps are left. Put the carried values back, then finish
    // in the plain way.
    let done = steps - lo_rest.len();
    for (v, cell) in carry.iter().zip(x[done..].iter()) {
        cell.set(*v);
    }
    for (lo, hi) in lo_rest.iter().zip(hi_rest.iter()) {
        let (y1, y2) = rotate_pair(lo.get(), hi.get(), c, s);
        hi.set(y2);
        lo.set(y1);
    }

    // Backward sweep: step `i` rotates `(x[i], x[i + S])` for `i` from
    // `steps - 1` down to 0. `carry[j]` holds the current value of the next
    // element that a step with `i % S == j` reads as `x[i + S]`. The steps
    // of one group of `S` do not depend on each other.
    if len < 2 * S {
        return;
    }
    let steps = len - 2 * S;
    for (v, cell) in carry.iter_mut().zip(x[steps..].iter()) {
        *v = cell.get();
    }
    let (lo_rest, lo) = x[..steps].as_rchunks::<S>();
    let (hi_rest, hi) = x[S..S + steps].as_rchunks::<S>();
    for (lo, hi) in lo.iter().zip(hi.iter()).rev() {
        for j in 0..S {
            let (y1, y2) = rotate_pair(lo[j].get(), carry[j], c, s);
            hi[j].set(y2);
            carry[j] = y1;
        }
    }
    let left = lo_rest.len();
    for (v, cell) in carry.iter().zip(x[left..].iter()) {
        cell.set(*v);
    }
    for (lo, hi) in lo_rest.iter().zip(hi_rest.iter()).rev() {
        let (y1, y2) = rotate_pair(lo.get(), hi.get(), c, s);
        hi.set(y2);
        lo.set(y1);
    }
}

/// The rotation pass for a stride of 4 or more on aarch64, four elements at
/// a time. With such a stride the four steps of a group do not depend on each
/// other. Each lane computes what `rotate_pair` computes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn exp_rotation1_neon(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    use core::arch::aarch64::*;

    debug_assert!(stride >= 4);
    let x = &mut x[..len];
    if len <= stride {
        return;
    }
    let p = x.as_mut_ptr();
    let vc = vdupq_n_f32(c);
    let vs = vdupq_n_f32(s);
    let vms = vdupq_n_f32(-s);

    // Forward sweep.
    let steps = len - stride;
    let mut i = 0;
    while i + 4 <= steps {
        // SAFETY: `i + 4 <= steps = len - stride`, so both `p[i..i + 4]` and
        // `p[i + stride..i + stride + 4]` are inside `x[..len]`.
        unsafe {
            let vx1 = vld1q_f32(p.add(i));
            let vx2 = vld1q_f32(p.add(i + stride));
            // x[i] = c*x1 + (ms*x2) and x[i + stride] = c*x2 + (s*x1),
            // with the products with `c` fused.
            let vy1 = vfmaq_f32(vmulq_f32(vms, vx2), vc, vx1);
            let vy2 = vfmaq_f32(vmulq_f32(vs, vx1), vc, vx2);
            vst1q_f32(p.add(i), vy1);
            vst1q_f32(p.add(i + stride), vy2);
        }
        i += 4;
    }
    for j in i..steps {
        (x[j], x[j + stride]) = rotate_pair(x[j], x[j + stride], c, s);
    }

    // Backward sweep, groups of four from the top down.
    if len < 2 * stride {
        return;
    }
    let mut j = len - 2 * stride;
    while j >= 4 {
        j -= 4;
        // SAFETY: `j + 4 <= len - 2 * stride`, so both `p[j..j + 4]` and
        // `p[j + stride..j + stride + 4]` are inside `x[..len]`.
        unsafe {
            let vx1 = vld1q_f32(p.add(j));
            let vx2 = vld1q_f32(p.add(j + stride));
            let vy2 = vfmaq_f32(vmulq_f32(vs, vx1), vc, vx2);
            let vy1 = vfmaq_f32(vmulq_f32(vms, vx2), vc, vx1);
            vst1q_f32(p.add(j + stride), vy2);
            vst1q_f32(p.add(j), vy1);
        }
    }
    for j in (0..j).rev() {
        (x[j], x[j + stride]) = rotate_pair(x[j], x[j + stride], c, s);
    }
}

/// The product `factor * k` of libopus `exp_rotation`, where `factor`
/// depends on the spreading decision. `None` when the band is not rotated:
/// libopus returns early for `2*K >= len` and for `spread == SPREAD_NONE`.
/// A stride of 0 is rejected too, because it would divide by zero in the
/// block size of `exp_rotation_with`.
#[inline(always)]
fn rotation_factor_k(length: usize, stride: usize, k: i32, spread: i32) -> Option<i32> {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if stride == 0 || 2 * k >= length as i32 || spread <= 0 || spread > 3 {
        return None;
    }
    Some(SPREAD_FACTOR[spread as usize - 1] * k)
}

/// The cosine `c` and the sine `s` of the angle by which `exp_rotation`
/// rotates a band of `length` values. `factor_k` comes from
/// `rotation_factor_k`.
#[inline]
fn rotation_cos_sin(length: usize, factor_k: i32) -> (f32, f32) {
    // C: `gain = celt_div(MULT16_16(Q15_ONE, len), len + factor*K)` and
    // `theta = HALF16(MULT16_16_Q15(gain, gain))`: the sum in integers, the
    // rest in single precision.
    let gain = length as f32 / (length as i32 + factor_k) as f32;
    let theta = 0.5 * (gain * gain);
    // C: `c = celt_cos_norm(theta)` and `s = celt_cos_norm(1 - theta)`, with
    // `celt_cos_norm(x) = (float)cos((.5f*PI)*(x))`: the cosine in double
    // precision. `1 - theta` is computed in single precision.
    const HALF_PI: f64 = 0.5 * core::f64::consts::PI;
    let c = (HALF_PI * theta as f64).cos() as f32;
    let s = (HALF_PI * (1.0f32 - theta) as f64).cos() as f32;
    (c, s)
}

/// Number of entries of a `RotationCache`, a power of two.
const ROTATION_CACHE_SIZE: usize = 64;

/// The results of `rotation_cos_sin` for the band sizes and pulse counts
/// the encoder used last.
///
/// `rotation_cos_sin` calls `cos` of the C library twice. A frame of 48 kHz
/// stereo CELT at complexity 10 quantizes about 54 bands with a rotation,
/// but most of them repeat a size and a pulse count of an earlier band or
/// frame. Over the 6 s clips of the harness, a cache of 64 entries answers
/// 82 % to 99 % of the lookups of the CELT and Hybrid scenarios (89 % for
/// 48 kHz stereo at complexity 10).
///
/// Each entry holds the key it was computed for and the result of
/// `rotation_cos_sin` for that key, so an entry gives the same values as a
/// new computation. The cache never changes what the codec computes.
pub(crate) struct RotationCache {
    /// `(length << 16) | factor_k` of each entry, or 0 for an empty entry.
    /// No key is 0, because `length` is at least 1.
    keys: [u32; ROTATION_CACHE_SIZE],
    /// `[c, s]` of each entry.
    cos_sin: [[f32; 2]; ROTATION_CACHE_SIZE],
}

impl RotationCache {
    pub(crate) const fn new() -> Self {
        RotationCache {
            keys: [0; ROTATION_CACHE_SIZE],
            cos_sin: [[0.0; 2]; ROTATION_CACHE_SIZE],
        }
    }

    /// `rotation_cos_sin(length, factor_k)`, from the cache when it holds
    /// the result.
    #[inline]
    fn cos_sin(&mut self, length: usize, factor_k: i32) -> (f32, f32) {
        // The key is exact only for `length < 2^16` and `0 <= factor_k <
        // 2^16`. The codec's bands have at most MAX_PVQ_N values and fewer
        // than `length / 2` pulses, so `factor_k` is below 15 * MAX_PVQ_N /
        // 2. Other values are computed without the cache.
        if length > MAX_PVQ_N || !(0..1 << 16).contains(&factor_k) {
            return rotation_cos_sin(length, factor_k);
        }
        let key = ((length as u32) << 16) | factor_k as u32;
        // Fibonacci hashing: the top bits of the product pick the entry.
        let slot = (key.wrapping_mul(0x9E37_79B1) >> (32 - ROTATION_CACHE_SIZE.trailing_zeros()))
            as usize
            & (ROTATION_CACHE_SIZE - 1);
        if self.keys[slot] == key {
            let [c, s] = self.cos_sin[slot];
            return (c, s);
        }
        let (c, s) = rotation_cos_sin(length, factor_k);
        self.keys[slot] = key;
        self.cos_sin[slot] = [c, s];
        (c, s)
    }
}

/// libopus `exp_rotation` (celt/vq.c, float build): spreads the pulses of a
/// band before the search (`dir` 1) and gathers them again after it (`dir`
/// -1).
///
/// The decoder calls this form. The encoder calls `alg_quant_with`, which
/// takes the angle from its `RotationCache`. Both rotate with
/// `exp_rotation_with`.
///
/// The angle is computed here as in `rotation_factor_k` and
/// `rotation_cos_sin`, but written out. With the two helpers the compiler
/// allocated the registers of `alg_unquant` differently, and 2.5 ms stereo
/// frames took 1.1 % to 1.6 % longer to decode on a Cortex-A55.
#[inline(always)]
pub fn exp_rotation(x: &mut [f32], length: usize, dir: i32, stride: usize, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    // A stride of 0 would divide by zero in the block size of
    // `exp_rotation_with`.
    if stride == 0 || 2 * k >= length as i32 || spread <= 0 || spread > 3 {
        return;
    }
    let factor = SPREAD_FACTOR[spread as usize - 1];
    // C: `gain = celt_div(MULT16_16(Q15_ONE, len), len + factor*K)` and
    // `theta = HALF16(MULT16_16_Q15(gain, gain))`: the sum in integers, the
    // rest in single precision.
    let gain = length as f32 / (length as i32 + factor * k) as f32;
    let theta = 0.5 * (gain * gain);
    // C: `c = celt_cos_norm(theta)` and `s = celt_cos_norm(1 - theta)`, with
    // `celt_cos_norm(x) = (float)cos((.5f*PI)*(x))`: the cosine in double
    // precision. `1 - theta` is computed in single precision.
    const HALF_PI: f64 = 0.5 * core::f64::consts::PI;
    let c = (HALF_PI * theta as f64).cos() as f32;
    let s = (HALF_PI * (1.0f32 - theta) as f64).cos() as f32;
    exp_rotation_with(x, length, dir, stride, c, s);
}

/// The rotation of `exp_rotation`, with the cosine and the sine of its
/// angle given (`rotation_cos_sin`). `stride` must not be 0.
#[inline(always)]
fn exp_rotation_with(x: &mut [f32], length: usize, dir: i32, stride: usize, c: f32, s: f32) {
    let mut stride2 = 0;
    if length >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < length {
            stride2 += 1;
        }
    }

    let block_len = length / stride;
    for i in 0..stride {
        let x_offset = i * block_len;
        let x_subset = &mut x[x_offset..x_offset + block_len];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(x_subset, block_len, stride2, s, c);
            }
            exp_rotation1(x_subset, block_len, 1, c, s);
        } else {
            exp_rotation1(x_subset, block_len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(x_subset, block_len, stride2, s, -c);
            }
        }
    }
}

/// Portable form of `extract_collapse_mask` for `b >= 2`. It is the
/// reference for the NEON form in the tests.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn extract_collapse_mask_scalar(iy: &[i32], n: usize, b: usize) -> u32 {
    let n0 = n / b;
    let mut collapse_mask = 0u32;
    for i in 0..b {
        let mut tmp = 0i32;
        let base = i * n0;
        for j in 0..n0 {
            tmp |= iy[base + j];
        }
        if tmp != 0 {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

/// NEON form of `extract_collapse_mask` for `b >= 2`: four values per test,
/// and a block is left at its first pulse.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn extract_collapse_mask_neon(iy: &[i32], n: usize, b: usize) -> u32 {
    use core::arch::aarch64::*;

    let n0 = n / b;
    let mut collapse_mask = 0u32;

    for i in 0..b {
        let base = i * n0;
        let slice = &iy[base..base + n0];

        let mut any_nonzero = false;
        let (quads, tail) = slice.as_chunks::<4>();
        for q in quads {
            // SAFETY: NEON is part of the baseline of every aarch64 target.
            // The load reads one `[i32; 4]`.
            let any = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(vld1q_s32(q.as_ptr()))) };
            if any != 0 {
                any_nonzero = true;
                break;
            }
        }

        if !any_nonzero {
            any_nonzero = tail.iter().any(|&v| v != 0);
        }

        if any_nonzero {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

/// Which of the `b` blocks of the pulse vector `iy[..n]` have a pulse: bit
/// `i` is set when one of the `n / b` values of block `i` is not zero. A
/// single block (`b <= 1`) always counts as filled.
#[inline(always)]
pub fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        extract_collapse_mask_neon(iy, n, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        extract_collapse_mask_scalar(iy, n, b)
    }
}

/// # Safety
///
/// `y` and `x` must hold at least `n` values each, and the CPU must have the
/// features this function is compiled for.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn alg_quant_resynth_avx2(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let yi = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        let yf = _mm256_cvtepi32_ps(yi);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), yf);
        acc0 = _mm256_fmadd_ps(yf, yf, acc0);
        i += 8;
    }

    let lo = _mm256_castps256_ps128(acc0);
    let hi = _mm256_extractf128_ps(acc0, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut ryy = _mm_cvtss_f32(s1);

    for j in i..n {
        let v = y[j] as f32;
        x[j] = v;
        ryy += v * v;
    }

    // C (normalise_residual): `g = (1.f/sqrt(Ryy)) * gain`, the reciprocal
    // first. `ryy` is a whole number of at least 1 for a band with pulses,
    // so the 1e-15 changes nothing there; it keeps an empty vector finite.
    let g = (1.0 / (1e-15f32 + ryy).sqrt()) * gain;
    let vg = _mm256_set1_ps(g);

    i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, vg));
        i += 8;
    }
    for j in i..n {
        x[j] *= g;
    }
}

/// # Safety
///
/// `n` must be at most 32, `x` must hold at least `n` values, and the CPU must
/// have the features this function is compiled for.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_search_scalar_init_avx2(
    x: &[f32],
    n: usize,
    abs_x: &mut [f32; 32],
    sign_x: &mut [i32; 32],
) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let a = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(abs_x.as_mut_ptr().add(i), a);
        acc = _mm256_add_ps(acc, a);

        for j in 0..8 {
            sign_x[i + j] = (x[i + j] < 0.0) as i32;
        }
        i += 8;
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);

    for j in i..n {
        let abs_xi = x[j].abs();
        abs_x[j] = abs_xi;
        sum += abs_xi;
        sign_x[j] = (x[j] < 0.0) as i32;
    }
    sum
}

/// Portable form of `alg_quant_resynth`; `y` and `x` have the same length.
/// The energy of `y` is summed one square after the other.
///
/// The squares and their sums are whole numbers. As long as the energy stays
/// below 2^24 every order of summation gives the same value, so this form
/// and the NEON form agree bit for bit. The tests check that.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn alg_quant_resynth_scalar(y: &[i32], x: &mut [f32], gain: f32) {
    debug_assert_eq!(y.len(), x.len());
    let mut ryy = 0.0f32;
    for (xv, &yv) in x.iter_mut().zip(y) {
        let v = yv as f32;
        *xv = v;
        ryy += v * v;
    }
    // C (normalise_residual): `g = (1.f/sqrt(Ryy)) * gain`, the reciprocal
    // first. `ryy` is a whole number of at least 1 for a band with pulses,
    // so the 1e-15 changes nothing there; it keeps an empty vector finite.
    let g = (1.0 / (1e-15 + ryy).sqrt()) * gain;
    for xv in x.iter_mut() {
        *xv *= g;
    }
}

/// NEON form of `alg_quant_resynth`. `y` must be at least as long as `x`;
/// the first `x.len()` values are used.
///
/// The energy is summed in four lanes with fused multiply-add over the
/// largest multiple of eight values, the lanes are added in pairs, and the
/// squares of the last values are added one by one.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn alg_quant_resynth_neon(y: &[i32], x: &mut [f32], gain: f32) {
    use core::arch::aarch64::*;

    let n = x.len();
    // One bounds check; the loads and stores below rely on it.
    let y = &y[..n];
    let n8 = n & !7;

    // SAFETY: NEON is part of the baseline of every aarch64 target. Both
    // slices hold exactly `n` values. Every vector load or store covers
    // `y[i..i + 8]` or `x[i..i + 8]` with `i + 8 <= n8 <= n`.
    unsafe {
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut i = 0;

        while i < n8 {
            let yi0 = vld1q_s32(y.as_ptr().add(i));
            let yi1 = vld1q_s32(y.as_ptr().add(i + 4));

            let yf0 = vcvtq_f32_s32(yi0);
            let yf1 = vcvtq_f32_s32(yi1);

            vst1q_f32(x.as_mut_ptr().add(i), yf0);
            vst1q_f32(x.as_mut_ptr().add(i + 4), yf1);

            sum_vec = vfmaq_f32(sum_vec, yf0, yf0);
            sum_vec = vfmaq_f32(sum_vec, yf1, yf1);

            i += 8;
        }

        let mut ryy = vaddvq_f32(sum_vec);
        for j in i..n {
            let v = y[j] as f32;
            x[j] = v;
            ryy += v * v;
        }

        // C (normalise_residual): `g = (1.f/sqrt(Ryy)) * gain`, the reciprocal
        // first. `ryy` is a whole number of at least 1 for a band with pulses,
        // so the 1e-15 changes nothing there; it keeps an empty vector finite.
        let g = (1.0 / (1e-15 + ryy).sqrt()) * gain;
        let vg = vdupq_n_f32(g);

        i = 0;
        while i < n8 {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            let vr0 = vmulq_f32(vx0, vg);
            let vr1 = vmulq_f32(vx1, vg);
            vst1q_f32(x.as_mut_ptr().add(i), vr0);
            vst1q_f32(x.as_mut_ptr().add(i + 4), vr1);
            i += 8;
        }

        for j in i..n {
            x[j] *= g;
        }
    }
}

/// Turns the pulse vector `y[..n]` into the band `x[..n]` with the energy
/// `gain * gain`: `x[i] = y[i] * ((1 / sqrt(energy of y)) * gain)`, as
/// libopus's `normalise_residual` (celt/vq.c) computes it.
///
/// Panics when `y` or `x` holds fewer than `n` values. This is the check
/// that keeps the vector kernels inside the two slices.
#[inline(always)]
fn alg_quant_resynth(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    let y = &y[..n];
    let x = &mut x[..n];
    #[cfg(target_arch = "aarch64")]
    alg_quant_resynth_neon(y, x, gain);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::compat::x86_has_avx2_fma() {
            // SAFETY: both slices hold exactly `n` values, and the CPU has
            // the features the kernel is compiled for.
            unsafe { alg_quant_resynth_avx2(y, x, n, gain) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    alg_quant_resynth_scalar(y, x, gain);
}

/// libopus `alg_quant` (celt/vq.c): codes the band `x[..n]` with `k` pulses.
/// The band is spread (`exp_rotation`), the pulse vector closest to it is
/// searched and coded. With `resynth` the band is replaced by what the
/// decoder will have. Returns the collapse mask of the `stride` blocks.
#[inline]
pub fn alg_quant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
    resynth: bool,
) -> u32 {
    alg_quant_with(x, n, k, spread, stride, rc, gain, resynth, None)
}

/// `alg_quant` that takes the cosine and the sine of the rotation from
/// `rotations` when the cache holds them. Without a cache it computes them.
/// Either way the results are those of `alg_quant`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn alg_quant_with(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
    resynth: bool,
    rotations: Option<&mut RotationCache>,
) -> u32 {
    // A band never has more than MAX_PVQ_N values. Checked in release
    // builds too, because the slice below must fit in the buffer.
    assert!(n <= MAX_PVQ_N, "band of {n} values exceeds MAX_PVQ_N");
    let mut y_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];
    // SAFETY: `n` values fit in the buffer (checked above). `pvq_search`
    // writes all `n` values before anything reads them (it panics if `x` is
    // shorter than `n`).
    let y = unsafe { core::slice::from_raw_parts_mut(y_mu.as_mut_ptr() as *mut i32, n) };

    // The rotation before the search and the one after the resynthesis use
    // the same angle, so its cosine and sine are found once.
    let rotation = rotation_factor_k(n, stride, k, spread).map(|factor_k| match rotations {
        Some(cache) => cache.cos_sin(n, factor_k),
        None => rotation_cos_sin(n, factor_k),
    });
    if let Some((c, s)) = rotation {
        exp_rotation_with(x, n, 1, stride, c, s);
    }
    pvq_search(x, y, k, n);
    let mask = extract_collapse_mask(y, n, stride);
    encode_pulses(y, n as u32, k as u32, rc);

    if resynth {
        alg_quant_resynth(y, x, n, gain);
        if let Some((c, s)) = rotation {
            exp_rotation_with(x, n, -1, stride, c, s);
        }
    }
    mask
}

#[inline]
pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
) -> u32 {
    // A band never has more than MAX_PVQ_N values. Checked in release
    // builds too, because the slice below must fit in the buffer.
    assert!(n <= MAX_PVQ_N, "band of {n} values exceeds MAX_PVQ_N");
    let mut y_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];
    // SAFETY: `n` values fit in the buffer (checked above). `decode_pulses`
    // writes all `n` values before anything reads them.
    let y = unsafe { core::slice::from_raw_parts_mut(y_mu.as_mut_ptr() as *mut i32, n) };
    decode_pulses(y, n as u32, k as u32, rc);

    let mask = extract_collapse_mask(y, n, stride);

    // This call checks that `x` holds `n` values. Without that check the
    // vector kernel wrote the decoded band through a pointer, past the end
    // of a shorter `x`.
    alg_quant_resynth(y, x, n, gain);

    exp_rotation(x, n, -1, stride, k, spread);

    mask
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

    /// Small deterministic generator, so that the tests need no dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        /// A value in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// The bits of each value. Every NaN counts as the same value: which
    /// operand a fused multiply-add takes the sign of a NaN from depends on
    /// how the compiler orders the operands, and no band holds a NaN.
    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter()
            .map(|v| if v.is_nan() { f32::NAN.to_bits() } else { v.to_bits() })
            .collect()
    }

    /// Input vectors of `total` values for the rotation tests: random values
    /// and a few patterns with zeros, large, tiny and alternating values.
    fn rotation_inputs(rng: &mut Lcg, total: usize) -> Vec<Vec<f32>> {
        let mut v = Vec::new();
        for _ in 0..3 {
            v.push((0..total).map(|_| rng.unit()).collect());
        }
        v.push(vec![0.0; total]);
        v.push(
            (0..total)
                .map(|i| if i % 2 == 0 { 1.0e30 } else { -1.0e30 })
                .collect(),
        );
        v.push(
            (0..total)
                .map(|i| if i % 3 == 0 { -0.0 } else { 1.0e-40 })
                .collect(),
        );
        v.push((0..total).map(|i| (i as f32 - 7.0) * 1.0e-20).collect());
        v
    }

    /// Indices to try for a codebook of `v` vectors: all of them when there
    /// are few, else both ends, the table values near them and random ones.
    fn indices(rng: &mut Lcg, v: u64) -> Vec<u32> {
        let v = v.min(1 << 32);
        if v <= 600 {
            return (0..v as u32).collect();
        }
        let mut out = vec![0, 1, 2, (v / 2) as u32, (v - 2) as u32, (v - 1) as u32];
        for _ in 0..40 {
            out.push((rng.next_u32() as u64 % v) as u32);
        }
        // Small indices decode to vectors with most pulses at the end.
        for _ in 0..10 {
            out.push((rng.next_u32() as u64 % v.min(4096)) as u32);
        }
        out
    }

    #[test]
    fn pvq_table_covers_matches_the_lookup_domain() {
        // `pvq_table_covers(n, k)` must hold exactly when the checked lookup
        // finds U(n, k + 1) in the table, for the whole table and beyond it.
        for n in 0..=400u32 {
            for k in 0..=400u32 {
                let r = n.min(k + 1) as usize;
                let c = n.max(k + 1) as usize;
                let in_table = r < CELT_PVQ_U_ROW.len() && c <= CELT_PVQ_U_CMAX[r];
                assert_eq!(pvq_table_covers(n, k), in_table, "n {n} k {k}");
            }
        }
        assert!(!pvq_table_covers(2, u32::MAX));
        assert!(!pvq_table_covers(u32::MAX, 1));
        assert!(!pvq_table_covers(u32::MAX, u32::MAX));
    }

    #[test]
    fn cwrsi_table_matches_checked_lookups() {
        let mut rng = Lcg(4);
        let mut pairs = 0;
        for n in 2..=176u32 {
            for k in 1..=176u32 {
                if !pvq_table_covers(n, k) {
                    continue;
                }
                pairs += 1;
                // SAFETY: the table covers `(n, k)`.
                let v = unsafe { pvq_v_unchecked(n, k) };
                assert_eq!(v.min(u32::MAX as u64) as u32, celt_pvq_v(n, k));
                for i in indices(&mut rng, v) {
                    let mut want = vec![7i32; n as usize + 2];
                    cwrsi_checked(n, k, i, &mut want);
                    let mut got = vec![7i32; n as usize + 2];
                    // SAFETY: the table covers `(n, k)` and `i < v`.
                    unsafe { cwrsi_table(n as usize, k as usize, i, &mut got[..n as usize]) };
                    assert_eq!(got, want, "n {n} k {k} i {i}");
                    let pulses: u32 = got[..n as usize].iter().map(|v| v.unsigned_abs()).sum();
                    assert_eq!(pulses, k, "n {n} k {k} i {i}");
                    // The public function takes the same path.
                    let mut public = vec![7i32; n as usize + 2];
                    cwrsi(n, k, i, &mut public);
                    assert_eq!(public, want);
                }
            }
        }
        // Rows 0 to 5 reach column 176; the higher rows are shorter.
        assert!(pairs > 1500, "only {pairs} pairs tested");
    }

    /// The search step written as the loop it replaces, with a plain `if`.
    fn best_position_plain(square: bool, abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32) -> usize {
        let num = |i: usize| {
            let rxy = xy + abs_x[i];
            if square { rxy * rxy } else { rxy }
        };
        let mut best_id = 0;
        let mut best_num = num(0);
        let mut best_den = yy + y2f[0];
        for i in 1..abs_x.len() {
            let den = yy + y2f[i];
            if best_den * num(i) > den * best_num {
                best_num = num(i);
                best_den = den;
                best_id = i;
            }
        }
        best_id
    }

    fn check_best_position(abs_x: &[f32], y2f: &[f32], xy: f32, yy: f32) {
        for square in [true, false] {
            let want = best_position_plain(square, abs_x, y2f, xy, yy);
            let (scalar, dispatched) = if square {
                (
                    pvq_best_position_scalar::<true>(abs_x, y2f, xy, yy),
                    pvq_best_position::<true>(abs_x, y2f, xy, yy),
                )
            } else {
                (
                    pvq_best_position_scalar::<false>(abs_x, y2f, xy, yy),
                    pvq_best_position::<false>(abs_x, y2f, xy, yy),
                )
            };
            assert_eq!(scalar, want, "scalar: {abs_x:?} {y2f:?} xy {xy} yy {yy}");
            assert_eq!(dispatched, want, "vector: {abs_x:?} {y2f:?} xy {xy} yy {yy}");
            // The check of `pvq_place_pulses` never accepts another
            // position than the scan's, whatever the candidate.
            #[cfg(target_arch = "aarch64")]
            if square {
                for m in 0..abs_x.len() {
                    // SAFETY: NEON is part of the baseline of every aarch64
                    // target.
                    let accepted = unsafe { pvq_candidate_is_scan_result(abs_x, y2f, xy, yy, m) };
                    assert!(
                        !accepted || m == want,
                        "check accepts {m}, scan gives {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn pvq_best_position_forms_agree() {
        let mut rng = Lcg(6);
        // Lengths 1..=33: every number of groups of four and every tail. The
        // longer ones cover both ends of the range of the checked form.
        for n in (1..=33usize).chain([40, 47, 63, 64, 65, 96, 176]) {
            for round in 0..300 {
                // Magnitudes as in a band, pulse counts as during a search.
                let mut abs_x: Vec<f32> = (0..n).map(|_| rng.unit().abs()).collect();
                let mut y2f: Vec<f32> =
                    (0..n).map(|_| 2.0 * (rng.next_u32() % 4) as f32).collect();
                let xy = rng.unit().abs() * 4.0;
                let yy = 1.0 + (rng.next_u32() % 40) as f32;
                match round % 8 {
                    // Many equal candidates: the first one must win.
                    1 => {
                        let v = abs_x[0];
                        abs_x.fill(v);
                        y2f.fill(2.0);
                    }
                    // Two values only, so that ties are common.
                    2 => {
                        for v in abs_x.iter_mut() {
                            *v = if *v > 0.5 { 0.75 } else { 0.25 };
                        }
                    }
                    // The best position is the last one, or the first one.
                    3 => abs_x[n - 1] = 3.0,
                    4 => abs_x[0] = 3.0,
                    // The longer buffer of the callers: extra values after n.
                    5 => y2f.push(123.0),
                    // Near ties: the largest value again, one or two float
                    // steps apart, before and after it.
                    6 | 7 => {
                        let (top, _) = abs_x
                            .iter()
                            .enumerate()
                            .fold((0, -1.0f32), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
                        let step = if round % 6 == 0 { 1 } else { 2 };
                        let v = f32::from_bits(abs_x[top].to_bits() + step);
                        abs_x[(top + n / 2) % n] = v;
                        abs_x[(top + 1) % n] = abs_x[top];
                        y2f[(top + n / 2) % n] = y2f[top];
                        y2f[(top + 1) % n] = y2f[top];
                    }
                    _ => {}
                }
                check_best_position(&abs_x, &y2f, xy, yy);
            }
            // Edge values: all zero, huge values whose products overflow,
            // and values that are not numbers.
            let zeros = vec![0.0f32; n];
            check_best_position(&zeros, &zeros, 0.0, 1.0);
            let huge: Vec<f32> = (0..n).map(|i| 1.0e30 * (1 + i % 3) as f32).collect();
            check_best_position(&huge, &zeros, 1.0e30, 1.0e30);
            check_best_position(&zeros, &huge, 1.0e-30, 1.0e-30);
            for at in [0, n / 2, n - 1] {
                let mut bad: Vec<f32> = (0..n).map(|_| rng.unit().abs()).collect();
                bad[at] = f32::NAN;
                check_best_position(&bad, &zeros, 0.5, 2.0);
                bad[at] = f32::INFINITY;
                check_best_position(&bad, &zeros, 0.5, 2.0);
            }
        }
    }

    /// The search, which places the pulses of bands of 8 to 64 values with
    /// `pvq_place_pulses`, puts every pulse where the search with the
    /// portable scan puts it. The bands are built to make the check fail:
    /// equal values, values one float step apart, many pulses, silence, and
    /// values that are not numbers.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn placing_pulses_in_one_loop_matches_the_scan() {
        let mut rng = Lcg(41);
        let mut compared = 0;
        for n in (1..=40usize).chain([44, 47, 48, 63, 64, 65, 72, 96]) {
            for round in 0..120 {
                let mut x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
                match round % 8 {
                    1 => x
                        .iter_mut()
                        .for_each(|v| *v = if *v > 0.0 { 0.5 } else { -0.5 }),
                    2 => {
                        // Values one or two float steps apart.
                        let base = 0.3f32;
                        for (i, v) in x.iter_mut().enumerate() {
                            *v = f32::from_bits(base.to_bits() + (i % 3) as u32);
                        }
                    }
                    3 => x.fill(0.0),
                    4 => x.iter_mut().for_each(|v| *v *= 1e-20),
                    5 => {
                        x.iter_mut().for_each(|v| *v *= 1e-3);
                        x[n / 2] = 1.0;
                    }
                    6 => x[n - 1] = f32::NAN,
                    7 => x.iter_mut().for_each(|v| *v *= 70.0),
                    _ => {}
                }
                for k in [1, 2, 3, 5, 9, 17, (n as i32) / 2 + 1, n as i32 + 2] {
                    let mut want = vec![0i32; n];
                    let want_yy = pvq_search_libopus_with(&x, &mut want, k, n, |a, b, xy, yy| {
                        pvq_best_position_scalar::<true>(a, b, xy, yy)
                    });
                    let mut got = vec![0i32; n];
                    let got_yy = pvq_search_libopus(&x, &mut got, k, n);
                    assert_eq!(got, want, "n {n} k {k} round {round}");
                    assert_eq!(got_yy.to_bits(), want_yy.to_bits(), "n {n} k {k}: yy");
                    compared += 1;
                }
            }
        }
        assert!(compared > 40_000);
    }

    /// A vector of `n` entries with `k` pulses in a few random positions.
    fn random_pulses(rng: &mut Lcg, n: usize, k: u32) -> Vec<i32> {
        let mut y = vec![0i32; n];
        let spots = 1 + rng.next_u32() as usize % n.min(5);
        for _ in 0..k {
            y[rng.next_u32() as usize % spots * (n / spots)] += 1;
        }
        for v in y.iter_mut() {
            if rng.next_u32() & 1 == 1 {
                *v = -*v;
            }
        }
        y
    }

    #[test]
    fn icwrs_table_matches_checked_lookups() {
        let mut rng = Lcg(5);
        let mut pairs = 0;
        for n in 2..=176u32 {
            for k in 0..=176u32 {
                if !pvq_table_covers(n, k) {
                    continue;
                }
                pairs += 1;
                // As many pulses as announced, and fewer.
                for pulses in [k, k / 2, k.saturating_sub(1), 0] {
                    for _ in 0..3 {
                        let y = random_pulses(&mut rng, n as usize, pulses);
                        let want = icwrs_checked(n, &y);
                        // SAFETY: the table covers `(n, k)`.
                        let got = unsafe { icwrs_table(k, &y) };
                        assert_eq!(got, Some(want), "n {n} k {k} y {y:?}");
                        assert_eq!(icwrs(n, k, &y), want);
                    }
                }
                // More pulses than announced: the table path must give up
                // before it reads, and the public function must still give
                // the checked result.
                for extra in [1u32, 2, 40] {
                    let mut y = random_pulses(&mut rng, n as usize, k);
                    let at = rng.next_u32() as usize % n as usize;
                    y[at] += if y[at] < 0 { -(extra as i32) } else { extra as i32 };
                    // SAFETY: the table covers `(n, k)`.
                    assert_eq!(unsafe { icwrs_table(k, &y) }, None, "n {n} k {k} y {y:?}");
                    if n <= 40 {
                        assert_eq!(icwrs(n, k, &y), icwrs_checked(n, &y));
                    }
                }
            }
        }
        assert!(pairs > 1500, "only {pairs} pairs tested");

        // Extreme entries. One of them alone is beyond any table row.
        for at in 0..4 {
            for big in [i32::MIN, i32::MAX, -i32::MAX] {
                let mut y = [1i32, -2, 0, 3];
                y[at] = big;
                // SAFETY: the table covers `(4, 6)`.
                assert_eq!(unsafe { icwrs_table(6, &y) }, None);
                assert_eq!(icwrs(4, 6, &y), icwrs_checked(4, &y));
            }
        }
    }

    #[test]
    fn cwrsi_falls_back_outside_the_table_and_for_invalid_indices() {
        // (n, k) outside the table: U(n, k + 1) is computed, not read. These
        // three have fewer than 2^32 vectors, so the computed values are
        // exact.
        for (n, k) in [(100u32, 5u32), (60, 6), (40, 7)] {
            assert!(!pvq_table_covers(n, k));
            let v = celt_pvq_v(n, k);
            assert!(v < u32::MAX);
            for i in [0, 1, v / 3, v - 1] {
                let mut want = vec![0i32; n as usize];
                cwrsi_checked(n, k, i, &mut want);
                let mut got = vec![0i32; n as usize];
                cwrsi(n, k, i, &mut got);
                assert_eq!(got, want, "n {n} k {k} i {i}");
                assert_eq!(icwrs(n, k, &got), i, "n {n} k {k} i {i}");
            }
        }
        // An index that is not below V(n, k) must not reach the table path.
        // The checked function defines the result. Only two and three
        // dimensions are tried: with more, the checked function can take
        // minutes for such an index.
        for (n, k) in [(3u32, 1u32), (3, 5), (3, 40), (2, 9)] {
            let v = celt_pvq_v(n, k);
            for i in [v, v + 1, 2 * v, v + 12345] {
                let mut want = vec![0i32; n as usize];
                cwrsi_checked(n, k, i, &mut want);
                let mut got = vec![0i32; n as usize];
                cwrsi(n, k, i, &mut got);
                assert_eq!(got, want, "n {n} k {k} i {i}");
            }
        }
    }

    /// Pairs `(c, s)`: real rotation angles in both directions and extremes.
    const ROTATION_CS: [(f32, f32); 8] = [
        (0.9238795, 0.38268343),
        (0.9238795, -0.38268343),
        (0.38268343, -0.9238795),
        (0.70710677, 0.70710677),
        (1.0, 0.0),
        (0.0, -1.0),
        (-0.0, 1.0),
        (1.0e-30, 1.0e20),
    ];

    /// Every band size and pulse count the codec can rotate, several times
    /// and in an order that makes entries replace each other: the cache
    /// always returns what `rotation_cos_sin` computes.
    #[test]
    fn rotation_cache_returns_the_computed_angles() {
        let mut cache = RotationCache::new();
        let mut rng = Lcg(7);
        let mut lookups = 0;
        for _ in 0..3 {
            for length in 1..=MAX_PVQ_N {
                for spread in 1..=3 {
                    for k in 0..=(length as i32 - 1) / 2 {
                        // Also a random key, so that the order is not only
                        // the increasing one.
                        let jump_len = 1 + rng.next_u32() as usize % MAX_PVQ_N;
                        let jump_k = rng.next_u32() as i32 % (1 + (jump_len as i32 - 1) / 2);
                        for (n, k) in [(length, k), (jump_len, jump_k)] {
                            let Some(fk) = rotation_factor_k(n, 1, k, spread) else {
                                continue;
                            };
                            let (c, s) = cache.cos_sin(n, fk);
                            let (want_c, want_s) = rotation_cos_sin(n, fk);
                            assert_eq!(
                                (c.to_bits(), s.to_bits()),
                                (want_c.to_bits(), want_s.to_bits()),
                                "length {n}, factor * k {fk}"
                            );
                            lookups += 1;
                        }
                    }
                }
            }
        }
        assert!(lookups > 100_000);
        // Keys outside the exact range are computed, not looked up.
        for (n, fk) in [(MAX_PVQ_N + 1, 3), (8, -5), (8, 1 << 16)] {
            let (c, s) = cache.cos_sin(n, fk);
            let (want_c, want_s) = rotation_cos_sin(n, fk);
            assert_eq!(
                (c.to_bits(), s.to_bits()),
                (want_c.to_bits(), want_s.to_bits())
            );
        }
    }

    /// `alg_quant` with a cache that already holds angles, and without one,
    /// writes the same bytes and leaves the same band.
    #[test]
    fn alg_quant_with_a_cache_matches_alg_quant() {
        let mut rng = Lcg(11);
        let mut cache = RotationCache::new();
        for &n in &[2usize, 4, 8, 12, 16, 18, 24, 36, 48, 72, 96, 144, 176] {
            for &stride in &[1usize, 2, 4] {
                if n % stride != 0 {
                    continue;
                }
                for spread in 0..=3 {
                    for k in 1..=(n as i32).min(12) {
                        for resynth in [false, true] {
                            let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
                            let (mut xa, mut xb) = (x.clone(), x.clone());
                            let mut ra = RangeCoder::new_encoder(256);
                            let mut rb = RangeCoder::new_encoder(256);
                            let ma =
                                alg_quant(&mut xa, n, k, spread, stride, &mut ra, 0.9, resynth);
                            let mb = alg_quant_with(
                                &mut xb,
                                n,
                                k,
                                spread,
                                stride,
                                &mut rb,
                                0.9,
                                resynth,
                                Some(&mut cache),
                            );
                            ra.done();
                            rb.done();
                            assert_eq!(ma, mb);
                            assert_eq!(bits(&xa), bits(&xb), "n {n} stride {stride} k {k}");
                            assert_eq!(ra.buf[..], rb.buf[..]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exp_rotation1_carry_matches_plain_form() {
        let mut rng = Lcg(1);
        // Lengths 0..=40 cover `len < S`, `len == S`, `len < 2 * S` and every
        // number of steps left over after the groups of `S`.
        for len in 0..=40usize {
            // Three more values than `len`: they must stay untouched.
            for input in rotation_inputs(&mut rng, len + 3) {
                for (c, s) in ROTATION_CS {
                    for stride in 1..=3usize {
                        let mut want = input.clone();
                        exp_rotation1_scalar(&mut want, len, stride, c, s);
                        let mut got = input.clone();
                        match stride {
                            1 => exp_rotation1_carry::<1>(&mut got, len, c, s),
                            2 => exp_rotation1_carry::<2>(&mut got, len, c, s),
                            _ => exp_rotation1_carry::<3>(&mut got, len, c, s),
                        }
                        assert_eq!(
                            bits(&got),
                            bits(&want),
                            "len {len} stride {stride} c {c} s {s}"
                        );
                        assert_eq!(bits(&got[len..]), bits(&input[len..]));
                        // The dispatching function gives the same result.
                        let mut via = input.clone();
                        exp_rotation1(&mut via, len, stride, c, s);
                        assert_eq!(bits(&via), bits(&want));
                    }
                }
            }
        }
    }

    /// One step of the rotation as libopus computes it: on aarch64 the
    /// products with `c` are fused into the adds, the other two products
    /// are rounded first.
    #[test]
    fn rotate_pair_rounds_as_libopus() {
        let mut rng = Lcg(4);
        for _ in 0..2000 {
            let (x1, x2) = (rng.unit(), rng.unit());
            let (c, s) = (rng.unit(), rng.unit());
            let (y1, y2) = rotate_pair(x1, x2, c, s);
            let (w1, w2) = if cfg!(target_arch = "aarch64") {
                (c.mul_add(x1, -s * x2), c.mul_add(x2, s * x1))
            } else {
                (c * x1 + -s * x2, c * x2 + s * x1)
            };
            assert_eq!(bits(&[y1, y2]), bits(&[w1, w2]), "x {x1} {x2} c {c} s {s}");
        }
    }

    /// The vector pass computes each value as the plain form does.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_rotation1_neon_matches_plain_form() {
        let mut rng = Lcg(2);
        for stride in 4..=13usize {
            // Up to `4 * stride + 7` covers every tail length of both sweeps.
            for len in 0..=(4 * stride + 7) {
                for input in rotation_inputs(&mut rng, len + 3) {
                    for (c, s) in ROTATION_CS {
                        let mut want = input.clone();
                        exp_rotation1_scalar(&mut want, len, stride, c, s);
                        let mut got = input.clone();
                        // SAFETY: NEON is part of the aarch64 baseline.
                        unsafe { exp_rotation1_neon(&mut got, len, stride, c, s) };
                        assert_eq!(
                            bits(&got),
                            bits(&want),
                            "len {len} stride {stride} c {c} s {s}"
                        );
                        assert_eq!(bits(&got[len..]), bits(&input[len..]));
                    }
                }
            }
        }
    }

    /// Lengths that cover every tail of the 16, 8 and 4 wide loops, and the
    /// largest band.
    fn kernel_lengths() -> impl Iterator<Item = usize> {
        (0..=41).chain([63, 64, 65, 88, 127, 176, MAX_PVQ_N])
    }

    #[test]
    fn alg_quant_resynth_forms_agree() {
        let mut rng = Lcg(31);
        for n in kernel_lengths() {
            for round in 0..40 {
                // Pulse vectors as the codec has them, and denser ones up
                // to an energy just below 2^24.
                let top: i32 = match round % 4 {
                    0 => 1,
                    1 => 3,
                    2 => 40,
                    _ => 200,
                };
                let mut y: Vec<i32> = (0..n)
                    .map(|_| {
                        let v = (rng.next_u32() % (2 * top as u32 + 1)) as i32 - top;
                        if round % 3 == 0 && !rng.next_u32().is_multiple_of(4) { 0 } else { v }
                    })
                    .collect();
                if round == 7 {
                    y.fill(0);
                }
                let energy: i64 = y.iter().map(|&v| v as i64 * v as i64).sum();
                assert!(energy < 1 << 24);
                let gain = [1.0f32, 0.37, 0.0, 1.0e-3][round % 4];
                let what = format!("n {n} round {round}");

                // The definition, with an indexed loop.
                let mut want = vec![0.0f32; n];
                let mut ryy = 0.0f32;
                for i in 0..n {
                    want[i] = y[i] as f32;
                    ryy += want[i] * want[i];
                }
                // C (normalise_residual): `g = (1.f/sqrt(Ryy)) * gain`, the reciprocal
                // first. `ryy` is a whole number of at least 1 for a band with pulses,
                // so the 1e-15 changes nothing there; it keeps an empty vector finite.
                let g = (1.0 / (1e-15 + ryy).sqrt()) * gain;
                for v in want.iter_mut() {
                    *v *= g;
                }

                let mut got = vec![7.0f32; n];
                alg_quant_resynth_scalar(&y, &mut got, gain);
                assert_eq!(bits(&got), bits(&want), "scalar, {what}");

                #[cfg(target_arch = "aarch64")]
                {
                    // A longer `y`: only the first `n` values count.
                    let mut y_long = y.clone();
                    y_long.extend([1000, -1000, 5]);
                    let mut got = vec![7.0f32; n];
                    alg_quant_resynth_neon(&y_long, &mut got, gain);
                    assert_eq!(bits(&got), bits(&want), "neon, {what}");
                }

                // The dispatching function leaves everything after `n` alone.
                let mut got = vec![7.0f32; n + 3];
                alg_quant_resynth(&y, &mut got, n, gain);
                assert_eq!(bits(&got[..n]), bits(&want), "dispatch, {what}");
                assert_eq!(&got[n..], &[7.0, 7.0, 7.0], "dispatch tail, {what}");
            }
        }
    }

    /// The resynthesis must not write past a band that is shorter than `n`.
    /// The vector kernels write through pointers, so `alg_quant_resynth`
    /// cuts `x` and `y` to `n` first, which panics when one is shorter.
    #[test]
    fn resynthesis_rejects_a_band_that_is_too_short() {
        let y = [1i32; 16];
        let short = std::panic::catch_unwind(|| {
            let mut x = [0.0f32; 15];
            alg_quant_resynth(&y, &mut x, 16, 1.0);
        });
        assert!(short.is_err(), "alg_quant_resynth with a short x");
        let short = std::panic::catch_unwind(|| {
            let mut x = [0.0f32; 16];
            alg_quant_resynth(&y[..15], &mut x, 16, 1.0);
        });
        assert!(short.is_err(), "alg_quant_resynth with a short y");

        for n in [8usize, 16, 24, 48] {
            let unquant = std::panic::catch_unwind(|| {
                let packet = [0x5au8; 64];
                let mut rc = RangeCoder::new_decoder(&packet);
                let mut x = vec![0.0f32; n - 1];
                alg_unquant(&mut x, n, 3, 0, 1, &mut rc, 1.0);
            });
            assert!(unquant.is_err(), "alg_unquant, n {n}");
            let quant = std::panic::catch_unwind(|| {
                let mut rc = RangeCoder::new_encoder(64);
                let mut x = vec![0.25f32; n - 1];
                alg_quant(&mut x, n, 3, 0, 1, &mut rc, 1.0, true);
            });
            assert!(quant.is_err(), "alg_quant, n {n}");
        }
        let renorm = std::panic::catch_unwind(|| {
            let mut x = [0.5f32; 15];
            crate::bands::renormalise_vector(&mut x, 16, 1.0);
        });
        assert!(renorm.is_err(), "renormalise_vector with a short x");
    }

    #[test]
    fn extract_collapse_mask_forms_agree() {
        let mut rng = Lcg(32);
        for n in 0..=72usize {
            for b in 2..=8usize {
                let n0 = n / b;
                // All zero, one pulse at every position in turn, and random
                // sparse vectors.
                let mut cases: Vec<Vec<i32>> = vec![vec![0; n]];
                for at in 0..n {
                    let mut y = vec![0i32; n];
                    y[at] = if at % 2 == 0 { 1 } else { -3 };
                    cases.push(y);
                }
                for _ in 0..8 {
                    cases.push(
                        (0..n)
                            .map(|_| if rng.next_u32().is_multiple_of(9) { (rng.next_u32() % 5) as i32 - 2 } else { 0 })
                            .collect(),
                    );
                }
                for y in &cases {
                    // The definition.
                    let mut want = 0u32;
                    for i in 0..b {
                        if y[i * n0..(i + 1) * n0].iter().any(|&v| v != 0) {
                            want |= 1 << i;
                        }
                    }
                    assert_eq!(extract_collapse_mask_scalar(y, n, b), want, "scalar n {n} b {b}");
                    #[cfg(target_arch = "aarch64")]
                    assert_eq!(extract_collapse_mask_neon(y, n, b), want, "neon n {n} b {b}");
                    assert_eq!(extract_collapse_mask(y, n, b), want, "dispatch n {n} b {b}");
                }
            }
        }
        // One block always counts as filled.
        assert_eq!(extract_collapse_mask(&[0, 0, 0], 3, 1), 1);
        assert_eq!(extract_collapse_mask(&[], 0, 0), 1);
    }

    /// The sum of magnitudes as the start-up code of the target adds it.
    fn magnitude_sum(abs_x: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            let n4 = abs_x.len() & !3;
            let mut lanes = [0.0f32; 4];
            for group in abs_x[..n4].as_chunks::<4>().0 {
                for l in 0..4 {
                    lanes[l] += group[l];
                }
            }
            let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            for &v in &abs_x[n4..] {
                sum += v;
            }
            sum
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            abs_x.iter().fold(0.0f32, |s, &v| s + v)
        }
    }

    /// The search of libopus (celt/vq.c, op_pvq_search_c) with indexed
    /// loops: the projection, then one pulse at a time with the exact
    /// comparison. Returns `None` when more than seven pulses are left after
    /// the projection; `pvq_search_fast_select` then places four per pass,
    /// which is a different rule.
    fn search_reference(x: &[f32], k: i32) -> Option<(Vec<i32>, f32)> {
        let n = x.len();
        let abs_x: Vec<f32> = x.iter().map(|v| v.abs()).collect();
        let sum = magnitude_sum(&abs_x);
        let mut y = vec![0i32; n];
        let (mut xy, mut yy) = (0.0f32, 0.0f32);
        let mut left = k;
        if k > (n >> 1) as i32 && sum > 1e-15 {
            let rcp = (k as f32 + 0.8) / sum;
            for i in 0..n {
                y[i] = (abs_x[i] * rcp) as i32;
                let yf = y[i] as f32;
                yy += yf * yf;
                xy += yf * abs_x[i];
                left -= y[i];
            }
            if left > n as i32 + 3 {
                let tmp = left as f32;
                yy += tmp * tmp + tmp * y[0] as f32;
                y[0] += left;
                left = 0;
            }
        }
        if left >= 8 && n >= 16 {
            return None;
        }
        for _ in 0..left {
            yy += 1.0;
            let mut best = 0;
            let mut best_num = (xy + abs_x[0]) * (xy + abs_x[0]);
            let mut best_den = yy + 2.0 * y[0] as f32;
            for i in 1..n {
                let num = (xy + abs_x[i]) * (xy + abs_x[i]);
                let den = yy + 2.0 * y[i] as f32;
                if best_den * num > den * best_num {
                    best = i;
                    best_num = num;
                    best_den = den;
                }
            }
            xy += abs_x[best];
            yy += 2.0 * y[best] as f32;
            y[best] += 1;
        }
        for i in 0..n {
            if x[i] < 0.0 {
                y[i] = -y[i];
            }
        }
        Some((y, yy))
    }

    /// The portable choice of position in `pvq_search_fast_select` gives the
    /// search of libopus. It did not on targets other than aarch64: there
    /// the one-pulse-at-a-time stage forgot the pulses that the projection
    /// had placed.
    #[test]
    fn fast_select_with_the_portable_position_is_the_libopus_search() {
        let mut rng = Lcg(34);
        let mut with_projection = 0;
        let mut without_projection = 0;
        for n in [1usize, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 40, 48, 64, 96, 144, 176, MAX_PVQ_N] {
            for round in 0..60 {
                // A few large coefficients among small ones, as in a band
                // that gets many pulses. Then the projection leaves only a
                // few pulses for the search.
                let peaks = 1 + (rng.next_u32() as usize % 4).min(n - 1);
                let floor = [0.0f32, 1.0e-4, 1.0e-2, 0.2][round % 4];
                let mut x: Vec<f32> = (0..n).map(|_| rng.unit() * floor).collect();
                for _ in 0..peaks {
                    let at = rng.next_u32() as usize % n;
                    x[at] = rng.unit() * 2.0;
                }
                for k in [1i32, 2, 5, 7, (n / 2) as i32 + 1, n as i32, 2 * n as i32, 100] {
                    let Some((want, want_yy)) = search_reference(&x, k) else {
                        continue;
                    };
                    if k > (n >> 1) as i32 {
                        with_projection += 1;
                    } else {
                        without_projection += 1;
                    }
                    let mut got = vec![7i32; n + 2];
                    let yy = pvq_search_fast_select_with(
                        &x,
                        &mut got,
                        k,
                        n,
                        pvq_best_position_scalar::<true>,
                    );
                    assert_eq!(&got[..n], &want[..], "n {n} k {k} round {round}");
                    assert_eq!(&got[n..], &[7, 7], "n {n} k {k}: wrote past n");
                    assert_eq!(yy.to_bits(), want_yy.to_bits(), "n {n} k {k}: energy");
                }
            }
        }
        // Both kinds of case must have been compared.
        assert!(with_projection > 1000 && without_projection > 1000);
    }

    #[test]
    fn fast_select_checks_its_sizes() {
        // No dimension: nothing is written and nothing is read.
        let mut y: [i32; 0] = [];
        assert_eq!(pvq_search_fast_select(&[], &mut y, 5, 0), 0.0);
        // No pulse: the vector is cleared.
        let mut y = [9i32; 40];
        assert_eq!(pvq_search_fast_select(&[0.5; 40], &mut y, 0, 40), 0.0);
        assert_eq!(y, [0; 40]);

        let short_x = std::panic::catch_unwind(|| {
            let mut y = [0i32; 40];
            pvq_search_fast_select(&[0.5; 39], &mut y, 3, 40);
        });
        assert!(short_x.is_err());
        let short_y = std::panic::catch_unwind(|| {
            let mut y = [0i32; 39];
            pvq_search_fast_select(&[0.5; 40], &mut y, 3, 40);
        });
        assert!(short_y.is_err());
        let too_long = std::panic::catch_unwind(|| {
            let x = vec![0.5f32; MAX_PVQ_N + 1];
            let mut y = vec![0i32; MAX_PVQ_N + 1];
            pvq_search_fast_select(&x, &mut y, 3, MAX_PVQ_N + 1);
        });
        assert!(too_long.is_err());
    }

    /// The reference for `pvq_magnitude_sum`: a sum of magnitudes with one
    /// register of four running sums over the blocks of 16, 8 and 4 values,
    /// a pairwise reduction, and the last values added one by one.
    #[cfg(target_arch = "aarch64")]
    fn magnitude_sum_old_kernel(abs_x: &[f32]) -> f32 {
        use core::arch::aarch64::*;
        let n = abs_x.len();
        let mut i = 0;
        // SAFETY: every load reads `abs_x[i..i + 4]` with `i + 4 <= n`.
        let mut sum = unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            while i + 16 <= n {
                for block in 0..4 {
                    sum_vec = vaddq_f32(sum_vec, vld1q_f32(abs_x.as_ptr().add(i + 4 * block)));
                }
                i += 16;
            }
            while i + 8 <= n {
                sum_vec = vaddq_f32(sum_vec, vld1q_f32(abs_x.as_ptr().add(i)));
                sum_vec = vaddq_f32(sum_vec, vld1q_f32(abs_x.as_ptr().add(i + 4)));
                i += 8;
            }
            while i + 4 <= n {
                sum_vec = vaddq_f32(sum_vec, vld1q_f32(abs_x.as_ptr().add(i)));
                i += 4;
            }
            vaddvq_f32(sum_vec)
        };
        for &v in &abs_x[i..] {
            sum += v;
        }
        sum
    }

    /// `pvq_magnitude_sum` adds in the order its target has always used.
    /// The projection step rounds with this sum, so its last bit counts.
    #[test]
    fn pvq_magnitude_sum_keeps_the_order_of_its_target() {
        let mut rng = Lcg(36);
        for n in kernel_lengths() {
            for round in 0..40 {
                let scale = [1.0f32, 1.0e-6, 3.0e4, 1.0e-20][round % 4];
                let abs_x: Vec<f32> = (0..n).map(|_| rng.unit().abs() * scale).collect();
                let got = pvq_magnitude_sum(&abs_x);
                assert_eq!(got.to_bits(), magnitude_sum(&abs_x).to_bits(), "model, n {n}");
                #[cfg(target_arch = "aarch64")]
                assert_eq!(
                    got.to_bits(),
                    magnitude_sum_old_kernel(&abs_x).to_bits(),
                    "old kernel, n {n}"
                );
            }
        }
    }
}
