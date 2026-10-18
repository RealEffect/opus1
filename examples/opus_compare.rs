//! The libopus test program `opus_compare`, on top of the port in
//! `opus1::internals::eval::compare`.
//!
//! ```text
//! cargo run --release --example opus_compare -- [--exact] [-s] [-r rate] reference.sw test.sw
//! ```
//!
//! The reference is 16-bit little-endian stereo at 48 kHz. The test signal
//! has one channel, or two with `-s`, at `rate` (default 48000). The
//! program prints the lines of the C program to stderr and exits with its
//! exit codes: 0 when the test vector passes, 1 otherwise.
//!
//! With `--exact` it also prints, to stdout, the internal weighted error
//! and the quality metric with all their digits and as bit patterns, and
//! the frame with the largest error.

#[cfg(not(feature = "std"))]
fn main() {
    println!("this example needs the `std` feature");
}

#[cfg(feature = "std")]
fn main() {
    std::process::exit(run());
}

/// C `read_pcm16`: whole frames of 16-bit little-endian samples; a partial
/// last frame is dropped.
#[cfg(feature = "std")]
fn read_pcm16(path: &str, channels: usize) -> Result<Vec<i16>, String> {
    let bytes = std::fs::read(path).map_err(|_| format!("Error opening '{path}'."))?;
    let frames = bytes.len() / (2 * channels);
    Ok(bytes[..frames * 2 * channels].as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect())
}

#[cfg(feature = "std")]
fn run() -> i32 {
    use opus1::internals::eval::compare::{Reference, opus_compare};

    let mut args: Vec<String> = std::env::args().collect();
    let exact = args.get(1).is_some_and(|a| a == "--exact");
    if exact {
        args.remove(1);
    }
    if args.len() < 3 || args.len() > 6 {
        eprintln!("Usage: {} [--exact] [-s] [-r rate2] <file1.sw> <file2.sw>", args[0]);
        return 1;
    }
    let mut a = 1;
    let mut channels = 1;
    if args[a] == "-s" {
        channels = 2;
        a += 1;
    }
    let mut rate = 48000;
    if args.get(a).is_some_and(|s| s == "-r") {
        // C `atoi`: a value that is not a number is 0, an invalid rate.
        rate = args.get(a + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
        if ![8000, 12000, 16000, 24000, 48000].contains(&rate) {
            eprintln!("Sampling rate must be 8000, 12000, 16000, 24000, or 48000");
            return 1;
        }
        a += 2;
    }
    let (Some(file1), Some(file2)) = (args.get(a), args.get(a + 1)) else {
        eprintln!("Usage: {} [--exact] [-s] [-r rate2] <file1.sw> <file2.sw>", args[0]);
        return 1;
    };
    let read = |path: &str, channels| {
        read_pcm16(path, channels).map_err(|e| {
            eprintln!("{e}");
            1
        })
    };
    let reference = match read(file1, 2) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let test = match read(file2, channels) {
        Ok(s) => s,
        Err(code) => return code,
    };
    match opus_compare(Reference::Stereo(&reference), &test, channels, rate) {
        Err(e) => {
            eprintln!("{e}");
            1
        }
        Ok(c) => {
            eprint!("{c}");
            if exact {
                println!("frames {}", c.frames);
                println!("err {:?} 0x{:016x}", c.err, c.err.to_bits());
                println!("q {:?} 0x{:08x}", c.quality, c.quality.to_bits());
                println!("worst_frame {} {:?}", c.worst_frame, c.worst_frame_error);
            }
            if c.passes() { 0 } else { 1 }
        }
    }
}
