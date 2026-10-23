# Tests opus1 natively on 64-bit Windows (Rust's MSVC toolchain) against
# libopus 1.6.1 built on that computer, and packs the results into one zip
# file to send back. It does what scripts/x86/test.sh does on Linux.
#
# What it does:
#   1. Records the CPU, the vector instruction sets that Windows reports,
#      the Windows version and the tool versions.
#   2. Downloads the libopus 1.6.1 release (or takes $env:OPUS_TARBALL),
#      checks its SHA-256 sum and builds it twice with CMake and Visual
#      Studio as static libraries (Release): the default build, and a build
#      with deep PLC, OSCE and DRED. libopus's CMake files do not define
#      ENABLE_OSCE_BWE, which its configure script defines with
#      --enable-osce, so the script adds it.
#   3. Runs the whole test suite linked with the default build.
#   4. Runs the neural network tests that compare every x86 kernel variant
#      the CPU has with the portable code, then the tests that compare with
#      the libopus built with its neural network code.
#   5. Checks the golden file of 64-bit ARM (x86 is expected to differ in
#      some streams; the number is recorded). Then compares opus1 with
#      libopus live, scenario by scenario (perf_harness modes): both
#      encoders and both decoders on the same input, as modes.md.
#   6. Times encoding and decoding against the default libopus build
#      (skipped with -Quick).
#      With -AbBaseline DIR it times this tree against the baseline tree
#      instead: scripts\x86\ab.ps1 builds perf_harness in both, runs the
#      bench set of each in turns (A B B A A B ...) and writes ab\ with a
#      report of time of this tree / time of the baseline (see ab.ps1). With
#      -Quick it only builds the two programs.
#   7. Writes x86-windows-results-<date>.zip in the repository root.
#
# Usage, from the repository root, in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\x86\test.ps1 [-Quick] [-NoDnn]
#       [-AbBaseline DIR [-Passes N]]
#
#   -Passes N  runs of each build in the A/B timing (default 5)
#
# Settings (environment variables):
#   OPUS_TARBALL           A local copy of opus-1.6.1.tar.gz.
#   OPUS1_CORPUS           A corpus folder with rfc8251\ for the conformance tests.
#   OPUS1_CMAKE_GENERATOR  Default: "Visual Studio 17 2022".
#
# Needs: Visual Studio 2022 Build Tools with "Desktop development with C++",
# CMake 3.16 or newer, and Rust (rustup-init.exe; host x86_64-pc-windows-msvc).
param([switch]$Quick, [switch]$NoDnn, [string]$AbBaseline = '', [int]$Passes = 5)

$ErrorActionPreference = 'Continue'
$OpusUrl = 'https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz'
$OpusSha256 = '6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1'

if ($AbBaseline) {
    $resolved = Resolve-Path -LiteralPath $AbBaseline -ErrorAction SilentlyContinue
    if (-not $resolved) { Write-Host "error: -AbBaseline: no folder $AbBaseline"; exit 2 }
    $AbBaseline = $resolved.Path
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Repo
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Out = Join-Path $Repo "x86-windows-results-$Stamp"
$Work = Join-Path $Repo 'target\x86-windows'
New-Item -ItemType Directory -Force -Path $Out, $Work | Out-Null
Start-Transcript -Path (Join-Path $Out 'run.log') | Out-Null
$Summary = Join-Path $Out 'summary.txt'

function Step([string]$text) { Write-Host ''; Write-Host ":: $text" }
function Die([string]$text) {
    Write-Host "error: $text"
    Stop-Transcript | Out-Null
    exit 1
}
function Note([string]$text) {
    Write-Host $text
    Add-Content -Path $Summary -Value $text
}

Step 'system'
Add-Type -Namespace Opus1 -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);'
$info = @()
$info += "date: $((Get-Date).ToUniversalTime().ToString('s'))Z"
$info += "windows: $([Environment]::OSVersion.VersionString)"
$info += "cpu: $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
$info += "logical cpus: $([Environment]::ProcessorCount)"
$features = [ordered]@{ 'sse3' = 13; 'ssse3' = 36; 'sse4_1' = 37; 'sse4_2' = 38; 'avx' = 39; 'avx2' = 40; 'avx512f' = 41 }
foreach ($f in $features.Keys) {
    $has = [Opus1.Cpu]::IsProcessorFeaturePresent($features[$f])
    $info += "feature ${f}: $(if ($has) { 'yes' } else { 'no' })"
}
$info += 'feature avx_vnni, avx512_vnni: not reported by Windows; the tests print the kernel variants they run'
foreach ($tool in @('rustc', 'cargo', 'cmake')) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($cmd) { $info += "${tool}: $((& $tool --version | Select-Object -First 1))" } else { $info += "${tool}: missing" }
}
$hostLine = & rustc -vV 2>$null | Select-String '^host:'
$host_triple = "$hostLine"
$info += "rust $host_triple"
$commitFile = Join-Path $Repo 'COMMIT'
if (Test-Path $commitFile) { $info += "git commit: $((Get-Content $commitFile -First 1))" }
$info | Tee-Object -FilePath (Join-Path $Out 'system.txt')

if (-not [Environment]::Is64BitOperatingSystem) { Die 'this is not 64-bit Windows' }
foreach ($tool in @('cargo', 'cmake')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Die "$tool is missing; see the header of this script" }
}
if ($host_triple -notmatch 'x86_64-pc-windows-msvc') { Die "Rust's host is not x86_64-pc-windows-msvc ($host_triple)" }
# opus1 needs Rust 1.98 or newer (rust-version in Cargo.toml). Check it now,
# before minutes are spent on libopus.
$rustcLine = "$(& rustc -V)"
if ($rustcLine -match 'rustc (\d+)\.(\d+)\.(\d+)') {
    if ([version]"$($Matches[1]).$($Matches[2]).$($Matches[3])" -lt [version]'1.98.0') {
        Die "$rustcLine is older than 1.98; run: rustup update stable"
    }
} else { Die "cannot read the Rust version ($rustcLine)" }

Step 'libopus 1.6.1 source'
$Tarball = $env:OPUS_TARBALL
if (-not $Tarball) { $Tarball = Join-Path $Work 'opus-1.6.1.tar.gz' }
if (-not (Test-Path $Tarball)) {
    try { Invoke-WebRequest -UseBasicParsing -Uri $OpusUrl -OutFile $Tarball } catch { Die "download of $OpusUrl failed" }
}
$hash = (Get-FileHash -Algorithm SHA256 -Path $Tarball).Hash
if ($hash -ne $OpusSha256) { Die "SHA-256 of $Tarball does not match ($hash)" }
Write-Host "SHA-256 ok"

$Generator = $env:OPUS1_CMAKE_GENERATOR
if (-not $Generator) { $Generator = 'Visual Studio 17 2022' }

function Build-Libopus([string]$name, [string[]]$extra) {
    $prefix = Join-Path $Work $name
    if (Test-Path (Join-Path $prefix 'lib\opus.lib')) { Write-Host "${name}: already built"; return }
    $src = Join-Path $Work "src-$name"
    $bld = Join-Path $Work "build-$name"
    Remove-Item -Recurse -Force -Path $src, $bld -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    & tar -xzf $Tarball -C $src --strip-components=1
    if ($LASTEXITCODE -ne 0) { Die "unpacking $Tarball failed" }
    $cmakeArgs = @('-S', $src, '-B', $bld, '-G', $Generator, '-A', 'x64',
        "-DCMAKE_INSTALL_PREFIX=$prefix", '-DOPUS_BUILD_SHARED_LIBRARY=OFF',
        '-DOPUS_BUILD_TESTING=OFF', '-DOPUS_BUILD_PROGRAMS=OFF',
        # configure turns FLOAT_APPROX on for x86; libopus's CMake files do not.
        '-DOPUS_FLOAT_APPROX=ON') + $extra
    & cmake @cmakeArgs > (Join-Path $Out "cmake-$name.log") 2>&1
    if ($LASTEXITCODE -ne 0) { Die "configuring libopus ($name) failed; see cmake-$name.log" }
    & cmake --build $bld --config Release --parallel > (Join-Path $Out "build-$name.log") 2>&1
    if ($LASTEXITCODE -ne 0) { Die "building libopus ($name) failed; see build-$name.log" }
    & cmake --install $bld --config Release > $null 2>&1
    if (-not (Test-Path (Join-Path $prefix 'lib\opus.lib'))) { Die "no $prefix\lib\opus.lib after the build ($name)" }
    Write-Host "${name}: built in $prefix"
}

Step 'building libopus (default)'
Build-Libopus 'libopus-default' @()
if (-not $NoDnn) {
    Step 'building libopus with deep PLC, OSCE and DRED (takes a few minutes)'
    Build-Libopus 'libopus-dnn' @('-DOPUS_DEEP_PLC=ON', '-DOPUS_OSCE=ON', '-DOPUS_DRED=ON', '-DCMAKE_C_FLAGS_INIT=/DENABLE_OSCE_BWE')
}

$env:RUST_MIN_STACK = '33554432'

# The RFC 8251 test vectors (74.6 MB), for the conformance tests and the
# quality checks below. Skipped with -Quick or when OPUS1_CORPUS is set.
if (-not $env:OPUS1_CORPUS -and -not $Quick) {
    Step 'RFC 8251 test vectors'
    $vt = Join-Path $Work 'opus_testvectors-rfc8251.tar.gz'
    $vdir = Join-Path $Work 'corpus\rfc8251'
    if (-not (Test-Path $vt)) {
        try { Invoke-WebRequest -UseBasicParsing -Uri 'https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz' -OutFile $vt } catch { Write-Host 'download of the test vectors failed; conformance tests will skip' }
    }
    if ((Test-Path $vt) -and ((Get-FileHash -Algorithm SHA256 -Path $vt).Hash -eq '6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55')) {
        New-Item -ItemType Directory -Force -Path $vdir | Out-Null
        if (-not (Test-Path (Join-Path $vdir 'opus_newvectors'))) { & tar -xzf $vt -C $vdir }
        $env:OPUS1_CORPUS = Join-Path $Work 'corpus'
        Write-Host "test vectors in $vdir"
    } else { Write-Host 'test vectors missing or with a wrong SHA-256; conformance tests will skip' }
}
$env:LIBOPUS_NO_PKG = '1'
$env:LIBOPUS_STATIC = '1'

# One cargo test run, with a summary of each test program.
function Run-Tests([string]$name, [string]$prefix, [string]$cargoArgs) {
    $env:LIBOPUS_LIB_DIR = $prefix
    $env:CARGO_TARGET_DIR = Join-Path $Repo "target\x86w-$name"
    $log = Join-Path $Out "$name.txt"
    Set-Content -Path $log -Value "cargo test --release --no-fail-fast $cargoArgs"
    cmd /c "cargo test --release --no-fail-fast $cargoArgs >> `"$log`" 2>&1"
    Add-Content -Path $log -Value "exit status: $LASTEXITCODE"
    $p = 0; $f = 0; $i = 0; $prog = ''
    $lines = Get-Content $log
    if ($lines | Where-Object { $_ -match '^error(\[E\d+\])?: (could not compile|failed to run custom build command)|^error\[E\d+\]' }) {
        Note "${name}: compile failed (see $name.txt)"
        return
    }
    $per = foreach ($l in $lines) {
        if ($l -match 'Running (\S+)') { $prog = $Matches[1] }
        elseif ($l -match '^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored') {
            $p += [int]$Matches[1]; $f += [int]$Matches[2]; $i += [int]$Matches[3]
            "${prog}: $l"
        }
    }
    $per | Set-Content -Path (Join-Path $Out "$name-summary.txt")
    $lines | Where-Object { $_ -match '^test .* FAILED$' } | Sort-Object -Unique | Set-Content -Path (Join-Path $Out "$name-failed.txt")
    Note "${name}: passed $p, failed $f, ignored $i"
}

Step 'test suite, default features (several minutes)'
Run-Tests 'default' (Join-Path $Work 'libopus-default') ''

Step 'neural network kernels: every x86 variant against the portable code'
Run-Tests 'dnn-kernels' (Join-Path $Work 'libopus-default') '--features osce-bwe,dred --lib --test dnn_edges --test dnn_props --test dnn_weights'

if (-not $NoDnn) {
    Step 'test suite with the neural network features, against libopus built with them'
    Run-Tests 'dnn' (Join-Path $Work 'libopus-dnn') '--features test-libopus-dnn,osce-bwe,dred'
}

$env:LIBOPUS_LIB_DIR = Join-Path $Work 'libopus-default'
$env:CARGO_TARGET_DIR = Join-Path $Repo 'target\x86w-default'

Step 'quality on x86: RFC 8251 conformance and the paired comparison with libopus'
# On x86 opus1 does not aim at libopus's bits; these check that the audio is
# right: the conformance limits of opus_compare, and opus1's coding error
# against libopus's on the same speech and music (examples/eval gates).
$clog = Join-Path $Out 'conformance.txt'
cmd /c "cargo run --release --example eval -- conformance > `"$clog`" 2>&1"
Note "conformance: exit status $LASTEXITCODE"
$qtxt = Join-Path $Out 'quality-quick.txt'; $qjson = Join-Path $Out 'quality-quick.json'; $qlog = Join-Path $Out 'quality-quick.log'
cmd /c "cargo run --release --example eval -- run --set quick --text `"$qtxt`" --json `"$qjson`" > `"$qlog`" 2>&1"
if (Test-Path $qtxt) { Note ("quality (quick set): " + ((Get-Content $qtxt | Select-String '^gates') -join ' ')) } else { Note 'quality (quick set): no report, see quality-quick.log' }

Step 'golden file of 64-bit ARM'
$golden = Join-Path $Repo 'scripts\x86\golden_r10.txt'
if (Test-Path $golden) {
    $glog = Join-Path $Out 'golden.txt'
    cmd /c "cargo run --release --example perf_harness -- golden --check `"$golden`" > `"$glog`" 2>&1"
    Note ("golden: " + (Get-Content $glog | Select-Object -Last 1))
} else {
    Note 'golden: no scripts\x86\golden_r10.txt in this copy; skipped'
}

# The golden file holds the output of an earlier run, so on x86 it can only
# show that the output changed. This step runs opus1 and libopus on the
# same input instead: decoder agreement in dB, and the coding error of each
# encoder with libopus decoding (see the legend of modes.md).
Step 'live comparison with libopus (perf_harness modes)'
$mmd = Join-Path $Out 'modes.md'; $mlog = Join-Path $Out 'modes.log'
cmd /c "cargo run --release --example perf_harness -- modes --md `"$mmd`" > `"$mlog`" 2>&1"
$mcode = $LASTEXITCODE
$attention = 0
if (Test-Path $mmd) {
    $tail = (Get-Content $mmd) -join "`n"
    $at = $tail.IndexOf('need attention')
    if ($at -ge 0) { $attention = ([regex]::Matches($tail.Substring($at), '(?m)^- ')).Count }
}
Note "modes: exit status $mcode, $attention scenarios need attention (see modes.md)"

if ($AbBaseline) {
    $abArgs = @{ Baseline = $AbBaseline; Out = (Join-Path $Out 'ab'); Passes = $Passes }
    if ($Quick) {
        Step "A/B: building perf_harness of the baseline ($AbBaseline) and of this tree (no timing with -Quick)"
        $abArgs.BuildOnly = $true
    } else {
        Step "A/B timing: the baseline ($AbBaseline) against this tree, $Passes runs of each, about $($Passes * 6 + 5) minutes; keep the computer idle"
    }
    & (Join-Path $PSScriptRoot 'ab.ps1') @abArgs
    Note "ab: exit status $LASTEXITCODE"
    $abSummary = Join-Path $Out 'ab\ab-summary.txt'
    if (Test-Path $abSummary) { Get-Content $abSummary | ForEach-Object { Note $_ } }
} elseif (-not $Quick) {
    Step 'timing against libopus (default build), a few minutes; keep the computer idle'
    $md = Join-Path $Out 'bench.md'; $json = Join-Path $Out 'bench.json'; $blog = Join-Path $Out 'bench.log'
    cmd /c "cargo run --release --example perf_harness -- bench --md `"$md`" --json `"$json`" > `"$blog`" 2>&1"
    Note "bench: exit status $LASTEXITCODE"
}

Step 'packing the results'
Stop-Transcript | Out-Null
$zip = "$Out.zip"
Compress-Archive -Path $Out -DestinationPath $zip -Force
Write-Host ''
Get-Content $Summary
Write-Host ''
Write-Host "Send this file back: $zip"
