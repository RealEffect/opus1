# Times two builds of opus1 in turns on this Windows computer, to see how
# much faster or slower a branch is than its baseline. It does what
# scripts/x86/ab.sh does on Linux:
#   A  the baseline, a source tree named with -Baseline;
#   B  the branch, the source tree this script is in.
# Both build examples/perf_harness.rs and link the same libopus, so the
# libopus times are a control: they should not change from A to B.
#
# Why in turns: the clock and the temperature of a computer change during a
# session. A run of A and a run of B one right after the other see the same
# state, so each such pair gives one comparison. The order turns from pair
# to pair (A B, then B A), so a slow drift does not favour one build. With
# 5 passes the order is A B B A A B B A A B. One run is the bench set of
# perf_harness: about 3 minutes on an Intel Core i7-12700.
#
# It writes into the output folder:
#   system.txt     CPU, vector instruction sets, Windows version, compilers
#   builds.txt     the two source trees, their commits and programs
#   build-A.log, build-B.log, build-report.log   the output of the builds
#   runs\pNN-A.*, runs\pNN-B.*   bench records of pass NN (.json, .md, .log)
#   runs.txt       order, start, duration and exit status of each run
#   ab-report.md   per scenario and as geometric means: time of B / time of
#                  A for opus1 encode and decode and for libopus (the
#                  control); see scripts\x86\ab_report.rs
#   ab-summary.txt the geometric means in a few lines
#   pairs.md       the same pairs as `perf_harness pairs` prints them
#
# Usage, in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\x86\ab.ps1 -Baseline DIR
#       [-Passes N] [-Out DIR] [-BuildOnly] [-BenchArgs '--filter celt_fb --reps 3']
#
#   -Baseline   the source tree of the baseline (A)
#   -Passes     runs of each build (default 5)
#   -Out        output folder (default: ab-results-<date> in the root of
#               this tree)
#   -BuildOnly  build the two programs and the report tool, check that they
#               start, and stop without timing
#   -BenchArgs  passed to `perf_harness bench`
#
# scripts\x86\test.ps1 runs this script when it gets -AbBaseline DIR.
#
# Settings (environment variables):
#   LIBOPUS_LIB_DIR  a libopus build: a folder with lib\opus.lib. Needed;
#                    test.ps1 sets it to its own build.
param(
    [string]$Baseline = '',
    [int]$Passes = 5,
    [string]$Out = '',
    [switch]$BuildOnly,
    [string]$BenchArgs = ''
)

$ErrorActionPreference = 'Continue'

function Die([string]$text) {
    Write-Host "error: $text"
    exit 1
}

if ($Passes -lt 1) { Die '-Passes needs a whole number of at least 1' }
if (-not $Baseline) { Die 'name the source tree of the baseline with -Baseline; see the header of this script' }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$resolved = Resolve-Path -LiteralPath $Baseline -ErrorAction SilentlyContinue
if (-not $resolved) { Die "no folder $Baseline" }
$Base = $resolved.Path
if ($Base -eq $Repo) { Die 'the baseline folder is this tree' }
foreach ($tree in @($Base, $Repo)) {
    if (-not (Test-Path (Join-Path $tree 'examples\perf_harness.rs'))) {
        Die "$tree is not an opus1 source tree (no examples\perf_harness.rs)"
    }
}
if (-not $env:LIBOPUS_LIB_DIR) { Die 'LIBOPUS_LIB_DIR is not set (the folder of a libopus build)' }
if (-not $env:LIBOPUS_NO_PKG) { $env:LIBOPUS_NO_PKG = '1' }
if (-not $env:LIBOPUS_STATIC) { $env:LIBOPUS_STATIC = '1' }
foreach ($tool in @('cargo', 'rustc')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Die "$tool is missing" }
}

if (-not $Out) { $Out = Join-Path $Repo "ab-results-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }
$RunsDir = Join-Path $Out 'runs'
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null
$Out = (Resolve-Path $Out).Path
$RunsDir = Join-Path $Out 'runs'
$BinDir = Join-Path $Repo 'target\ab-bin'
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# The commit of a source tree: the file COMMIT of a kit, or git.
function Get-Commit([string]$tree) {
    $file = Join-Path $tree 'COMMIT'
    if (Test-Path $file) { return "$((Get-Content $file -First 1))".Trim() }
    $head = & git -C $tree rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $head) { return "$head".Trim() }
    return 'unknown'
}
$CommitA = Get-Commit $Base
$CommitB = Get-Commit $Repo

Write-Host ':: system'
if (-not ('Opus1.Cpu' -as [type])) {
    Add-Type -Namespace Opus1 -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);'
}
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$info = @()
$info += "date: $((Get-Date).ToUniversalTime().ToString('s'))Z"
$info += "windows: $([Environment]::OSVersion.VersionString)"
$info += "cpu: $($cpu.Name)"
$info += "cores: $($cpu.NumberOfCores), logical cpus: $([Environment]::ProcessorCount), base clock: $($cpu.MaxClockSpeed) MHz"
$memory = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$info += "memory: $([math]::Round($memory / 1GB, 1)) GiB"
$features = [ordered]@{ 'sse3' = 13; 'ssse3' = 36; 'sse4_1' = 37; 'sse4_2' = 38; 'avx' = 39; 'avx2' = 40; 'avx512f' = 41 }
foreach ($f in $features.Keys) {
    $has = [Opus1.Cpu]::IsProcessorFeaturePresent($features[$f])
    $info += "feature ${f}: $(if ($has) { 'yes' } else { 'no' })"
}
$plan = & powercfg /getactivescheme 2>$null | Select-Object -First 1
$info += "power plan: $plan"
$hostLine = "$(& rustc -vV 2>$null | Select-String '^host:')".Trim()
$info += "rustc: $(& rustc -V 2>$null) ($hostLine)"
$info += "cargo: $(& cargo -V 2>$null)"
$info += "libopus: $env:LIBOPUS_LIB_DIR"
$info += "baseline (A): $Base, commit $CommitA"
$info += "branch (B): $Repo, commit $CommitB"
$info | Set-Content -Path (Join-Path $Out 'system.txt')
$info | ForEach-Object { Write-Host $_ }

# Builds perf_harness of a tree in its own target folder and copies it to
# $BinDir\perf_harness_NAME.exe.
$Builds = Join-Path $Out 'builds.txt'
New-Item -ItemType File -Force -Path $Builds | Out-Null
function Build-Harness([string]$name, [string]$tree) {
    Write-Host ":: building $name ($tree)"
    $targetDir = Join-Path $Repo "target\x86w-ab-$name"
    $log = Join-Path $Out "build-$name.log"
    # The caller's CARGO_TARGET_DIR comes back after the build.
    $savedTargetDir = $env:CARGO_TARGET_DIR
    $env:CARGO_TARGET_DIR = $targetDir
    Push-Location $tree
    cmd /c "cargo build --release --example perf_harness > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Pop-Location
    $env:CARGO_TARGET_DIR = $savedTargetDir
    if ($code -ne 0) { Die "the build of $name failed; see $log" }
    $exe = Join-Path $targetDir 'release\examples\perf_harness.exe'
    if (-not (Test-Path $exe)) { Die "no $exe after the build" }
    Copy-Item -Force -Path $exe -Destination (Join-Path $BinDir "perf_harness_$name.exe")
    Add-Content -Path $Builds -Value "${name}: $tree, commit $(Get-Commit $tree), perf_harness.exe $((Get-Item $exe).Length) bytes"
}
Build-Harness 'A' $Base
Build-Harness 'B' $Repo
Write-Host ':: building the report tool'
$reportSource = Join-Path $Repo 'scripts\x86\ab_report.rs'
$reportLog = Join-Path $Out 'build-report.log'
Push-Location $BinDir
cmd /c "rustc --edition 2024 -O -o ab_report.exe `"$reportSource`" > `"$reportLog`" 2>&1"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Die "the report tool did not build; see $reportLog" }

# The programs are called by their bare names from $BinDir: cmd /c then
# needs no quotes around the program.
Push-Location $BinDir
foreach ($name in @('A', 'B')) {
    cmd /c "perf_harness_$name.exe list > NUL 2>&1"
    if ($LASTEXITCODE -ne 0) { Pop-Location; Die "perf_harness_$name.exe does not start" }
}
cmd /c "ab_report.exe --help > NUL 2>&1"
if ($LASTEXITCODE -ne 0) { Pop-Location; Die 'the report tool does not start' }
Pop-Location
Get-Content $Builds | ForEach-Object { Write-Host $_ }

if ($BuildOnly) {
    Write-Host 'built both programs and the report tool; no timing (-BuildOnly)'
    exit 0
}

# The order of the runs: A B in odd passes, B A in even ones.
$orders = @()
for ($pass = 1; $pass -le $Passes; $pass++) {
    if ($pass % 2 -eq 1) { $orders += , @('A', 'B') } else { $orders += , @('B', 'A') }
}
$orderText = ($orders | ForEach-Object { $_ -join ' ' }) -join ' '
Write-Host ":: timing in the order $orderText; keep the computer idle"
$RunsTable = Join-Path $Out 'runs.txt'
Set-Content -Path $RunsTable -Value @('| pass | build | start (UTC) | seconds | exit status |', '|---:|---|---|---:|---:|')
$pairFiles = @()
for ($pass = 1; $pass -le $Passes; $pass++) {
    $p = 'p{0:D2}' -f $pass
    foreach ($name in $orders[$pass - 1]) {
        $file = Join-Path $RunsDir "$p-$name"
        $start = Get-Date
        Write-Host "pass $pass of ${Passes}, ${name}: started $($start.ToString('HH:mm:ss'))"
        Push-Location $BinDir
        cmd /c "perf_harness_$name.exe bench --json `"$file.json`" --md `"$file.md`" $BenchArgs > `"$file.log`" 2>&1"
        $code = $LASTEXITCODE
        Pop-Location
        $seconds = [int]((Get-Date) - $start).TotalSeconds
        Add-Content -Path $RunsTable -Value "| $pass | $name | $($start.ToUniversalTime().ToString('HH:mm:ss')) | $seconds | $code |"
        if ($code -ne 0) { Write-Host "pass $pass, ${name}: exit status $code; see $file.log" }
    }
    $jsonA = Join-Path $RunsDir "$p-A.json"
    $jsonB = Join-Path $RunsDir "$p-B.json"
    if ((Test-Path $jsonA) -and (Test-Path $jsonB)) {
        $pairFiles += $jsonA
        $pairFiles += $jsonB
    } else {
        Write-Host "pass ${pass}: a bench record is missing; this pair is left out of the report"
    }
}
if ($pairFiles.Count -eq 0) { Die 'no pass has both bench records' }

Write-Host ':: report'
$quoted = ($pairFiles | ForEach-Object { "`"$_`"" }) -join ' '
$report = Join-Path $Out 'ab-report.md'
$summary = Join-Path $Out 'ab-summary.txt'
$pairs = Join-Path $Out 'pairs.md'
$reportErrors = Join-Path $Out 'report-errors.log'
Push-Location $BinDir
cmd /c "ab_report.exe --a `"baseline, commit $CommitA`" --b `"branch, commit $CommitB`" --order `"$orderText`" --summary `"$summary`" $quoted > `"$report`" 2> `"$reportErrors`""
$code = $LASTEXITCODE
cmd /c "perf_harness_B.exe pairs $quoted > `"$pairs`" 2>&1"
Pop-Location
if ($code -ne 0) { Die "the report tool failed; see $reportErrors" }
Get-Content $summary | ForEach-Object { Write-Host $_ }
Write-Host "report: $report"
exit 0
