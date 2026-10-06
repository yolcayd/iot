# Latency benchmark in a Docker container limited to 2 CPUs and 2 GB RAM (docs/DECISIONS.md K-047).
# Requires Docker Desktop to be running. Resumable: targets already measured for the session are skipped.
#
#   .\docker\run_benchmark.ps1 -Session 1
#   .\docker\run_benchmark.ps1 -Session 2
#   python -m src.benchmark --profile container --compare     # on the host afterwards
param(
    [int]$Session = 1,
    [switch]$Force,
    [switch]$NoBuild
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$image = "can-ids-benchmark:1"

docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker engine is not running; start Docker Desktop first." }

$commit = (git -C $repo rev-parse HEAD).Trim()
# generated results are excluded from the dirty check, as in src/benchmark.py
$dirtyFiles = git -C $repo status --porcelain -- . ":(exclude)results/raw" ":(exclude,glob)results/tables/quant_*.csv" ":(exclude,glob)results/tables/benchmark_*.csv" ":(exclude)results/tables/benchmark_manifest.json"
$dirty = if ($dirtyFiles) { "1" } else { "0" }
if ($dirty -eq "1") { throw "Working tree has uncommitted changes (CLAUDE.md rule 7); commit first." }

if (-not $NoBuild) {
    docker build -f (Join-Path $repo "docker\benchmark.Dockerfile") -t $image (Join-Path $repo "docker")
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

$extra = @()
if ($Force) { $extra += "--force" }
docker run --rm --cpus=2 --memory=2g --memory-swap=2g `
    -e GIT_COMMIT=$commit -e GIT_DIRTY=$dirty `
    -v "${repo}:/work:ro" -v "${repo}\results:/work/results" `
    $image --profile container --threads 2 --session $Session @extra
if ($LASTEXITCODE -ne 0) { throw "benchmark failed (exit $LASTEXITCODE)" }
