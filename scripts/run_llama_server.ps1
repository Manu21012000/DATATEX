# Start llama.cpp's OpenAI-compatible HTTP server for the repo's Mistral GGUF.
#
# If `llama-server` is not on PATH, set the full path once (current session or User env):
#   $env:LLAMA_SERVER = "C:\path\to\llama.cpp\build\bin\Release\llama-server.exe"
#
# Optional:
#   LLAMA_EXTRA    — extra args, e.g. "-ngl 99" (GPU) or "-t 8" (threads)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Gguf = Join-Path $ProjectRoot "mistral-7b-instruct-v0.3-q4_k_m.gguf"

if (-not (Test-Path -LiteralPath $Gguf)) {
    Write-Error "GGUF not found: $Gguf"
}

function Resolve-LlamaServerExe {
    $fromEnv = $env:LLAMA_SERVER
    if ($fromEnv -and (Test-Path -LiteralPath $fromEnv)) {
        return (Resolve-Path -LiteralPath $fromEnv).Path
    }
    if ($fromEnv) {
        Write-Warning "LLAMA_SERVER is set but file not found: $fromEnv"
    }

    foreach ($name in @("llama-server.exe", "llama-server")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source) {
            return $cmd.Source
        }
    }

    # Common CMake output locations (sibling clone or folder inside project)
    $candidates = @(
        (Join-Path $ProjectRoot "llama.cpp\build\bin\Release\llama-server.exe")
        (Join-Path $ProjectRoot "llama.cpp\build\bin\llama-server.exe")
        (Join-Path $ProjectRoot "llama.cpp\build\Release\llama-server.exe")
        (Join-Path (Split-Path $ProjectRoot -Parent) "llama.cpp\build\bin\Release\llama-server.exe")
    )
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) {
            Write-Host "Using llama-server at: $p" -ForegroundColor DarkGray
            return (Resolve-Path -LiteralPath $p).Path
        }
    }

    return $null
}

$exe = Resolve-LlamaServerExe
if (-not $exe) {
    Write-Host @"

llama-server was not found.

Build llama.cpp (https://github.com/ggml-org/llama.cpp) with the server target, then either:
  - Add the folder that contains llama-server.exe to your PATH, or
  - Point to the executable explicitly, e.g.:

    `$env:LLAMA_SERVER = 'C:\path\to\llama.cpp\build\bin\Release\llama-server.exe'`
    powershell -ExecutionPolicy Bypass -File scripts\run_llama_server.ps1

Typical Visual Studio CMake build output:
  ...\llama.cpp\build\bin\Release\llama-server.exe

"@ -ForegroundColor Yellow
    exit 1
}

$extra = @()
if ($env:LLAMA_EXTRA) {
    $extra = $env:LLAMA_EXTRA -split '\s+' | Where-Object { $_ }
}

$argList = @(
    "-m", $Gguf,
    "--alias", "mistral-7b-local",
    "--host", "127.0.0.1",
    "--port", "8080",
    "-c", "8192"
) + $extra

Write-Host "Starting: $exe $($argList -join ' ')"
& $exe @argList
