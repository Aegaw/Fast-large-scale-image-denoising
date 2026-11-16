# build.ps1 -- 用 MSVC 编译 agf
# 用法:  .\build.ps1 bench_box
#        .\build.ps1 agf
param([string[]]$Targets = @("agf"))

$ErrorActionPreference = "Stop"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found" }

$flags  = "/nologo /O2 /arch:AVX2 /fp:precise /EHsc /std:c++17 /W3 /utf-8 /DNOMINMAX"
$outDir = Join-Path $PSScriptRoot "bin"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$defs = @{
    "agf" = @{ src = @("agf.cpp", "image_io.cpp", "config.cpp", "pipeline.cpp", "main_agf.cpp"); libs = @("gdiplus.lib") }
}

Push-Location $outDir
try {
    foreach ($t in $Targets) {
        if (-not $defs.ContainsKey($t)) { throw "unknown target: $t" }
        $srcs = ($defs[$t].src | ForEach-Object { "`"$(Join-Path $PSScriptRoot $_)`"" }) -join " "
        $libs = $defs[$t].libs -join " "
        Write-Host "== building $t.exe"
        cmd /c "call `"$vcvars`" >nul 2>&1 && cl $flags /Fe:$t.exe $srcs /link $libs"
        if ($LASTEXITCODE -ne 0) { throw "build failed: $t" }
    }
} finally {
    Get-ChildItem "*.obj" -ErrorAction SilentlyContinue | Remove-Item -Force
    Pop-Location
}
Write-Host "done."
