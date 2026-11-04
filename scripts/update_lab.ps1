# Сдвинуть пин субмодуля lab на тег или коммит. Аналог scripts/update_lab.sh.
# Запуск из корня репозитория:
#   powershell -ExecutionPolicy Bypass -File scripts\update_lab.ps1 lab-v0.1.0
param(
    [Parameter(Mandatory = $true)]
    [string]$Ref
)

$ErrorActionPreference = "Stop"
$Root = (git rev-parse --show-toplevel).Trim()
Set-Location $Root

if (-not (Test-Path "lab/.git")) {
    git submodule update --init lab
}

$Old = ""
try {
    $Old = (git rev-parse "HEAD:lab").Trim()
} catch {
    $Old = (git -C lab rev-parse HEAD).Trim()
}
Write-Host "текущий пин: $Old"
Write-Host "запрошено: $Ref"

function Restore-Lab {
    Write-Host "обновление не принято, возвращаю lab на $Old"
    git -C lab checkout --detach $Old
    if ($LASTEXITCODE -ne 0) {
        git -C lab checkout $Old
    }
}

git -C lab fetch --tags origin
if ($LASTEXITCODE -ne 0) { throw "git fetch lab не удался" }
git -C lab checkout --detach $Ref
if ($LASTEXITCODE -ne 0) { throw "не удалось перейти на $Ref" }
$New = (git -C lab rev-parse HEAD).Trim()
Write-Host "lab HEAD: $New"

$env:PELLET_LAB_PATH = Join-Path $Root "lab"
$env:PYTHONPATH = $env:PELLET_LAB_PATH
$env:PELLET_MODEL_DIR = Join-Path $Root "models\v2"
$env:PELLET_LAB_COMMIT = $New

python -m pytest (Join-Path $Root "lab\tests\test_public_api.py") -q --tb=short
if ($LASTEXITCODE -ne 0) { Restore-Lab; exit 1 }

$prod = @(
    "tests\test_v2_parser.py",
    "tests\test_v2_contract.py",
    "tests\test_v2_switch.py",
    "tests\test_v2_auth.py",
    "tests\test_v2_evaluation.py",
    "tests\test_v2_submodule.py"
) | ForEach-Object { Join-Path $Root $_ }
python -m pytest @prod -q --tb=short
if ($LASTEXITCODE -ne 0) { Restore-Lab; exit 1 }

python (Join-Path $Root "scripts\build_v2_artifact.py") --out $env:PELLET_MODEL_DIR
if ($LASTEXITCODE -ne 0) { Restore-Lab; exit 1 }

python (Join-Path $Root "scripts\compare_v2_baseline.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Если смена прогноза осознанная, обновите tests\fixtures\v2_baseline.json и повторите."
    Restore-Lab
    exit 1
}

Set-Content -Path (Join-Path $Root "app\services\ml_v2\PIN") -Value $New -NoNewline -Encoding ascii
Add-Content -Path (Join-Path $Root "app\services\ml_v2\PIN") -Value ""
git add lab app/services/ml_v2/PIN
Write-Host "Пин подготовлен: $Old -> $New"
Write-Host "В индекс добавлены gitlink lab и PIN. Коммит делает владелец."
Write-Host "Откат: git checkout $Old -- lab ; восстановите прежний PIN и снова git add."
