# ==========================================
# Script PowerShell para clonar repositorios
# ==========================================

$repos = @(
    "https://github.com/KimJongSung/jPetStore.git",
    "https://github.com/acmeair/acmeair.git",
    "https://github.com/WASdev/sample.daytrader7.git",
    "https://github.com/WASdev/sample.plantsbywebsphere.git"
)

foreach ($repo in $repos) {
    Write-Host "Clonando $repo ..."
    git clone $repo
    Write-Host "------------------------------------------"
}

Write-Host "✅ Todos los repositorios han sido clonados."

# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# .\clone_repos.ps1
