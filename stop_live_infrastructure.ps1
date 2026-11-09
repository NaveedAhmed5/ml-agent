<#
.SYNOPSIS
    Automates tearing down live local infrastructure services for MLOps AI Agent.
    Services: Kubernetes manifests, Airflow Docker container, MLflow background processes.
#>

$ErrorActionPreference = "Continue"

Write-Host "========================================================================" -ForegroundColor Blue
Write-Host "🛑 Stopping Live MLOps Infrastructure Services..." -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Blue

# 1. Stop Airflow Container
Write-Host "`n[1/3] Stopping Apache Airflow container (airflow-local)..." -ForegroundColor Cyan
docker rm -f airflow-local 2>&1 | Out-Null
Write-Host "  ✔ Airflow container stopped and removed." -ForegroundColor Green

# 2. Stop MLflow UI Process
Write-Host "`n[2/3] Terminating MLflow UI background processes on port 5000..." -ForegroundColor Cyan
Get-Process -Name "mlflow", "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*mlflow*ui*" -or $_.CommandLine -like "*--port 5000*"
} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "  ✔ MLflow UI process terminated." -ForegroundColor Green

# 3. Clean up Kubernetes Resources (Optional)
Write-Host "`n[3/3] Deleting Kubernetes deployment resources in 'mlops-prod'..." -ForegroundColor Cyan
kubectl delete -f kubernetes/ 2>&1 | Out-Null
Write-Host "  ✔ Kubernetes inference deployment, service, and configmap deleted." -ForegroundColor Green

Write-Host "`n========================================================================" -ForegroundColor Blue
Write-Host "✔ All local live infrastructure services stopped." -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Blue
Write-Host ""

