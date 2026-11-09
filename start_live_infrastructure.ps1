<#
.SYNOPSIS
    Automates standing up live local infrastructure for MLOps AI Agent demonstrations.
    Services: Kubernetes (mlops-prod namespace, deployment, service, configmap),
              Dockerized FastAPI inference image,
              Apache Airflow (Docker webserver container),
              MLflow Tracking UI (background process).

.DESCRIPTION
    1. Creates 'mlops-prod' Kubernetes namespace (idempotent).
    2. Builds 'mlops-inference-api:1.2.0' Docker image from local Dockerfile.
    3. Applies all Kubernetes manifests in kubernetes/.
    4. Starts Apache Airflow 2.10.2 in Docker on port 8080 (cleans up existing first).
    5. Starts MLflow UI in the background on port 5000.
    6. Displays endpoint URLs and operational status summary.
#>

$ErrorActionPreference = "Continue"

Write-Host "========================================================================" -ForegroundColor Blue
Write-Host "🚀 Starting Live MLOps Infrastructure Services..." -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Blue

# ------------------------------------------------------------------------------
# 1. Create the Kubernetes Namespace (mlops-prod)
# ------------------------------------------------------------------------------
Write-Host "`n[1/5] Ensuring Kubernetes namespace 'mlops-prod' exists..." -ForegroundColor Yellow
$nsCheck = kubectl get namespace mlops-prod 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✔ Namespace 'mlops-prod' already exists." -ForegroundColor Green
} else {
    kubectl create namespace mlops-prod 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✔ Namespace 'mlops-prod' created successfully." -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Note: kubectl command returned code $LASTEXITCODE. Proceeding with deployment..." -ForegroundColor DarkYellow
    }
}

# ------------------------------------------------------------------------------
# 2. Build the FastAPI Docker Image for Kubernetes
# ------------------------------------------------------------------------------
Write-Host "`n[2/5] Building FastAPI Docker image 'mlops-inference-api:1.2.0'..." -ForegroundColor Yellow
docker build -t mlops-inference-api:1.2.0 .
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✔ Docker image 'mlops-inference-api:1.2.0' built successfully." -ForegroundColor Green
} else {
    Write-Host "  ✖ Failed to build Docker image (exit code $LASTEXITCODE). Ensure Docker daemon is running." -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# 3. Apply the Kubernetes Manifests
# ------------------------------------------------------------------------------
Write-Host "`n[3/5] Applying Kubernetes manifests from 'kubernetes/'..." -ForegroundColor Yellow
kubectl apply -f kubernetes/
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✔ Kubernetes ConfigMap, Deployment, and Service applied." -ForegroundColor Green
} else {
    Write-Host "  ⚠ Warning: kubectl apply returned exit code $LASTEXITCODE. Check cluster connectivity." -ForegroundColor DarkYellow
}

# ------------------------------------------------------------------------------
# 4. Start Apache Airflow in Docker (Port 8080)
# ------------------------------------------------------------------------------
Write-Host "`n[4/5] Starting Apache Airflow in Docker (container: airflow-local)..." -ForegroundColor Yellow
# Remove existing container if it exists
docker rm -f airflow-local 2>&1 | Out-Null

docker run -d --name airflow-local -p 8080:8080 -e AIRFLOW__CORE__LOAD_EXAMPLES=False apache/airflow:2.10.2 webserver
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✔ Airflow container 'airflow-local' started in background on port 8080." -ForegroundColor Green
} else {
    Write-Host "  ✖ Failed to start Airflow container (exit code $LASTEXITCODE)." -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# 5. Start the MLflow UI in the Background (Port 5000)
# ------------------------------------------------------------------------------
Write-Host "`n[5/5] Launching MLflow Tracking UI on port 5000..." -ForegroundColor Yellow
try {
    Start-Process -NoNewWindow -FilePath "mlflow" -ArgumentList "ui --port 5000"
    Write-Host "  ✔ MLflow UI process spawned in background." -ForegroundColor Green
} catch {
    Write-Host "  ⚠ 'mlflow' command not found in PATH, trying with python -m mlflow..." -ForegroundColor Yellow
    try {
        Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m mlflow ui --port 5000"
        Write-Host "  ✔ MLflow UI process spawned via Python in background." -ForegroundColor Green
    } catch {
        Write-Host "  ✖ Failed to start MLflow UI: $_" -ForegroundColor Red
    }
}

# ------------------------------------------------------------------------------
# 6. Final Status & Instructions
# ------------------------------------------------------------------------------
Write-Host "`n========================================================================" -ForegroundColor Blue
Write-Host "✅ Live Infrastructure is running!" -ForegroundColor Green
Write-Host "- MLflow: http://localhost:5000" -ForegroundColor Cyan
Write-Host "- Airflow: http://localhost:8080" -ForegroundColor Cyan
Write-Host "- Kubernetes FastAPI Pods are deploying to mlops-prod namespace." -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Blue
Write-Host "`nUseful commands to verify live status:" -ForegroundColor White
Write-Host "  kubectl get pods -n mlops-prod -w" -ForegroundColor Gray
Write-Host "  kubectl get svc -n mlops-prod" -ForegroundColor Gray
Write-Host "  docker ps --filter name=airflow-local" -ForegroundColor Gray
Write-Host "  curl http://localhost:8000/healthz" -ForegroundColor Gray
Write-Host ""

