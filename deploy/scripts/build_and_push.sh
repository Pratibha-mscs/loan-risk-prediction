#!/bin/bash
set -euo pipefail

# Build and push Docker images to ECR
# Usage: ./build_and_push.sh <aws-account-id> <aws-region>

ACCOUNT_ID="${1:?Usage: $0 <aws-account-id> <aws-region>}"
REGION="${2:-us-east-1}"
PROJECT="credit-risk"
TAG="${3:-latest}"

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "=== Authenticating with ECR ==="
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_BASE}"

echo "=== Building API image ==="
docker build -t "${PROJECT}-api:${TAG}" \
    -f Dockerfile \
    --target api \
    .

echo "=== Building Dashboard image ==="
docker build -t "${PROJECT}-dashboard:${TAG}" \
    -f Dockerfile \
    --target dashboard \
    .

echo "=== Tagging and pushing ==="
for SERVICE in api dashboard; do
    docker tag "${PROJECT}-${SERVICE}:${TAG}" \
        "${ECR_BASE}/${PROJECT}-${SERVICE}:${TAG}"
    docker push "${ECR_BASE}/${PROJECT}-${SERVICE}:${TAG}"
    echo "Pushed ${ECR_BASE}/${PROJECT}-${SERVICE}:${TAG}"
done

echo "=== Done ==="
