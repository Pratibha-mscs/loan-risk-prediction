#!/bin/bash
set -euo pipefail

# Full deployment: build images, push to ECR, update ECS services
# Usage: ./deploy.sh <aws-account-id> <aws-region>

ACCOUNT_ID="${1:?Usage: $0 <aws-account-id> <aws-region>}"
REGION="${2:-us-east-1}"
CLUSTER="credit-risk-cluster"

echo "=== Step 1: Build and Push ==="
bash deploy/scripts/build_and_push.sh "${ACCOUNT_ID}" "${REGION}"

echo "=== Step 2: Upload Model ==="
BUCKET=$(aws s3api list-buckets --query "Buckets[?starts_with(Name, 'credit-risk-model')].Name | [0]" --output text)
if [ "${BUCKET}" != "None" ]; then
    bash deploy/scripts/upload_model.sh "${BUCKET}"
fi

echo "=== Step 3: Update ECS Services ==="
aws ecs update-service --cluster "${CLUSTER}" --service credit-risk-api \
    --force-new-deployment --region "${REGION}"
aws ecs update-service --cluster "${CLUSTER}" --service credit-risk-dashboard \
    --force-new-deployment --region "${REGION}"

echo "=== Step 4: Wait for Stability ==="
aws ecs wait services-stable --cluster "${CLUSTER}" \
    --services credit-risk-api credit-risk-dashboard --region "${REGION}"

ALB_DNS=$(aws elbv2 describe-load-balancers \
    --query "LoadBalancers[?LoadBalancerName=='credit-risk-alb'].DNSName | [0]" \
    --output text --region "${REGION}")

echo ""
echo "=== Deployment Complete ==="
echo "Dashboard: http://${ALB_DNS}/"
echo "API:       http://${ALB_DNS}/predict"
echo "Health:    http://${ALB_DNS}/health"
echo "Docs:      http://${ALB_DNS}/docs"
