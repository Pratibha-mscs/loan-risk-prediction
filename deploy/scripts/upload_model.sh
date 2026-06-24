#!/bin/bash
set -euo pipefail

# Upload trained model to S3
# Usage: ./upload_model.sh <s3-bucket>

BUCKET="${1:?Usage: $0 <s3-bucket-name>}"
MODEL_DIR="models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Uploading model artifacts to s3://${BUCKET}/ ==="

aws s3 cp "${MODEL_DIR}/best_model.joblib" \
    "s3://${BUCKET}/models/best_model.joblib"

aws s3 cp "${MODEL_DIR}/best_model.joblib" \
    "s3://${BUCKET}/models/archive/best_model_${TIMESTAMP}.joblib"

for MODEL in CatBoost_tuned LightGBM_tuned XGBoost_tuned; do
    if [ -f "${MODEL_DIR}/${MODEL}.joblib" ]; then
        aws s3 cp "${MODEL_DIR}/${MODEL}.joblib" \
            "s3://${BUCKET}/models/${MODEL}.joblib"
        echo "Uploaded ${MODEL}.joblib"
    fi
done

echo "=== Model artifacts uploaded ==="
echo "Latest: s3://${BUCKET}/models/best_model.joblib"
echo "Archive: s3://${BUCKET}/models/archive/best_model_${TIMESTAMP}.joblib"
