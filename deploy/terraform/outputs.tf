output "alb_dns_name" {
  description = "Public URL of the application"
  value       = aws_lb.main.dns_name
}

output "api_ecr_url" {
  description = "ECR repository URL for API image"
  value       = aws_ecr_repository.api.repository_url
}

output "dashboard_ecr_url" {
  description = "ECR repository URL for Dashboard image"
  value       = aws_ecr_repository.dashboard.repository_url
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "s3_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.models.id
}
