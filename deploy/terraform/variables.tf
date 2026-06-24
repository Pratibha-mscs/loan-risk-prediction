variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "credit-risk"
}

variable "environment" {
  default = "production"
}

variable "db_username" {
  default   = "credit_risk_admin"
  sensitive = true
}

variable "db_password" {
  sensitive = true
}

variable "db_name" {
  default = "loan_risk_db"
}

variable "api_container_port" {
  default = 8000
}

variable "dashboard_container_port" {
  default = 8501
}

variable "api_cpu" {
  default = 512
}

variable "api_memory" {
  default = 1024
}

variable "dashboard_cpu" {
  default = 256
}

variable "dashboard_memory" {
  default = 512
}
