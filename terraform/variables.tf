variable "cloud_id" {
  description = "id облака yandex cloud"
  type        = string
  sensitive   = true
}

variable "folder_id" {
  description = "id каталога yandex cloud"
  type        = string
  sensitive   = true
}

variable "zone" {
  description = "зона доступности"
  type        = string
  default     = "ru-central1-a"
}

variable "ssh_public_key_path" {
  description = "путь к публичному ssh-ключу"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}
