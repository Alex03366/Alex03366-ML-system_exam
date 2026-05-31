terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.140"
    }
  }
  required_version = ">= 1.0"
}

provider "yandex" {
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
}

# сервисный аккаунт для всех компонентов
resource "yandex_iam_service_account" "ml_sa" {
  name        = "ml-pipeline-sa"
  description = "сервисный аккаунт для ml пайплайна"
}

# роль editor для сервисного аккаунта
resource "yandex_resourcemanager_folder_iam_member" "ml_sa_editor" {
  folder_id = var.folder_id
  role      = "editor"
  member    = "serviceAccount:${yandex_iam_service_account.ml_sa.id}"
}

# ключ доступа для сервисного аккаунта
resource "yandex_iam_service_account_static_access_key" "ml_sa_key" {
  service_account_id = yandex_iam_service_account.ml_sa.id
}

# object storage — бакет для данных и артефактов
resource "yandex_storage_bucket" "data_bucket" {
  bucket     = "churn-model-data-${var.folder_id}"
  access_key = yandex_iam_service_account_static_access_key.ml_sa_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.ml_sa_key.secret_key

  anonymous_access_flags {
    read        = false
    list        = false
    config_read = false
  }
}

# виртуальная машина для mlflow
resource "yandex_compute_instance" "mlflow_vm" {
  name        = "mlflow-server"
  platform_id = "standard-v3"
  zone        = var.zone

  resources {
    cores         = 2
    memory        = 4
    core_fraction = 50
  }

  boot_disk {
    initialize_params {
      image_id = "fd8emfm7o1bsjq5kevoc"  # ubuntu 22.04 lts
      size     = 30
      type     = "network-ssd"
    }
  }

  network_interface {
    subnet_id = yandex_vpc_subnet.ml_subnet.id
    nat       = true
  }

  metadata = {
    ssh-keys = "ubuntu:${file(var.ssh_public_key_path)}"
  }
}

# виртуальная машина для airflow
resource "yandex_compute_instance" "airflow_vm" {
  name        = "airflow-server"
  platform_id = "standard-v3"
  zone        = var.zone

  resources {
    cores         = 4
    memory        = 8
    core_fraction = 50
  }

  boot_disk {
    initialize_params {
      image_id = "fd8emfm7o1bsjq5kevoc"  # ubuntu 22.04 lts
      size     = 50
      type     = "network-ssd"
    }
  }

  network_interface {
    subnet_id = yandex_vpc_subnet.ml_subnet.id
    nat       = true
  }

  metadata = {
    ssh-keys = "ubuntu:${file(var.ssh_public_key_path)}"
  }
}

# облачная сеть
resource "yandex_vpc_network" "ml_network" {
  name = "ml-network"
}

# подсеть
resource "yandex_vpc_subnet" "ml_subnet" {
  name           = "ml-subnet"
  zone           = var.zone
  network_id     = yandex_vpc_network.ml_network.id
  v4_cidr_blocks = ["10.10.0.0/24"]
}

# container registry для docker-образов
resource "yandex_container_registry" "ml_registry" {
  name = "ml-models-registry"
}

# serverless containers для сервинга модели
resource "yandex_serverless_container" "churn_api" {
  name        = "churn-prediction-api"
  description = "api для прогнозирования оттока"
  memory      = 512

  image {
    url = "${yandex_container_registry.ml_registry.id}/churn-model:latest"
  }
}
