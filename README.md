Here's a formatted version of your provided content turned into a `README.md`:

---

# End-to-End Chest Cancer Classification using MLflow & DVC

This project demonstrates a complete machine learning pipeline for Chest Cancer Classification using MLflow for experiment tracking and DVC for data versioning. Additionally, the project outlines an AWS CI/CD deployment process using GitHub Actions.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Setup](#project-setup)
  - [Configuration](#configuration)
- [MLflow Workflow](#mlflow-workflow)
  - [Running MLflow UI](#running-mlflow-ui)
  - [Setting up Environment Variables](#setting-up-environment-variables)
- [DVC Workflow](#dvc-workflow)
- [AWS CI/CD Deployment](#aws-cicd-deployment)
  - [Setting Up AWS for Deployment](#setting-up-aws-for-deployment)
- [License](#license)

---

## Project Overview

The goal of this project is to build a machine learning pipeline for classifying chest diseases using MLflow for experiment tracking and DVC for lightweight orchestration of the pipeline. The project also involves using AWS services like EC2 and ECR for deployment using GitHub Actions.

---

## Project Setup

### Configuration

- **Update the configuration files:**
  1. `config.yaml`: Contains general configuration settings.
  2. `secrets.yaml`: (Optional) Contains sensitive information.
  3. `params.yaml`: Used for experiment parameters and hyperparameters.

- **Steps to update the code:**
  1. Update the entity files.
  2. Update the configuration manager in the `src/config` directory.
  3. Update components as required.
  4. Modify and update the pipeline.
  5. Update `main.py` to run the model pipeline.
  6. Update the `dvc.yaml` to orchestrate the DVC pipelines.

---

## MLflow Workflow

MLflow is used for tracking experiments, logging, and tagging models during the development process.

### Running MLflow UI

You can run the MLflow UI to visually track and manage your experiments:

```bash
mlflow ui
```

### Setting up Environment Variables

To connect MLflow to Dagshub for tracking experiments, export the following environment variables:

After setting up the environment, you can run your script:

```bash
python script.py
```

---

## DVC Workflow

DVC (Data Version Control) is used for lightweight orchestration of the pipeline and for tracking dataset versions.

### DVC Commands:

1. **Initialize DVC**:

   ```bash
   dvc init
   ```

2. **Run Pipeline**:

   ```bash
   dvc repro
   ```

3. **Visualize Pipeline DAG**:

   ```bash
   dvc dag
   ```

---

## AWS CI/CD Deployment

This project utilizes AWS services (EC2 and ECR) and GitHub Actions for CI/CD deployment. Below are the steps to set up AWS for deployment.

### Setting Up AWS for Deployment

1. **Login to AWS Console** and create an IAM user with the following permissions:
   - **EC2 Access**: Virtual machine deployment.
   - **ECR Access**: Elastic Container Registry for storing Docker images.

2. **Deployment Steps**:
   - Build a Docker image of the source code.
   - Push the Docker image to ECR.
   - Launch an EC2 instance.
   - Pull the Docker image from ECR into the EC2 instance.
   - Run the Docker image in EC2.

### IAM Policies Required:

1. `AmazonEC2ContainerRegistryFullAccess`
2. `AmazonEC2FullAccess`

---

### AWS Deployment Instructions:

1. **Create ECR Repository**:
   - Create an ECR repo to store the Docker image.
   - Save the repository URI:  
     `566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken`

2. **Create EC2 Instance (Ubuntu)**:
   - Launch an EC2 instance running Ubuntu.

3. **Install Docker on EC2**:

   - Update and install Docker:

   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. **Configure EC2 as a Self-Hosted Runner for GitHub Actions**:
   - Navigate to `Settings > Actions > Runners` in your GitHub repository.
   - Add a new self-hosted runner for EC2 by following the commands provided in GitHub.

5. **Set Up GitHub Secrets** for AWS:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (e.g., `us-east-1`)
   - `AWS_ECR_LOGIN_URI`  
     Example: `566373416292.dkr.ecr.ap-south-1.amazonaws.com`
   - `ECR_REPOSITORY_NAME` (e.g., `lstname in the above url`)

---

