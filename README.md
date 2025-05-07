
# Decentralized Smart Grid Energy Demand Forecasting and Transformer Fault Detection using Federated Learning

This project is a Dockerized, full-stack web application designed to perform decentralized energy demand forecasting and transformer fault detection using Federated Learning (FL). The system simulates multiple clients (energy substations or smart meters) uploading datasets to a centralized server where model aggregation, fault detection, and demand forecasting are performed.

The application is deployed on AWS and built with:
- **Frontend**: [Next.js](https://nextjs.org/) (TypeScript)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **ML Frameworks**: PyTorch, Scikit-learn
- **Orchestration**: Docker
- **Deployment**: AWS EC2

---

## 🚀 Features

- **Federated Learning Architecture**: Decentralized training where multiple clients contribute to a global model without sharing raw data.
- **Transformer Fault Detection**: Isolation Forest-based local models identify anomalies in transformer performance.
- **Energy Demand Forecasting**: LSTM-based neural networks predict future energy demand patterns.
- **Client-Server Simulation**: Multiple clients can simulate local training and send model updates to the server.
- **Interactive Dashboards**: Visualize forecasting and fault detection via responsive frontend.
- **Scalable Architecture**: Easily deployable on AWS with Docker containers.

---

## 🛠️ Tech Stack

| Layer       | Technology         |
|------------|--------------------|
| Frontend   | Next.js (TypeScript), Tailwind CSS |
| Backend    | FastAPI, Uvicorn |
| ML         | PyTorch, Sklearn |
| Container  | Docker |
| Deployment | AWS EC2 (Ubuntu), Nginx (optional) |

---

## 🗂️ Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   └── models/
│   └── Dockerfile
├── frontend/
│   ├── pages/
│   ├── components/
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Local Development Setup

### Prerequisites

- Docker & Docker Compose
- Node.js
- Python 3.10+
- AWS CLI (for deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/smartgrid-fl.git
cd smartgrid-fl
```

### Step 2: Environment Configuration

Create `.env` files in both `backend` and `frontend` with the necessary environment variables like database URL, API endpoints, etc.

### Step 3: Build and Run Locally with Docker

```bash
docker-compose up --build
```

Access:
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:8000](http://localhost:8000)

---

## ☁️ AWS Deployment (Dockerized)

### Step 1: Provision EC2

- Launch an EC2 instance (Ubuntu 22.04)
- Allow ports 22 (SSH), 3000 (Frontend), and 8000 (Backend) in security group
- SSH into the instance:
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-dns
```

### Step 2: Install Docker & Docker Compose

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 3: Clone Repo & Deploy

```bash
git clone https://github.com/your-username/smartgrid-fl.git
cd smartgrid-fl
docker-compose up --build -d
```

### Step 4: Optional - Setup Nginx Reverse Proxy

Use Nginx to proxy requests from ports 80/443 to backend/frontend for production environments.

---

## 🤖 Federated Learning Overview

1. **Clients** simulate smart meters or substations.
2. Each client trains a local model (Isolation Forest for fault detection or LSTM for demand forecasting).
3. Clients share gradients/model weights with the server.
4. Server aggregates them into a global model and returns updated weights to clients.
5. Global model's output is visualized in the web interface.

---

## 📈 Visualization & UI

- View transformer fault anomalies
- Compare local and global model predictions
- Monitor energy demand trends

---

## 🧪 Sample Data

- Smart grid energy consumption dataset
- Transformer performance logs with anomaly labels
- Time-series structured CSVs uploaded via the frontend

---

## 🔐 Security Considerations

- CORS handled via FastAPI settings
- Option to enable HTTPS with Let's Encrypt (Nginx)
- Secure API routes with authentication (JWT or OAuth2 - optional)

---

## 📚 Resources & References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Next.js Docs](https://nextjs.org/docs)
- [PySyft - Federated Learning](https://github.com/OpenMined/PySyft)
- [AWS EC2 Docs](https://docs.aws.amazon.com/ec2/)

---

## 🧑‍💻 Authors

- Harshit Sinha

---

## 📄 License

MIT License

---

## 🏁 Future Improvements

- Real-time MQTT client data ingestion
- Support for more FL algorithms (FedAvg, FedProx)
- CI/CD pipeline for auto-deployment
- Kubernetes-based scalability for production

---
