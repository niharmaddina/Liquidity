# Real-Time Stock Forecasting Application

This application predicts stock prices in real-time using a Liquid Neural Network (LNN) with Neural ODEs. It includes a backend server built with FastAPI and a frontend application built with React.js.

## Project Structure

- **backend/**: Contains the backend server code and machine learning model.
- **frontend/**: Contains the frontend React application.
- **docker-compose.yml**: Docker Compose configuration to run both backend and frontend services.
- **README.md**: This documentation file.

## Setup Instructions

### Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- (Optional) NVIDIA GPU and CUDA drivers for GPU acceleration.

### Backend Setup

1. Navigate to the `backend/` directory.
2. Build the Docker image:

   ```bash
   docker build -t stock-backend .
