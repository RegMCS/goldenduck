# Backend - GoldenDuck

This directory contains the backend services for the GoldenDuck project, featuring a PostgreSQL database and a FastAPI service for GARCH modeling.

## Services

- **db**: PostgreSQL 15 database.
- **garch**: FastAPI application serving the GARCH model.
- **flyway**: Database migration tool.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Services

1. Navigate to this directory (if you aren't already here):

   ```bash
   cd backend
   ```

2. Start the services (specifying the env file):

   ```bash
   docker-compose --env-file docker-compose.env up -d --build
   ```

3. Verify:
   - FASTAPI Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

## GARCH Model Design

> [!NOTE]
> This section is a placeholder for the GARCH model design specification.

### Objectives

- [TODO: Define the primary objectives of the GARCH model]

### Mathematical Formulation

[TODO: Insert mathematical formula for the GARCH model here]

### Data Requirements

- Input Data: [TODO: Describe input data source and format]
- Frequency: [TODO: e.g., Daily, Hourly]

### Model Architecture

- **Variance Equation**: [TODO: Describe the variance equation]
- **Distribution**: [TODO: e.g., Normal, Student-t]

### Implementation Details

- Library: `arch` (Python) [Proposed]
- Parameters: p=?, q=?

### Integration

- The model is exposed via the FastAPI service in the `GARCH` folder.
