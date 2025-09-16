# YAPAT Setup Instructions

This document describes how to set up and run the YAPAT (Yet Another PAM Annotation Tool) application using Docker.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git to clone the repository

## Initial Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yapat-app/yapat.git
   cd yapat
   ```

2. **Set Up Environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Create necessary directories
   mkdir -p data/audio data/embeddings models instance
   
   # Copy environment file to instance directory
   cp .env instance/
   
   # Set proper permissions
   chmod -R 777 instance/
   touch instance/user.db instance/pipeline.db
   chmod 666 instance/user.db instance/pipeline.db
   ```

3. **Configure Environment Variables**
   
   Edit the `.env` file to set your specific configuration:
   ```env
   HOST=0.0.0.0
   PORT=1050
   DEBUG=true  # Set to false in production
   SECRET_KEY=your-secure-secret-key
   API_KEY=your-api-key
   ```

## Running the Application

1. **Build and Start the Container**
   ```bash
   # Build and start the application
   docker-compose up --build
   
   # To run in detached mode (background)
   docker-compose up -d
   ```

2. **Verify the Application**
   - The application should be available at `http://localhost:1050`
   - Default login credentials:
     - Username: admin
     - Password: admin

3. **Stop the Application**
   ```bash
   # If running in foreground, use Ctrl+C
   # If running in background:
   docker-compose down
   ```

## Directory Structure

- `data/audio/`: Store audio files for processing
- `data/embeddings/`: Store generated embeddings
- `models/`: Store ML models
- `instance/`: Contains database files and runtime configurations

## Troubleshooting

1. **Database Access Issues**
   - Verify permissions on instance directory and database files
   - Ensure the .env file is present in both root and instance directories

2. **Port Conflicts**
   - If port 1050 is already in use, modify the PORT variable in .env

3. **Container Won't Start**
   - Check logs: `docker-compose logs`
   - Verify all required directories exist with proper permissions
   - Ensure all required environment variables are set

## Development Setup

For development work, you might want to mount the source code directory:

```yaml
# Add to docker-compose.yml volumes:
- ./src:/app/src
```

## Production Deployment Notes

1. Set `DEBUG=false` in .env
2. Use strong, unique values for SECRET_KEY and API_KEY
3. Consider using a proper reverse proxy (nginx/apache) in front of the application
4. Configure proper backup strategy for the instance directory
5. Ensure proper security measures are in place for the Docker environment
