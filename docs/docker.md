# Docker Configuration

This project uses Docker for containerization, providing consistent development and production environments.

## Docker Images

The project has a multi-stage Dockerfile that creates two types of images:

1. **Development Image** - Includes all dependencies, including development tools.
2. **Production Image** - Contains only the necessary production dependencies.

## Using Docker Compose

### Development Environment

To start a development environment with Docker:

```bash
docker-compose up dev
```

This will:
- Build the development Docker image
- Mount your local directory to the container
- Start a bash shell for development work

### Running the Application

To run the application in a production-like environment:

```bash
docker-compose up app
```
