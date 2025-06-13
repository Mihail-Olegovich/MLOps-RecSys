"""Run API server with proper environment configuration."""

import sys
from pathlib import Path


def setup_environment() -> None:
    """Setup environment variables and paths."""
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))


def main() -> None:
    """Main function to run the API server."""
    setup_environment()

    import uvicorn

    from service.api.app import app
    from service.api.config import settings

    print("Starting MLOps RecSys API Server")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Reload: {settings.reload}")
    print(f"Log Level: {settings.log_level}")

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("Warning: data/ directory not found")
        print("   Run data preparation first or ensure data files are available")

    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        print("Warning: models/ directory not found")
        print("   Train models first or ensure model files are available")

    print("\nAPI Documentation available at:")
    print(f"   - Swagger UI: http://{settings.host}:{settings.port}/docs")
    print(f"   - ReDoc: http://{settings.host}:{settings.port}/redoc")
    print(f"   - Health Check: http://{settings.host}:{settings.port}/health")

    print("\nStarting server...")

    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
