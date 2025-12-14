#!/bin/bash
# Stop the objective server (gunicorn)
echo "Stopping the objective server..."
pkill -f "gunicorn.*uvicorn.workers.UvicornWorker.*main:app" || echo "Server was not running."
