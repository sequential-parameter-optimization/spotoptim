#!/bin/bash
cd /home/objective/objective_server
./venv/bin/gunicorn -D \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --access-logfile gunicorn_access.log \
    --error-logfile gunicorn_error.log \
    main:app
