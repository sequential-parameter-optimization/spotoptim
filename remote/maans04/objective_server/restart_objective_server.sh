#!/bin/bash
# Restart the objective server
echo "Restarting the objective server..."
./stop_objective_server.sh
sleep 2
./start_objective_server.sh
