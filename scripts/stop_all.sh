#!/bin/bash
for pidfile in logs/*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped $(basename "$pidfile" .pid) (PID $pid)"
        fi
        rm "$pidfile"
    fi
done
echo "All services stopped."
