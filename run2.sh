#!/bin/bash

for j in `seq 0 1 2`; do
    echo "Starting server"
    python3 server2.py $j &
    sleep 3  # Sleep for 3s to give the server enough time to start

    for i in `seq 0 1 2`; do
        echo "Starting client $i"
        python3 client2.py $i &
    done

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

    # Wait for all background processes to complete
    wait
done
