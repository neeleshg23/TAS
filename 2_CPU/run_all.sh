#!/bin/bash

base_command="python main.py -m r18 -d c10 -c "

for i in {1001..1049}
do
    command="$base_command $i"
    echo "Running command: $command"
    nohup $command &
done