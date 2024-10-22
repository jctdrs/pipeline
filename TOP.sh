#!/usr/bin/bash
while pgrep 'pipeline' > /dev/null; do
  top -b -n 1 | grep 'pipeline' | awk '{print $6}'
  sleep 1
done
