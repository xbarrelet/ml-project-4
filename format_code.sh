#!/bin/bash

for file in $(ls *.py | grep -v BAK); do
  echo "Formatting $file"
  cp $file $file_BAK &&
  autopep8 --in-place --aggressive --aggressive $file
done
