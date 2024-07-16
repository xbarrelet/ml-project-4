#!/bin/bash

cp script.py script_BAK.py &&
autopep8 --in-place --aggressive --aggressive script.py