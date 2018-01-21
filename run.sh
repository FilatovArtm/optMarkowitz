#!/bin/bash
python3 run_experiments.py;
../usr/bin/pdflatex paper.tex && cp paper.pdf ../results/;
