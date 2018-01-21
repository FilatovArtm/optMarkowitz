#!/bin/bash
cd python run_experiments.py;
cd ../latex && pdflatex paper.tex && cp paper.pdf ../results/;
