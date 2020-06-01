#!/bin/bash

# Install packages from requirements list
cat r_requirements.txt | while read line || [[ -n $line ]]; do
    wget "https://cran.r-project.org/src/contrib/$line.tar.gz"
    R CMD INSTALL "$line.tar.gz"
    rm "$line.tar.gz"
done
