#!/bin/bash

# Set user library
mkdir "./r-library"
echo "~/.Renviron" > "R_LIBS_USER=./r-library"

# Install packages from requirements list
cat r_requirements.txt | while read line || [[ -n $line ]]; do
    wget "https://cran.r-project.org/src/contrib/$line.tar.gz"
    R CMD INSTALL --library='./r-library' "$line.tar.gz"
    rm "$line.tar.gz"
done
