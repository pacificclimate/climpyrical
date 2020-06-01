#!/bin/bash

# Set user library
echo ".libPaths(c( .libPaths(), '~/.' ))" > ".Rprofile"
echo ".libPaths()" > "config.R"
R CMD BATCH "config.R"

# Install packages from requirements list
cat r_requirements.txt | while read line || [[ -n $line ]]; do
    wget "https://cran.r-project.org/src/contrib/$line.tar.gz"
    R CMD INSTALL --library='.' "$line.tar.gz"
    rm "$line.tar.gz"
done
