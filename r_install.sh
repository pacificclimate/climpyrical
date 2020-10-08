#!/bin/bash

# Set user library
mkdir "./r-library"
echo "R_LIBS_USER=./r-library" > ".Renviron"

# Install packages from requirements list
cat r_requirements.txt | while read line || [[ -n $line ]]; do

wget "https://cran.r-project.org/src/contrib/$line.tar.gz"
if [[ "$line" == *"$Archive"* ]]; then
R CMD INSTALL --library='./r-library' "$(echo $line.tar.gz| cut -d'/' -f 3)"
rm "$(echo $line.tar.gz| cut -d'/' -f 3)"
else
R CMD INSTALL --library='./r-library' "$line.tar.gz"
rm "$line.tar.gz"
fi

done
