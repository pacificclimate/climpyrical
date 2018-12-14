# Stable version of RStudio provided by Rocker (Docker environments for R)
FROM rocker/rstudio:3.5.1
# Multilevel stack to include common geospatial packages requirements
FROM rocker/geospatial:latest

LABEL maintainer="Nic Annau <nannau@uvic.ca>"

# Install packages onto debian base level machine
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
    liblzma-dev \
    libbz2-dev \
    clang  \
    ccache \
    default-jdk \
    default-jre \
    git \
    unzip \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*

# Copy in required geospatial data (user dependent)
COPY data/ /home/rstudio/data/

# Copy in R code
COPY *.R /home/rstudio/

# Set home directory and make user root
USER root
WORKDIR /home/rstudio

# Set environment variable for spatial detail.
ARG size=50m

# Copy in required geospatial data from natural earth data (user dependent)
ADD https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/${size}/physical/ne_${size}_coastline.zip support/

ADD https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/${size}/physical/ne_${size}_land.zip support/

# Unzip geospatial data and delete downloaded zip file
RUN unzip -q support/ne_${size}_coastline.zip -d support/ne_${size}_coastline \
    && rm support/ne_${size}_coastline.zip

RUN unzip -q support/ne_${size}_land.zip -d support/ne_${size}_land \
    && rm support/ne_${size}_land.zip
