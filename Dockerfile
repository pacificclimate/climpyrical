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
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*

# Copy in required geospatial data (user dependent)
COPY data/ /home/rstudio/data/

# Copy in required geospatial data (user dependent)
COPY support/ /home/rstudio/support/

# Set working directory
WORKDIR home/rstudio

# Clone git repo with R code. Pulls latest commits each build
RUN git init . \
    && git remote add -t \* -f origin https://github.com/pacificclimate/map-xtreme.git \
    && git checkout master
