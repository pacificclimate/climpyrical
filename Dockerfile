FROM rocker/rstudio:3.5.1
FROM rocker/geospatial:latest

LABEL maintainer="Nic Annau <nannau@uvic.ca>"

# Install packages
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

COPY data/ /home/rstudio/data/

COPY support/ /home/rstudio/support/

# Set working directory to match local machine
WORKDIR home/rstudio

# Clone git repo 
RUN git init . \
    && git remote add -t \* -f origin https://github.com/pacificclimate/map-xtreme.git \
    && git checkout master
