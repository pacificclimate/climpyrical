FROM jupyter/scipy-notebook

LABEL maintainer="Nic Annau <nannau@uvic.ca>"

USER root
WORKDIR /home/jovyan

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
COPY ./data/ data/ 

# Set environment variable for spatial detail.
ARG size=50m

# Copy in required geospatial data from natural earth data (user dependent)
ADD https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/${size}/physical/ne_${size}_coastline.zip support/

ADD https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/${size}/physical/ne_${size}_land.zip support/

# Install dependencies from requirements file
ADD requirements.txt /home/jovyan/

USER root

# Install python requirements
RUN pip install --verbose -r requirements.txt
RUN rm /home/jovyan/requirements.txt

# Unzip geospatial data and delete downloaded zip file
RUN unzip -q support/ne_${size}_coastline.zip -d support/ne_${size}_coastline \
    && rm support/ne_${size}_coastline.zip

RUN unzip -q support/ne_${size}_land.zip -d support/ne_${size}_land \
    && rm support/ne_${size}_land.zip


RUN chmod -R 777 /home/*