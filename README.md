![logo.png](https://images.zenhubusercontent.com/5bc02597fcc72f27390ed1f9/c2cf2ba4-edb1-4b47-856e-20338712d4a7)
## Extreme Isopleth Mapping Tool
A mapping tool for displaying North American design value isopleths.

<a href="url"><img src="https://images.zenhubusercontent.com/5bc02597fcc72f27390ed1f9/a07326c9-8e16-4faa-9056-89ebcfdb7c2a" align="left" width="270" ></a>


## Getting started
The following instructions will guide installation and implementation on a local machine, including a stable Docker environment in which to use the software. It is recommended to use this Dockerized implementation, unless the user has confidence in their local environment. These instructions will only cover implementation using Docker images on Ubuntu OS.

### Prerequisites 
The only required software is `Docker`. Carefully choose the correct installation for the OS that you have from the [Docker Community Edition (CE) website](https://docs.docker.com/install/#supported-platforms). Desktop clients are available for both Mac and Windows.

### Deployment
To get started, clone this repository to your local machine.
```
git clone https://github.com/pacificclimate/map-xtreme.git
```

Change directories to cloned repo, i.e. `cd map-xtreme`

The size of physical geospatial data required for plotting, by default is set to 1:50. However, if one requires larger or smaller scales, this can be changed by setting the environment variable in `.env` to either 10m, 50m or 110m as provided by [Natural Earth Data](https://www.naturalearthdata.com/downloads/). To do this, change `.env` to have one of the selected scales. 
```
export size=<10m, 50, 110m>
```
To activate the new scale choice, use 
```source .env``` 

And `echo $size` should display the correct scale selected. 

Within this directory contains the Dockerfile necessary for R environment. If Docker has been installed correctly, build with
```
docker build --build-arg size --rm -t <DOCKER IMAGE NAME> .
```
Where `--build-arg size` sends the host environment variable `size` into the build, `--rm` tells Docker to remove intermediate containers after a successful build, and `-t` will automatically tag the image that was created. `<DOCKER IMAGE NAME>` can be any logical name for the iamge. We specify that the Dockerfile is in the current directory with `.`

Now that the Docker image is built, with name `<DOCKER IMAGE NAME>`, it is time to run the image to create a container.

```
docker run -e PASSWORD="<CHOOSE A PASSWORD>" -p 8787:8787 --rm -it <DOCKER IMAGE NAME>:latest
```

This runs the built docker image in a container. The `-e` flag allows us to set a user defined password as an environment variable (more on that later). The `-p` flag lets Docker make the exposed port accessible on the host. These ports will be available to any client that can reach the host. The `--rm` flag is to automatically clean up the container and remove the file system when the container exits. The `-it` flag makes the container interactive.

If all steps were successful, then the previous command will make an RStudio GUI and interactive session available at 
```
http://localhost:8787
```

You will be prompted to enter a username and password. The default username is `rstudio` and the password is the same that you set while running the docker image, i.e. `<CHOOSE A PASSWORD>`. 

## About the Environment
This Docker environment is a multilayered stack from [rocker](https://hub.docker.com/r/rocker/rstudio/)'s r-ver. This stack contains RStudio 3.5.1 as well as the R geospatial libraries required by map-xtreme.

### Visualization and Data
In addition to CanRCM data (which should be placed in `map-xtreme/data`).

## Authors
* **Chao Li** - *map.xtreme.pcic R software* - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
* **Nic Annau** - *Dockerized implementation of R environment* - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
