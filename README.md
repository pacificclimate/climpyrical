# Xtreme Isopleth Mapping Tool - map.xtreme.pcic
 
 A mapping tool for displaying North American design value isopleths.

## Getting started
The following instructions will guide installation and implementation on a local machine. 

### Prerequisites 

#### Software
A stable installation of R is required. The preferred software is [RStudio ](https://www.rstudio.com/products/rstudio/). *map.xtreme.pcic* requires the following external R packages.

`abind` for data array combinations, `akima` for bilinear interpolcation, and `kohonen` for SOM analysis. The *map.xtreme.pcic* will check for these dependencies upon running.
 
To install these packages, within RStudio and the R shell, run
 
 ```
install.packages("abind")
install.packages("akima")
install.packages("kohonen")
 ```

#### Data

Coastline and land data found in support

pr\_24hr\_50yrs 

