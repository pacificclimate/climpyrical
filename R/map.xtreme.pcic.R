map.xtreme.pcic <- function(CanRCM4.lens, obs, res=NULL, method=c('eof', 'som')) {
  # MAP.XTREME.PCIC maps design values over North America
  # 
  # Arguments
  # CanRCM4.lens : information data list for CanRCM4 modelled design values over North America
  # obs : data array of observed design values over North America, [lon, lat, data] three columns 
  # res : resolution (in km) of the map
  # method : whether EOF or SOM-based method is employed for mapping
  # 
  # Value
  # rlon : vector of longitude coordinates of the map
  # rlat : vector of latitude coordinates of the map
  # xtreme : data array of the mapped design values
  # sp.basis : data array of the spatial basis functions estimated from CanRCM4 modelled data
  # obs.grid : data array of the gridded observations 
  #
  # Note: the coordinate system is in polar rotated projection for all involved arrays. The projection
  # is "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
  # 
  # 
  # Author : Chao Li at PCIC, University of Victoria, chaoli@uvic.ca
  #
  # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  
  land.mask <- function(lon, lat) {
    # LAND.MASK creates land mask for North America
    # Arguments
    # lon : vector of longitude coordinates for making the land mask
    # lat : vector of latitude coordinates for making the land mask
    # Value
    # a matrix with TRUE for land grid cells
    
    num.lon <- length(lon)
    num.lat <- length(lat)
    
    # all grid cells
    lon <- rep(lon, each=num.lat)
    lat <- rep(lat, times=num.lon)
    pts <- cbind.data.frame(lon, lat)
    pts.idx <- rownames(pts)
    
    # grid cells over land
    nam.coast.shp <- suppressWarnings(coastlineCanRCM4()) 
    coordinates(pts) <- ~lon+lat
    projection(pts) <- proj4string(nam.coast.shp)
    pts.land <- pts[!is.na(over(pts, as(nam.coast.shp, "SpatialPolygons"))),]
    pts.land <- data.frame(pts.land@coords)
    
    # where are the land grid cells 
    mask <- pts.idx %in% rownames(pts.land)
    
    # formate as a matrix
    mask <- matrix(mask, nrow=num.lon, byrow=TRUE)
    
    return(mask)
    
  }
  # = = = = = 
  
  # check if the required external packages are installed
  if (!suppressPackageStartupMessages(require("abind"))) {      # for data array combination
    install.packages("abind")
    suppressPackageStartupMessages(require("abind"))
  }
  
  if (!suppressPackageStartupMessages(require("akima"))) {      # for bilinear interpolcation
    install.packages("akima")
    suppressPackageStartupMessages(require("akima"))
  }
  
  if (!suppressPackageStartupMessages(require("kohonen"))) {    # for SOM analysis
    install.packages("kohonen")
    suppressPackageStartupMessages(require("kohonen"))
  }
  
  # input check
  if (dim(obs)[1]<100) stop("map.xtreme.pcic: observed sample is too small.")
  
  res.library <- c(50, 25, 10, 5, 1)
  if (is.null(res))  res <- 50
  if (length(res)>1) stop("map.xtreme.pcic: mapping resolution must be an integer.")
  res <- res.library[which.min(abs(res.library-res))]
  res.factor <- 50/res
  
  method <- match.arg(method)
  
  # organize (35) maps to (35) vectors
  xtreme <- CanRCM4.lens$xtreme
  rlon <- CanRCM4.lens$rlon
  rlat <- CanRCM4.lens$rlat
  
  xtreme.dim <- dim(xtreme)
  num.rlon <- xtreme.dim[1]
  num.rlat <- xtreme.dim[2]
  num.run <- xtreme.dim[3]
  
  xtreme.matrix <- matrix(NA, nrow=num.run, ncol=(num.rlon*num.rlat))
  for (i in 1:num.run) {
    xtreme.matrix[i, ] <- as.vector(t(xtreme[, , i]))
  }
  
  rlon.matrix <- rep(rlon, each=num.rlat)
  rlat.matrix <- rep(rlat, times=num.rlon)
  
  # mask grid cells with missing values, e.g., oceans
  idx <- is.na(xtreme.matrix[1, ])
  xtreme.matrix <- xtreme.matrix[, !idx]
  rlon.matrix <- rlon.matrix[!idx]
  rlat.matrix <- rlat.matrix[!idx]
  
  # spatial basis function
  if (method=="eof") {
    sp.basis <- t(svd(xtreme.matrix)$v[, 1:4])
  } else {
    sp.basis <- getCodes(som(X=xtreme.matrix, grid=somgrid(2, 2, "hexagonal", "gaussian"), rlen=10000, keep.data=FALSE))
  }
  
  # create the mapping grid
  num.map.rlon <- res.factor*num.rlon
  num.map.rlat <- res.factor*num.rlat
  map.rlon <- seq(from=min(rlon), to=max(rlon), length.out=num.map.rlon)
  map.rlat <- seq(from=min(rlat), to=max(rlat), length.out=num.map.rlat)
  
  # interpolate the spatial basis function
  map.sp.basis <- array(NA, dim=c(num.map.rlon, num.map.rlat, 4))
  for (i in 1:4) {
    map.sp.basis[, , i] <- interp(x=rlon.matrix, y=rlat.matrix, z=sp.basis[i, ], xo=map.rlon, yo=map.rlat, extrap=TRUE)$z
  }
  
  # mask oceans in the mapping grid
  mask <- land.mask(map.rlon, map.rlat)
  mask <- replicate(4, mask, simplify=FALSE)
  mask <- do.call(abind, c(mask, along = 3))
  
  map.sp.basis[!mask] <- NA
  
  # create gridded observations
  dif.map.rlon <- (map.rlon[2]-map.rlon[1])*0.5
  dif.map.rlat <- (map.rlat[2]-map.rlat[1])*0.5
  
  obs.grid <- array(NA, dim=c(num.map.rlon, num.map.rlat))
  for (i in 1:num.map.rlon) {
    for (j in 1:num.map.rlat) {
      if (mask[i, j, 1]) {
        idx1 <- obs[, 1]>=(map.rlon[i]-dif.map.rlon) & obs[, 1]<(map.rlon[i]+dif.map.rlon)
        idx2 <- obs[, 2]>=(map.rlat[j]-dif.map.rlat) & obs[, 2]<(map.rlat[j]+dif.map.rlat)
        obs.grid[i, j] <- mean(obs[idx1&idx2, 3])
      }
    }
  }
  # Note:
  # The grid value is the mean of the observations falling in a grid cell.
  
  # estimate 'temporal' basis function
  idx <- is.na(obs.grid)
  y <- as.matrix(obs.grid[!idx])
  x <- array(NA, dim=c(length(y), 4))
  for (i in 1:4) {
    x.ith <- map.sp.basis[, , i]
    x[, i] <- x.ith[!idx]
  }
  map.tp.basis <- lm(y~x+1)$coefficients
  
  # mapping
  map.xtreme <- map.tp.basis[1]
  for (i in 1:4) {
    map.xtreme <- map.xtreme+map.sp.basis[, , i]*map.tp.basis[i+1]
  }
  
  # output
  return(list(rlon=map.rlon,
              rlat=map.rlat,
              xtreme=map.xtreme,
              sp.basis=map.sp.basis,
              obs.grid=obs.grid))
}


colorpalette <- function(breaks, col=NULL) {
  # COLORPALETTE creates color palette
  # BREAKS : vector of break points between clor levels
  # COL : vector of colors used for creating the color palette
  #
  # Note:
  # There are several different way to manually or automatically create a
  # color vector, for example,
  # [1] mannually create using palettes in, for example, NCL color scales, e.g.,
  #     c('#FE4D01', '#B56A27', '#CE8540', '#E1A664', '#F5CD85', '#F5E09D', '#FFF5B8')
  # [2] automatically create by built-in R functions, e.g.
  #     rainbow(n, alpha=1)
  #     heat.colors(n, alpha=1)
  #     terrain.colors(n, alpha=1)
  #     topo.colors(n, alpha=1)
  #     cm.colors(n, alpha=1)
  #     'n' is the number of colors to be in the palette
  #     'ALPHA' is the alpha transparency, a number in [0, 1]
  # [3] automatically create by functions in 'oce' package, e.g.,
  #     oce.colorsTwo(n, alpha=1)
  #     oce.colorsJet(n, alpha=1)
  #     oce.colorsGebco(n, alpha=1)
  #     oce.colorsPalette(n, alpha=1)
  #     oce.colorsViridis(n, alpha=1)
  #     oce.colorsCDOM(n, alpha=1)
  #     oce.colorsChlorophyll(n, alpha=1)
  #     oce.colorsDensity(n, alpha=1)
  #     oce.colorsFreesurface(n, alpha=1)
  #     oce.colorsOxygen(n, alpha=1)
  #     oce.colorsPAR(n, alpha=1)
  #     oce.colorsPhase(n, alpha=1)
  #     oce.colorsSalinity(n, alpha=1)
  #     oce.colorsTemperature(n, alpha=1)
  #     oce.colorsTurbidity(n, alpha=1)
  #     oce.colorsVelocity(n, alpha=1)
  #     oce.colorsVorticity(n, alpha=1)
  #     'n' is the number of colors to be in the palette
  #     'ALPHA' is the alpha transparency, a number in [0, 1]
  #
  #
  # Author : Chao Li at PCIC, University of Victoria, chaoli@uvic.ca
  #
  # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  
  # check if package 'oce' is installed, if not, install it
  if (!suppressPackageStartupMessages(require("oce"))) {
    install.packages("oce")
    suppressPackageStartupMessages(require("oce"))
  }
  
  # create color palette
  if (is.null(col)) {
    zcol <- oceColorsTemperature(length(breaks)-1)
  } else {
    col.fun <- colorRampPalette(col)
    zcol <- col.fun(length(breaks)-1)
  }
  
  # return created colour palette and breaks
  return(list(col=zcol, breaks=breaks))
}


val2col <- function(val, breaks, col=NULL) {
  # VAL2COL converts values to colors
  # VAL : array of numerical values to be converted
  # BREAKS : vector of break points between color levels
  # COL : vector of colors for labelling the values
  # Note:
  # The color vector can be generated by 'colorpalette'
  
  # check if package 'oce' is installed, if not, install it
  if (!suppressPackageStartupMessages(require("oce"))) {
    install.packages("oce")
    suppressPackageStartupMessages(require("oce"))
  }
  
  if (is.null(col)) {
    col <- oceColorsTemperature(length(breaks)-1)
  }
  
  if((length(breaks)-1) != length(col)) stop("val2col: must have one more breaks than colors.")
  
  # remember dimensions of VAL
  size <- dim(val)
  if (length(size) > 3) stop("val2col: VAL shoud be an array with at most 3 dimensions.")
  val <- as.vector(val)
  
  # convert values to colors
  cutval <- cut(val, breaks=breaks)
  valcol <- col[match(cutval, levels(cutval))]
  
  # recover the data structure
  if (is.null(size)) {
    valcol <- valcol
  } else {
    valcol <- array(valcol, dim=size)
  }
  
  return(valcol)
}


colorbar <- function(breaks, col, horiz=TRUE) {
  # COLORBAR plots a color bar scale
  # BREAKS : vector of break points between color levels
  # COL : vector of colors labelling intervals of breaks
  # HORIZ : whether or not the color bar is placed horizontally
  
  if((length(breaks)-1) != length(col)) stop("colorbar: must have one more breaks than colors.")
  
  # setup the plot
  xaxt <- ifelse(horiz, "s", "n")
  yaxt <- ifelse(horiz, "n", "s")
  if (horiz) {
    ylim <- c(0, 1)
    xlim <- range(breaks)
  } else {
    ylim=range(breaks)
    xlim=c(0, 1)
  }
  plot(1, 1, t="n", ylim=ylim, xlim=xlim, xlab='', ylab='', main='', 
       xaxs="i", yaxs="i", axes=FALSE)  
  
  # plot polygons filled with colors 
  for(i in seq(col)){
    polybd <- c(breaks[i], breaks[i+1], breaks[i+1], breaks[i])
    if (horiz) {
      polygon(polybd, c(0,0,1,1), col=col[i], border=NA)
    } else {
      polygon(c(0,0,1,1), polybd, col=col[i], border=NA)
    }
  }
  
}


coastlineCanRCM4 <- function(coastlineWorld.file=NULL) {
  # coastlineCanRCM4 creates the coastline object of the CanRCM4 domain
  # coastlineWorld.file : directory of the shp file for coastline of the world
  # Note:
  # The coastline object is in polar rotated coordinate system, that is, the
  # coordinate system used in the CanRCM4 simulations.
  #
  # Author : Chao Li at PCIC, University of Victoria, chaoli@uvic.ca
  #
  # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  
  
  # check the availability of the raster package
  if (!suppressPackageStartupMessages(require("raster"))) {
    install.packages("raster")
    suppressPackageStartupMessages(require("raster"))
  }
  
  # check the availability of the maptools package
  if (!suppressPackageStartupMessages(require("maptools"))) {
    install.packages("maptools")
    suppressPackageStartupMessages(require("maptools"))
  }
  
  # check the availability of the rgdal package
  if (!suppressPackageStartupMessages(require("rgdal"))) {
    install.packages("rgdal")
    suppressPackageStartupMessages(require("rgdal"))
  }
  
  # check the availability of the rgeos package
  if (!suppressPackageStartupMessages(require("rgeos"))) {
    install.packages("rgeos")
    suppressPackageStartupMessages(require("rgeos"))
  }
  
  # projections used by coastlineWorld and CanRCM4
  regular <- "+proj=longlat +ellps=WGS84"
  rotated <- "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
  
  # coastlineWorld
  if (is.null(coastlineWorld.file)) coastlineWorld.file <- "./support/ne_110m_land/ne_110m_land.shp"
  
  if (file.exists(coastlineWorld.file)) {
    landWorld <- readShapePoly(coastlineWorld.file)
    projection(landWorld) <- regular
  } else {
    stop(paste(coastlineWorld.file, "does not exist.", sep=' '))
  }
  
  # a polygon around the CanRCM4 domain for cropping coastlineWorld
  rlon <- seq(from=-35, to=35, by=0.01)
  rlat <- seq(from=-29, to=29, by=0.01)
  # Note:
  # The domain of CanRCM4 simulations spans from -34.1 to 34.1 in longitude and from
  # -28.82 to 28.38 in latitude, both in the polar rotated coordinate system.
  
  rbox.top <- cbind(rlon, rep(max(rlat), length(rlon))) 
  rbox.right <- cbind(rep(max(rlon), length(rlat)), rev(rlat)) 
  rbox.bottom <- cbind(rev(rlon), rep(min(rlat), length(rlon))) 
  rbox.left <- cbind(rep(min(rlon), length(rlat)), rlat) 
  rbox <- rbind(rbox.top, rbox.right, rbox.bottom, rbox.left) 
  rbox <- Polygon(rbox) 
  rbox <- Polygons(list(rbox), 1) 
  rbox <- SpatialPolygons(list(rbox))   # create 'box' of class SpatialPolygons
  projection(rbox) <- rotated
  
  # project 'rbox' from rotated to regular system (as the coastline shp file is in regular system)
  box <- spTransform(rbox, CRS(regular))
  
  # crop the coastline in regular system
  landCanRCM4 <- crop(landWorld, box) 
  
  # project back to rotated system
  rlandCanRCM4 <- spTransform(landCanRCM4, CRS(rotated))
  
  return(rlandCanRCM4)
}


CanRCM4image <- function(rlon, rlat, z, breaks, col) {
  # CanRCM4image plots an image over the domain and with the projection of CanRCM4
  # RLON : vector of longitudes corresponding to z matrix (in polar rotated coordinates)
  # RLAT : vector of latitudes corresponding to z matrix (in polar rotated coordinates)
  # Z : matrix to be represented as an image
  # BREAKS : z values for breaks in the color scheme
  # COL : vector of colors corresponding to the breaks
  #
  # Author : Chao Li at PCIC, University of Victoria, chaoli@uvic.ca
  #
  # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  
  
  if (!is.matrix(z)) stop("z must be matrix-like")
  
  size <- dim(z)
  if (length(rlon) != size[1] | length(rlat) != size[2]) {
    stop("Dimensions of RLON, RLAT, and Z must agree.")
  }
  
  if((length(breaks)-1) != length(col)) stop("Must have one more breaks than colors.")
  
  # extent of the NAM domain of CanRCM4 simulations
  xlim <- range(rlon)
  ylim <- range(rlat)
  
  # plot the image field
  plot(x=1, y=1, type="n", xlim=xlim, ylim=ylim, xlab='', ylab='', main='', axes=FALSE,
       xaxs="i", yaxs="i")
  image(x=rlon, y=rlat, z=z, col=col, breaks=breaks, add=TRUE)
}


points.reg2rotated <- function(lon, lat) {
  # points.reg2rotated projects from regular lon-lat coordinates to polar rotated coordinates
  # lon : vector of the logitude 
  # lat : vector of the latitude
  #
  # Author : Chao Li at PCIC, University of Victoria, chaoli@uvic.ca
  #
  # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  
  # check the availability of the maptools package
  if (!suppressPackageStartupMessages(require('maptools'))) {
    install.packages("maptools")
    suppressPackageStartupMessages(require('maptools'))
  }
  
  # projections used by coastlineWorld and CanRCM4
  regular <- "+proj=longlat +ellps=WGS84"
  rotated <- "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
  
  # a polygon around the CanRCM4 domain for cropping coastlineWorld
  xy <- cbind(lon, lat)
  spdf <- SpatialPoints(coords = xy, proj4string = CRS(regular))
  
  # project to rotated system
  xy.rotated <- spTransform(spdf, CRS(rotated))
  
  # get the projected coordinates
  xy.rotated <- coordinates(xy.rotated)
  
  return(xy.rotated)
}

