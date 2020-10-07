function(latlon, z, nx, ny, extrap){

	obj <- spatialProcess(latlon, z, Distance = "rdist.earth", cov.args = list(Covariance="Exponential"), verbose=FALSE, REML=FALSE)
			
	ps <- predictSurface(obj, grid.list = NULL, extrap = extrap, chull.mask = NA,
	        nx = nx, ny = ny, xy = c(1, 2), verbose = FALSE, ZGrid = NULL,
	        drop.Z = FALSE, just.fixed=FALSE)
			rlist <- list('x' = ps$x, 'y' = ps$y, 'z' = ps$z)
	return(rlist)
}