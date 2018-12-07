rm(list=ls())
library(ncdf4)
library(abind)

source("map.xtreme.pcic.R")


# CanRCM4 modelled return levels
# = = = = = = = = = = = = = = = = =
ncfile.path <- "./data/pr_24hr_50yrs"
ncfiles <- list.files(ncfile.path, full.names=T)
pr <- c()
for (i in 1:length(ncfiles)) {
  ncfile.ith <- ncfiles[i]
  nc <- nc_open(ncfile.ith)
  pr <- abind(pr, ncvar_get(nc, "pr"), along=3)
  nc_close(nc)
}

nc <- nc_open(ncfiles[1])
rlon <- ncvar_get(nc, "rlon")
rlat <- ncvar_get(nc, "rlat")
nlon <- length(rlon)
nlat <- length(rlat)
nc_close(nc)

CanRCM4.lens <- list(xtreme=pr, rlon=rlon, rlat=rlat)


# pseudo observations
# = = = = = = = = = = = = 
prx <- CanRCM4.lens$xtreme[, , 1]
prx <- as.vector(t(prx))

prx.rlon <- rep(rlon, each=nlat)
prx.rlat <- rep(rlat, times=nlon)

idx <- is.na(prx)
prx <- prx[!idx]
prx.rlon <- prx.rlon[!idx]
prx.rlat <- prx.rlat[!idx]
prx <- cbind(as.matrix(prx.rlon), as.matrix(prx.rlat), as.matrix(prx))
n <- dim(prx)[1]

set.seed(125)
idx.pseudo <- sample(x=1:n, size=ceiling(n*0.02), replace=F)
obs.pseudo <- prx[idx.pseudo, ]


# mapping...
res <- 5
method <- 'som'
mp <- map.xtreme.pcic(CanRCM4.lens=CanRCM4.lens, obs=obs.pseudo, res, method) 


# plot the mapped design values
breaks <- pretty(range(CanRCM4.lens$xtreme[, , 1], na.rm=T), n=10)
col <- c('#e11900', '#ff7d00', '#ff9f00', '#ffc801', '#ffff01', '#c8ff32', '#98ff00',
         '#64ff01', '#00c834', '#009695', '#0065ff', '#3232c8', '#dc00dc', '#ae00b1')
breaks.col <- colorpalette(breaks, col)
breaks <- breaks.col$breaks
colors <- breaks.col$col

coastline <- coastlineCanRCM4()


pdf("Fig_isopleth.pdf", width=3.5, height=4.0)

rlon <- mp$rlon
rlat <- mp$rlat
xtreme <- mp$xtreme

par(fig = c(0.05, 0.95, 0.25, 0.98), mar = c(0.5, 0.0, 0.5, 0.0), cex=1.0, lwd=0.5)
CanRCM4image(rlon=rlon, rlat=rlat, z=xtreme, breaks=breaks, col=colors)
lines(coastline, col='black', lwd=0.5)
box()

par(fig = c(0.05, 0.95, 0.07, 0.20), new = TRUE, mar = c(1.5, 0.0, 0.0, 0.0), cex=1.0, lwd=0.5)
colorbar(breaks=breaks, col=colors)
axis(1, tck=-0.25, mgp=c(0, 0.5, 0), lwd=0.5, lwd.tick=0.5)
mtext("50-year daily precipitation [mm/h]", side=1, line=1.5, cex=1.0)
box()
dev.off()


# other plots of possible interest : in-site observations
pdf("Fig_obs.pdf", width=3.5, height=4.0)

par(fig = c(0.05, 0.95, 0.25, 0.98), mar = c(0.5, 0.0, 0.5, 0.0), cex=1.0, lwd=0.5)
lon <- obs.pseudo[, 1]
lat <- obs.pseudo[, 2]
obs <- obs.pseudo[, 3]
cols <- val2col(obs, breaks, col=colors) 
plot(x=lon, y=lat, type='p', xlab='', ylab='', main='', axes=FALSE, xaxs="i", yaxs="i", 
     xlim=range(lon), ylim=range(lat), pch=20, col=cols, cex=0.8)
lines(coastline, col='black', lwd=0.5)
box()

par(fig = c(0.05, 0.95, 0.07, 0.20), new = TRUE, mar = c(1.5, 0.0, 0.0, 0.0), cex=1.0, lwd=0.5)
colorbar(breaks=breaks, col=colors)
axis(1, tck=-0.25, mgp=c(0, 0.5, 0), lwd=0.5, lwd.tick=0.5)
mtext("50-year daily precipitation [mm/h]", side=1, line=1.5, cex=1.0)
box()
dev.off()


# other plots of possible interest : gridded observations
pdf("Fig_grid.obs.pdf", width=3.5, height=4.0)

rlon <- mp$rlon
rlat <- mp$rlat
obs.grid <- mp$obs.grid

par(fig = c(0.05, 0.95, 0.25, 0.98), mar = c(0.5, 0.0, 0.5, 0.0), cex=1.0, lwd=0.5)
CanRCM4image(rlon=rlon, rlat=rlat, z=obs.grid, breaks=breaks, col=colors)
lines(coastline, col='black', lwd=0.5)
box()

par(fig = c(0.05, 0.95, 0.07, 0.20), new = TRUE, mar = c(1.5, 0.0, 0.0, 0.0), cex=1.0, lwd=0.5)
colorbar(breaks=breaks, col=colors)
axis(1, tck=-0.25, mgp=c(0, 0.5, 0), lwd=0.5, lwd.tick=0.5)
mtext("50-year daily precipitation [mm/h]", side=1, line=1.5, cex=1.0)
box()
dev.off()


# other plots of possible interest : reference where the pseudo observations are generated
pdf("Fig_ref.pdf", width=3.5, height=4.0)

rlon <- CanRCM4.lens$rlon
rlat <- CanRCM4.lens$rlat
ref <- CanRCM4.lens$xtreme[, , 1]

par(fig = c(0.05, 0.95, 0.25, 0.98), mar = c(0.5, 0.0, 0.5, 0.0), cex=1.0, lwd=0.5)
CanRCM4image(rlon=rlon, rlat=rlat, z=ref, breaks=breaks, col=colors)
lines(coastline, col='black', lwd=0.5)

rlon <- obs.pseudo[, 1]
rlat <- obs.pseudo[, 2]
points(x=rlon, y=rlat, pch=21, cex=0.5, lwd=1.0)
box()

par(fig = c(0.05, 0.95, 0.07, 0.20), new = TRUE, mar = c(1.5, 0.0, 0.0, 0.0), cex=1.0, lwd=0.5)
colorbar(breaks=breaks, col=colors)
axis(1, tck=-0.25, mgp=c(0, 0.5, 0), lwd=0.5, lwd.tick=0.5)
mtext("50-year daily precipitation [mm/h]", side=1, line=1.5, cex=1.0)
box()
dev.off()

