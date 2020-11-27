# Usage:
# Rscript install_pgks.R r_requirements.txt
# r_requirements delimited by '==' as in python requirements.txt

# Create user library
dir.create(Sys.getenv('R_LIBS_USER'), recursive = TRUE);
.libPaths(Sys.getenv('R_LIBS_USER'));

# Install devtools and its dependencies
install.packages('devtools', dependencies=TRUE);

# Install packages from requirements list
args <- commandArgs(trailingOnly = TRUE)
req_filename <- args[1]
requirements_file <- file(req_filename,open="r")
data <-readLines(requirements_file)
for (i in 1:length(data)){
    pkg_ver_pair <- unlist(stringr::str_split(data[i], "=="))
    pkg<-pkg_ver_pair[1]
    ver<-pkg_ver_pair[2]
    if (is.na(ver)){
        devtools::install_version(pkg)
    } else {
        devtools::install_version(pkg, version = ver);
    }
}
close(requirements_file)