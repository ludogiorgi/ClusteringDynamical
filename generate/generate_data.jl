using MultiDimensionalClustering, Random, Enzyme, HDF5, ProgressBars, GoogleDrive

# random seed for reproducibility
Random.seed!(12345)

# create data directory if it's not there
isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")

# potential well
if isfile(pwd() * "/data/potential_well.hdf5")
    @info "potential well data already exists. skipping data generation"
else 
    include("potential_well.jl")
end

# lorenz
if isfile(pwd() * "/data/lorenz.hdf5")
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end

# newton-leipnik
if isfile(pwd() * "/data/newton.hdf5")
    @info "newton data already exists. skipping data generation"
else
    include("newton.jl")
end

# There are problems with the GoogleDrive.jl package...
# kuramoto
#=
if isfile(pwd() * "/data/kuramoto.hdf5")
    @info "kuramoto data already exists. skipping data generation"
else
    drive_download("https://drive.google.com/file/d/11FN55Kr0Ue3JI5RDon60Iwquvoc-j5SW/view?usp=share_link", pwd() * "/data/kuramoto.hdf5")
end

# PIV
if isfile(pwd() * "/data/PIV.hdf5")
    @info "PIV data already exists. skipping data generation"
else
    drive_download("https://drive.google.com/file/d/11M2fCihaDXdW-_Kzm2XWChKP0qqOXDBp/view?usp=share_link", pwd() * "/data/PIV.hdf5")
end
=#