##############################################################################
##
## VcovSpec - Integration with model + vcov() syntax
##
## VcovSpec is now imported from CovarianceMatrices.jl.
## The single-argument vcov(estimator) -> VcovSpec method is defined there.
##
## This file is kept for documentation purposes but no longer extends vcov.
##
##############################################################################

# Note: VcovSpec is imported from CovarianceMatrices.jl
# The vcov(::AbstractAsymptoticVarianceEstimator) -> VcovSpec method is defined in CovarianceMatrices.jl

# The Base.:+ operators for OLSEstimator, OLSMatrixEstimator, and IVEstimator
# are defined in their respective model files (LinearModel.jl, IVModel.jl).
# Those are NOT type piracy because Regress owns those model types.
