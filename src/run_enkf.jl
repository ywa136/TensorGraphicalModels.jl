"""
Scripts to run Ensemble Kalman Filter experiments with tensor graphical models

Author: Wayne Wang
Last modified: 08/05/2021
"""

dynamic_type = "poisson"
obs_type = "identity"
method_list = ["glasso", "kpca", "kglasso", "teralasso", "sg_palm"]
# method_list = ["sg_palm"]
T = 20
N = 25
px = py = (64, 64)
obs_noise = 0.1
process_noise = 0.1
add_process_noise = false

using SparseArrays
using LinearAlgebra
using SpecialMatrices
using BandedMatrices
using Distributions
using Random
using DataFrames
using MatrixEquations
using PDMats
using TensorToolbox
using Printf
using Plots
using PyPlot
using MIRT: jim
ENV["GKSwstype"] = 100
ENV["GKSwstype"] = "nul"

include("utils.jl")
include("glasso.jl")
include("kglasso.jl")
include("sg_palm.jl")
include("kron_pca.jl")
include("teralasso.jl")
include("utils_sim.jl")
include("enkf.jl")


# generate ground truth data 
X, Y = gen_kalmanfilter_data(dynamic_type, obs_type, T, px, py, obs_noise, process_noise, add_process_noise)

# run enkf with tensor graphical models
NRMSEs_list = []
time_list = []
Omegahat_list = []
for method in method_list
    starttime = time()
    ## run enkf 
    Xhat, Xhat_bar, Omegahat = enkf(Y, 
                            method_str_to_type(method),
                            dynamic_type,
                            obs_type,
                            px,
                            py,
                            N,
                            obs_noise, 
                            process_noise,
                            add_process_noise)
    ## timer
    stoptime = time() - starttime
    push!(time_list, stoptime)
    ## compute NRMSEs
    NRMSEs = compute_nrmse(X, Xhat)
    push!(NRMSEs_list, NRMSEs)
    ## store est. precision matrix
    push!(Omegahat_list, Omegahat)
end

# set up plots of nrmses
plt = plot()
xlabel!("Time step")
ylabel!("RMSE")
for method in method_list
    if method == "sg_palm"
        NRMSEs = NRMSEs_list[end] 
    elseif method == "teralasso"
        NRMSEs = NRMSEs_list[4] 
    elseif method == "kglasso"
        NRMSEs = NRMSEs_list[3] 
    elseif method == "glasso"
        NRMSEs = NRMSEs_list[1] 
    elseif method == "kpca"
        NRMSEs = NRMSEs_list[2] 
    end
    ## plot rmse progression for each method
    plot_nrmse!(NRMSEs, method)
end
display(plt)


