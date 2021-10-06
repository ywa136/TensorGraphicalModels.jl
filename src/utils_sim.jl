"""
Collections of utility functions for data simulation and evaluations

Author: Wayne Wang
Last modified: 10/06/2021
"""


# Precision estimation methods type
struct SG_PALM end
struct TERALASSO end
struct KGLASSO end
struct GLASSO end
# Covariance estimation methods type
struct KPCA end
# Customized precision matrix structure types
struct AR1 end
struct SB end
struct UNIF end
# Customized Kalman Filter process types
struct POISSON end
struct POISSON_AR end
struct CONVECTION_DIFFUSION end
# Customized Kalman Filter observation types
struct IDENTITY end
struct LINEAR_PERM end
struct LINEAR_PERM_MISS end


function method_str_to_type(method::AbstractString)
    if method == "sg_palm"
        method = SG_PALM()
    elseif method == "teralasso"
        method = TERALASSO()
    elseif method == "kglasso"
        method = KGLASSO()
    elseif method == "glasso"
        method = GLASSO()
    elseif method == "kpca"
        method = KPCA()
    else
        print("Precision/covariance matrix estimation method unsupported!")
    end
    return method
end


function method_type_to_str(method::Union{SG_PALM, TERALASSO, KGLASSO, GLASSO, KPCA})
    if method == SG_PALM()
        method = "sg_palm"
    elseif method == TERALASSO()
        method = "teralasso"
    elseif method == KGLASSO()
        method = "kglasso"
    elseif method == GLASSO()
        method = "glasso"
    elseif method == KPCA()
        method = "kpca"
    else
        print("Precision/covariance matrix estimation method unsupported!")
    end
    return method
end


function inv_cov_model(model_type::AR1, p::Int; ρ::Real = 0.5)
    """
    Create sparse precision matrix according to one of Autoregressive1 (AR),
        Uniform distributed weights (UNIF), and Start-Block (SB) models
    In:
        - p::Int: dimension of the cov matrix
        - ρ::Real: weight param between 0 and 1
    Out:
        - inv_cov::SparseMatrixCSC
    """
    inv_cov = sparse(Diagonal(repeat([1 + ρ^2], p)))
    for i = 1:(p-1)
        inv_cov[i, i+1] = -ρ
        inv_cov[i+1, i] = -ρ
    end

    return inv_cov
end


function inv_cov_model(model_type::UNIF, p::Int; dens::Real = 0.1, min_eigen::Real = 1e-3)
    """
    Create sparse precision matrix according to one of Autoregressive (AR),
        Uniform distributed weights (UNIF), and Start-Block (SB) models
    In:
        - p: the desired dimension of the (inv)covariance matrix
        - dens: percentage of non-zeros of the final inverse covariance
        - min_eigen: the minimum bound for the eigenvalues of the desired (inv)covariance matrix
    Out:
        - inv_cov::SparseMatrixCSC
    """
    U = sprand(p, p, dens)
    for i = 1:p 
        for j = 1:p
            if U[i, j] != 0
                U[i, j] = rand([-1.0, 1.0], 1)[1]
            end
        end
    end
    U .= copy(U') * U
    Udiag = Diagonal(U) .+ Diagonal(ones(p))
    Uoff = max.(min.(U .- Diagonal(U), 1.0), -1.0)
    inv_cov = Uoff .+ Udiag
    eigs, _ = eigen(collect(inv_cov))
    inv_cov .+= Diagonal(repeat([max(-1.2 * minimum(eigs), min_eigen)], p))

    return inv_cov
end


function inv_cov_model(model_type::SB, p::Int; ρ::Real = 0.5, num_subgraph::Int = 4,
    tol::Real = 1e-5)
    """
    Create sparse precision matrix according to one of Autoregressive (AR),
        Uniform distributed weights (UNIF), and Start-Block (SB) models
    In:
        - p::Int: dimension of the cov matrix
        - ρ::Real: weight param between 0 and 1
        - num_subgraph::Int: number of star structures
        - tol::Real: thresholding value for zeros
    Out:
        - inv_cov::SparseMatrixCSC
    """
    p_subgraph = Int(floor(p / num_subgraph))
    uneq_graph = (p != p_subgraph * num_subgraph)

    list_inv_subgraph = []

    for i = 1:num_subgraph
        if i == num_subgraph && uneq_graph
            p_subgraph = p - p_subgraph * (i - 1)
        end
        central_node = Int(rand(Tuple(collect(1:p_subgraph))))
        subgraph = repeat([ρ^2], p_subgraph, p_subgraph)
        subgraph[:, central_node] .= ρ
        subgraph[central_node, :] .= ρ
        subgraph .= subgraph .- Diagonal(subgraph) .+ Diagonal(ones(p_subgraph))
        inv_subgraph = inv(subgraph)
        push!(list_inv_subgraph, sparse(inv_subgraph)) 
    end

    inv_cov = SparseArrays.blockdiag(list_inv_subgraph...)
    inv_cov[abs.(inv_cov) .< tol] .= 0

    return inv_cov
end


function metric_edge(true_edge, est_edge)
    """
    Computes FNR, FPR, MCC, etc
    In
    - true_edge::AbstractArrary: true sparse precision matrix
    - est_edge::AbstractArray: estimated sparse precision matrix
    Out
    - metric_res::Array: FP, FN, TP, TN, FPR, FNR, Precision, Recall, MCC
    """
    p = size(true_edge, 1)
    nz_pos = (true_edge .!= 0)
    z_pos = (true_edge .== 0)

    FP = (sum((est_edge[z_pos] .!= 0)))/2
    FN = (sum((est_edge[nz_pos] .== 0)))/2
    TP = (sum((est_edge[nz_pos] .!= 0)) - p)/2
    TN = (sum((est_edge[z_pos] .== 0)))/2

    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    metric_res = [FP, FN, TP, TN, FPR, FNR, Precision, Recall, MCC]

    return metric_res
end


function gen_kronecker_data(model::AbstractString, inv_cov_model_type::AbstractString, K::Int,
    N::Int, d_list::AbstractVector; Ψ_list::AbstractArray = Array{Array{Float64, 2}, 1}(),
    tensorize_out::Bool = true)
    """
    Generate multivariate Gaussian data with precision matrix satisfying either the
        Kronecker product, Kronecker sum, or Sylvester structre.
    """
    if inv_cov_model_type == "ar1"
        inv_cov_model_type = AR1()
    elseif inv_cov_model_type == "sb"
        inv_cov_model_type = SB()
    elseif inv_cov_model_type == "unif"
        inv_cov_model_type = UNIF()
    else
        print("Unsupported inverse covariance model type")
    end

    # create each factor matrix if not provided
    if length(Ψ_list) == 0
        for k = 1:K
            push!(Ψ_list, inv_cov_model(inv_cov_model_type, d_list[k]))
        end
        d = prod(d_list)
    else
        K = length(Ψ_list)
        d_list = size.(Ψ_list, 1)  
        d = prod(d_list)  
    end
    
    # eigen decomposition for each Ψ_k
    U_list = []; Λ_list = []
    for k = 1:K
        Λk, Uk = eigen(collect(Ψ_list[k])) 
        push!(Λ_list, Diagonal(Λk))
        push!(U_list, Uk)
    end
    reverse!(U_list)
    eigvecsSigmaSqrt = kron(U_list...)

    if model == "kp"
        eigsSigmaSqrt = inv(Diagonal(kron(Λ_list...)).^(1/2))
    elseif model == "ks"
        eigsSigmaSqrt = inv(Diagonal(kroneckersum_list(Λ_list)).^(1/2))
    elseif model == "sylvester"
        eigsSigmaSqrt = inv(Diagonal(kroneckersum_list(Λ_list)))
    else
        print("Unsupported model type")
    end

    # generate fake data
    X = zeros(d, N)
    z = zeros(d)
    xtilde = zeros(d)
    x = zeros(d)
    for i = 1:N
        randn!(z)
        mul!(xtilde, eigsSigmaSqrt, z)
        mul!(x, eigvecsSigmaSqrt, xtilde)
        X[:, i] .= x
    end

    if tensorize_out
        return matten(copy(X'), K + 1, push!(d_list, N))
    end

    return X
end


function gen_kalmanfilter_data(dynamic_type::AbstractString, obs_type::AbstractString, 
    T::Int, px::Tuple, py::Tuple,
    obs_noise::Real, process_noise::Real,
    add_process_noise::Bool)
    """
    Generate Kalman Filtering obervations
    Input:
        - T: number of time steps
        - px: 2-tuple for process dimension
        - py: 2-tuple for observation dimension
        - obs_noise: obervations will be subject to Gaussian noise N(0,obs_noise*I)
        - process_noise: processes will be subject to Gaussian noise N(0,process_noise*I)
        - add_process_noise: whether to introduce process noise
    Output:
        - (X, Y): T true processes and observations
    """
    # initialization
    X0 = rand(MatrixNormal(2 * randn(px), ScalMat(px[1], 2), ScalMat(px[2], 2)))
    X = zeros((prod(px), T + 1))
    Y = (obs_type == "linear_perm_miss") ? zeros((Int(0.5 * prod(py)), T)) : zeros((prod(py), T))
    X[:, 1] .= X0[:]

    # swtich observation types
    if obs_type == "identity"
        obs_type = IDENTITY()
    elseif obs_type == "linear_perm"
        obs_type = LINEAR_PERM()
    elseif obs_type == "linear_perm_miss"
        obs_type = LINEAR_PERM_MISS()
    else
        print("Observation type type unsupported!")
    end
    H = measure_operator(obs_type, py, px)
    
    # evolution
    for t = 1:T
        # dynamics update
        X[:, t + 1] .= kalmanfilter_dynamic_update(dynamic_type, X[:, t], px, add_process_noise, process_noise)
        # observations update
        Y[:, t] .= kalmanfilter_observation_update(H, X[:, t + 1], py, px, obs_noise)
    end

    return X, Y, H
end


function kalmanfilter_dynamic_update(dynamic_type::AbstractString, X_curr::AbstractArray,
    px::Tuple,
    add_process_noise::Bool,
    process_noise::Real;
    α::Real = 0.05,
    ϵ::Real = 0.01,
    Δx::Real = 0.005,
    Δt::Real = 0.0005)
    # step sizes for 2D convection-diffusion PDE discretization
    # convergence/stability criteria: Δt <= (Δx^2)/(4*α)
    # diffusion constant α = 0.5 
    # convection constant ϵ = 0.5 
    # assume same spatial steps in x&y Δx = 1/200 
    # time steps Δt = 0.0005 

    # swtich observation types
    if dynamic_type == "poisson"
        dynamic_type = POISSON()
    elseif dynamic_type == "poisson_ar"
        dynamic_type = POISSON_AR()
    elseif dynamic_type == "convection_diffusion"
        dynamic_type = CONVECTION_DIFFUSION()
    else
        print("Dynamic type unsupported!")
    end

    # dynamics update, i.e., solve the Sylvester equation
    px = size(X_curr)
    X_new = similar(X_curr)
    sylv_eqn_solver!(X_new, X_curr, px, dynamic_type; α = α, ϵ = ϵ, Δx = Δx, Δt = Δt)

    if add_process_noise
        w = zeros(prod(px))
        rand!(MvNormal(prod(px), process_noise), w)
        X_new .+= w
    end

    return X_new
end


function sylv_eqn_solver!(X_new::AbstractArray, X_curr::AbstractArray, px::Tuple,
    type::CONVECTION_DIFFUSION;
    α::Real = 0.05,
    ϵ::Real = 0.01,
    Δx::Real = 0.005,
    Δt::Real = 0.0005)
    # step sizes for 2D convection-diffusion PDE discretization
    # convergence/stability criteria: Δt <= (Δx^2)/(4*α)
    # diffusion constant α = 0.5 
    # convection constant ϵ = 0.5 
    # assume same spatial steps in x&y Δx = 1/200 
    # time steps Δt = 0.0005 
    γ = (α * Δt) / Δx^2
    # 2D oncvection-diffusion equation operator
    A = spdiagm(-1 => [-(1 + ϵ / (2 * α) * Δx) * γ for _ = 1:(px[1] - 1)],
                0 => [(4 + 1 / γ) * γ / 2 for _ = 1:px[1]],
                1 => [-(1 - ϵ / (2 * α) * Δx) * γ for _ = 1:(px[1] - 1)])
    # X = sylvc(A,B,C)
    # Solve the continuous Sylvester matrix equation AX + XB = C
    X_new .= sylvc(A, A, reshape(X_curr, px))[:]
    return nothing
end


function sylv_eqn_solver!(X_new::AbstractArray, X_curr::AbstractArray, px::Tuple,  
    type::POISSON_AR;
    α::Real = 0.05,
    ϵ::Real = 0.01,
    Δx::Real = 1/200,
    Δt::Real = 0.0005)
    # 2D oncvection-diffusion equation operator
    A = spdiagm(-1 => [-1.0 for _ = 1:(px[1] - 1)],
                0 => [2.0 for _ = 1:px[1]],
                1 => [-1.0 for _ = 1:(px[1]-1)])
    B = spdiagm(-1 => [-1.0 for _ = 1:(px[1] - 1)],
                0 => [2.0 for _ = 1:px[1]],
                1 => [-1.0 for _ = 1:(px[1] - 1)])
    # X = gsylv(A,B,E)
    # Solve the generalized Sylvester matrix equation AXB  = E.
    X_new .= gsylv(A, B, randn(px))[:]
    return nothing
end


function sylv_eqn_solver!(X_new::AbstractArray, X_curr::AbstractArray, px::Tuple, 
    type::POISSON;
    α::Real = 0.05,
    ϵ::Real = 0.01,
    Δx::Real = 1/200,
    Δt::Real = 0.0005)
    # 2D oncvection-diffusion equation operator
    A = spdiagm(-1 => [-1.0 for _ = 1:(px[1] - 1)],
                0 => [2.0 for _ = 1:px[1]],
                1 => [-1.0 for _ = 1:(px[1] - 1)]) 
    # X = sylvc(A,B,C)
    # Solve the continuous Sylvester matrix equation AX + XB = C
    # X_new .= sylvc(A, A, (Δx^2) * randn(px))[:]
    X_new .= sylvc(A, A, randn(px))[:]
    return nothing
end


function kalmanfilter_observation_update(H::AbstractArray, X::AbstractArray,
    py::Tuple,
    px::Tuple,
    obs_noise::Real)
    # measurement operation plus noise 
    v = zeros(size(H, 1))
    rand!(MvNormal(length(v), obs_noise), v)
    Y = H * X .+ v
    return Y
end


function measure_operator(type::IDENTITY,
    py::Tuple,
    px::Tuple)
    # measurement operator
    H = 1.0 * I(prod(px)) 
    return H
end


function measure_operator(type::LINEAR_PERM,
    py::Tuple,
    px::Tuple)
    # measurement operator
    H = zeros(prod(py), prod(px)) 
    perm = randperm(prod(px))
    for i = 1:prod(px)
        H[i, perm[i]] = 1.0
    end
    return H
end


function measure_operator(type::LINEAR_PERM_MISS,
    py::Tuple,
    px::Tuple)
    # measurement operator
    H = zeros(prod(py), prod(px)) 
    perm = randperm(prod(px))
    for i = 1:prod(px)
        H[i, perm[i]] = 1.0
    end
    H = H[randperm(prod(px))[1:Int(prod(px)/2)], :] # half of the rows missing
    return H
end


# # gif plot all time steps
# function plot_state_gif(X::AbstractArray, px::Tuple;
#     out_dir::AbstractString = "",
#     clim::Tuple = ())
#     """
#     Plot all time steps of the state images 
#     """
#     anim = @animate for i = 2:(T + 1)
#         plot_state_img(X, px, i, clim = clim)
#     end

#     if length(out_dir) != 0
#         gif(anim, out_dir, fps = 5)
#     else
#         gif(anim, fps = 5)
#     end

#     return nothing
# end


# function plot_state_img(X::AbstractArray, px::Tuple, time_step::Int;
#     clim::Tuple = ())
#     """
#     Plot a single state img
#     """
#     if length(clim) != 0
#         display(plot(jim(reshape(X[:, time_step], px), clim = clim), 
#                     title = string("Time step: ", time_step)))
#     end

#     display(plot(jim(reshape(X[:, time_step], px)), 
#                 title = string("Time step: ", time_step)))

#     return nothing
# end


function plot_state_img(X::AbstractArray, px::Tuple, time_step::Int;
    vmin = nothing, vmax = nothing, cmap::AbstractString = "inferno")
    """
    Plot a single state img *using PyPlot*
    """
    fig = figure()

    if !isnothing(vmin) && !isnothing(vmax)
        PyPlot.imshow(reshape(X[:, time_step], px), cmap = cmap, vmin = vmin, vmax = vmax)
    else
        PyPlot.imshow(reshape(X[:, time_step], px), cmap = cmap)
    end
    
    PyPlot.axis("off")
    PyPlot.title(string("Time step: ", time_step))
    PyPlot.colorbar()
    PyPlot.display(fig)

    return nothing
end


# plot nrmse
function plot_nrmse!(NRMSEs::AbstractArray, method::AbstractString)
    T, N = size(NRMSEs)
    ## NRMSEs over time steps
    ## showing the mean NRMSEs and the 90% CI
    Plots.plot!(1:T, mean(NRMSEs, dims=2),
            yaxis = :log,
            ribbon = sort(NRMSEs, dims = 2),
            fillalpha = .2,
            markerstrokecolor = :auto,
            label = string("Mean RMSE ", method)
        )

    return nothing
end


# compute nrmse
function compute_nrmse(X::AbstractArray, Xhat::AbstractArray)
    ## compute NRMSEs for all ensemble members
    T = size(X, 2)
    N = size(Xhat, 3)
    NRMSEs = zeros((T - 1, N))

    for t = 2:T
        for i = 1:N
            NRMSEs[t - 1, i] = sqrt(mean((Xhat[t, :, i] .- X[:, t]).^2)) #/(maximum(X[:])-minimum(X[:]))
        end
    end

    return NRMSEs
end
