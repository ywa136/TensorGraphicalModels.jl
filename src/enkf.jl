"""
Ensemble Kalman Filter (EnKF) using (sparse) tensor graphical models for state    
    covariance / inverse covariance estimation

Author: Wayne Wang
Last modified: 10/06/2021
"""


function enkf(Y::AbstractMatrix{<:Real}, 
    inv_cov_method::Union{SG_PALM, TERALASSO, KGLASSO, GLASSO},
    dynamic_type::AbstractString,
    H::AbstractArray,
    px::Tuple,
    py::Tuple,
    N::Int,
    obs_noise::Real, 
    process_noise::Real,
    add_process_noise::Bool)
    # initialization
    T = size(Y, 2)
    X = zeros((T + 1, prod(px), N)) # T × p × N
    Xbar = zeros((T + 1, prod(px))) # T × p: mean latest process each time
    X0 = randn((prod(px), N))
    X[1, :, :] .= X0
    Omega = spzeros(prod(px), prod(px))
    obs_noise_vec = zeros(size(H, 1))
    Ht_Rinv_H = (obs_noise^-1) * copy(H') * H #assuming R is diagonal w/ obs_noise*I
    X_shift = similar(X[1, :, 1])

    # evolution
    for t = 1:T
        @printf("###### Time step: %d ######\n", t)
        # estiamte state precision matrix
        state_inv_cov_est!(Omega, inv_cov_method, X, Xbar, px, t, N)

        for i = 1:N
            # forecast step
            X[t + 1, :, i] .=  kalmanfilter_dynamic_update(dynamic_type, 
                                    X[t, :, i], px, add_process_noise, process_noise)
            # update step
            rand!(MvNormal(size(obs_noise_vec, 1), obs_noise), obs_noise_vec)
            X_shift .= (obs_noise^-1) * copy(H') * (view(Y, :, t) .+ obs_noise_vec .- H * view(X, t + 1, :, i))
            X[t + 1, :, i] .+= (Omega .+ Ht_Rinv_H) \ X_shift
        end
    end

    return X, Xbar, Omega
end


function enkf(Y::AbstractMatrix{<:Real}, 
    cov_method::KPCA,
    dynamic_type::AbstractString,
    H::AbstractArray,
    px::Tuple,
    py::Tuple,
    N::Int,
    obs_noise::Real, 
    process_noise::Real,
    add_process_noise::Bool)
    # initial ensemble
    T = size(Y, 2)
    X = zeros((T + 1, prod(px), N)) # T × p × N
    Xbar = zeros((T + 1, prod(px))) # T × p: mean latext process each time
    X0 = randn((prod(px), N))
    X[1, :, :] .= X0
    obs_noise_vec = zeros(size(H, 1))
    K = zeros((size(H, 1), size(H, 1)))
    R = ScalMat(size(H, 1), obs_noise)
    Sigma = zeros((prod(px), prod(px)))
    X_shift = similar(Y[:, 1])

    # evolution
    for t = 1:T
        @printf("###### Time step: %d ######\n", t)
        # estiamte state precision matrix
        state_cov_est!(Sigma, cov_method, X, Xbar, px, t, N)

        ## Kalman gain matrix: Sigma H^T (R + H Sigma H^T)^-1 
        ## form the invertible part of the Kalman gain matrix
        K .= H * Sigma * copy(H') .+ R
        # droptol!(K, sp_thres) # threshold K since it has many small values
        
        for i = 1:N
            # forecast step
            X[t + 1, :, i] .=  kalmanfilter_dynamic_update(dynamic_type, 
                                    X[t, :, i], px, add_process_noise, process_noise)
            # update step
            rand!(MvNormal(size(obs_noise_vec, 1), obs_noise), obs_noise_vec)
            X_shift .= K \ (view(Y,:,t) .+ obs_noise_vec .- H * view(X, t + 1, :, i))
            X[t + 1, :, i] .+= Sigma * copy(H') * X_shift
        end
    end

    return X, Xbar, Sigma
end


function state_inv_cov_est!(Omega::AbstractArray, method::SG_PALM, X::AbstractArray, Xbar::AbstractArray, 
    px::Tuple, t::Int, N::Int;
    K::Int = 2,
    λ::AbstractVector{<:Real} = Array{Float64, 1}(),
    regtype::String = "L1",
    a::Real = 3,
    niter::Int = 100, 
    ninner::Int = 10,
    η0::Real = 0.01, 
    c::Real = 0.01, 
    lsrule::String = "constant",
    ϵ::Real = 1e-5)
    ## compute mode-k Gram matrices
    Xbar[t, :] .= mean(view(X, t, :, :), dims = 2)[:]
    X_kGram = [zeros(px[k], px[k]) for k = 1:K]
    Xk = [zeros(px[k], Int(prod(px) / px[k])) for k = 1:K]
    for k = 1:K
        for i = 1:N
            copy!(Xk[k], tenmat(reshape(view(X, t, :, i) - Xbar[t, :], Tuple(px)), k))
            mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0 / N, 1.0)
        end
    end

    ## run PALM for precision matrix estimation
    Psi0 = [sparse(eye(px[k])) for k = 1:K]
    fun = (iter, Psi) -> [1, time()] # NULL func
    lambda = (length(λ) == 0) ? [sqrt(px[k] * log(prod(px)) / N) for k = 1:K] : λ
    PsiH, _ = syglasso_palm(X[t, :, :] .- Xbar[t, :], X_kGram, lambda, Psi0,
                    regtype = regtype, a = a, niter = niter, ninner = ninner,
                    η0 = η0, c = c, lsrule = lsrule, ϵ = ϵ, fun = fun)

    ## form Omega
    Omega .= kron(I(px[2]), PsiH[1]^2) .+ kron(PsiH[2]^2, I(px[1])) .+ 2 * kron(PsiH[2], PsiH[1])
    return nothing
end


function state_inv_cov_est!(Omega::AbstractArray, method::TERALASSO, X::AbstractArray, Xbar::AbstractArray, 
    px::Tuple, t::Int, N::Int;
    K::Int = 2,
    niter::Int = 50, 
    ninner::Int = 11,
    λ::AbstractVector{<:Real} = Array{Float64, 1}())
    ## compute mode-k Gram matrices
    Xbar[t, :] .= mean(view(X, t, :, :), dims = 2)[:]
    X_kGram = [zeros(px[k], px[k]) for k = 1:K]
    Xk = [zeros(px[k], Int(prod(px) / px[k])) for k = 1:K]
    for k = 1:K
        for i = 1:N
            copy!(Xk[k], tenmat(reshape(view(X, t, :, i) - Xbar[t, :], Tuple(px)), k))
            mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0 / N, 1.0)
        end
    end

    ## run teralasso
    # lambda = (length(λ) == 0) ? [10 * sqrt(log(prod(px)) / (N * px[k])) for k = 1:K] : λ
    PsiH, _ = teralasso(X_kGram)

    ## form Omega
    Omega .= kroneckersum(PsiH[1], PsiH[2])
    return nothing
end


function state_inv_cov_est!(Omega::AbstractArray, method::KGLASSO, X::AbstractArray, Xbar::AbstractArray,
    px::Tuple, t::Int, N::Int;
    K::Int = 2,
    λ::AbstractVector{<:Real} = Array{Float64, 1}())
    ## convert data matrix to multi-dimensional array
    str_to_eval = string("(", join([string("px[", i, "], ") for i = 1:K]), "N)")
    Xt = zeros(eval(Meta.parse(str_to_eval))) #d_1 × … × d_K × N tensor
    for i = 1:N
        ## K=2 only! TODO: modify to work with various K ##
        Xt[:, :, i] .= reshape(view(X, t, :, i), px)
    end
    # run kglasso algorithm
    lambda = (λ == 0) ? [5 * sqrt(px[k] * log(prod(px)) / N) for k = 1:K] : λ
    Ψ_hat_list = kglasso(Xt, λ_list = lambda)
    Omega .= kron(Ψ_hat_list...)
    return nothing
end


function state_inv_cov_est!(Omega::AbstractArray, method::GLASSO, X::AbstractArray, Xbar::AbstractArray,
    px::Tuple, t::Int, N::Int;
    λ::Real = 0,
    penalize_diag::Bool = false,
    tol::Real = 1e-5)
    ## compute the sample covariance
    Xbar[t,:] .= mean(view(X, t, :, :), dims = 2)[:]
    S = zeros(prod(px), prod(px))
    for i = 1:N
        mul!(S, 
            view(X, t, :, i) - view(Xbar, t, :), 
            copy(transpose(view(X, t, :, i) - view(Xbar, t, :))), 1.0 / N, 1.0)
    end

    ## run Glasso for precision matrix estimation
    lambda = (λ == 0) ? 100 * sqrt(log(prod(px)) / N) : λ
    Omega .= glasso(S, lambda; tol = tol, penalize_diag = penalize_diag)
    return nothing
end


function state_cov_est!(Sigma::AbstractArray, method::KPCA, X::AbstractArray, Xbar::AbstractArray,
    px::Tuple, t::Int, N::Int;
    r::Int = 5, 
    λ::Real = 0, 
    lambdaL::Real = 0, 
    lambdaS::Real = 0,
    tau::Real = 0.5, 
    robust_pca_method::String = "SVT",
    iter::Int = 5)
    ## compute the sample covariance
    Xbar[t, :] .= mean(view(X, t, :, :), dims = 2)[:]
    S = zeros(prod(px), prod(px))
    for i = 1:N
        mul!(S, 
            view(X, t, :, i) - view(Xbar, t, :), 
            copy(transpose(view(X, t, :, i) - view(Xbar, t, :))), 1.0 / N, 1.0)
    end

    lambdaL = (lambdaL == 0) ? (px[1]^2+px[2]^2+log(max(px[1],px[2],N)))/N : lambdaL 
    lambdaS = (lambdaS == 0) ? 5*sqrt(log(px[1]*px[2])/N) : lambdaS

    ## run Glasso for precision matrix estimation
    Sigma .= robust_kron_pca(S, px[1], px[2], lambdaL, lambdaS, robust_pca_method;
                             tau = tau, iter = iter, r = r)
    return nothing
end
