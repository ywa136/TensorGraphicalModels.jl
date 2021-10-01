"""
Implementation of the Graphical Lasso (Glasso) for l1-penalized precision matrix estimation

Code credit: https://github.com/bnaul/GraphLasso.jl
"""


function glasso(S::Matrix{Float64}, α::Float64; tol::Float64 = 1e-5,
                    maxit::Int = 1000, penalize_diag::Bool = true, verbose::Bool = false)
    p = size(S, 1)
    adj = abs.(S) .> α
    blocks = connected_components(SimpleGraph(max.(adj, I(p))))
    if(verbose)
        print(blocks)
    end
    Θ = zeros(p, p)
    for block = blocks
        Θ[block,block] .= fit_block(S[block,block], α, tol, maxit, penalize_diag)
    end
    return Θ
end


function fit_block(S::Matrix{Float64}, α::Float64, tol::Float64,
                   maxit::Int, penalize_diag::Bool)
    p = size(S, 1)
    W = copy(S)
    if penalize_diag
        W .+= α * I(p)
    end
    if size(S, 1) == 1
        return inv(W)
    end
    Θ = zeros(p, p)
    i = 0
    β = zeros(p-1, p)
    while i < maxit
        i += 1
        W_old = copy(W)
        for j = 1:p
            inds = collect(1:p)
            splice!(inds, j)
            W11 = W[inds, inds]
            sqrtW11 = sqrt(W11)
            # β[:,j] = fit(LassoPath, sqrtW11, sqrtW11 \ S[inds,j], λ=[α/(p-1)],
            #              standardize=false, intercept=false).coefs
            β[:,j] = glmnet(sqrtW11, sqrtW11 \ S[inds, j], lambda=[α / (p-1)],
                         standardize = false, intercept = false).betas
            W[inds, j] .= W11 * β[:, j]
            W[j, inds] .= W[inds, j]
        end
        if norm(W - W_old) < tol
            break
        end
    end
    if i == maxit
        @warn "Maximum number of iterations reached, graphical lasso failed to converge"
    end
    for j in 1:p
        inds = collect(1:p)
        splice!(inds, j)
        Θ[j, j] = 1 / (W[j, j] - dot(W[inds, j], β[:, j]))
        Θ[inds, j] .= -Θ[j, j] * β[:, j]
        Θ[j, inds] .= Θ[inds, j]
    end
    return Θ
end


# # Testing
# S_iris = cov(Array(dataset("datasets", "iris")[:,1:4]))

# α = 1.0

# W_R = [1.68569351 0.000000 0.2743155 0.01969989
#        0.00000000 1.189979 0.0000000 0.00000000
#        0.27431545 0.000000 4.1162779 0.29560940
#        0.01969989 0.000000 0.2956094 1.58100626]

# Θ_R = [ 0.59973155 0.0000000 -0.03996709  0.00000000
#         0.00000000 0.8403507  0.00000000  0.00000000
#        -0.03996708 0.0000000  0.24890787 -0.04604166
#         0.00000000 0.0000000 -0.04604166  0.64111722]

# W_iris, Θ_iris = graphlasso(S_iris, α; tol=1e-6, penalize_diag=true)

# @test isapprox(W_R, W_iris, atol=1e-5)
# @test isapprox(Θ_R, Θ_iris, atol=1e-5)