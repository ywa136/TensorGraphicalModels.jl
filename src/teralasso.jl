using LinearAlgebra
using Statistics
using Debugger
include("utils_teralasso.jl")

break_on(:error)

"""
    teralasso(S_kGram, regtype, regparam, regcoef, tol, niter, ninner)

Implementation of Tensor Graphical Lasso (TeraLasso) precision matrix estimator.
Learns a K-order Kronecker sum covariance model.

# Arguments
- `S_kGram::AbstractVector`: length K array of factorwise
    Gram matrices (S_k in the paper).
    NOTE: use the most general accepted type (UnionAll):
    ```julia-repl
    julia> AbstractVector{<:AbstractMatrix{<:Real}} <: AbstractVector === AbstractArray{T, 1} where T
    true

    julia> AbstractVector{<:AbstractMatrix{<:Real}} <: AbstractVector{Any}
    false
    ```
- `regtype::Symbol=:L1`: type of regularization, :L1, :SCAD, or :MCP.
- `regparam::Real=20`: parameter for SCAD/MCP (a in the paper).
- `regcoef::AbstractVector`: length K vector of regularization
    coefficients for each factor (λ).

# Keywords
- `zeta::Real=0.1`: initial step size.
- `c::Real=0.5`: backtracking constant.
    NOTE: 0.5 too large for active region data; use 0.05.
- `tol::Real=1e-6`: tolerance for convergence criterion (ε).
- `niter::Int=50`: maximum number of iterations allowed.
- `ninner::Int=1`: maximum number of backtracking line search iteration.

# Returns:
- `Psi::AbstractVector{AbstractMatrix{Real}}`: length K array of factorwise
    precision matrices
- `out::AbstractVector{AbstractVector{Any}}`: iteration history.
"""
function teralasso(
    S_kGram::AbstractVector,
    regtype::Symbol=:L1,
    regparam::Real=20,
    regcoef::Union{Nothing, AbstractVector}=nothing,
    ;
    zeta::Real=0.1,
    c::Real=0.5,
    tol::Real=1e-6,
    niter::Int=50,
    ninner::Int=11,
)
    # Set eigenvalue upper bound for nonconvex contraints.
    if regtype == :L1
        mu = 1e-6
    elseif regtype == :SCAD
        mu = 1 / (regparam - 1)
    elseif regtype == :MCP
        mu = 1 / regparam
    end
    kappa = sqrt(2 / mu) * 2

    # Preparation
    K = length(S_kGram)
    ps = [size(S_kGram[k], 1) for k in 1:K]
    p = prod(ps)
    regcoef = isnothing(regcoef) ? sqrt.(log.(ps) ./ (p ./ ps)) : regcoef
    #regcoef .+= 1e-9  # inv(cov) may be ill conditioned if p < n and lambda = 0
    S_tilde = [S - (K - 1) / K * mean(diag(S)) * I for S in S_kGram]

    # Initlalization
    logdet = 0
    logdetN = 0 # updated outside inner loop
    Psi = 1 / K * [Matrix{Float64}(I, ps[k], ps[k]) for k = 1:K]
    PsiN = 1 / K * [Matrix{Float64}(I, ps[k], ps[k]) for k = 1:K]
    M = 1 / K * [Matrix{Float64}(I, ps[k], ps[k]) for k = 1:K]
    out = Array{Any}(undef, niter)

    for iter = 1:niter
        # Backtracking line search
        for cnt = 1:ninner
            for k = 1:K
                PsiN[k] = shrink_regularizer(
                    Psi[k] - zeta * (S_tilde[k] - M[k]),
                    regcoef[k],
                    zeta,
                    regtype,
                    regparam,
                )
            end

            result, logdetN = eval_cond(PsiN, Psi, M, S_kGram, S_tilde, ps, zeta, logdet, kappa)
            #println([iter, cnt, zeta, logdetN])

            if result == 1
                break
            end

            if cnt < ninner - 1
                zeta *= c
            elseif cnt == ninner - 1 # safe step size is not actually safe
                min_eig = sum([minimum(eigvals(Psi[k])) for k = 1:K])
                zeta = min_eig ^ 2 / 2
            else # cnt == ninner
                @warn "Backtracking fails. Use more inner iterations / smaller c."
            end
        end

        # Compute convergence
        dPsi = [PsiN[k] - Psi[k] for k = 1:K]
        nrm = norm([dPsi[k] - I * mean(diag(dPsi[k])) for k = 1:K])^2
        out[iter] = [fun(S_kGram, PsiN), reg(PsiN, regcoef), nrm, PsiN, zeta]
        if iter > 3 && all([out[i][3] <= tol for i in iter-3:iter])
            break
            # Stopping criterion based on fun val doesn't work. f keeps decreasing.
            # The problem is the inner product in f doesn't work properly.
            # Is there a reason why Kristjan uses nrm as the criterion?
            # if abs(out[iter][1] - out[iter-1][1]) < tol
            #     break
            # end
        end

        # Set next initial step
        ~, ~, MsN = project_ksum(PsiN, ps)
        zeta = barzilai_borwein(PsiN, Psi, MsN, M, ps)

        # Update iterate
        M .= MsN  # (copy, deepcopy, vs .=) Use === to check if identical
        Psi .= PsiN
        logdet = logdetN

        #println(iter, ": ", [out[iter][1], out[iter][2]])
        #display(heatmaps(PsiN))
        #println("-----------------------------")
    end
    return Psi, out
end

function barzilai_borwein(PsiN, Psi, MsN, M, ps)
    Num = 0
    Den = 0
    dgI = zeros(K)
    for k = 1:K
        dgI[k] = mean(diag((PsiN[k] - Psi[k])))
        Num = Num + prod(ps[[1:k-1; k+1:K]]) * sum((PsiN[k] - Psi[k] - I * dgI[k]) .^ 2)
        dK = PsiN[k] - Psi[k]
        for j = 1:K
            dKK = M[j] - MsN[j]
            if k == j
                Den = Den + prod(ps[[1:k-1; k+1:K]]) * sum(sum(dK .* dKK))
            else
                Den = Den + tr(dK) * tr(dKK) * prod(ps) / (ps[k] * ps[j])
            end
        end
    end
    Num = Num + mean(dgI)^2 * prod(ps)
    zeta = Num / Den
    if Den == 0
        zeta = 0
    elseif abs(zeta) > 1e2
        zeta = 1e2 * sign(zeta)
    elseif zeta < 0
        zeta = 1
    end
    return zeta
end

function eval_cond(PsiN, Psi, M, S_kGram, S_tilde, ps, zeta, logdet, kappa)
    K = length(ps)
    p = prod(ps)
    ms = [p / ps[k] for k = 1:K]
    ergNvec = zeros(p, 1) # eigval of the Kronecker sum of PsiN
    for k = 1:K
        ergNvec = ergNvec + repeat(
            kron(eigvals(PsiN[k]), ones(prod(ps[k+1:end]),1)),
            prod(ps[1:k-1]),
            1,
        ) # repmat is eqv to
    end
    #ergNvec = reduce(⊕, [eigvals(PsiN[k]) for k in 1:K])

    if minimum(ergNvec) <= 0
        logdetN = NaN
        result = 0
        return result, logdetN
    end

    logdetN = -sum(log.(ergNvec))
    if maximum(ergNvec) > kappa
        result = 0
        return result, logdetN
    end

    fNew = logdetN + sum([ms[k] * sum(S_kGram[k] .* PsiN[k]) for k = 1:K])
    # Ψ = PsiN[1] ⊕ PsiN[2]
    # SSSS = S_kGram[1] ⊕ S_kGram[2]
    # my_fNew = -log(det(Ψ)) + sum(Ψ .* SSSS)
    # @bp # sum(Ψ .* SSSS) != sum([ms[k] * sum(S_kGram[k] .* PsiN[k]) for k = 1:K])
    # println([fNew, fun(S_kGram, PsiN), my_fNew])

    dPsi = [PsiN[k] - Psi[k] for k = 1:K]
    tau = [mean(diag(dPsi[k])) for k = 1:K] # dgI, tr(D_k)/d_k
    Num = sum([ms[k] * norm(dPsi[k] - tau[k] * I)^2 for k = 1:K])
    Num += mean(tau)^2 * p  # Norm of change
    Q = logdet# + sum([ms[k] * sum(S_kGram[k].*Psi[k]) for k = 1:K]) + Num*K/(2*zeta)
    for k = 1:K #
        Q += sum(S_kGram[k] .* Psi[k]) * ms[k] + Num / (2 * zeta)
        for j = 1:K
            if j == k
                Q += sum((S_tilde[k] - M[k]) .* dPsi[k]) * prod(ps) / ps[k]
            else
                Q -= tr(M[k] - S_tilde[k]) * tr(dPsi[j]) * prod(ps) / (ps[k]*ps[j])
            end
        end
    end

    result = fNew <= Q

    return result, logdetN
end

function fun(S_kGram, Ψs)
    # - logdet(Ω) + < Ω,S >
    ps = [size(S, 1) for S in S_kGram]
    K = length(ps)
    p = prod(ps)
    T = zeros(ps...)
    for (k, Ψ) in enumerate(Ψs)
        dims = ones(Int, length(ps))
        dims[k] = ps[k]
        T .+= reshape(eigvals(Ψ), dims...)
    end
    # 2x faster than this:
    # T = diag(reduce(⊕, [Diagonal(eigvals(Ψs[k])) for k in 1:K]))
    logdet = sum(log.(T))

    trace = sum([sum(S_kGram[k] .* Ψs[k]) * p / ps[k] for k in 1:K])

    return -logdet + trace
end

function reg(Ψs, ρs)
    ps = [size(Ψ, 1) for Ψ in Ψs]
    p = prod(ps)
    result = 0
    for k in 1:length(Ψs)
        mk = p ÷ ps[k]
        offdiag = Ψs[k] - diagm(diag(Ψs[k]))
        result += mk * ρs[k] * sum(abs.(offdiag))
    end
    return result
end
