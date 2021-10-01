"""
Proximal Alternating Linearized Minimization method for solving
    \\argmin_{Ψ_1,...Ψ_K} H(Ψ_1,...Ψ_K) + ∑_{k=1}^K G_k(Ψ_k);
The main function is: Ψ_1,...Ψ_K, out = syglasso_palm(...)

In
- X: data matrix of dimention N × p, i.e., data ternsor unfolded by the sample dimention
- X_kGram: order-k Gram matrices
- λ: penalty parameters
- Ψ0: initial iterates
Optional
- regtype: type of regularization, one of "L1", "SCAD", "MCP"
- a: parameter for SCAD and MCP
- niter: number of max iterations
- ninner: number of max linesearch step
- η0: initial stepsize
- c: discounting factor for line search
- lsrule: linesearch method, one of "bb": Barzilai-Borwein, prev: previous feasible step; constant: initial step size(=η0);
- ϵ: convergence threshold
- fun: user-defined function value to be returned
Out
- Ψ_1,...Ψ_K: final estimates
- out: [(0,fun(Ψ_1^0,...Ψ_K^0)),...(niter,fun(Ψ_1^niter,...Ψ_K^niter))]

Author: XXXX XXXX
Last modified: 02/25/2021
"""


function syglasso_palm(X::AbstractArray{<:Real}, X_kGram::AbstractVector{<:AbstractArray}, λ::Vector{<:Real},
    Ψ0::AbstractVector{<:AbstractArray}; regtype::String = "L1", a::Real=3, niter::Int=100, ninner::Int=10,
    η0::Real=0.01, c::Real=0.01, lsrule::String="bb", ϵ::Real=1e-5,
    fun::Function=(iter::Int,Psi::AbstractVector{<:AbstractArray})->undef)
    """
    Main PALM algorithm function
    """
    # Initializations
    K = length(Ψ0)
    Ψ = deepcopy(Ψ0)
    Ψnew = deepcopy(Ψ0)
    grad = [zeros(size(Ψ[k])) for k=1:K]
    gradnew = deepcopy(grad)
    η = copy(η0)
    out = undef
    # push!(out, fun(0, Ψ))
    iter = 1
    Δfunc = 1

    # Loop until max iter reached or converged
    while (Δfunc > ϵ) & (iter <= niter)
        # @printf("Iter: %d\n", iter)
        ηK = [0.0 for k=1:K]
        @inbounds for k=1:K
            # Compute block-wise gradient
            # h_grad!(grad[k], X_kGramTilde, Ψ, k, Xt, X_kGram[k])
            grad[k], X_kGramTilde = h_grad(Ψ, k, X, X_kGram[k])
            # Proximal mapping 
            g_prox!(Ψnew[k], Ψ[k] - η*grad[k], λ[k], η, a, regtype)
            # Backtracking line search
            res = linesearch_cond(Ψ, Ψnew, k, grad[k], η, X_kGram[k], X_kGramTilde)
            t = 1 # line search step counts, max. ninner
            while !res && t <= ninner
                # @printf("lineasearch step: %d\n", t)
                η = c*η
                # Compute the next iterate using Proximal mapping
                g_prox!(Ψnew[k], Ψ[k] - η*grad[k], λ[k], η, a, regtype)
                res = linesearch_cond(Ψ, Ψnew, k, grad[k], η, X_kGram[k], X_kGramTilde)
                t += 1
            end
            # Set next initial linesearch step
            # prev: previous feasible step; constant: initial step size(=η0); bb: Barzilai-Borwein
            if lsrule == "constant"
                ηK[k] = η0
            elseif lsrule == "bb"
                # Compute new gradient for kth coordinate
                gradnew[k], _ = h_grad(Ψnew, k, X, X_kGram[k])
                ηK[k] = BB_init(Ψ[k], Ψnew[k], grad[k], gradnew[k])
            end
            # Update coordinates
            copy!(Ψ[k], Ψnew[k])
        end
        # # Compute user-defined func output
        # push!(out, fun(iter,Ψ))
        # # Compute convergence criteria
        # Δfunc = (out[iter+1][1] - out[iter][1])/out[iter][1]
        # Set next initial linesearch step
        # (a)Constant initial step size(=η0);(b)previous step;(c)Barzilai-Borwein
        if (lsrule == "constant") | (lsrule == "bb")
            η = minimum(ηK)
        end
        # Increment iter number
        iter += 1
    end

    return Ψ, out
end


function BB_init(Ψk::AbstractArray, Ψnewk::AbstractArray, gradk::AbstractArray, gradnewk::AbstractArray)
    """
    Compute the Barzilai-Borwein stepsize as an initial value for the next iteration
    """
    Diff = Ψnewk - Ψk
    gradDiff = gradnewk - gradk
    num = norm(Diff)^2
    den = tr(copy(transpose(Diff))*gradDiff)
    return num/den
end


function linesearch_cond(Ψ::AbstractVector{<:AbstractArray}, Ψnew::AbstractVector{<:AbstractArray}, k::Int, 
    h_grad_k::AbstractArray{<:Real}, η::Real, X_kGram_k::AbstractArray{<:Real}, X_kGramTilde::AbstractVector{<:AbstractArray})
    """
    Check whether line search is successful, i.e., check whether
        h(Ψnew) ≤ h(Ψ) + <Ψnew[k]-Ψ[k], ∇_{Ψk}(Ψ[k])> + (1/2η)*\\|Ψnew[k]-Ψ[k]\\|^2
    In
    - Ψ: old iterates
    - Ψnew: new iterates using current step size η
    - k: coordinate
    - h_grad_k: gradient evaluated at kth coordinate of old iterates
    - η: step size
    - X: data tensor
    - X_kGram: Gram matrices of data
    Out
    - res::Bool: true if the condition is satisfied
    """
    K = length(Ψ)
    d = size.(Ψ,1)
    dk = size(Ψ[k],1)
    Diff = similar(Ψ[k])
    sqrDiff = similar(Ψ[k])
    copy!(Diff, Ψnew[k] - Ψ[k])
    copy!(sqrDiff, Ψnew[k]^2 - Ψ[k]^2)
    N = size(X,2)
    # RHS: quadratic term
    rhs = tr(Diff*h_grad_k) + (0.5/η)*norm(Diff)
    # LHS: difference in h function
    lhs = -logdet(Diagonal(kroneckersum_list([Diagonal(Ψnew[j]) for j=1:K]))^2) +
            logdet(Diagonal(kroneckersum_list([Diagonal(Ψ[j]) for j=1:K]))^2) +
            tr(sqrDiff*X_kGram_k)

    Xk = zeros(dk,Int(prod(d)/dk)) # pre-allocate
    XkT = zeros(Int(prod(d)/dk),dk) # pre-allocate
    kp_k = spzeros(Int(prod(d)/dk),Int(prod(d)/dk)) # pre-allocate
    crossTerm = zeros(dk,dk) # pre-allocate
    @inbounds for j=1:K
        if j != k
            mul!(crossTerm, Diff, X_kGramTilde[j])
            lhs += 2*tr(crossTerm)
        end
    end

    if lhs <= rhs
        res = true
    else
        res = false
    end

    return res
end


function h_grad(Ψ::AbstractVector{<:AbstractArray}, k::Int, X::AbstractArray{<:Real},
    X_kGram_k::AbstractArray{<:Real})
    """
    Block-wise gradient of the function h with respect Ψ_K
    In
    - Ψ::AbstractVector{<:Any}: current iterates 
    - k::Int: coordinate to take derivative
    - X::AbstractArray: data tensor with the first mode being samples
    - X_kGram::AbstractVector{<:Any}: mode-k Gram matrices for all modes
    Out
    - kth block-wise gradient evaluated at the current iterates
    """ 
    K = length(Ψ)
    d = size.(Ψ,1)
    dk = size(Ψ[k],1)
    N = size(X,2)
    # Dlogdet: grad of the log-det term
    Dlogdet = -2*Diagonal([tr(inv(Diagonal(repeat([Ψ[k][i,i]],Int(prod(d)/dk))) + 
                Diagonal(kroneckersum_list([Diagonal(Ψ[j]) for j=1:K if j!=k])))) for i=1:dk])
    # Dtr: grad of trace term 
    # crossTerm = zeros(dk,dk) # pre-allocate
    # sumKGramTilde = zeros(dk,dk) # pre-allocate
    X_kGramTilde = [zeros(dk,dk) for j=1:K] # pre-allocate
    Xk = zeros(dk,Int(prod(d)/dk)) # pre-allocate
    XkT = zeros(Int(prod(d)/dk),dk) # pre-allocate
    kp_k = spzeros(Int(prod(d)/dk),Int(prod(d)/dk)) # pre-allocate
    @inbounds for j=1:K
        if j != k
            Ψ_k = [(l==j) ? Ψ[j] : Diagonal(ones(d[l])) for l=1:K]
            deleteat!(Ψ_k,k)
            reverse!(Ψ_k)
            ### TODO: Use LuxurySparse? ###
            ### TODO: Use LinearMap? ###
            copy!(kp_k, (length(Ψ_k)==1) ? Ψ_k[1] : kron(Ψ_k...)) # handles case K=2
            @inbounds for i=1:N
                copy!(Xk, tenmat(reshape(view(X,:,i),Tuple(d)),k))
                mul!(XkT, kp_k, copy(transpose(Xk)))
                # mul!(sumKGramTilde, Xk, XkT, 1.0/N, 1.0)
                mul!(X_kGramTilde[j], Xk, XkT, 1.0/N, 1.0)
            end
        end
    end
    # mul!(crossTerm, sumKGramTilde, Diagonal(eye(dk)))
    Dtr = X_kGram_k*Ψ[k] + Ψ[k]*X_kGram_k + 2*sum(X_kGramTilde)
    
    return Dlogdet + Dtr, X_kGramTilde
end


# function h_grad!(D::AbstractArray{<:Real}, G::AbstractVector{<:AbstractArray}, Ψ::AbstractVector{<:AbstractArray}, k::Int, 
#     X::AbstractArray{<:Real}, X_kGram_k::AbstractArray{<:Real})
#     """
#     In place version of h_grad, store the gradient values in D
#     """
#     D, G .= h_grad(Ψ, k, X, X_kGram_k)
#     nothing
# end


function scad_pen(x::Real, reg::Real, a::Real)
    """
    SCAD penalty funciton
    """
    if abs(x) <= reg
        return reg*abs(x)
    elseif (abs(x) > reg) & (abs(x) <= a*reg)
        return (2*a*reg*abs(x) - reg^2 - x^2)/(2*(a - 1))
    else
        return reg^2*(a + 1)/2
    end
end


function mcp_pen(x::Real, reg::Real, a::Real)
    """
    MCP penalty funciton
    """
    if abs(x) <= a*reg
        return reg*abs(x) - x^2/(2*a)
    else
        return a*reg^2/2
    end
end


function l1_grad_mat(Ψk::AbstractArray, reg::Real)
    """
    Sub-derivative of an l1 penalty function on the off-diagonal of a matrix
    """
    return reg.*sign.(Ψk - Diagonal(Ψk))
end


function scad_grad(x::Real, reg::Real, a::Real)
    """
    Function to evaluate derivative of (SCAD penalty - L1 penalty) with
    parameter a>2 and regularization parameter reg
    """
    if abs(x) <= reg
        return 0
    elseif (abs(x) > reg) & (abs(x) <= a*reg)
        return (-x + reg*sign(x))/(a - 1)
    else
        return -reg*sign(x)
    end
end


function mcp_grad(x::Real, reg::Real, a::Real)
    """
    Function to evaluate derivative of (MCP penalty - L1 penalty) with
    parameter a>2 and regularization parameter reg
    """
    if abs(x) <= a*reg
        return -x/a
    else
        return -reg*sign(x)
    end
end


function l1_prox(Ψk::AbstractArray, λk::Real)
    """
    Off-diagonal soft-thresholding
    In
    - Ψk: matrix to be thresheld
    - λk: regularization parameter
    Out
    - thresholed matrix
    """
    soft = (x,reg) -> sign(x)*max(abs(x)-reg,0)
    return Diagonal(Ψk) + soft.(Ψk - Diagonal(Ψk), λk)
end


function g_prox(Ψk::AbstractArray, λk::Real, η::Real, a::Real, regtype::String)
    """
    General elementwise proximal operator for L1, SCAD, or MCP shrinkage
    In
    - Ψk: matrix to be thresheld
    - λk: regularization parameter
    - η: step size
    - a: parameter for SCAD and MCP
    - regtype: type of thresholding
    Out
    - thresholded matrix
    """
    if regtype == "L1"
        return l1_prox(Ψk, λk*η)
    elseif regtype == "SCAD"
        return l1_prox(Ψk - η.*scad_grad.(Ψk - Diagonal(Ψk), λk, a), λk*η)
    elseif regtype == "MCP"
        return l1_prox(Ψk - η.*mcp_grad.(Ψk - Diagonal(Ψk), λk, a), λk*η)
    end
end


function g_prox!(ΨkTilde::AbstractArray, Ψk::AbstractArray, λk::Real, η::Real, a::Real, regtype::String)
    """
    In place off-diagonal soft-thresholding
    """
    ΨkTilde .= g_prox(Ψk, λk, η, a, regtype)
    nothing
end


function cost(Ψ::AbstractVector{<:AbstractArray}, λ::Vector{<:Real}, a::Real, regtype::String,
    X::AbstractArray{<:Real}, X_kGram::AbstractVector{<:AbstractArray})
    """
    Compute the SyGlasso cost function
    """
    d = size.(Ψ,1)
    K = length(X_kGram)
    N = size(X,2)
    # logdet term
    logDet = -logdet(Diagonal(kroneckersum_list([Diagonal(Ψ[k]) for k=1:K]))^2)
    # trace term
    crossTerm = 0
    sqrTerm = sum([tr(Ψ[k]^2*X_kGram[k]) for k=1:K])
    @inbounds for k=1:(K-1)
        crossTermMat = zeros(d[k],d[k]) # pre-allocate
        Xk = zeros(d[k],Int(prod(d)/d[k])) # pre-allocate
        XkT = zeros(Int(prod(d)/d[k]),d[k]) # pre-allocate
        kp_k = spzeros(Int(prod(d)/d[k]),Int(prod(d)/d[k])) # pre-allocate
        @inbounds for l=(k+1):K
            sumKGramTilde = zeros(size(Ψ[k])) # pre-allocate
            Ψ_k = [(i==l) ? Ψ[l] : Diagonal(ones(d[i])) for i=1:K]
            deleteat!(Ψ_k,k)
            reverse!(Ψ_k)
            ### TODO: Use LuxurySparse? ###
            ### TODO: Use LinearMap? ###
            copy!(kp_k, (length(Ψ_k)==1) ? Ψ_k[1] : kron(Ψ_k...)) # handles case K=2
            @inbounds for i=1:N
                copy!(Xk, tenmat(reshape(view(X,:,i),Tuple(d)),k))
                mul!(XkT, kp_k, copy(transpose(Xk)))
                mul!(sumKGramTilde, Xk, XkT, 1.0/N, 1.0)
            end
            mul!(crossTermMat,Ψ[k],sumKGramTilde)
            crossTerm += 2*tr(crossTermMat)
        end
    end
    # penalty term
    if regtype == "L1"
        pen = sum([λ[k]*sum(abs.(Ψ[k] - Diagonal(Ψ[k]))) for k=1:K])
    elseif regtype == "SCAD"
        pen = sum([sum(scad_pen.(Ψ[k] - Diagonal(Ψ[k]), λ[k], a)) for k=1:K])
    elseif regtype == "MCP"
        pen = sum([sum(mcp_pen.(Ψ[k] - Diagonal(Ψ[k]), λ[k], a)) for k=1:K])
    end
    # logdet + trace + pen
    return logDet + sqrTerm + crossTerm + pen
end
