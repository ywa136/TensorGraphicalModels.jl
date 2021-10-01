"""
Implementation of the Kronecker Graphical Lasso algorithm for tensor graphical model estimation

Author: Wayne Wang
Last modified: 07/29/2021
"""


function kglasso(X::AbstractArray{<:Real}; Ψ0_list::AbstractArray = Array{Array{Float64, 2}, 1}(),
    λ_list::AbstractArray = Array{Float64, 1}(), N_iter::Int = 2, ϵ::Real=1e-5)
    """
    The Kronecker graphical lasso algorithm, which applies the Glasso algorithm alternatively
    Input:
        - X: multi-dimensional array (tensor) of dimension d_1 × … × d_K × N
    Output:
        - Ψ_list: {Ψ_1,…,Ψ_k}
    """
    # initial setup
    dimensions = size(X)
    K = length(dimensions) - 1
    d_list = dimensions[1:K]
    d = prod(d_list)
    N = dimensions[end]

    # initilize the precision matrices if not provided
    if length(Ψ0_list) == 0
        for k = 1:K
            push!(Ψ0_list, 1.0 * I(d_list[k]))
        end
    end

    # set penalty parameters if not provided
    if length(λ_list) == 0
        for k = 1:K
            push!(λ_list, 5 * sqrt(d_list[k] * log(d) / N))
        end
    end

    S_list = [zeros((dk, dk)) for dk in d_list]
    Ψ_list = copy(Ψ0_list)

    for _ = 1:N_iter
        for k = 1:K
            S_list[k] .= Tilde_S_k(k, N, X, Ψ_list)
            Ψ_list[k] .= glasso(S_list[k], λ_list[k])
        end
    end

    return Ψ_list
end


function Tilde_S_k(k::Int, N::Int, X::AbstractArray, Ψ_list::AbstractArray)
    """
    Function to compute:
        (d_k / N*d) * ∑_{i=1}^N V_i^k (V_i^k)^T
    where V_i^k = [X_i × {Ψ_1^(1/2),…,Ψ_(k-1)^(1/2),I,Ψ_(k+1)^(1/2),…,Ψ_K^(1/2)}]_(k);
    here × is the tensor product and []_(k) the mode-k matricication. 
    """
    Ψ_list = copy(Ψ_list)
    d_list = size.(Ψ_list, 1)
    dk = d_list[k]
    d = prod(d_list)
    K = length(d_list)
    S = zeros((dk, dk))
    X_mode_k = zeros(dk, Int(d / dk))
    X_mode_k_trans = zeros(Int(d / dk), dk)

    deleteat!(Ψ_list, k)
    reverse!(Ψ_list)
    KP_Ψ_list = (length(Ψ_list) == 1) ? Ψ_list[1] : kron(Ψ_list...) # handles case K=2

    @inbounds for i = 1:N
        ## TODO: modify to work with various K ##
        # str_to_eval = string("X[", join([":," for _ = 1:K]), "i]") # adapt to different K
        # X_mode_k .= tenmat(eval(Meta.parse(str_to_eval)), k)
        X_mode_k .= tenmat(X[:, :, i], k)
        mul!(X_mode_k_trans, KP_Ψ_list, copy(X_mode_k'))
        mul!(S, X_mode_k, X_mode_k_trans, dk / (d * N), 1.0)
    end

    return S
end


# # Testing
# include("utils.jl")
# include("utils_sim.jl")
# include("glasso.jl")

# X = gen_kronecker_data("kp", "sb", 2, 1000, [5,10,15])
# Ψ_hat_list = kglasso(X)