"""
Implementation of the Kronecker PCA method for covariance estimation

Paper: Tsiligkaridis T, Hero AO. 
    Covariance estimation in high dimensions via kronecker product expansions.
     IEEE Transactions on Signal Processing. 2013 Aug 22;61(21):5347-60.

Author: Wayne Wang
Last modified: Apr 1, 2021
"""


function rearrange(A::AbstractMatrix{<:Real},
    m1::Int, n1::Int, m2::Int, n2::Int)
    """
    Rearrange operator of (Van Loan, 1993)

    Inputs:
        A: input matrix comprising m1 × n1 blocks of m2 × n2 matrices 
    Output:
        R: rearranged matrix comprising n1 × n2 blocks of m1 × m2 matrices
    """
    R = zeros((m1*n1,m2*n2))
    Aij = zeros((m2,n2))

    nrow = 1
    for j=1:n1
        for i=1:m1
            Aij .= view(A, ((i-1)*m2+1):i*m2, ((j-1)*n2+1):j*n2)
            R[nrow,:] .= Aij[:]
            nrow += 1
        end
    end 

    return R
end


function rearrange_adj(R::AbstractMatrix{<:Real},
    m1::Int, n1::Int, m2::Int, n2::Int)
    """
    Adjoint operator of the rearrange operator (Van Loan, 1993)

    Inputs:
        R: rearranged matrix comprising n1 × n2 blocks of m1 × m2 matrices  
    Output:
        A: input matrix comprising m1 × n1 blocks of m2 × n2 matrices 
    """
    A = zeros((m1*m2,n1*n2))
    Aij = zeros(m2*n2)

    nrow = 1
    for j=1:n1
        for i=1:m1
            Aij .= view(R, nrow, :)
            A[((i-1)*m2+1):i*m2, ((j-1)*n2+1):j*n2] .= reshape(Aij,(m2,n2))
            nrow += 1
        end
    end 

    return A
end


function kron_pca(S::AbstractMatrix{<:Real}, k::Int,
    px::Int, py::Int; λ::Real=0)
    """
    PCA on the rearranged sample covariance of data matrix X

    Inputs:
        S: sample covariance matrix d × d, where d = pxpy
        k: top-k PCA/SVD
    Optional:
        λ: threshold value for singular values
    Output:
        A, B: top-k left (A of size px²×k) and right (B of size py²×k)
         singular vectors
    """
    # rearrange the sample covariance
    R = rearrange(S, px, px, py, py)

    # topc-k SVD for rearranged sample covariance
    F = svds(R, nsv=k)[1]
    U = F.U
    Vt = F.Vt
    svdvals = F.S

    # form covariance estiamte
    SigmaH = zeros(size(S))
    for j=1:k
        SigmaH .+= max(svdvals[j]-λ/2,0)*kron(reshape(view(U,:,j),(px,px)),
         reshape(view(Vt,j,:),(py,py)))
    end

    return SigmaH
end


function robust_pca(X::AbstractMatrix{<:Real}, lambdaL::Real, lambdaS::Real,
     method::String; tau::Real=0.5, iter::Int=5, r::Int=5)
    """
    Perform robust PCA as in Nadakuditi et al 2014 (SSP). Proximal
    gradient method.
    """
    # dims
    c = size(X,2)/size(X,1)
    q = minimum(size(X))

    # initialize
    L = zeros(size(X))
    S = spzeros(size(X,1),size(X,2))
    M = deepcopy(X) 
    U = zeros((size(X,1),q)) 
    Vt = zeros((q,size(X,2))) 
    s = zeros(q) 

    for i=1:iter 
        # SVT
        if method == "SVT"
            F = svd(M-S)
            U .= F.U
            Vt .= F.Vt
            s .= F.S
            r = count(i -> i>lambdaL*tau, s)
            
            if r == 0
                r = 1
            end

            if s[1] == 0
                s[1] = 1e-10
            end
        elseif method == "OptShrink"
            F = svd(M-S)
            U .= F.U
            Vt .= F.Vt
            s .= F.S

            for i = 1:r
                sumV = 1/(q-r)*sum(s[i]./(s[i]^2 .- s[r+1:q].^2))
                D = sumV*(c*sumV+(1-c)/s[i])
                sumVprime = 1/(q-r)*sum((-s[i]^2 .- s[r+1:q].^2)./(s[i]^2 .- s[r+1:q].^2).^2)
                Dprime = (c*sumV+(1-c)/s[i])*sumVprime + (c*sumVprime-(1-c)/s[i]^2)*sumV
                s[i] = -2*D/Dprime
            end   
        end
      
        L = U[:,1:r]*Diagonal(s[1:r])*Vt[1:r,:]

        # Soft
        S .= M .- L
        S .= sign.(S).*max.(abs.(S).-lambdaS*tau,0)
        
        # Update M
        M .= L .+ S .- tau*(L .+ S .- X)
    end

    return L, S, U[:,1:r], s[1:r], Vt[1:r,:]
end


function robust_kron_pca(S::AbstractMatrix{<:Real}, px::Int, py::Int, 
    lambdaL::Real, lambdaS::Real, method::String;
    tau::Real=0.5, iter::Int=5, r::Int=5)
    """
    Robust Kronecker PCA method for covariance estimation
    """
    # rearrange the sample covariance
    R = rearrange(S, px, px, py, py)

    # robust pca for rearranged sample covariance
    L, S, _, _, _ = robust_pca(R, lambdaL, lambdaS, method; tau, iter, r)

    # form covariance estiamte
    SigmaH = rearrange_adj(L+S, px, px, py, py)

    return SigmaH
end


# ## Testing
# using Distributions
# using PDMats
# using BenchmarkTools

# px = py = 25
# N = 100
# X = zeros((px*py,N))

# for i=1:N
#     X[:,i] .= vec(rand(MatrixNormal(zeros((px,py)),ScalMat(px,2.0),ScalMat(py,4.0))))
# end

# S = cov(copy(X'))
# lambdaL = 20*(px^2+py^2+log(max(px,py,N)))/N
# lambdaS = 20*sqrt(log(px*py)/N)

# begin
#     @btime SigmaH = robust_kron_pca(S,px,py,lambdaL,lambdaS,"OptShrink";tau=0.5,r=5)
#     @btime SigmaH_kron_pca = kron_pca(S,5,px,py)
# end

# norm(8.0*I(px*py) - SigmaH)
# norm(8.0*I(px*py) - SigmaH_kron_pca)

# isposdef((SigmaH+SigmaH')/2.0)