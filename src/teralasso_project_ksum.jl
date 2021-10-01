using LinearAlgebra # tr
using Kronecker
using Plots

"""
PROJECT_KSUM Projects the inverse of a Kronecker sum matrix [stored as factor
matrices Xs, where Xs[k] is of size ps[k] x ps[k]] onto a
Kronecker sum of dimensions also given by the vector ps.
"""
function project_ksum(Xs, ps)
    K = length(Xs)
    SigInvs = Array{Any}(undef, K)
    Uinvs = Array{Any}(undef, K)
    Xinvs = Array{Any}(undef, K)

    if K > 1 # If K = 1 no projection required
        diagmat = zeros(ps[end:-1:1]...)
        for i = 1:K
            s, U = eigen(Xs[i])
            Uinvs[i] = U
            less = prod(ps[1:i-1])
            more = prod(ps[i+1:end])

            # more & less should swap in the following line.
            diagmat = diagmat + reshape(
                kron(ones(less,1), kron(s, ones(more,1))),
                ps[end:-1:1]...
            )
        end
        diagmat = permutedims(diagmat,K:-1:1)

        trA = undef
        for i = 1:K
            # matricize. Column is ps[i]-dim. Each row is a fiber.
            mtt = reshape(
                permutedims(1 ./ diagmat, [(1:i-1)... (i+1:K)... i]),
                prod(ps) ÷ ps[i],
                ps[i],
            )
            SigInvs[i] = dropdims(mean(mtt, dims=1); dims=1) # mean(mtt,1) # mtt is not callable.
            if i == 1
               trA = mean(SigInvs[i])
            end
            SigInvs[i] = SigInvs[i] .- trA * (K - 1) / K
        end

        for i = 1:K
            Xinvs[i] = Uinvs[i] * diagm(SigInvs[i]) * Uinvs[i]'
        end
    else
        #[U,s,V] = svd(Xs[1])
        s, U = eigen(Xs[1])
        Uinvs[1] = U
        SigInvs[1] = 1 ./ s
        Xinvs[1] = inv(Xs[1])
    end
    return Uinvs, SigInvs, Xinvs
end

"""
Submatrix notation A(i,j|k)

A(i,j|k)_{r,s} = (1/mk) < A, I ⊗ E_{r,s} ⊗ I >
"""
function submatrix(A, ps, k, i, j)
    K = length(ps)
    pre, post = prod(ps[1:k-1]), prod(ps[k+1:K])

     # Ksum def in the paper requires reversed order of Julia dim
    i_post, i_pre = Base._ind2sub((post, pre), i)
    j_post, j_pre = Base._ind2sub((post, pre), j)
    # @vectorize_2nd_arg? Ref?
    # (j_pre - 1) * ps[k] * post + j_post .+ post * (0 : ps[k]-1)
    i_grid = [Base._sub2ind((post, ps[k], pre), i_post, r, i_pre) for r = 1:ps[k]]
    j_grid = [Base._sub2ind((post, ps[k], pre), j_post, s, j_pre) for s = 1:ps[k]]
    # println((i, j), ' ',
    #         collect((i_pre - 1) * ps[k] .+ i_post * (1 : ps[k])), ' ',
    #         collect((j_pre - 1) * ps[k] .+ j_post * (1 : ps[k])))
    return A[i_grid, j_grid]
end

function test_submatrix_repr()
    ps = [2, 2, 2]
    p = prod(ps)

    #A = reshape(1:p^2, p, p)
    A = rand(p, p)

    K = length(ps)
    for k = 1:K #1:K
        mk = p ÷ ps[k]
        pre, post = prod(ps[1:k-1]), prod(ps[k+1:end])
        B = zeros(size(A))
        for i = 1:mk
            i_post, i_pre = Base._ind2sub((post, pre), i)
            for j = 1:mk
                j_post, j_pre = Base._ind2sub((post, pre), j)
                sub = submatrix(A, ps, k, i, j)
                E_pre = zeros(pre, pre)
                E_pre[i_pre, j_pre] = 1
                E_post = zeros(post, post)
                E_post[i_post, j_post] = 1
                B += reduce(kron, [E_pre, sub, E_post])
                T = reduce(kron, [E_pre, sub, E_post])
                display(heatmap([A T],
                        yflip=true,
                        aspect_ratio=:equal,
                        title=string([i, j])))
            end
        end
        print(all(A == B))
    end
end

"""
The projection operation in Lemma 33 in the paper
"""
function project_ksum_v1(A, ps)
    p = prod(ps)
    K = length(ps)
    Ak = Array{Any}(undef, K)
    for k = 1:K
        mk = p ÷ ps[k]
        sub = [submatrix(A, ps, k, i, i) for i = 1:mk] # Diag mode-k submatrices
        Ak[k] = sum(sub) / mk  # Statistics.mean
    end
    tau = tr(A) / p
    A_proj = reduce(⊕, Ak) - (K - 1) * tau * I
    return A_proj
end
