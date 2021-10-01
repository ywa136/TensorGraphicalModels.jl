"""
Utility functions for Tensor Graphical Models

Author: Wayne Wang
Last modified: 07/14/2021
"""

function kroneckersum_alt(A::AbstractMatrix, B::AbstractMatrix)
    """
    Alternate Kronecker sum defined as I ⊗ A + B ⊗ I
    """
    return kroneckersum(B,A)
end


function kroneckersum_list(mat_list::AbstractVector{<:Any})
    """
    Kronecker sum for a list of matrices using package Kronecker.jl
    In
        - mat_list::AbstractVector{<:Any}: a list of matrices 
    Out
        - mat::KroneckerSum: mat_list[1] ⊕ ... ⊕ mat_list[end] 
    """
    K = length(mat_list)
    mat = copy(mat_list[1])
    if K == 1
        return mat
    else
        for j=2:K
            mat = kroneckersum_alt(mat, mat_list[j])
        end
    end
    return mat
end


function mul_alt!(C::StridedMatrix, X::StridedMatrix, A::SparseMatrixCSC)
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(C, zero(eltype(C)))
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mX
            C[multivec_row, col] += X[multivec_row, ki] * kv
        end
    end
    C
end


import Base.*
function *(B::StridedMatrix, A::SparseMatrixCSC)
    mB, nB = size(B)
    nB == A.m || throw(DimensionMismatch())
    C = zeros(mB,A.n)
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mB
            C[multivec_row, col] += B[multivec_row, ki] * kv
        end
    end
    C
end


function offdiag(X::AbstractMatrix)
    """
    Extract the off-diagonal matrix of X
    """
    return X - Diagonal(X)
end


function offdiag!(X::AbstractMatrix)
    """
    In placce version of offdiag
    """
    X .= X .- Diagonal(X)
    return X
end
