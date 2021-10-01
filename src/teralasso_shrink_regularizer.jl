function shrink_regularizer(A, lambda,stepSz, type,a)
    #type = L1; SCAD; MCP
    if type == "L1"
        a_shr = shrinkL1(A, lambda * stepSz)
    elseif type == "SCAD"
        a_shr = shrinkL1(A - stepSz * SCAD_derivative(A, a, lambda), lambda * stepSz)
    elseif type == "MCP"
        a_shr = shrinkL1(A - stepSz * MCP_derivative(A, a, lambda), lambda * stepSz)
    end
    return a_shr
end

function shrinkL1(A, lambda)
    dg = diag(A)
    soft = (x, t) -> sign(x) * max(0, abs(x) - t)
    #A = soft.(A, lambda) # A .= modifies Psi in place.
    B = soft.(A, lambda)
    B[diagind(A)] = dg
    return B
end

function SCAD_derivative(A, a, rho)
    #Function to evaluate derivative of [SCAD penalty - L1] with parameter a .> 2 &
    #regularization parameter rho. Matricized with no offdiagonal penalty.
    A = A - diagm(diag(A))
    trm1 = (abs.(A) .> rho) .& (abs.(A) .<= a*rho).*(-2*A./(2*(a-1)) + rho.*sign.(A)/(a-1))
    trm2 = (abs.(A) .> a*rho).*(-rho.*sign.(A))
    dy = trm1 + trm2
    return dy
end

function MCP_derivative(A, a, rho)
    #Function to evaluate derivative of [MCP penalty - L1] with parameter a .> 0 &
    #regularization parameter rho. Matricized with no offdiagonal penalty.
    A = A - diagm(diag(A))
    dy = (abs.(A) .< rho*a).*sign.(A).*rho.*(-abs.(A)/(rho*a))
    return dy
end
