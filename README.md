# TensorGraphicalModels
TensorGraphicalModels.jl is a suite of Julia tools for estimating high-dimensional multiway (tensor-variate) covariance and inverse covariance matrices.

# WORK IN PROGRESS
TODO: add a Julia colab page.

## Installation
```julia
] add https://github.com/ywa136/TensorGraphicalModels.jl
```

## Examples
Example code for fitting a KP inverse covariance model:
```julia
using TensorGraphicalModels

model_type = "kp"
sub_model_type = "sb" #this defines the structure of the Kronecker factors, sb = star-block
K = 3
N = 1000
d_list = [5, 10, 15]

X = gen_kronecker_data(model_type, sub_model_type, K, N, d_list) #multi-dimensional array (tensor) of dimension d_1 × … × d_K × N
Ψ_hat_list = kglasso(X)
```

Example code for fitting a KS inverse covariance model:
```julia
using TensorGraphicalModels

model_type = "ks"
sub_model_type = "sb" #this defines the structure of the Kronecker factors, sb = star-block
K = 3
N = 1000
d_list = [5, 10, 15]

X = gen_kronecker_data(model_type, sub_model_type, K, N, d_list, tensorize_out = false) #matrix of dimension d × N

# compute the mode-k Gram matrices (the sufficient statistics for TeraLasso)
X_kGram = [zeros(d_list[k], d_list[k]) for k = 1:K]
Xk = [zeros(d_list[k], Int(prod(d_list) / d_list[k])) for k = 1:K]
for k = 1:K
    for i = 1:N
        copy!(Xk[k], tenmat(reshape(view(X, :, i), d_list), k))
        mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0 / N, 1.0)
    end
end

Ψ_hat_list, _ = teralasso(X_kGram)
```

Example code for fitting a Sylvester inverse covariance model:
```julia
using TensorGraphicalModels

model_type = "sylvester"
sub_model_type = "sb" #this defines the structure of the Kronecker factors, sb = star-block
K = 3
N = 1000
d_list = [5, 10, 15]

X = gen_kronecker_data(model_type, sub_model_type, K, N, d_list, tensorize_out = false) #matrix of dimension d × N

# compute the mode-k Gram matrices (the sufficient statistics for TeraLasso)
X_kGram = [zeros(d_list[k], d_list[k]) for k = 1:K]
Xk = [zeros(d_list[k], Int(prod(d_list) / d_list[k])) for k = 1:K]
for k = 1:K
    for i = 1:N
        copy!(Xk[k], tenmat(reshape(view(X, :, i), d_list), k))
        mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0 / N, 1.0)
    end
end

Psi0 = [sparse(eye(d_list[k])) for k = 1:K]
fun = (iter, Psi) -> [1, time()] # NULL func
lambda = [sqrt(px[k] * log(prod(d_list)) / N) for k = 1:K] 

Ψ_hat_list, _ = syglasso_palm(X, X_kGram, lambda, Psi0, fun = fun)
```

Example code for fitting a KPCA covariance model:
```julia
using TensorGraphicalModels

px = py = 25 #works for K=2 modes only
N = 100
X = zeros((px * py, N))

for i=1:N
    X[:, i] .= vec(rand(MatrixNormal(zeros((px, py)), ScalMat(px, 2.0), ScalMat(py, 4.0))))
end

S = cov(copy(X')) #sample covariance matrix
lambdaL = 20 * (px^2 + py^2 + log(max(px, py, N))) / N
lambdaS = 20 * sqrt(log(px * py)/N)

# robust Kronecker PCA methods using singular value thresholding
Sigma_hat = robust_kron_pca(S, px, py, lambdaL, lambdaS, "SVT"; tau = 0.5, r = 5)
```
