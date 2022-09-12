### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE.jl")
#include("../NeuralODE2.jl")
using Plots
using JLD

### ============================================================================
### Data loading
### ============================================================================

X = load("./Test Example/data.jld", "X")
Y = load("./Test Example/data.jld", "Y")

n_in = size(X)[1]

### ============================================================================
### Basic Lotka-Volterra Model
### ============================================================================
#=
# Lotka-Volterra equations
function dzdt_lv(z, θ, t)
    x, y = z
    α, β, δ, γ = θ
    return [α*x - β*x*y,
            -δ*y + γ*x*y]
end

# Initial Parameters
#θ = [1.0f0, 1.0f0, 1.0f0, 1.0f0]
θ = rand(Float32, 4, 1)
=#
### ============================================================================
### n-Dimensional Lotka-Volterra Model
### ============================================================================

#= function dzdt_nlv_gen(dim)
    # Definition of the derivative model
    function dzdt_nlv(z, θ, t)
        A = reshape(θ[1:dim*dim], dim, dim)
        λ = θ[(dim*dim+1):end]
        return z .* (λ .+ A*z)
    end

    # Initial parameters
    θ_nlv = (rand(Float32, dim * (dim+1)) .- 0.5f0) * 1f-6

    return dzdt_nlv, θ_nlv
end

dzdt_nlv, θ = dzdt_nlv_gen(n_in) =#



### ============================================================================
### Neural Network Model
### ============================================================================

# Derivative model
nn_dense = Chain(Dense(n_in, 8, relu), Dense(8, n_in))
θ, re = Flux.destructure(nn_dense)
dzdt_nn(z, θ, t) = re(θ)(z)


### ============================================================================
### Initialization of the Neural ODE 
### ============================================================================

# Define Neural ODE layer
node = NeuralODE(dzdt_nn, θ, [0.0f0, 1.0f0])

# Parameters to be optimized
params = Flux.params(node)

### ============================================================================
### Definition of the Loss Functions
### ============================================================================

# Sum of squared error
function loss()
    return Flux.mse(node(X)[end][1,:], Y)
end

### ============================================================================
### Definition of the Training 
### ============================================================================

# Predict outputs
function predict(X)
    return node(X)[end][1,:]
end



# Optimizer
opt = ADAM(0.01)
data = Iterators.repeated((), 1000)

# Store and visualize losses
losses = Float32[]
tol = 1.0f-5

ode_cb = () -> begin
    l = loss()
    println(l)
    push!(losses, l)
    l < tol && Flux.stop()
end
throttled_cb = Flux.throttle(ode_cb, 0.1)


# Training loop
Flux.train!(loss, params, data, opt, cb=throttled_cb)



#Test
X_test = rand(Float32, 500) * 2 .- 1 |> sort!
X_test = hcat(X_test, zeros(eltype(X_test), length(X_test), size(X,1)-1)) |> permutedims
#X_test = repeat(X_test, 1, size(X,1)) |> permutedims

plt1 = plot(X_test[1,:], predict(X_test))
plot!(plt1, X_test[1,:], (X_test[1,:]).^3)
plt2 = plot(1:length(losses[40:end]), losses[40:end])
plot(plt1,plt2)


