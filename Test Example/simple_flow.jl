### Here we are trying to generate a simple flow by solving a SODE
### whose solution we know. ### We have found that if the function
### that we want to obtain and its derivative are strictly positive,
### then the solution converges quickly and well.
### In that case, the optimization of the parameters is possible if 
### the starting guesses are close enough to the actual generating ones.

### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE.jl")
using Plots
using JLD

### ============================================================================
### Data loading
### ============================================================================
sample_size = 4
X = rand(Float32, sample_size) * Float32(π/6) .+ Float32(π/3)   |> sort!
Y = hcat(sin.(X), 2*X.^3 + X.^2) |> permutedims
X = repeat(X, 1, 2)
X = hcat(X, zeros(eltype(X), sample_size)) |> permutedims



### ============================================================================
### Model
### ============================================================================
f1(z, A) = sin(A * z)
df1(z, A) = gradient(f1, z, A)[1]

f2(z, B) = B * z^3 + z^2
df2(z, B) = gradient(f2, z, B)[1]

# Derivative model
function dzdt(z, θ, t)
    z1 = @view z[1,:]
    z2 = @view z[2,:]
    z3 = @view z[3,:]
    θ1 = @view θ[1]
    θ2 = @view θ[2]


    dz1 = @. (f1(z1, θ1) - z1) / (z3 + (1 - z3) * df1(z1, θ1))
    dz2 = @. (f2(z2, θ2) - z2) / (z3 + (1 - z3) * df2(z2, θ2))
    dz3 = ones(Float32, length(dz1))
    return hcat(dz1, dz2, dz3) |> permutedims
end

θ = [0.5f0, 0.5f0]
#θ = rand(Float32, 2)

### ============================================================================
### Initialization of the Neural ODE 
### ============================================================================

# Define Neural ODE layer
node = NeuralODE(dzdt, θ, [0.0f0, 1.0f0])

# Parameters to be optimized
params = Flux.params(node)

### ============================================================================
### Definition of the Loss Functions
### ============================================================================

# Sum of squared error
function loss()
    return Flux.mse(node(X)[end][1:2,:], Y)
end

### ============================================================================
### Definition of the Training 
### ============================================================================

# Optimizer and options
opt = ADAM(0.01)
tol = 1.0f-4

# Max Iterations
max_iter = 3000
data = Iterators.repeated((), max_iter)

# Store and visualize losses
losses = Float32[]

# Callback
ode_cb = () -> begin
    l = loss()
    println("Loss: ", l)
    println(θ)
    push!(losses, l)
    l < tol && Flux.stop()
end
throttled_cb = Flux.throttle(ode_cb, 0.5)


# Training loop
Flux.train!(loss, params, data, opt, cb=throttled_cb)


### ============================================================================
### Testing & Plotting
### ============================================================================


# Predict outputs
function predict(X)
    return node(X)[end]
end

#= 
#Test
X_test = rand(Float32, 500) * 2 .- 1 |> sort!
X_test = hcat(X_test, zeros(eltype(X_test), length(X_test), size(X,1)-1)) |> permutedims
#X_test = repeat(X_test, 1, size(X,1)) |> permutedims

plt1 = plot(X_test[1,:], predict(X_test))
plot!(plt1, X_test[1,:], (X_test[1,:]).^3)
plt2 = plot(1:length(losses[40:end]), losses[40:end])
plot(plt1,plt2)


 =#