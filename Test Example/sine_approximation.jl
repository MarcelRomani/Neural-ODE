### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE.jl")
using Plots
using JLD


### ============================================================================
### Data loading
### ============================================================================

X = load("./Test Example/data.jld", "X")
Y = load("./Test Example/data.jld", "Y")

n_in = size(X,1)



### ============================================================================
### Mathematical Neural ODE Model
### ============================================================================

# Derivative model for [Y, dY]
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
    θ_nlv = (rand(Float32, dim * (dim+1)) .- 0.5f0)* 1.0f-3

    return dzdt_nlv, θ_nlv
end

dzdt_nlv, θ = dzdt_nlv_gen(n_in) =#

### ============================================================================
### Neural Network Model
### ============================================================================

# Derivative model
nn_dense = Chain(Dense(n_in, 4, relu), Dense(4, 4, relu), Dense(4, n_in))
θ, re = Flux.destructure(nn_dense)
dzdt_nn(z, θ, t) = re(θ)(z)

# Define ODE
ode_model = NeuralODE(dzdt_nn, θ, [X[1], X[end]])

# Parameters to be optimized
ode_params = Flux.params(ode_model)

# Set up to run for 200 loops
epochs = 50
ode_data = Iterators.repeated((), epochs)


# Solve for all solutions at each time step X
function predict()
    p = ode_model([Y[1]], saveat=X)
    return hcat(p...)[1,:]
end

# Sum of squared error as loss function
ode_loss() = begin
    predicted = predict()
    Flux.mse(predicted, Y)
end

# Optimizer
ode_opt = ADAM(0.1)

# Store losses for visualization
ode_losses = Float32[]
ode_cb = () -> begin
    push!(ode_losses, ode_loss())
end

# Training loop
Flux.train!(ode_loss, ode_params, ode_data, ode_opt, cb=ode_cb)




# Visualization of sine function and training data
xgrid = -2π:0.1:2π
function plot_data()
    scatter(X[1,:], Y, label=:none, ms=3, alpha=0.5)    # training data
    plot!(xgrid, sin.(xgrid), label="sine", lw=2, c=:red)    # sine function
end


# Plot losses versus epochs
pl3 = plot(1:epochs, ode_losses, label="Neural ODE Model Loss",
           c=:orange, xlabel="Epoch", ylabel="Loss", ylim=(0,1))
           
# Mathematical Neural ODE visualization
pl4 = plot_data()
Ŷ = predict()
plot!(pl4, vec(X), Ŷ, lw=2, c=:orange,
      xlabel="x", ylabel="y", label="Neural ODE Model")
      
# Show plots together
plot(pl4, pl3, size=(800, 400))