### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE.jl")
using Plots

### ============================================================================
### Generate Training Data
### ============================================================================

# Random input from -5 to 5
sample_size = 300
X = rand(Float32, sample_size) * 10 .- 5 |> sort!

# Sine function output with added noise
Y = sin.(X) .+ Float32(1e-3) * (rand(Float32, sample_size) .- 0.5f0)

# Visualization of sine function and training data
xgrid = -5:0.1:5
function plot_data()
    scatter(X, Y, label=:none, ms=3, alpha=0.5)    # training data
    plot!(xgrid, sin.(xgrid), label="sine", lw=2, c=:red)    # sine function
end
pl1 = plot_data()


### ============================================================================
### Augmented Neural ODE Model
### ============================================================================

# Dimension of the model
dim = 5
# Derivative model
function model(z, θ, t)
    z_aug = hcat(X,zeros(length(X),dim-1),ones(length(X))) |> transpose
    return dropdims(sum((z_aug*z_aug') .* reshape(θ, dim+1, dim+1, dim), dims=(1,2)), dims=(1,2))
end
θ = (rand(Float32, (dim+1)*(dim+1)*dim) .- 0.5e0) * 1.0e-1


# Define Augmented Neural ODE
anode_model = NeuralODE(model, θ, [0.0f0, 1.0f0])

# Parameters to be optimized, including time span
anode_params = Flux.params(anode_model)

anode_model(X)
# Sum of squared error as loss function
anode_loss() = sum(abs2, anode_model(X)[end][1, :] - Y)

# Set up to run for 100 epochs
anode_data = Iterators.repeated((), 100)

# Optimizer
anode_opt = ADAM(0.01)

# Store losses for visualization
anode_losses = Float32[]
anode_cb = () -> begin
    push!(anode_losses, anode_loss())
end

# Training loop
Flux.train!(anode_loss, anode_params, anode_data, anode_opt, cb=anode_cb)

# Augmented Neural ODE visualization
aug_xgrid = hcat(xgrid, zeros(eltype(xgrid), length(xgrid), 3)) |> transpose
plot!(pl1, xgrid, anode_model(aug_xgrid)[end][1, :], lw=2, c=:purple,
      xlabel="x", ylabel="y", label="Augmented Neural ODE Model")

# Plot losses versus epochs
plot!(pl1, 1:100, anode_losses, label="Augmented Neural ODE Loss",
xlabel="Epoch", ylabel="Loss", c=:purple)
      
# Plot all visualizations
plot(pl1, pl2, legendfontsize=6, size=(800, 400))




for d in anode_data
    # `d` should produce a collection of arguments
    # to the loss function
  
    # Calculate the gradients of the parameters
    # with respect to the loss function
    grads = Flux.gradient(anode_params) do
      loss(d...)
    end
  
    # Update the parameters based on the chosen
    # optimiser (opt)
    Flux.Optimise.update!(opt, anode_params, grads)
  end

