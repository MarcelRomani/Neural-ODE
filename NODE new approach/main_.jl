### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE2.jl")
using Plots
using JLD

### ============================================================================
### Data loading
### ============================================================================

X = load("./L-V New approach/data.jld", "X")
Y = load("./L-V New approach/data.jld", "Y")

### ============================================================================
### Define and Solve Lotka-Volterra Model
### ============================================================================

# Lotka-Volterra equations
function model(z, θ, t)
    x, y = z
    α, β, δ, γ = θ
    return [α*x - β*x*y,
            γ*x*y - δ*y]
end




### ============================================================================
### Neural ODE with Lotka-Volterra equations
### ============================================================================
# Starting populations at time 0
z0 = [1.4f0, 0.9f0]

# Parameters
θ = [0.95f0, 2.7f0, 1.6f0, 1.5f0]


# Time span
tspan = [X[1], X[end]]

# Define initial ODE problem
problem = ODEProblem(model, z0, tspan, θ)



# Define Neural ODE layer
node = NeuralODE(model, z0, θ, tspan)

# Parameters to be optimized
params = Flux.params(node)

# Sum of squared error from 1 as loss function
function loss()
    zs = node(saveat=X)
    return Flux.mse(vcat(zs...), vcat(Y...))
end


# Set up the Neural ODE 100 to run for 100 epochs
data = Iterators.repeated((), 200)

# Optimizer
opt = ADAM(0.1)


# Update the plot of our populations each epoch
anim = Animation()
cb = () -> begin
    plot(solve(remake(problem; p=node.θ, u0=node.z_t0), Tsit5()),
         xlims=(0, 10), ylims=(0, 5))
    frame(anim)
end

# Training Loop
Flux.train!(loss, params, data, opt)#, cb=cb)
println("Training finished")

# Training visualization
#gif(anim)
