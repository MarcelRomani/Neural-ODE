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

# Set up the Neural ODE 100 to run for 100 epochs
X = load("./Test Example/data.jld", "X")
Y = load("./Test Example/data.jld", "Y")
data = zip(X, Y)
n_in = length(X[1])

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
θ = [1.0f0, 1.0f0, 1.0f0, 1.0f0]
#θ = vec(rand(4))
=#
### ============================================================================
### n-Dimensional Lotka-Volterra Model
### ============================================================================

function dzdt_nlv_gen(dim)
    # Definition of the derivative model
    function dzdt_nlv(z, θ, t)
        A = reshape(θ[1:dim*dim], dim, dim)
        λ = θ[(dim*dim+1):end]
        return z .* (λ + A*z)
    end

    # Initial parameters
    θ_nlv = rand(Float32, dim * (dim+1)) .- 0.5f0

    return dzdt_nlv, θ_nlv
end

dzdt, θ = dzdt_nlv_gen(n_in)



### ============================================================================
### Neural Network Model
### ============================================================================
#=
# Derivative model
nn_dense = Chain(Dense(nn_size, 4, tanh), Dense(4, nn_size))
θ, re = Flux.destructure(nn_dense)
dzdt_nn(z, θ, t) = re(θ)(z)
=#

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
function loss(z0, y)
    zs = node(z0, saveat=1.0f0)[end]
    return Flux.mse(zs, y)
end

function mean_squared_error(input, gt)
    tse = 0.0f0
    for (z0,y) in zip(input,gt)
        tse += loss(z0,y)
    end
    return tse/length(input)
end


### ============================================================================
### Definition of the Training 
### ============================================================================

function my_custom_train!(loss, ps, data, opt, cb=()->())
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss
    local gs
    local gs_key
    grads = zeros(Float32, length(collect(keys(ps.params.dict))[1]))
    for d in data
        # back is a method that computes the product of the gradient so far with its argument.
        training_loss, back = Zygote.pullback(() -> loss(d...), ps)
        # Apply back() to the correct type of 1.0 to get the gradient of loss.
        gs = back(one(training_loss))
        # In order to do batch gradient descent, we need to store the gradients at each data point
        # and update at the end.
        gs_key = (typeof(collect(keys(gs.grads))[1]) == Vector{Float32}) ? collect(keys(gs.grads))[1] : collect(keys(gs.grads))[2]
        grads += gs.grads[gs_key]
    end
    # Gradient object created as the average of all the data points
    new_gs = copy(gs)
    new_gs.grads[gs_key] = grads/length(data)
    # Update with the data of the whole batch
    Flux.update!(opt, ps, new_gs)
    cb()
end

# Predict outputs
function predict(X)
    sol = Vector{Vector{Float32}}()
    for z0 in X
        push!(sol,node(z0, saveat=1.0f0)[end])
    end
    return sol
end

# Update the plot of our populations each epoch
anim = Animation()
cb = () -> begin
    scatter(mapreduce(permutedims, vcat, Y)[:,1], 
            mapreduce(permutedims, vcat, predict(X))[:,1])#, 
            #xlims=(-1,1), ylims=(-1,1))
    scatter!(mapreduce(permutedims, vcat, Y)[:,2],
            mapreduce(permutedims, vcat, predict(X))[:,2])
    scatter!(mapreduce(permutedims, vcat, Y)[:,3],
            mapreduce(permutedims, vcat, predict(X))[:,3])
            
    #=plot(solve(remake(ODEProblem(dzdt_lv, [0.5f0, 0.38f0], (0.0f0, 1.0f0), θ); p=node.θ), Tsit5()),
         xlims=(0, 1), ylims=(0, 1))=#
    frame(anim)
end
throttled_cb = Flux.throttle(cb, 0.01)


# Optimizer
opt = ADAM(0.1)

# Training loop
for i in 1:2000
    my_custom_train!(loss, params, data, opt, cb)
    if i % 20 == 0
        l = mean_squared_error(X, Y)
        println("epoch: ", i, ", loss = ", l)
        if l < 1.0f-4
            break
        end
    end
end

# Training visualization
gif(anim)


#scatter(mapreduce(permutedims, vcat, Y)[:,1], mapreduce(permutedims, vcat, predict(X))[:,1])
#scatter!(mapreduce(permutedims, vcat, Y)[:,2], mapreduce(permutedims, vcat, predict(X))[:,2])
