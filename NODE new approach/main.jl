### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE2.jl")
using Plots
using JLD
using PreallocationTools
using Random

### ============================================================================
### Data loading
### ============================================================================

X = load("./NODE New approach/data.jld", "X")
Y = load("./NODE New approach/data.jld", "Y")
#dim = size(Y[1], 1)
dim = 5

### ============================================================================
### Define Model
### ============================================================================

# Lotka-Volterra equations
#= function model(z, θ, t)
    return z .* (reshape(θ,dim,dim+1)*vcat(z,1e0))
end =#

# Riccati equations
function model(z, θ, t)
    z_aug = vcat(z,1e0)
    return dropdims(sum((z_aug*z_aug') .* reshape(θ, dim+1, dim+1, dim), dims=(1,2)), dims=(1,2))
end

# S-system equations
#= function model(z, θ, t)
    a = θ[1:dim]
    b = θ[dim+1:2*dim]
    g = reshape(θ[(2*dim+1):(2*dim+dim*dim)], dim, dim)
    h = reshape(θ[dim*(2+dim)+1 : 2dim*(1+dim)], dim, dim)
    dz = a .* prod(z'.^g, dims=2) - b .* prod(z'.^h, dims=2)
    return dz
end =#


### ============================================================================
### Training parameters
### ============================================================================

# Set up the Neural ODE to run for ### epochs
epochs = 5000
data = Iterators.repeated((), epochs)

reltol = 0.001
tol = reltol * max(Y...)

# Optimizer
opt = ADAM(0.01)


### ============================================================================
### Neural ODE initialization
### ============================================================================
    
# Initialization of L-V parameters and starting populations at time 0
#θ = (rand(Float32, dim*(dim+1)) .- 0.5e0) * 1.0e-1
#z0 = (rand(Float32, dim)) * 1.0e-1

# Initialization of Ricatti parameters and starting populations at time 0
θ = (rand(Float32, (dim+1)*(dim+1)*dim) .- 0.5e0) * 1.0e-1
z0 = (rand(Float32, dim)) * 1.0e-1

# Initialization of S-system parameters and starting populations at time 0
#θ = (rand(Float32, 2*dim*(dim+1)) .- 0.5e0) * 1.0e-1
#z0 = 1 .+ (rand(Float32, dim)) * 1.0e-1

    
# Define Neural ODE layer
tspan = [X[1], X[end]]
node = NeuralODE(model, z0, θ, tspan)

# Parameters to be optimized
params = Flux.params(node)

### ============================================================================
### Training 
### ============================================================================
  
# Loss for 1-dimensional functions
function loss()
    zs = node(saveat=X, abstol=10e-12, reltol=10e-12, maxiters=1e5)#, alg=Rosenbrock23())
    return Flux.mse(hcat(zs...)[1,:], Y) 
end


# Update plots at each epoc
anim = Animation()
losses = []
cb = () -> begin
    plot(X,Y, linewidth=3, labels="sin(3x)")
    plot!(solve(remake(ODEProblem(model, z0, tspan, θ); p=node.θ, u0=node.z_t0)), legend=:right,#, Tsit5()),
    ylims=(-1,1), vars=(1), label="NODE ouput")
    
    frame(anim)
    l = loss()
    append!(losses, l)
    l < tol && Flux.stop()
end

# Training Loop
Flux.train!(loss, params, data, opt, cb=Flux.throttle(cb, .5))



# Visualizations
gif(anim)
plot(losses, ylim=(0, .01), legend=nothing, xaxis="epoch", yaxis="loss")

plot(solve(remake(ODEProblem(model, z0, tspan, θ); p=node.θ, u0=node.z_t0), alg=Tsit5()), legend=nothing,
     xlab="x")#, label=["output1" "output2"])#ylims=(-1.5,1.5))#, 
plot!(X,Y)#, labels="sin(3x)")

#savefig("output5dims.png")

print(length(losses))




    