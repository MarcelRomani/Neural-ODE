### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE2.jl")
include("model_gen.jl")
using Plots
using JLD
using PreallocationTools
using Random

### ============================================================================
### Data loading
### ============================================================================

X = load("./NODE New approach/data.jld", "X")
Y = load("./NODE New approach/data.jld", "Y")



### ============================================================================
### Define Model
### ============================================================================

# Define size of the model. This parameter is similar to the width of a NN.
dim = 5
# Generate model: Riccati, Lotka-Volterra or S-system
modelName = "Riccati"


model = genModel(modelName, dim)


### ============================================================================
### Training parameters
### ============================================================================

# Set up the Neural ODE to run for ### epochs
epochs = 5000
data = Iterators.repeated((), epochs)

reltol = 0.01
tol = reltol * max(Y...)

# Optimizer
opt = ADAM(0.01)


### ============================================================================
### Neural ODE initialization
### ============================================================================
    
# Initialization of model parameters and starting populations at time 0
θ, z0 = paramsInit(modelName, dim)
    
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

#print(length(losses))




    