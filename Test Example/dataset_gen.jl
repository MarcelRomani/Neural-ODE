### ============================================================================
### Load Packages
### ============================================================================

include("../NeuralODE.jl")
using Plots
using JLD

### ============================================================================
### Lotka-Volterra Model Data Generation
### ============================================================================
function gen_lotka_volterra(sample_size=50)
    # Lotka-Volterra equations
    function model(z, θ, t)
        x, y = z
        α, β, δ, γ = θ
        return [α*x - β*x*y,
                -δ*y + γ*x*y]
    end

    # Definition of the parameters that generate the dataset
    θ = [1.2f0, 2.9f0, 1.5f0, 1.7f0]

    # Define and solve ODE problems
    function solve_ode(z0)
        z0 = vec(z0)
        sol = solve(ODEProblem(model, z0, (0.0f0, 1.0f0), θ), Tsit5())
        return sol.u[end]
    end

    # Starting populations at time 0
    X = Vector{Vector{Float32}}()
    for i in 1:sample_size
        push!(X, vec(rand(Float32, (2,1))*2))
    end
    # Generating outputs by solving the correspondent differential equation
    Y = map(solve_ode, X)

    return X, Y
end

### ============================================================================
### N-Dimensional Lotka-Volterra Model Data Generation
### ============================================================================
function gen_n_lotka_volterra(sample_size=50, n=3)
    # Lotka-Volterra equations
    function model(z, θ, t)
        A = reshape(θ[1:n*n], n, n)
        λ = θ[(n*n+1):end]
        return z .* (λ + A*z)
    end

    # Definition of the parameters that generate the dataset
    θ = rand(Float32, n*(n+1)) .- 0.5f0

    # Define and solve ODE problems
    function solve_ode(z0)
        z0 = vec(z0)
        sol = solve(ODEProblem(model, z0, (0.0f0, 1.0f0), θ), Tsit5())
        return sol.u[end]
    end

    # Starting populations at time 0
    X = Vector{Vector{Float32}}()
    for i in 1:sample_size
        push!(X, vec(rand(Float32, (n,1))))
    end
    # Generating outputs by solving the correspondent differential equation
    Y = map(solve_ode, X)

    return X, Y
end



### ============================================================================
### Sine Data Generation
### ============================================================================
function gen_sine(sample_size=500, depth=4)
    X = rand(Float32, sample_size) * Float32(2π) |> sort!
    Y = sin.(X)
    X_aug = hcat(X, zeros(eltype(X), length(X), depth-1)) |> permutedims
    #X_aug = repeat(X, 1, depth) |> permutedims
    
    return X_aug, Y
end


### ============================================================================
### Cosine Data Generation
### ============================================================================
function gen_cosine(sample_size=500, depth=4)
    X = rand(Float32, sample_size) * Float32(4π) .- Float32(2π) |> sort!
    Y = cos.(X)
    X_aug = hcat(X, zeros(eltype(X), length(X), depth-1)) |> permutedims
    #X_aug = repeat(X, 1, depth) |> permutedims
    
    return X_aug, Y
end

### ============================================================================
### SineCosine Data Generation
### ============================================================================
function gen_sine_cosine(sample_size=500)
    X = rand(Float32, sample_size) * Float32(4π) .- Float32(2π) |> sort!
    Y = hcat(sin.(X), cos.(X)) |> permutedims
    X_aug = repeat(X, 1, 2) |> permutedims
    
    return X_aug, Y
end


### ============================================================================
### X^2 Data Generation
### ============================================================================
function gen_poly(sample_size=50, depth=4, d=2)
    X = rand(Float32, sample_size) * 30 .- 15 |> sort!
    Y = X .^ d
    #X_aug = hcat(X, zeros(eltype(X), length(X), 3)) |> permutedims
    X_aug = hcat(X, zeros(eltype(X), length(X), depth-1)) |> permutedims
    #X_aug = repeat(X, 1, depth) |> permutedims
    
    return X_aug, Y
end




sample_size = 100
depth = 4
#X, Y = gen_lotka_volterra(sample_size)
#X, Y = gen_n_lotka_volterra(sample_size, 3)
#X, Y = gen_sine(sample_size, 1)
#X, Y = gen_cosine(sample_size)
#X, Y = gen_poly(sample_size, depth, 2)
X, Y = gen_sine_cosine(sample_size)


save("./Test Example/data.jld","X", X, "Y", Y)
println("Done generating data!")
