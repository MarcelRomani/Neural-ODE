using Flux, DifferentialEquations
using JLD, Plots

### ============================================================================
### Lotka-Volterra Model Data Generation
### ============================================================================
function gen_lotka_volterra()
    # Lotka-Volterra equations
    function model(z, θ, t)
        x, y = z
        α, β, δ, γ = θ
        return [α*x - β*x*y,
                -δ*y + γ*x*y]
    end

    # Definition of the parameters that generate the dataset
    θ = [1.2f0, 2.9f0, 1.5f0, 1.7f0]
    z0 = [1.3, 0.7]
    tspan = (0.0f0, 30.0f0)

    problem = ODEProblem(model, z0, tspan, θ)
    sol = solve(problem, saveat=0.1)

    return sol.t, sol.u
end

### ============================================================================
### N-Dimensional Lotka-Volterra Model Data Generator
### ============================================================================
function gen_n_lotka_volterra(dim=3)
    # Lotka-Volterra equations
    function model(z, θ, t)
        A = reshape(view(θ,1:dim*dim), dim, dim)
        λ = view(θ, (dim*dim+1):dim * (dim+1))
        return z .* (λ + A*z)
    end

    # Definition of the parameters that generate the dataset
    θ = (rand(Float32, dim*(dim+1)) .- 0.5f0) * 0.5f0
    z0 = rand(Float32, dim) * 1.0f0
    tspan = (0.0f0, 2.0f0)

    problem = ODEProblem(model, z0, tspan, θ)
    sol = solve(problem, saveat=0.05, Rosenbrock23())

    return sol.t, map(v->v+rand(dim)*2f-2, sol.u), sol
end

### ============================================================================
### Sine Data Generator
### ============================================================================
function gen_sin()
    X = collect(0:0.05:pi)
    Y = map(sin, X)
    
    return X, Y
end

### ============================================================================
### Arbitrari function Data Generator
### ============================================================================
function generate_data()
    X = collect(0:0.05:3)
    #f(x) = 1 + 0.8x - x^2 +0.3x^3
    f(x) = sin(3x)
    Y = map(f, X)    
    return X, Y
end

###########################################################
###########################################################
###########################################################

#X, Y, sol = gen_n_lotka_volterra(10)
X, Y = generate_data()


save("./L-V New approach/data.jld","X", X, "Y", Y)
println("Done generating data!")