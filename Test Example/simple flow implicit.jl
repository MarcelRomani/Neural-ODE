### Simple flow with implicit solutions W for Lotka-Volterra

using DiffEqBase, Markdown, DifferentialEquations, Sundials, Zygote, Plots, LinearAlgebra


# Derivative model
z0 = [1.2, 1.3]

θ = [0.666, 1.333, 1, 1]
a, b, g, d = θ
K = d*z0[1] - g*log(z0[1]) + b*z0[2] - a*log(z0[2])


function f(y, θ, x)
    a, b, g, d = θ

    M = [x*(d-g/y[1]) + (1-x)       x*(b-a/y[2])                       
         x*(d-g/y[1])               x*(b-a/y[2]) + (1-x)]
    #println(M)
    dy = inv(M) * [K - d*y[1] + g*log(y[1]) - b*y[2] + a*log(y[2]) + y[1],
                   K - d*y[1] + g*log(y[1]) - b*y[2] + a*log(y[2]) + y[2]]
    return dy
end




# initial conditions (t,t)
t = 4

prob = ODEProblem(f, [t, t], (0,0.999999), θ)
sol = solve(prob, Rosenbrock23(), abstol=1e-10, dtmin=1e-10)

#= for s in 1:20
    probs = ODEProblem(f, [s, s], (0,0.99999f0), θ)
    sols = solve(probs, Rosenbrock23(), abstol=1f-12, dtmin=1f-9)[end]
    Kf = d*sols[1] - g*log(sols[1]) + b*sols[2] - a*log(sols[2])
    println(Kf-s)
end =#


