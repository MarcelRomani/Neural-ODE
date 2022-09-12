### Testing DAE solving methods on the simple flow system

using DiffEqBase, Markdown, DifferentialEquations, Sundials, Zygote


### DAE Problems

f1(z, A) = sin(A * z)
df1(z, A) = gradient(f1, z, A)[1]

f2(z, B) = B * z^3 + z^2
df2(z, B) = gradient(f2, z, B)[1]

# Derivative model
function f(r, zp, z, θ, t)
    r[1] = (f1(z[1], θ[1]) - z[1]) / (z[3] + (1 - z[3]) * df1(z[1], θ[1])) - zp[1]
    r[2] = (f2(z[2], θ[2]) - z[2]) / (z[3] + (1 - z[3]) * df2(z[2], θ[2])) - zp[2]
    r[3] = 1.0 - zp[3]
end

r = ones(3)
u0 = [π/3, 1.0, 0]
du0 = [-0.362, 0.2, 1.0]
θ = [1.0f0, 1.0f0]

prob_dae_resrob = DAEProblem(f,du0,u0,(0.0,1.0),θ)
sol = solve(prob_dae_resrob, IDA(linear_solver=:Band,jac_upper=3,jac_lower=3))