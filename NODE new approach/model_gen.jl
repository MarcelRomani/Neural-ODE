
function genModel(model_name, dim)
    # Lotka-Volterra equations
    function LVModel(z, θ, t)
        return z .* (reshape(θ,dim,dim+1)*vcat(z,1e0))
    end 

    # Riccati equations
    function RiccatiModel(z, θ, t)
        z_aug = vcat(z,1e0)
        return dropdims(sum((z_aug*z_aug') .* reshape(θ, dim+1, dim+1, dim), dims=(1,2)), dims=(1,2))
    end

    # S-system equations
    function SSysModel(z, θ, t)
        a = θ[1:dim]
        b = θ[dim+1:2*dim]
        g = reshape(θ[(2*dim+1):(2*dim+dim*dim)], dim, dim)
        h = reshape(θ[dim*(2+dim)+1 : 2dim*(1+dim)], dim, dim)
        dz = a .* prod(z'.^g, dims=2) - b .* prod(z'.^h, dims=2)
        return dz
    end

    if model_name == "Riccati"
        return RiccatiModel
    elseif model_name == "Lotka-Volterra"
        return LVModel
    elseif model_name == "S-system"
        return SSysModel
    else 
        println("Model not supported.")
        exit()

    end
end


function paramsInit(model_name, dim)
    if model_name == "Riccati"
        θ = (rand(Float32, (dim+1)*(dim+1)*dim) .- 0.5e0) * 1.0e-1
        z0 = (rand(Float32, dim)) * 1.0e-1
        return θ, z0
    elseif model_name == "Lotka-Volterra"
        θ = (rand(Float32, dim*(dim+1)) .- 0.5e0) * 1.0e-1
        z0 = (rand(Float32, dim)) * 1.0e-1
        return θ, z0
    elseif model_name == "S-system"
        θ = (rand(Float32, 2*dim*(dim+1)) .- 0.5e0) * 1.0e-1
        z0 = 1 .+ (rand(Float32, dim)) * 1.0e-1
        return θ, z0
    else 
        println("Model not supported.")
        exit()

    end

end
