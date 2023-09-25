using DifferentialEquations

function nl(du,u,p,t)
    du[1] = -p[1]*u[1]+u[2]+p[2]*u[2]*u[3]
    du[2] = -u[1]-p[1]*u[2]+5*u[1]*u[3]
    du[3] = p[3]*u[3]-5*u[1]*u[2]
end
  
function σ_nl(du,u,p,t)
    du[1] = 0.04
    du[2] = 0.04
    du[3] = 0.04
end

dt = 0.005
u0 = [0.; 0.1; 0.01]
p = [0.4, 10., 0.175]
prob_sde_newton = SDEProblem(nl,σ_nl,u0,(dt,Int(10^6*dt)),p)
trj = solve(prob_sde_newton,EM(),dt=dt)[:]
x = zeros(Float64,3,length(trj))
for i in eachindex(trj)
    x[:,i] = trj[i]
end

@info "saving data for Newton-Leipnik"
hfile = h5open(pwd() * "/data/newton.hdf5", "w")
hfile["x"] = x
hfile["dt"] = dt
close(hfile)
@info "done saving data for Newton-Leipnik"