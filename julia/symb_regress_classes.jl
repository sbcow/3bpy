using SymbolicRegression
using Random: MersenneTwister
using Zygote
using MLJBase: machine, fit!, predict, report
using Test
using PyCall

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"
out = load_pickle( "../data/test_3/output_lyapunov.pkl")
periods = load_pickle( "../data/test_3/periods.pkl")


relevant_indices = []
for (index, value) in out
  if value !== nothing
    push!(relevant_indices, index)
  end
end
out = filter!(row -> row[2] !== nothing, out)
out = filter!(row -> row[2] !== nothing, out)

# ntimesteps = size(out[relevant_indices[1]], 1)
ntimesteps = size(out[1], 1)

time_histories = Dict{Int, Vector{Float64}}() # class, vector of timehistories
for key in relevant_indices
    time_histories[key] = range(0, stop=periods[key], length=ntimesteps)
end

time_histories_arr = hcat(collect(values(time_histories))...)
out_arr = hcat(collect(values(out))...)

n_of_traj = 5
time_histories_arr = time_histories_arr[:, 1:n_of_traj]
out_arr = out_arr[:, 1:42:size(out_arr, 1)]
out_arr = out_arr[:, 1:n_of_traj]

classes = repeat(1:n_of_traj, size(out_arr, 1))
X = (; x1=vec(time_histories_arr), class=classes)
y = vec(out_arr)

## Setting up the Search

# We'll configure the symbolic regression search to:
# - Use parameterized expressions with up to 2 parameters
# - Use Zygote.jl for automatic differentiation during parameter optimization (important when using parametric expressions, as it is higher dimensional)
# =#

stop_at = Ref(1e-4)  #src

model = SRRegressor(;
    niterations=100,
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=30,
    expression_type=ParametricExpression,
    expression_options=(; max_parameters=2),
    autodiff_backend=:Zygote,
    early_stop_condition=(loss, _) -> loss < stop_at[],  #src
);

mach = machine(model, X, y)


fit!(mach)

report(mach).equations[end]

# ## Key Takeaways
#
# 1. [`ParametricExpression`](@ref)s allows us to discover symbolic expressions with optimizable parameters
# 2. The parameters can capture class-dependent variations in the underlying model
#
# This approach is particularly useful when you suspect your data follows a common
# functional form, but with varying parameters across different conditions or class!
# =#
# #literate_end
#
# fit!(mach)
# idx1 = lastindex(report(mach).equations)
# ypred1 = predict(mach, (data=X, idx=idx1))
# loss1 = sum(i -> abs2(ypred1[i] - y[i]), eachindex(y)) / length(y)
#
# # Should keep all parameters
# stop_at[] = loss1 * 0.999
# mach.model.niterations *= 2
# fit!(mach)
# idx2 = lastindex(report(mach).equations)
# ypred2 = predict(mach, (data=X, idx=idx2))
# loss2 = sum(i -> abs2(ypred2[i] - y[i]), eachindex(y)) / length(y)
#
# # Should get better:
# @test loss1 > loss2
