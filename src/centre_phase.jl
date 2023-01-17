
# Correct phase accumulated due to B0 eddy currents (Moussavi2013)
function centre_phase(kspace::AbstractArray{<: Number, N} where N)
	centre = (size(kspace, 1) ÷ 2 + 1)
	@views ϕ = angle.(kspace[centre, axes(kspace)[2:end]...])
	return ϕ
end

function fit_b0_eddy_current_parameters(φ::AbstractArray{<: Real, N}, centre_phase::AbstractVector{<: Real}) where N
	@assert N in (1,2)
	num_spokes = size(φ, 1)
	@assert num_spokes == length(centre_phase)
	# Build matrix for linear regression
	M = Matrix{Float64}(undef, num_spokes, N+2)
	M[:, 1] .= 1
	fit_b0_eddy_current_parameters!(M, φ)
	return M \ centre_phase
end
function fit_b0_eddy_current_parameters!(M::AbstractMatrix{<: Real}, φ::AbstractVector{<: Real})
	for i in axes(M, 1)
		sine, cosine = sincos(φ[i])
		M[i, 2] = cosine
		M[i, 3] = sine
	end
	return
end
function fit_b0_eddy_current_parameters!(M::AbstractMatrix{<: Real}, φ::AbstractMatrix{<: Real})
	@assert size(φ, 1) == 2
	for i in axes(M, 1)
		(sine1, cosine1), (sine2, cosine2) = sincos.(φ[:, i])
		M[i, 2] = cosine1 * sine2
		M[i, 3] = sine1 * sine2
		M[i, 4] = cosine2
	end
	return
end

"""
	[constant, cos, sin]
	[constant, cos sin, sin sin, cos]
"""
function b0_eddy_current_model(φ::AbstractVector{<: Real}, params::AbstractVector{<: Real})
	num_spokes = length(φ)
	out = Vector{Float64}(undef, num_spokes)
	for i = 1:num_spokes
		sine, cosine = sincos(φ[i])
		out[i] = params[1] + cosine*params[2] + sine*params[3]
	end
	return out
end
function b0_eddy_current_model(φ::AbstractMatrix{<: Real}, params::AbstractVector{<: Real})
	@assert size(φ, 1) == 2
	num_spokes = length(φ)
	out = Vector{Float64}(undef, num_spokes)
	for i = 1:num_spokes
		(sine1, cosine1), (sine2, cosine2) = sincos.(φ[:, i])
		out[i] = params[1] + cosine1*sine2*params[2] + sine1*sine2*params[3] + cosine2*params[4]
	end
	return out
end

