
module MRIRadialCorrections

	using FFTW	
	import DSP
	import MRIConst: γ

	# Correct phase accumulated due to B0 eddy currents (Moussavi2013)
	function centre_phase(kspace::AbstractArray{<: Number, N}) where N
		centre = (size(kspace, 1) ÷ 2)
		ϕ = angle.(kspace[centre, axes(kspace)[2:end]...])
		return ϕ
	end

	function b0_eddy_current_parameters(φ::AbstractArray{<: Real, N}, centre_phase::AbstractVector{<: Real}) where N
		@assert N in (1,2)
		num_spokes = size(φ, 1)
		@assert num_spokes == length(centre_phase)
		# Build matrix for linear regression
		M = Matrix{Float64}(undef, num_spokes, N+2)
		M[:, 1] .= 1
		b0_eddy_current_parameters!(M, φ)
		return M \ centre_phase
	end
	function b0_eddy_current_parameters!(M::AbstractMatrix{<: Real}, φ::AbstractVector{<: Real})
		for i in axes(M, 1)
			sine, cosine = sincos(φ[i])
			M[i, 2] = cosine
			M[i, 3] = sine
		end
		return
	end
	function b0_eddy_current_parameters!(M::AbstractMatrix{<: Real}, φ::AbstractMatrix{<: Real})
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


		#function phase
		#factor = Vector{Float64}(undef, 1, size(kspace, 2))
		#@. factor = exp(-im * phase)



	"""
		Find radial spokes shift according to
		@inproceedings{Block2011,
				year = {2011},
				pages = {2816},
				author = {K. T. Block and M. Uecker},
				title = {Simple Method for Adaptive Gradient Delay Correction in Radial {MRI}},
				booktitle = {Proceedings 19th Scientific Meeting, International Society for Magnetic Resonance in Medicine}
		}
	"""
	function signal_shifts(a::AbstractMatrix{<: Number}, b::AbstractMatrix{<: Number}, w::Real)
		# First axis is time, second axis is signal index
		@assert size(a) == size(b)
		# Get copies of input arrays
		a = copy(a)
		b = b[end:-1:1, :] # Reverse b along time
		# 1/2 FOV shift to move object into centre
		a[1:2:end, :] *= -1
		b[1:2:end, :] *= -1
		# Transform into the Fourier domain (of time along readout, not kspace)
		# "Convolve" by multiplying and get phase
		f, g = bfft.(ifftshift.((a, b), 1), 1)
		# Select relevant window which contains the linear phase ramp
		num_signals = size(f, 2)
		ramp_centres = let
			indicator = @. abs(f) + abs(g)
			indices = reshape(1:num_signals, num_signals, 1)
			ramp_centres::Matrix{Int64} = floor.(sum(indicator .* indices; dims=1) ./ sum(indicator; dims=1))
		end
		num_samples = size(a, 1)
		half_ramp_length = Int(ceil(w * num_samples)) ÷ 2
		ramp_length = 2 * half_ramp_length
		p = similar(f, ramp_length, num_signals)
		q = similar(g, ramp_length, num_signals)
		for i = 1:num_signals
			centre = ramp_centres[i][1]
			lower = centre - half_ramp_length
			upper = centre + half_ramp_length - 1
			if lower < 1 || upper > num_samples
				error("Region of interest does not fit into array, weird imaging subject?")
			end
			p[:, i] = @view f[lower:upper, i]
			q[:, i] = @view g[lower:upper, i]
		end
		ϕ = @. angle(p * conj(q))
		DSP.Unwrap.unwrap!(ϕ; dims=1)
		# Compute phase
		# Create matrix for linear fitting
		M = Matrix{Float64}(undef, ramp_length, 2)
		M[:, 1] .= 1
		M[:, 2] = (1:ramp_length) .* (4π/num_samples)
		# Linear fit to get slope (global phase useless due to unwrapping)
		δ = (M \ ϕ)[2, :]
		return δ
	end

	# See Peters2003
	# TODO: Use linreg and return errors
	function canonical_axes_shifts(φ::AbstractVector{<: Real}, δ::AbstractVector{<: Real})
		# Get δxi from parallel shifts fitting sines
		M = Matrix{Float64}(undef, length(φ), 4)
		M[:, 1] .= 1
		for i in axes(M, 1)
			sine, cosine = sincos(φ[i])
			M[i, 2] = cosine^2
			M[i, 3] = sine^2
			M[i, 4] = cosine * sine
		end
		return M \ δ
	end
	function canonical_axes_shifts(φ::AbstractMatrix{<: Real}, δ::AbstractVector{<: Real})
		# Get δxi from parallel shifts fitting sines
		M = Matrix{Float64}(undef, length(φ), 3)
		for i in axes(M, 1)
			(sine1, cosine1), (sine2, cosine2) = sincos.(φ[:, i])
			sine1 = sine1^2
			sine2 = sine2^2
			cosine1 = cosine1^2
			cosine2 = cosine2^2
			M[i, 1] = cosine1 * sine2
			M[i, 2] = sine1 * sine2
			M[i, 3] = cosine2
		end
		return M \ δ
	end

	#function gradient_delay(δ::Real, fov::Real, dwelltime::Real)
	#	fov = 2π/kres
	#	1/dwelltime * Δ = δ / kres
	#end



	# TODO: naming of phi and theta
	#@inline function spoke_parallel_shift(φ::Real, δx::Real, δy::Real)
	#	sine, cosine = sincos(φ)
	#	return δx*cosine^2 + δy*sine^2
	#end
	#@inline function spoke_parallel_shift(φ::Real, θ::Real, δx::Real, δy::Real, δz::Real)
	#	sine, cosine = sincos.((φ, θ))
	#	sine2 = sine.^2
	#	cosine2 = cosine.^2
	#	return (
	#		δx * cosine2[1] * sine2[2]
	#		+ δy * sine2[1] * sine2[2]
	#		+ δz * cosine2[2]
	#	)
	#end
	#@inline function spoke_perpendicular_shift(φ::Real, δx::Real, δy::Real)
	#	sine, cosine = sincos(φ)
	#	δx*cosine*sine + δy*sine*cosine
	#end
	#@inline function spoke_perpendicular_shift(φ::Real, θ::Real, δx::Real, δy::Real, δz::Real)
	#	sine, cosine = sincos.((φ, θ))
	#	sine2 = sine.^2
	#	cosine2 = cosine.^2
	#	parallel_shift = δx * cosine2[1] * sine2[2] + δy * sine2[1] * sine2[2] + δz * cosine2[2]
	#	return [
	#		(δx - parallel_shift) * cosine[1] * sine[2],
	#		(δy - parallel_shift) * sine[1] * sine[2],
	#		(δz - parallel_shift) * cosine[2]
	#	]
	#end


end

