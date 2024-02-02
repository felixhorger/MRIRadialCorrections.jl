"""
	Find parallel shift of radial spokes according to

	@inproceedings{Block2011,
			year = {2011},
			pages = {2816},
			author = {K. T. Block and M. Uecker},
			title = {Simple Method for Adaptive Gradient Delay Correction in Radial {MRI}},
			booktitle = {Proceedings 19th Scientific Meeting, International Society for Magnetic Resonance in Medicine}
	}

	k-space expected to be NOT ifftshifted (i.e. peak must be in the centre of the array)
	Object in image space must be in the middle of the array after ifft(ifftshift(kspace))

	a: forward
	b: backward
	w: the absolute signal is `w * maxium(abs, signal)` at the borders of the window used for fitting 
"""
function estimate_parallel_shift(a::AbstractVector{<: Number}, b::AbstractVector{<: Number}, w::Real)
	num_samples = length(a)
	@assert num_samples == length(b)
	f, g = let
		# Get magnitude signals
		ma = abs.(a)
		mb = reverse(abs.(b)) # Further reverse b along time
		# Transform into the Fourier domain (of time along readout, not kspace)
		f, g = rfft.(ifftshift.((ma, mb)))
	end
	# Select relevant window which contains the linear phase ramp
	lower = 0 # No idea why this doesn't work with local lower, upper
	upper = 0
	let
		indicator = @. abs(f) + abs(g)
		# Centre of mass not used because it fails sometimes, maybe useful though, use max and centre of mass, but how?
		#ramp_centre = floor.(Int, sum(indicator .* (1:length(indicator))) ./ sum(indicator))
		maxi, maxi_index = findmax(abs, indicator)
		ramp_centre = maxi_index
		#if abs(ramp_centre - maxi_index) > 0.2num_samples
		#	error("Centre of mass and position of maximum signal differ more than 20% of the array length")
		#end
		# Find lower and upper boundary
		threshold = w * maxi
		for outer lower = ramp_centre-1:-1:1
			indicator[lower] < threshold && break
		end
		for outer upper = ramp_centre+1:length(indicator)
			indicator[upper] < threshold && break
		end
		# Remove the border cases (only non-optimal if object fills the whole field of view, but then not significant)
		lower += 1
		upper -= 1
		abs(upper - lower) < 0.05num_samples && error("Window is shorter than 5% of the array length")
	end
	# Extract phase ramp
	@views ϕ = @. angle(f[lower:upper] * conj(g[lower:upper]))
	DSP.Unwrap.unwrap!(ϕ)
	# Create matrix for linear fitting
	ramp_length = upper - lower + 1
	M = Matrix{Float64}(undef, ramp_length, 2)
	M[:, 1] .= 1
	M[:, 2] = (1:ramp_length) .* (4π/num_samples)
	# Linear fit to get slope (global phase useless due to unwrapping)
	return (M \ ϕ)[2]
end

"""
	Convenience, vectorised version
"""
function estimate_parallel_shift(a::AbstractMatrix{<: Number}, b::AbstractMatrix{<: Number}, w::Real)
	num_signals = size(a, 2)
	@assert num_signals == size(b, 2) "a and b must have the same number of signals (second dimension)"
	δ_parallel = Vector{Float64}(undef, num_signals)
	@views for i = 1:num_signals
		δ_parallel[i] = estimate_parallel_shift(a[:, i], b[:, i], w)
	end
	return δ_parallel
end

function linreg(X::AbstractMatrix{<: Number}, y::Vector{<: Number}, w::Vector{<: Number}) # TODO put in separate package? Or just get the relevant bits out
	W = diagm(w.^2)
	A = X' * W
	H = A * X
	if det(H) == 0
		θ = Vector{Float64}(undef, size(X, 1))
		fill!(θ, NaN)
		Δθ = θ
	else
		θ = H \ (A * y)
		Δθ = sqrt.(diag(inv(H)))
	end
	return θ, Δθ
end
"""
	Fit a model to parallel delays in depenence of the spoke angle (Peters2003).
"""
function fit_parallel_shift(φ::AbstractVector{<: Real}, δ_parallel::AbstractVector{<: Real})
	M = Matrix{Float64}(undef, length(φ), 3)
	w = Vector{Float64}(undef, length(φ))
	for i in axes(M, 1)
		sine, cosine = sincos(φ[i])
		M[i, 1] = cosine^2
		M[i, 2] = sine^2
		M[i, 3] = cosine * sine
		w[i] = cos(2*φ[i])^2
	end
	return Tuple(linreg(M, δ_parallel, w)[1]), w #Tuple(M \ δ_parallel)
end
# 3D version not functional
#function fit_parallel_shift(φ::AbstractMatrix{<: Real}, δ::AbstractVector{<: Real})
#	# Get δxi from parallel shifts fitting sines
#	M = Matrix{Float64}(undef, length(φ), 3)
#	for i in axes(M, 1)
#		(sine1, cosine1), (sine2, cosine2) = sincos.(φ[:, i])
#		sine1 = sine1^2
#		sine2 = sine2^2
#		cosine1 = cosine1^2
#		cosine2 = cosine2^2
#		M[i, 1] = cosine1 * sine2
#		M[i, 2] = sine1 * sine2
#		M[i, 3] = cosine2
#	end
#	return M \ δ
#end

"""
	Evaluate the model for parallel spoke shift as a function of spoke angle.
	δ is obtained from `fit_parallel_shift()`.
"""
function parallel_shift_model(φ::Real, δ1::Real, δ2::Real, δ3::Real)
	sine, cosine = sincos(φ)
	return δ1 * cosine^2 + δ2 * sine^2 + 2δ3 * sine * cosine
end

"""
	Δk in device coordinate system in units of δi
	R: gradient coordinate system (read, phase, partition) -> device coordinate system (x, y, z)
"""
function gcs2dcs(δ::NTuple{3, Real}, R::AbstractMatrix{<: Real}, pattern::Val{:StackOfStars})
	# It's more complicated than a simple coordinate transform because the actual equation goes
	# something like R' * T * R * k where T is the delay operator and k the trajectory
	M = [
		R[1,1]^2		R[2,1]^2		R[3,1]^2;
		R[1,2]^2		R[2,2]^2		R[3,2]^2;
		R[1,1]*R[1,2]	R[2,2]*R[2,1]	R[3,1]*R[3,2]
	]
	# Use normal equations because the fitted δi might not be consistent with the model
	Δk = pinv(M) * collect(δ)
	return Δk
end

"""
	Δk in device coordinate system
	R: gradient coordinate system (read, phase, partition) -> device coordinate system (x, y, z)
	realised k = R' * (Δk .* (R * e_r(φ))) + k
	beware of the units of Δk
"""
function compute_spoke_shift(
	φ::AbstractVector{<: Real},
	Δk::AbstractVector{<: Real},
	R::AbstractMatrix{<: Real},
	pattern::Val{:StackOfStars}
)
	e_r = Vector{Float64}(undef, 2)
	δk = Matrix{Float64}(undef, 3, length(φ))
	tmp = Vector{Float64}(undef, 3)
	for (i, ϕ) in enumerate(φ)
		sine, cosine = sincos(ϕ)
		e_r[1] = cosine
		e_r[2] = sine
		@views mul!(tmp, R[:, 1:2], e_r)
		tmp .*= Δk
		@views mul!(δk[:, i], R', tmp)
	end
	return δk
end

"""
	beware of the units of Δk and k
"""
function shift_spokes!(k::AbstractArray{<: Real, 3}, φ::AbstractVector{<: Real}, ΔkⅡ::AbstractVector{<: Real}, Δk⊥::AbstractVector{<: Real})
	@assert length(φ) == length(ΔkⅡ) == length(Δk⊥) == size(k, 3)
	@assert size(k, 1) == 2
	shift = Vector{Float64}(undef, 2)
	for i in eachindex(φ)
		sine, cosine = sincos(φ[i])
		shift[1] = ΔkⅡ[i] * cosine	- Δk⊥[i] * sine
		shift[2] = ΔkⅡ[i] * sine	+ Δk⊥[i] * cosine
		k[:, :, i] .+= shift
	end
	return k
end

