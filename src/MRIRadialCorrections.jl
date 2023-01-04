
module MRIRadialCorrections

	using FFTW	
	using LinearAlgebra
	import DSP
	import MRIConst: γ
	import MRIRecon

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


	"""
		Find radial spokes shift according to
		@inproceedings{Block2011,
				year = {2011},
				pages = {2816},
				author = {K. T. Block and M. Uecker},
				title = {Simple Method for Adaptive Gradient Delay Correction in Radial {MRI}},
				booktitle = {Proceedings 19th Scientific Meeting, International Society for Magnetic Resonance in Medicine}
		}

		Kspace expected to be not ifftshifted (i.e. peak must be in the centre of the array)
		Object in image space must be in the middle of the array after ifft(ifftshift(kspace))
	"""
	function parallel_shift(a::AbstractVector{<: Number}, b::AbstractVector{<: Number}, w::Real)
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
			for outer upper = ramp_centre+1:num_samples
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
	function parallel_shift(a::AbstractMatrix{<: Number}, b::AbstractMatrix{<: Number}, w::Real)
		num_signals = size(a, 2)
		@assert num_signals == size(b, 2) "a and b must have the same number of signals (second dimension)"
		δ_parallel = Vector{Float64}(undef, num_signals)
		@views for i = 1:num_signals
			δ_parallel[i] = parallel_shift(a[:, i], b[:, i], w)
		end
		return δ_parallel
	end

	# See Peters2003
	# TODO: Use linreg and return errors
	function fit_parallel_shift(φ::AbstractVector{<: Real}, δ_parallel::AbstractVector{<: Real})
		# Get δxi from parallel shifts fitting sines
		M = Matrix{Float64}(undef, length(φ), 3)
		for i in axes(M, 1)
			sine, cosine = sincos(φ[i])
			M[i, 1] = cosine^2
			M[i, 2] = sine^2
			M[i, 3] = cosine * sine
		end
		return M \ δ_parallel
	end
	#function canonical_axes_shifts(φ::AbstractMatrix{<: Real}, δ::AbstractVector{<: Real})
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

	#function gradient_delay(δ::Real, fov::Real, dwelltime::Real)
	#	fov = 2π/kres
	#	1/dwelltime * Δ = δ / kres
	#end

	function parallel_shift_model(φ::Real, δ1::Real, δ2::Real, δ3::Real)
		sine, cosine = sincos(φ)
		return δ1 * cosine^2 + δ2 * sine^2 + 2δ3 * sine * cosine
	end

	"""
		Δk in device coordinate system in units of δi
		R: gradient coordinate system (read, phase, partition) -> device coordinate system (x, y, z)
	"""
	function compute_delay(δ1::Real, δ2::Real, δ3::Real, R::AbstractMatrix{<: Real}, pattern::Val{:StackOfStars})
		M = [
			R[1,1]^2		R[2,1]^2		R[3,1]^2;
			R[1,2]^2		R[2,2]^2		R[3,2]^2;
			R[1,1]*R[1,2]	R[2,2]*R[2,1]	R[3,1]*R[3,2]
		]
		Δk = pinv(M) * [δ1, δ2, δ3]
		# Note: Use normal equations because the fitted δi might not be consistent with the model
		return Δk
	end

	"""
		Δk in device coordinate system
		R: gradient coordinate system (read, phase, partition) -> device coordinate system (x, y, z)
		realised k = R' * (Δk .* (R * e_r(φ))) + k
		beware of the units of Δk
	"""
	function compute_shift(
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

	"""
		beware of the units of Δk and k
	"""
	function shift_spokes!(k::AbstractArray{<: Real, 3}, φ::AbstractVector{<: Real}, ΔkⅡ::AbstractVector{<: Real}, Δk⊥::AbstractVector{<: Real})
		@assert length(φ) == length(ΔkⅡ) == length(Δk⊥) == size(k, 3)
		@assert size(k, 1) == 2
		shift = Vector{Float64}(undef, 2)
		for i in eachindex(φ)
			sine, cosine = sincos(φ[i])
			shift[1] =	ΔkⅡ[i] * cosine	- Δk⊥[i] * sine
			shift[2] =	ΔkⅡ[i] * sine	+ Δk⊥[i] * cosine
			k[:, :, i] .+= shift
		end
		return k
	end

	function compute_shift(
		φ::AbstractVector{<: Real},
		S::AbstractMatrix{<: Real},
		pattern::Val{:StackOfStars}
	)
		e_r = Vector{Float64}(undef, 2)
		δk = Matrix{Float64}(undef, 2, length(φ))
		for (i, ϕ) in enumerate(φ)
			sine, cosine = sincos(ϕ)
			e_r[1] = cosine
			e_r[2] = sine
			@views mul!(δk[:, i], S, e_r)
		end
		return δk
	end

	"""
		kspace[readout, channel, spoke]
		kspace so that ifft(ifftshift(kspace, 1), 1) gives projections with object in the centre.
	"""
	function compute_delays(φ::AbstractVector{<: Real}, kspace::AbstractArray{T, N}; cutoff::Real=0.2, upsampling::Integer=64, search::Real=1.5) where {T <: Complex, N}
		@assert length(φ) == size(kspace, 3)
		# Sinc-interpolate spokes
		interpolated = sinc_interpolate_denoise(kspace, cutoff, upsampling, search)

		# Extract centre for searching
		num_centre = floor(Int, upsampling * search)
		#extracted, start = find_centre_of_mass_window(interpolated, num_centre)
		i = MRIRecon.centre_indices(size(interpolated, 1), num_centre)
		extracted = interpolated[i, :, :]

		# Find pairs of (almost) perpendicular spokes
		spoke_pairs = perpendicular_spoke_pairs(φ)

		# Find intersections
		intersections = find_intersections(extracted, spoke_pairs)

		# Add start indices
		#expected_centre = size(interpolated, 1) ÷ 2 + 1
		#for p in eachindex(spoke_pairs)
		#	intersections[p] = (
		#		start[p] - expected_centre + intersections[p][1],
		#		start[spoke_pairs[p]] - expected_centre + intersections[p][2]
		#	)
		#end

		# Set up linear system of equations
		A, b = RING_linear_system(φ, spoke_pairs, intersections)

		# Solve linear system of equations
		S = A \ b
		return S ./ upsampling, spoke_pairs, intersections, extracted, interpolated
	end

	function sinc_interpolate_denoise(kspace::AbstractArray{T, N}, cutoff::Real, upsampling::Integer, search::Real) where {T <: Complex, N}
		# Transform
		transformed = ifft!(ifftshift(kspace, 1), 1)
		# Set outer regions to zero, defined by cutoff
		num_columns, num_channels, num_spokes = size(kspace)
		i = floor(Int, cutoff * num_columns)
		transformed[1:i, :, :] .= 0
		transformed[num_columns-i+1:num_columns, :, :] .= 0
		# Zero pad
		new_num_columns = num_columns * upsampling
		padded = zeros(T, new_num_columns, num_channels, num_spokes)
		centre_indices = MRIRecon.centre_indices(new_num_columns, num_columns)
		@views padded[centre_indices, :, :] .= transformed
		# Transform back
		interpolated = fftshift(fft!(padded, 1), 1)
		return interpolated
	end

	function find_centre_of_mass_window(a::AbstractArray{T, N}, num::Integer) where {T <: Complex, N}
		# TODO: more general
		num_columns, num_channels, num_spokes = size(a)
		lower = num ÷ 2
		upper = lower-1
		lower = -lower
		extracted = Array{T, 3}(undef, num, num_channels, num_spokes)
		start = Vector{Int}(undef, num_spokes)
		position = (1:num_columns)'
		mass = Matrix{Float64}(undef, num_columns, num_channels)
		mass_times_position = Matrix{Float64}(undef, 1, num_channels)
		for s = 1:num_spokes
			spoke = @view a[:, :, s]
			@. mass = abs(spoke)
			mul!(mass_times_position, position, mass)
			centre_of_mass = sum(mass_times_position) / sum(mass)
			indices = round(Int, centre_of_mass) .+ (lower:upper)
			@views extracted[:, :, s] .= spoke[indices, :]
			start[s] = indices.start
		end
		return extracted, start
	end

	function perpendicular_spoke_pairs(φ::AbstractVector{<: Real})
		pairs = Vector{Int}(undef, length(φ))
		i = 1
		for ϕ in φ
			ϕ > 1.5π && break # This should be when all spokes have one "partner"
			_, j = findmin(Φ -> abs(mod2pi(abs(Φ - ϕ)) - 0.5π), φ)
			pairs[i] = j
			i += 1
		end
		return @view pairs[1:i-1]
	end

	function find_intersections(a::AbstractArray{<: Complex, 3}, pairs::AbstractVector{<: Integer})
		num_columns = size(a, 1)
		num_channels = size(a, 2)
		intersections = Vector{NTuple{2, Float64}}(undef, length(pairs))
		tmp = Vector{Float64}(undef, num_channels)
		for (s1, s2) in enumerate(pairs)
			intersection = (0, 0)
			Δ = Inf
			for i = 1:num_columns, j = 1:num_columns
				@views @. begin
					tmp = abs2(a[i, :, s1] - a[j, :, s2])
					tmp /= 0.5 * (abs2(a[i, :, s1]) + abs2(a[j, :, s2]))
				end
				δ = sum(tmp)
				if δ < Δ
					Δ = δ
					intersection = (i, j)
				end
			end
			intersections[s1] = intersection .- (num_columns ÷ 2 + 0.5)
			# Offset due to having the centre at zero and no sample at k = 0 (half index shift)
		end
		return intersections
	end

	function RING_linear_system(φ::AbstractVector{<: Real}, pairs::AbstractVector{<: Int}, intersections::AbstractVector{<: NTuple{2, <: Real}})
		num_equations = length(pairs) * 2
		A = Matrix{Float64}(undef, num_equations, 3)
		b = Vector{Float64}(undef, num_equations)
		for (s1, s2) in enumerate(pairs)
			ϕ = (φ[s1], φ[s2])
			((sine1, cosine1), (sine2, cosine2)) = sincos.(ϕ)
			# Difference of spokes' unit vectors
			Δn = (
				cosine1 - cosine2,
				sine1 - sine2
			)
			# Fill matrix
			A[2s1-1, 1]	= Δn[1]
			A[2s1-1, 2]	= 0
			A[2s1-1, 3]	= Δn[2]
			A[2s1, 1]	= 0
			A[2s1, 2]	= Δn[2]
			A[2s1, 3]	= Δn[1]
			# Fill right hand side
			a1 = intersections[s1][1]
			a2 = intersections[s1][2]
			b[2s1-1]	= a2 * cosine2 - a1 * cosine1 # cos
			b[2s1]		= a2 * sine2 - a1 * sine1 # sin
		end
		return A, b
	end

end

