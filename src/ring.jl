
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

function compute_spoke_shift(
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
