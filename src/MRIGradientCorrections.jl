
module MRIGradientCorrections

	using FFTW	
	import DSP


	"""
		Find radial delay according to
		@inproceedings{Block2011,
				year = {2011},
				pages = {2816},
				author = {K. T. Block and M. Uecker},
				title = {Simple Method for Adaptive Gradient Delay Correction in Radial {MRI}},
				booktitle = {Proceedings 19th Scientific Meeting, International Society for Magnetic Resonance in Medicine}
		}
	"""
	function radial_parallel_shift(a::AbstractMatrix{<: Number}, b::AbstractMatrix{<: Number})
		# First axis is time, second axis is signal
		@assert size(a) == size(b)
		# Transform into the Fourier domain (of time along readout, not kspace)
		# "Convolve" by multiplying and get phase
		samples = size(a, 1)
		f, g = fft.(ifftshift.((a, b), 1), 1)
		ϕ = @. angle(f * conj(g))
		DSP.Unwrap.unwrap!(ϕ; dims=1)
		# Find support of f
		support = let m = abs.(f)
			m .> 0.1 * maximum(m; dims=1)
		end
		# Create matrix for linear fitting
		M = Matrix{Float64}(undef, samples, 2)
		M[:, 1] .= 1
		let x0 = samples ÷ 2
			M[:, 2] = (-x0:x0-1+mod(samples, 2)) .* (4π/samples)
		end
		# Linear fit to get slope (global phase useless due to unwrapping)
		β = Matrix{Float64}(undef, size(ϕ, 2))
		@views for i in axes(ϕ, 2)
			β[i] = last(M[support[:, i], :] \ ϕ[support[:, i], i])
		end
		return β
	end
end

