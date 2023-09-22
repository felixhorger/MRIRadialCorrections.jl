
module MRIRadialCorrections

	using FFTW	
	using LinearAlgebra
	import DSP
	import MRIConst: γ
	import MRIRecon

	include("centre_phase.jl")
	include("simple.jl")
	include("ring.jl")

	"""
		Δk_from_Δt(Δt::Real, dwelltime::Real, num_columns::Integer) = Δk = Δt / dwelltime * 2π / num_columns

	Compute trajectory shift corresponding to a delay.

	"""
	Δk_from_Δt(Δt::Real, dwelltime::Real, num_columns::Integer) = Δk = Δt / dwelltime * 2π / num_columns

end

