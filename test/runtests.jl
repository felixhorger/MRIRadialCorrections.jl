
using Revise
import MRIRadialCorrections
import MRITrajectories
import MRIPhantoms
import MRIRecon
using Noise
using ImageView
using FFTW
using UnicodePlots

Δ = [1.0 0.5; 0.5 -0.75]

num_columns = 128
num_lines = 128
num_channels = 8
num_angles = floor(Int, 0.5π * num_lines)
if iseven(num_angles)
	num_angles += 1
end
_, spoke_indices = MRITrajectories.golden_angle_incremented(num_angles, num_angles)
k = MRITrajectories.radial_spokes(num_angles, num_columns)
φ = range(0, 2π; length=num_angles+1)[1:end-1]
k_flat = reshape(k, 2, num_columns * num_angles);
k .+= reshape(MRIRadialCorrections.compute_shift(φ, Δ, Val(:StackOfStars)), 2, 1, num_angles) * (2π / num_columns)


sensitivities = MRIPhantoms.coil_sensitivities((num_columns, num_lines), (num_channels,), 0.3)
phantom, noisy_kspace = let snr = 100
	upsampling = (4, 4)
	# Generate spatial profile and ground truth in temporal LR domain
	shutter, highres_shutter = MRIPhantoms.homogeneous((num_columns, num_lines), upsampling)
	coil_images = highres_shutter .* MRIRecon.upsample(sensitivities, upsampling .* (num_columns, num_lines))
	kspace = MRIPhantoms.measure(coil_images, upsampling, k_flat, 1e-8)
	# Noise
	signal_level = maximum(abs, kspace)
	noise_level = signal_level / snr
	# Add noise in kspace
	if isinf(snr)
		noisy_kspace = kspace
	else
		noisy_kspace = Noise.add_gauss(kspace, noise_level)
	end
	shutter, permutedims(reshape(noisy_kspace, num_columns, num_angles, num_channels), (1, 3, 2))
end
noisy_kspace[1:2:end, :, :] .*= -1


S, Sx = MRIRecon.plan_sensitivities(sensitivities, 1)
F = MRIRecon.plan_fourier_transform!(reshape(Sx, num_columns, num_lines, num_channels), k_flat[1, :], k_flat[2, :], (num_columns, num_lines, num_channels))

Δ_estimated, spoke_pairs, intersections, extracted, interpolated = MRIRadialCorrections.compute_delays(φ, noisy_kspace; upsampling=96, search=5);

imshow(permutedims(abs.(extracted), (1, 3, 2)))

lineplot(abs.(extracted)[:, 1, spoke_pairs[1]])
lineplot(abs.(extracted)[:, 1, 1])

#imshow(permutedims(abs.(ifft!(ifftshift(noisy_kspace, 1), 1)), (1, 3, 2)))

spoke_pairs[1], intersections[1]

