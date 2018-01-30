import numpy as np
try:
    from scipy.optimize import minimize
    from scipy.ndimage import map_coordinates
    from scipy.ndimage.interpolation import rotate
except:
    minimize = None #Won't be able to use 'phase_fitting' in stack_registration
    map_coordinates = None # Won't be able to use stack_rotational_registration
    rotate = None
try:
    import np_tif
except:
    np_tif = None #Won't be able to use the 'debug' option of stack_registration

def stack_registration(
    s,
    align_to_this_slice=0,
    refinement='spike_interpolation',
    register_in_place=True,
    fourier_cutoff_radius=None,
    background_subtraction=None,
    debug=False,):
    """Calculate shifts which would register the slices of a
    three-dimensional stack `s`, and optionally register the stack in-place.

    Axis 0 is the "z-axis", axis 1 is the "up-down" (Y) axis, and axis 2
    is the "left-right" (X) axis. For each XY slice, we calculate the
    shift in the XY plane which would line that slice up with the slice
    specified by `align_to_this_slice`. If `align_to_this_slice` is a
    number, it indicates which slice of `s` to use as the reference
    slice. If `align_to_this_slice` is a numpy array, it is used as the
    reference slice, and must be the same shape as a 2D slice of `s`.

    `refinement` is one of `integer`, `spike_interpolation`, or
    `phase_fitting`, in order of increasing precision/slowness. I don't
    yet have any evidence that my implementation of phase fitting gives
    any improvement over (faster, simpler) spike interpolation, so
    caveat emptor.

    `register_in_place`: If `True`, modify the input stack `s` by
    shifting its slices to line up with the reference slice.

    `fourier_cutoff_radius`: Ignore the Fourier phases of spatial
    frequencies higher than this cutoff, since they're probably lousy
    due to aliasing and noise anyway. If `None`, attempt to estimate a
    resonable cutoff.

    'background_subtraction': One of None, 'mean', 'min', or
    'edge_mean'. Image registration is sensitive to edge effects. To
    combat this, we multiply the image by a real-space mask which goes
    to zero at the edges. For dim images on large DC backgrounds, the
    registration can end up mistaking this mask for an important image
    feature, distorting the registration. Sometimes it's helpful to
    subtract a background from the image before registration, to reduce
    this effect. 'mean' and 'min' subtract the mean and minimum of the
    stack 's', respectively, and 'edge_mean' subtracts the mean of the
    edge pixels. Use None (the default) for no background subtraction.
    """
    assert len(s.shape) == 3
    try:
        assert align_to_this_slice in range(s.shape[0])
        align_to_this_slice = s[align_to_this_slice, :, :]
    except ValueError:
        align_to_this_slice = np.squeeze(align_to_this_slice)
    assert align_to_this_slice.shape == s.shape[-2:]
    assert refinement in ('integer', 'spike_interpolation', 'phase_fitting')
    if refinement == 'phase_fitting' and minimize is None:
        raise UserWarning("Failed to import scipy minimize; no phase fitting.")
    assert register_in_place in (True, False)
    # What background should we subtract from each slice of the stack?
    assert background_subtraction in (None, 'mean', 'min', 'edge_mean')
    if background_subtraction is None:
        bg = 0
    elif background_subtraction is 'min':
        bg = s.min()
    elif background_subtraction is 'mean':
        bg = s.mean()
    elif background_subtraction is 'edge_mean':
        bg = np.mean((s[:, 0, :].mean(), s[:, -1, :].mean(),
                      s[:, :, 0].mean(), s[:, :, -1].mean()))
    if fourier_cutoff_radius is None:
        fourier_cutoff_radius = estimate_fourier_cutoff_radius(s, bg, debug)
    assert (0 < fourier_cutoff_radius <= 0.5)
    assert debug in (True, False)
    if debug and np_tif is None:
        raise UserWarning("Failed to import np_tif; no debug mode.")
    ## Multiply each slice of the stack by an XY mask that goes to zero
    ## at the edges, to prevent periodic boundary artifacts when we
    ## Fourier transform.
    mask_ud = np.sin(np.linspace(0, np.pi, s.shape[1])).reshape(s.shape[1], 1)
    mask_lr = np.sin(np.linspace(0, np.pi, s.shape[2])).reshape(1, s.shape[2])
    masked_reference_slice = (align_to_this_slice - bg) * mask_ud * mask_lr
    ## We'll base our registration on the phase of the low spatial
    ## frequencies of the cross-power spectrum. We'll need the complex
    ## conjugate of the Fourier transform of the masked reference slice,
    ## and a mask in the Fourier domain to pick out the low spatial
    ## frequencies:
    ref_slice_ft_conj = np.conj(np.fft.rfftn(masked_reference_slice))
    k_ud = np.fft.fftfreq(s.shape[1]).reshape(ref_slice_ft_conj.shape[0], 1)
    k_lr = np.fft.rfftfreq(s.shape[2]).reshape(1, ref_slice_ft_conj.shape[1])
    fourier_mask = (k_ud**2 + k_lr**2) < (fourier_cutoff_radius)**2
    ## Now we'll loop over each slice of the stack, calculate our
    ## registration shifts, and optionally apply the shifts to the
    ## original stack.
    registration_shifts = []
    if debug:
        ## Save some intermediate data to help with debugging
        masked_stack = np.zeros_like(s)
        masked_stack_ft = np.zeros(
            (s.shape[0],) + ref_slice_ft_conj.shape, dtype=np.complex128)
        masked_stack_ft_vs_ref = np.zeros_like(masked_stack_ft)
        cross_power_spectra = np.zeros_like(masked_stack_ft)
        spikes = np.zeros(s.shape, dtype=np.float64)
    for which_slice in range(s.shape[0]):
        if debug: print("Calculating registration for slice", which_slice)
        ## Compute the cross-power spectrum of our slice, and mask out
        ## the high spatial frequencies.
        current_slice = (s[which_slice, :, :] - bg) * mask_ud * mask_lr
        current_slice_ft = np.fft.rfftn(current_slice)
        cross_power_spectrum = current_slice_ft * ref_slice_ft_conj
        cross_power_spectrum = (fourier_mask *
                                cross_power_spectrum /
                                np.abs(cross_power_spectrum))
        ## Inverse transform to get a 'spike' in real space. The
        ## location of this spike gives the desired registration shift.
        ## Start by locating the spike to the nearest integer:
        spike = np.fft.irfftn(cross_power_spectrum, s=current_slice.shape)
        loc = np.array(np.unravel_index(np.argmax(spike), spike.shape))
        if refinement in ('spike_interpolation', 'phase_fitting'):
            ## Use (very simple) three-point polynomial interpolation to
            ## refine the location of the peak of the spike:
            neighbors = np.array([-1, 0, 1])
            ud_vals = spike[(loc[0] + neighbors) %spike.shape[0], loc[1]]
            lr_vals = spike[loc[0], (loc[1] + neighbors) %spike.shape[1]]
            lr_fit = np.poly1d(np.polyfit(neighbors, lr_vals, deg=2))
            ud_fit = np.poly1d(np.polyfit(neighbors, ud_vals, deg=2))
            lr_max_shift = -lr_fit[1] / (2 * lr_fit[2])
            ud_max_shift = -ud_fit[1] / (2 * ud_fit[2])
            loc = loc + (ud_max_shift, lr_max_shift)
        ## Convert our shift into a signed number near zero:
        loc = ((np.array(spike.shape)//2 + loc) % np.array(spike.shape)
               -np.array(spike.shape)//2)
        if refinement == 'phase_fitting':
            if debug: print("Phase fitting slice", which_slice, "...")
            ## (Attempt to) further refine our registration shift by
            ## fitting Fourier phases. I'm not sure this does any good,
            ## perhaps my implementation is lousy?
            def minimize_me(loc, cross_power_spectrum):
                disagreement = np.abs(
                    expected_cross_power_spectrum(loc, k_ud, k_lr) -
                    cross_power_spectrum
                    )[fourier_mask].sum()
                if debug: print(" Shift:", loc, "Disagreement:", disagreement)
                return disagreement
            loc = minimize(minimize_me,
                           x0=loc,
                           args=(cross_power_spectrum,),
                           method='Nelder-Mead').x
        registration_shifts.append(loc)
        if debug:
            ## Save some intermediate data to help with debugging
            masked_stack[which_slice, :, :] = current_slice
            masked_stack_ft[which_slice, :, :] = (
                np.fft.fftshift(current_slice_ft, axes=0))
            masked_stack_ft_vs_ref[which_slice, :, :] = (
                np.fft.fftshift(current_slice_ft * ref_slice_ft_conj, axes=0))
            cross_power_spectra[which_slice, :, :] = (
                np.fft.fftshift(cross_power_spectrum, axes=0))
            spikes[which_slice, :, :] = np.fft.fftshift(spike)
    if register_in_place:
        ## Modify the input stack in-place so it's registered.
        if refinement == 'integer':
            registration_type = 'nearest_integer'
        else:
            registration_type = 'fourier_interpolation'
        apply_registration_shifts(
            s, registration_shifts, registration_type=registration_type)
    if debug:
        np_tif.array_to_tif(masked_stack, 'DEBUG_masked_stack.tif')
        np_tif.array_to_tif(np.log(np.abs(masked_stack_ft)),
                            'DEBUG_masked_stack_FT_log_magnitudes.tif')
        np_tif.array_to_tif(np.angle(masked_stack_ft),
                            'DEBUG_masked_stack_FT_phases.tif')
        np_tif.array_to_tif(np.angle(masked_stack_ft_vs_ref),
                            'DEBUG_masked_stack_FT_phase_vs_ref.tif')
        np_tif.array_to_tif(np.angle(cross_power_spectra),
                            'DEBUG_cross_power_spectral_phases.tif')
        np_tif.array_to_tif(spikes, 'DEBUG_spikes.tif')
        if register_in_place:
            np_tif.array_to_tif(s, 'DEBUG_registered_stack.tif')
    return np.array(registration_shifts)

mr_stacky = stack_registration #I like calling it this.

def apply_registration_shifts(
    s,
    registration_shifts,
    registration_type='fourier_interpolation',
    edges='zero',
    ):
    """Modify the input stack `s` in-place so it's registered.

    If you used `stack_registration` to calculate `registration_shifts`
    for the stack `s`, but didn't use the `register_in_place` option to
    apply the registration correction, you can use this function to
    apply the registration correction later.
    """
    assert len(s.shape) == 3
    assert len(registration_shifts) == s.shape[0]
    assert registration_type in ('fourier_interpolation', 'nearest_integer')
    assert edges in ('sloppy', 'zero')
    n_y, n_x = s.shape[-2:]
    k_ud, k_lr = np.fft.fftfreq(n_y), np.fft.rfftfreq(n_x)
    k_ud, k_lr = k_ud.reshape(k_ud.size, 1), k_lr.reshape(1, k_lr.size)
    for which_slice, loc in enumerate(registration_shifts):
        y, x = -int(np.round(loc[0])), -int(np.round(loc[1]))
        top, bot = max(0, y), min(n_y, n_y+y)
        lef, rig = max(0, x), min(n_x, n_x+x),
        if registration_type == 'nearest_integer':
            s[which_slice, top:bot, lef:rig] = (
                s[which_slice, top-y:bot-y, lef-x:rig-x])
        elif registration_type == 'fourier_interpolation':
            phase_correction = expected_cross_power_spectrum(loc, k_ud, k_lr)
            shift_me = s[which_slice, :, :].astype(np.float64, copy=False)
            s[which_slice, :, :] = np.fft.irfftn(
                np.fft.rfftn(shift_me) / phase_correction, s=(n_y, n_x)).real
        if edges == 'sloppy':
            pass
        elif edges == 'zero':
            s[which_slice, :top, :].fill(0), s[which_slice, bot:, :].fill(0)
            s[which_slice, :, :lef].fill(0), s[which_slice, :, rig:].fill(0)
    return None #`s` is modified in-place.

def stack_rotational_registration(
    s,
    align_to_this_slice=0,
    refinement='spike_interpolation',
    register_in_place=True,
    fourier_cutoff_radius=None,
    fail_180_test='fix_but_print_warning',
    debug=False,):
    """Calculate rotations which would rotationally register the slices
    of a three-dimensional stack `s`, and optionally rotate the stack
    in-place.

    Axis 0 is the "z-axis", axis 1 is the "up-down" (Y) axis, and axis 2
    is the "left-right" (X) axis. For each XY slice, we calculate the
    rotation in the XY plane which would line that slice up with the
    slice specified by `align_to_this_slice`. If `align_to_this_slice`
    is a number, it indicates which slice of `s` to use as the reference
    slice. If `align_to_this_slice` is a numpy array, it is used as the
    reference slice, and must be the same shape as a 2D slice of `s`.

    `refinement` is pretty much just `spike_interpolation`, until I get
    around to adding other refinement techniques.

    `register_in_place`: If `True`, modify the input stack `s` by
    shifting its slices to line up with the reference slice.

    `fourier_cutoff_radius`: Ignore the Fourier amplitudes of spatial
    frequencies higher than this cutoff, since they're probably lousy
    due to aliasing and noise anyway. If `None`, attempt to estimate a
    resonable cutoff.

    'fail_180_test': One of 'fix_but_print_warning', (the default),
    'fix_silently', 'ignore_silently', or 'raise_exception'. The
    algorithm employed here has a fundamental ambiguity: due to
    symmetries in the Fourier domain under 180 degree rotations, the
    rotational registration won't return rotations bigger than +/- 90
    degrees. We currently have a lousy correlation-based check to detect
    if slice(s) in your stack seem to need bigger rotations to align
    with the reference slice. What would you like to happen when this
    warning triggers?
    """
    assert len(s.shape) == 3
    try:
        assert align_to_this_slice in range(s.shape[0])
        align_to_this_slice = s[align_to_this_slice, :, :]
    except ValueError: # Maybe align_to_this_slice is a numpy array?
        align_to_this_slice = np.squeeze(align_to_this_slice)
    assert align_to_this_slice.shape == s.shape[-2:]
    ## Create a square-cropped view 'c' of the input stack 's'. We're
    ## going to use a circular mask anyway, and this saves us from
    ## having to worry about nonuniform k_x, k_y sampling in Fourier
    ## space:
    delta = s.shape[1] - s.shape[2]
    y_slice, x_slice = slice(s.shape[1]), slice(s.shape[2]) # Default: no crop
    if delta > 0: # Crop Y
        y_slice = slice(delta//2, delta//2 - delta)
    elif delta < 0: # Crop X
        x_slice = slice(-delta//2, delta - delta//2)
    c = s[:, y_slice, x_slice]
    align_to_this_slice = align_to_this_slice[y_slice, x_slice]
    assert c.shape[1] == c.shape[2] # Take this line out in a few months!
    assert refinement in ('spike_interpolation',)
    assert register_in_place in (True, False)
    if fourier_cutoff_radius is None:
        fourier_cutoff_radius = estimate_fourier_cutoff_radius(c, debug=debug)
    assert (0 < fourier_cutoff_radius <= 0.5)
    assert fail_180_test in ('fix_but_print_warning',
                             'fix_silently',
                             'ignore_silently',
                             'raise_exception')
    assert debug in (True, False)
    if debug and np_tif is None:
        raise UserWarning("Failed to import np_tif; no debug mode.")
    if map_coordinates is None:
        raise UserWarning("Failed to import scipy map_coordinates;" +
                          " no stack_rotational_registration.")
    ## We'll multiply each slice of the stack by a circular mask to
    ## prevent edge-effect artifacts when we Fourier transform:
    mask_ud = np.arange(-c.shape[1]/2, c.shape[1]/2).reshape(c.shape[1], 1)
    mask_lr = np.arange(-c.shape[2]/2, c.shape[2]/2).reshape(1, c.shape[2])
    mask_r_sq = mask_ud**2 + mask_lr**2
    max_r_sq = (c.shape[1]/2)**2
    mask =  (mask_r_sq <= max_r_sq) * np.cos((np.pi/2)*(mask_r_sq/max_r_sq))
    del mask_ud, mask_lr, mask_r_sq
    masked_reference_slice = align_to_this_slice * mask
    small_ref = bucket(masked_reference_slice, (4, 4))
    ## We'll base our rotational registration on the logarithms of the
    ## amplitudes of the low spatial frequencies of the reference and
    ## target slices. We'll need the amplitudes of the Fourier transform
    ## of the masked reference slice:
    ref_slice_ft_log_amp = np.log(np.abs(np.fft.fftshift(np.fft.rfftn(
        masked_reference_slice), axes=0)))
    ## Transform the reference slice log amplitudes to polar
    ## coordinates. Note that we avoid some potential subtleties here
    ## because we've cropped to a square field of view (which was the
    ## right thing to do anyway):
    n_y, n_x = ref_slice_ft_log_amp.shape
    k_r = np.arange(1, 2*fourier_cutoff_radius*n_x)
    k_theta = np.linspace(-np.pi/2, np.pi/2,
                          np.ceil(2*fourier_cutoff_radius*n_y))
    k_theta_delta_degrees = (k_theta[1] - k_theta[0]) * 180/np.pi
    k_r, k_theta = k_r.reshape(len(k_r), 1), k_theta.reshape(1, len(k_theta))
    k_y = k_r * np.sin(k_theta) + n_y//2
    k_x = k_r * np.cos(k_theta)
    del k_r, k_theta, n_y, n_x
    polar_ref = map_coordinates(ref_slice_ft_log_amp, (k_y, k_x))
    polar_ref_ft_conj = np.conj(np.fft.rfft(polar_ref, axis=1))
    n_y, n_x = polar_ref_ft_conj.shape
    y, x = np.arange(n_y).reshape(n_y, 1), np.arange(n_x).reshape(1, n_x)
    polar_fourier_mask = (y > x) # Triangular half-space of good FT phases
    del n_y, n_x, y, x
    ## Now we'll loop over each slice of the stack, calculate our
    ## registration rotations, and optionally apply the rotations to the
    ## original stack.
    registration_rotations_degrees = []
    if debug:
        ## Save some intermediate data to help with debugging
        n_z = c.shape[0]
        masked_stack = np.zeros_like(c)
        masked_stack_ft_log_amp = np.zeros((n_z,) + ref_slice_ft_log_amp.shape)
        polar_stack = np.zeros((n_z,) + polar_ref.shape)
        polar_stack_ft = np.zeros((n_z,) + polar_ref_ft_conj.shape,
                                  dtype=np.complex128)
        cross_power_spectra = np.zeros_like(polar_stack_ft)
        spikes = np.zeros_like(polar_stack)
    for which_slice in range(c.shape[0]):
        if debug:
            print("Calculating rotational registration for slice", which_slice)
        ## Compute the polar transform of the log of the amplitude
        ## spectrum of our masked slice:
        current_slice = c[which_slice, :, :] * mask
        current_slice_ft_log_amp = np.log(np.abs(np.fft.fftshift(
            np.fft.rfftn(current_slice), axes=0)))
        polar_slice = map_coordinates(current_slice_ft_log_amp, (k_y, k_x))
        ## Register the polar transform of the current slice against the
        ## polar transform of the reference slice, using a similar
        ## algorithm to the one in mr_stacky:
        polar_slice_ft = np.fft.rfft(polar_slice, axis=1)
        cross_power_spectrum = polar_slice_ft * polar_ref_ft_conj
        cross_power_spectrum = (polar_fourier_mask *
                                cross_power_spectrum /
                                np.abs(cross_power_spectrum))
        ## Inverse transform to get a 'spike' in each row of fixed k_r;
        ## the location of this spike gives the desired rotation in
        ## theta pixels, which we can convert back to an angle. Start by
        ## locating the spike to the nearest integer:
        spike = np.fft.irfft(cross_power_spectrum,
                             axis=1, n=polar_slice.shape[1])
        spike_1d = spike.sum(axis=0)
        loc = np.argmax(spike_1d)
        if refinement is 'spike_interpolation':
            ## Use (very simple) three-point polynomial interpolation to
            ## refine the location of the peak of the spike:
            neighbors = np.array([-1, 0, 1])
            lr_vals = spike_1d[(loc + neighbors) %spike.shape[1]]
            lr_fit = np.poly1d(np.polyfit(neighbors, lr_vals, deg=2))
            lr_max_shift = -lr_fit[1] / (2 * lr_fit[2])
            loc += lr_max_shift
        ## Convert our shift into a signed number near zero:
        loc = (loc + spike.shape[1]//2) % spike.shape[1] - spike.shape[1]//2
        ## Convert this shift in "theta pixels" back to a rotation in degrees:
        angle = loc * k_theta_delta_degrees
        ## There's a fundamental 180-degree ambiguity in the rotation
        ## determined by this algorthim. We need to check if adding 180
        ## degrees explains our data better. This is a half-assed fast
        ## check that just downsamples the bejesus out of the relevant
        ## slices and looks at cross-correlation.
        ## TODO: FIX THIS FOR REAL! ...or does it matter? Testing, for now.
        small_current_slice = bucket(current_slice, (4, 4))
        small_cur_000 = rotate(small_current_slice, angle=angle, reshape=False)
        small_cur_180 = rotate(small_current_slice,
                               angle=angle+180, reshape=False)
        if (small_cur_180*small_ref).sum() > (small_cur_000*small_ref).sum():
            if fail_180_test is 'fix_but_print_warning':
                angle += 180
                print(" **Warning: potentially ambiguous rotation detected**")
                print("   Inspect the registration for rotations off by 180"+
                      " degrees, at slice %i"%(which_slice))
            elif fail_180_test is 'fix_silently':
                angle += 180
            elif fail_180_test is 'ignore_silently':
                pass
            elif fail_180_test is 'raise_exception':
                raise UserWarning(
                    "Potentially ambiguous rotation detected.\n"+
                    "One of your slices needed more than 90 degrees rotation")
        ## Pencils down.
        registration_rotations_degrees.append(angle)
        if debug:
            ## Save some intermediate data to help with debugging
            masked_stack[which_slice, :, :] = current_slice
            masked_stack_ft_log_amp[which_slice, :, :] = (
                current_slice_ft_log_amp)
            polar_stack[which_slice, :, :] = polar_slice
            polar_stack_ft[which_slice, :, :] = polar_slice_ft
            cross_power_spectra[which_slice, :, :] = cross_power_spectrum
            spikes[which_slice, :, :] = np.fft.fftshift(spike, axes=1)
    if register_in_place:
        ## Rotate the slices of the input stack in-place so it's
        ## rotationally registered:
        for which_slice in range(s.shape[0]):
            s[which_slice, :, :] = rotate(
                s[which_slice, :, :],
                angle=registration_rotations_degrees[which_slice],
                reshape=False)
    if debug:
        np_tif.array_to_tif(masked_stack, 'DEBUG_masked_stack.tif')
        np_tif.array_to_tif(masked_stack_ft_log_amp,
                            'DEBUG_masked_stack_FT_log_magnitudes.tif')
        np_tif.array_to_tif(polar_stack,
                            'DEBUG_polar_stack.tif')
        np_tif.array_to_tif(np.angle(polar_stack_ft),
                            'DEBUG_polar_stack_FT_phases.tif')
        np_tif.array_to_tif(np.log(np.abs(polar_stack_ft)),
                            'DEBUG_polar_stack_FT_log_magnitudes.tif')
        np_tif.array_to_tif(np.angle(cross_power_spectra),
                            'DEBUG_cross_power_spectral_phases.tif')
        np_tif.array_to_tif(spikes, 'DEBUG_spikes.tif')
        if register_in_place:
            np_tif.array_to_tif(s, 'DEBUG_registered_stack.tif')
    return np.array(registration_rotations_degrees)

mr_spinny = stack_rotational_registration #I like calling it this.

def expected_cross_power_spectrum(shift, k_ud, k_lr):
    """A convenience function that gives the expected spectral phase
    associated with an arbitrary subpixel shift.

    `k_ud` and `k_lr` are 1D Fourier frequencies generated by
    `np.fft.fftfreq` (or equivalent). Returns a 2D numpy array of
    expected phases.
    """
    shift_ud, shift_lr = shift
    return np.exp(-2j*np.pi*(k_ud*shift_ud + k_lr*shift_lr))

def estimate_fourier_cutoff_radius(s, bg=0, debug=False):
    """Estimate the radius in the Fourier domain which divides signal
    from pure noise.

    The Fourier transform amplitudes of most microscope images show a
    clear circular edge, outside of which there is no signal. This
    function tries to estimate the position of this edge. The estimation
    is not especially precise, but seems to be within the tolerance of
    `stack_registration`.
    """
    # We only need one slice for this estimate:
    if len(s.shape) == 3: s = s[0, :, :]
    assert len(s.shape) == 2
    s = s - bg # Background subtraction
    # Mask `s` to avoid fourier-domain artifacts:
    mask_ud = np.sin(np.linspace(0, np.pi, s.shape[0])).reshape(s.shape[0], 1)
    mask_lr = np.sin(np.linspace(0, np.pi, s.shape[1])).reshape(1, s.shape[1])
    s = s * mask_ud * mask_lr
    # We use pixels in the 'corners' of the Fourier domain to estimate
    # our noise floor:
    k_ud, k_lr = np.fft.fftfreq(s.shape[-2]), np.fft.rfftfreq(s.shape[-1])
    k_ud, k_lr = k_ud.reshape(k_ud.size, 1), k_lr.reshape(1, k_lr.size)
    ft_radius = np.sqrt(k_ud**2 + k_lr**2)
    deplorables = ft_radius > 0.5
    ft_mag = np.abs(np.fft.rfftn(s))
    noise_floor = np.median(ft_mag[deplorables])
    # We use the brightest pixels in the Fourier domain (except the DC
    # term) to estimate the peak signal:
    ft_mag[0, 0] = 0
    peak_signal = ft_mag.max()
    # Our cutoff radius is the highest spatial frequency with an
    # amplitude that exceeds the geometric mean of the noise floor and
    # the peak signal:
    cutoff_signal = np.sqrt(noise_floor * peak_signal)
    cutoff_radius = ft_radius[(ft_mag > cutoff_signal) &
                              (ft_radius < 0.45)].max()
    if debug: print("Estimated Fourier cutoff radius:", cutoff_radius)
    return cutoff_radius

def bucket(x, bucket_size):
    """'Pixel bucket' a numpy array.

    By 'pixel bucket', I mean, replace groups of N consecutive pixels in
    the array with a single pixel which is the sum of the N replaced
    pixels. See: http://stackoverflow.com/q/36269508/513688
    """
    for b in bucket_size: assert float(b).is_integer()
    bucket_size = [int(b) for b in bucket_size]
    x = np.ascontiguousarray(x)
    new_shape = np.concatenate((np.array(x.shape) // bucket_size, bucket_size))
    old_strides = np.array(x.strides)
    new_strides = np.concatenate((old_strides * bucket_size, old_strides))
    axis = tuple(range(x.ndim, 2*x.ndim))
    return np.lib.stride_tricks.as_strided(x, new_shape, new_strides).sum(axis)

def coinflip_split(a, photoelectrons_per_count):
    """Split `a` into two subarrays with the same shape as `a` but
    roughly half the mean value, while attempting to preserve Poisson
    statistics.

    Interpret `a` as an image; for each photoelectron in the image, we
    flip a coin to decide which subarray the photoelectron ends up in.
    If the input array obeys Poisson statistics, the output arrays
    should too.

    To determine `photoelectrons_per_count`, consult your camera's
    documentation. Possibly, you could inspect the relationship between
    the mean and variance for some projection of `a` along which the
    only variation is due to noise.
    """
    photoelectrons = np.round(photoelectrons_per_count * a).astype(np.int64)
    # Flip a coin for each photoelectron to decide which substack it
    # gets assigned to:
    out_1 = np.random.binomial(photoelectrons, 0.5)
    return out_1, photoelectrons - out_1

if __name__ == '__main__':
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        def gaussian_filter(x, sigma): return x # No blurring; whatever.
    ## Simple debugging tests. Put a 2D TIF where python can find it.
    print("Loading test object...")
    obj = np_tif.tif_to_array('blobs.tif').astype(np.float64)
    print(" Done.")
    shifts = [
        [0, 0],
        [-1, 1],
        [-2, 1],
        [-3, 1],
        [-4, 1],
        [-5, 1],
        [-1, 15],
        [-2, 0],
        [-3, 21],
        [-4, 0],
        [-5, 0],
        ]
    bucket_size = (1, 4, 4)
    crop = 8
    blur = 4
    brightness = 0.1
    print("Creating shifted stack from test object...")
    shifted_obj = np.zeros_like(obj)
    stack = np.zeros((len(shifts),
                      obj.shape[1]//bucket_size[1] - 2*crop,
                      obj.shape[2]//bucket_size[2] - 2*crop))
    expected_phases = np.zeros((len(shifts),) +
                               np.fft.rfft(stack[0, :, :]).shape)
    k_ud, k_lr = np.fft.fftfreq(stack.shape[1]), np.fft.rfftfreq(stack.shape[2])
    k_ud, k_lr = k_ud.reshape(k_ud.size, 1), k_lr.reshape(1, k_lr.size)
    for s, (y, x) in enumerate(shifts):
        top = max(0, y)
        lef = max(0, x)
        bot = min(obj.shape[-2], obj.shape[-2] + y)
        rig = min(obj.shape[-1], obj.shape[-1] + x)
        shifted_obj.fill(0)
        shifted_obj[0, top:bot, lef:rig] = obj[0, top-y:bot-y, lef-x:rig-x]
        stack[s, :, :] = np.random.poisson(
            brightness *
            bucket(gaussian_filter(shifted_obj, blur), bucket_size
                   )[0, crop:-crop, crop:-crop])
        expected_phases[s, :, :] = np.angle(np.fft.fftshift(
            expected_cross_power_spectrum((y/bucket_size[1], x/bucket_size[2]),
                                          k_ud, k_lr), axes=0))
    np_tif.array_to_tif(expected_phases, 'DEBUG_expected_phase_vs_ref.tif')
    np_tif.array_to_tif(stack, 'DEBUG_stack.tif')
    print(" Done.")
    print("Registering test stack...")
    calculated_shifts = stack_registration(
        stack,
        refinement='spike_interpolation',
        debug=True)
    print(" Done.")
    for s, cs in zip(shifts, calculated_shifts):
        print('%0.2f (%i)'%(cs[0] * bucket_size[1], s[0]),
              '%0.2f (%i)'%(cs[1] * bucket_size[2], s[1]))

    # Now we test rotational alignment, on top of translational alignment
    input("Hit enter to test rotational registration...\n" + 
          " (This will overwrite some of the DEBUG_* files" + 
          " from translational registration)")
    print("Creating rotated and shifted stack from test object...")
    angles_deg = [180*np.random.random()-90 for s in shifts]
    angles_deg[0] = 0
    rotated_shifted_obj = np.zeros_like(obj)
    stack = np.zeros((len(shifts),
                      obj.shape[1]//bucket_size[1] - 2*crop,
                      obj.shape[2]//bucket_size[2] - 2*crop))
    for s, (y, x) in enumerate(shifts):
        top = max(0, y)
        lef = max(0, x)
        bot = min(obj.shape[-2], obj.shape[-2] + y)
        rig = min(obj.shape[-1], obj.shape[-1] + x)
        rotated_shifted_obj.fill(0)
        rotated_shifted_obj[0, top:bot, lef:rig
                            ] = obj[0, top-y:bot-y, lef-x:rig-x]
        rotated_shifted_obj = rotate(rotated_shifted_obj, angles_deg[s],
                                     reshape=False, axes=(2, 1))
        rotated_shifted_obj[rotated_shifted_obj <= 0] = 0
        stack[s, :, :] = np.random.poisson(
            brightness *
            bucket(gaussian_filter(rotated_shifted_obj, blur), bucket_size
                   )[0, crop:-crop, crop:-crop])
    np_tif.array_to_tif(stack, 'DEBUG_stack.tif')
    print(" Done.")
    print("Rotationally registering test stack...")
    calculated_angles = stack_rotational_registration(
        stack,
        refinement='spike_interpolation',
        register_in_place=True,
        debug=True)
    print("estimated angle (true angle)")
    for a, ca in zip(angles_deg, calculated_angles):
        print('%0.2f (%0.2f)'%(ca, -a))
    input("Second round...")
    stack_rotational_registration(
            stack,
            refinement='spike_interpolation',
            register_in_place=True,
            debug=True)
