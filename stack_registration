import numpy as np
try:
    from scipy.optimize import minimize
except:
    minimize = None #Won't be able to use 'phase_fitting' in stack_registration
try:
    import np_tif
except:
    np_tif = None #Won't be able to use the 'debug' option of stack_registration

def stack_registration(
    s,
    align_to_this_slice=0,
    refinement='spike_interpolation',
    register_in_place=True,
    registered_stack_is_masked=False,
    fourier_mask_magnitude=0.15,
    debug=False):
    """Calculate shifts which would register the slices of a
    three-dimensional stack 's', and optionally register the stack in-place.

    Axis 0 is the "z-axis", axis 1 is the "up-down" (Y) axis, and axis 2
    is the "left-right" (X) axis. For each XY slice, we calculate the
    shift in the XY plane which would line that slice up with the slice
    specified by 'align_to_this_slice'.

    'refinement' is one of 'integer', 'spike_interpolation', or
    'phase_fitting', in order of increasing precision/slowness. I don't
    yet have any evidence that my implementation of phase fitting gives
    any improvement (faster, simpler) simple spike interpolation, so
    caveat emptor.

    'register_in_place': If 'True', modify the input stack 's' by
    shifting its slices to line up with the reference slice.

    'registered_stack_is_masked': We mask each slice of the stack so
    that it goes to zero at the edges, which reduces Fourier artifacts
    and improves registration accuracy. If we're also modifying the
    input stack, we can save one Fourier transform per iteration if
    we're willing to substitute the 'masked' version of each slice for
    its original value.

    'fourier_mask_magnitude': Ignore the Fourier phases of spatial
    frequencies above this cutoff, since they're probably lousy due to
    aliasing and noise anyway.
    """
    assert len(s.shape) == 3
    assert align_to_this_slice in range(s.shape[0])
    assert refinement in ('integer', 'spike_interpolation', 'phase_fitting')
    if refinement == 'phase_fitting' and minimize is None:
        raise UserWarning("Failed to import scipy minimize; no phase fitting.")
    assert register_in_place in (True, False)
    assert registered_stack_is_masked in (True, False)
    assert 0 < fourier_mask_magnitude < 0.5
    assert debug in (True, False)
    if debug and np_tif is None:
        raise UserWarning("Failed to import np_tif; no debug mode.")
    ## Multiply each slice of the stack by an XY mask that goes to zero
    ## at the edges, to prevent periodic boundary artifacts when we
    ## Fourier transform.
    mask_ud = np.sin(np.linspace(0, np.pi, s.shape[1])).reshape(s.shape[1], 1)
    mask_lr = np.sin(np.linspace(0, np.pi, s.shape[2])).reshape(1, s.shape[2])
    masked_reference_slice = s[align_to_this_slice, :, :] * mask_ud * mask_lr
    ## We'll base our registration on the phase of the low spatial
    ## frequencies of the cross-power spectrum. We'll need the complex
    ## conjugate of the Fourier transform of the masked reference slice,
    ## and a mask in the Fourier domain to pick out the low spatial
    ## frequencies:
    ref_slice_ft_conj = np.conj(np.fft.rfftn(masked_reference_slice))
    k_ud = np.fft.fftfreq(s.shape[1]).reshape(ref_slice_ft_conj.shape[0], 1)
    k_lr = np.fft.rfftfreq(s.shape[2]).reshape(1, ref_slice_ft_conj.shape[1])
    fourier_mask = (k_ud**2 + k_lr**2) < (fourier_mask_magnitude)**2
    ## We can also use these Fourier frequencies to define a convenience
    ## function that gives the expected spectral phase associated with
    ## an arbitrary subpixel shift:
    def expected_cross_power_spectrum(shift):
        shift_ud, shift_lr = shift
        return np.exp(-2j*np.pi*(k_ud*shift_ud + k_lr*shift_lr))
    ## Now we'll loop over each slice of the stack, calculate our
    ## registration shifts, and optionally apply the shifts to the
    ## original stack.
    registration_shifts = []
    if debug:
        ## Save some intermediate data to help with debugging
        masked_stack = np.zeros_like(s)
        masked_stack_ft = np.zeros(
            (s.shape[0],) + ref_slice_ft_conj.shape, dtype=np.complex128)
        cross_power_spectra = np.zeros(
            (s.shape[0],) + ref_slice_ft_conj.shape, dtype=np.complex128)
        spikes = np.zeros_like(s)
    for which_slice in range(s.shape[0]):
        if debug: print("Slice", which_slice)
        if which_slice == align_to_this_slice and not debug:
            registration_shifts.append(np.array((0, 0)))
            if register_in_place and registered_stack_is_masked:
                s[which_slice, :, :] = masked_reference_slice
            continue
        ## Compute the cross-power spectrum of our slice, and mask out
        ## the high spatial frequencies.
        current_slice = s[which_slice, :, :] * mask_ud * mask_lr
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
                    expected_cross_power_spectrum(loc) -
                    cross_power_spectrum
                    )[fourier_mask].sum()
                if debug: print(" Shift:", loc, "Disagreement:", disagreement)
                return disagreement
            loc = minimize(minimize_me,
                           x0=loc,
                           args=(cross_power_spectrum,),
                           method='Nelder-Mead').x
        registration_shifts.append(loc)
        if register_in_place:
            ## Modify the input stack in-place so it's registered.
            phase_correction = expected_cross_power_spectrum(loc)
            if registered_stack_is_masked:
                ## If we're willing to tolerate a "masked" result, we
                ## can save one Fourier transform:
                s[which_slice, :, :] = np.fft.irfftn(
                    current_slice_ft / phase_correction,
                    s=current_slice.shape).real
            else:
                ## Slower, but probably the right way to do it:
                shift_me = s[which_slice, :, :]
                if not shift_me.dtype == np.float64:
                    shift_me = shift_me.astype(np.float64)
                s[which_slice, :, :] = np.fft.irfftn(
                    np.fft.rfftn(shift_me) / phase_correction,
                    s=current_slice.shape).real
        if debug:
            ## Save some intermediate data to help with debugging
            masked_stack[which_slice, :, :] = current_slice
            masked_stack_ft[which_slice, :, :] = (
                np.fft.fftshift(current_slice_ft, axes=0))
            cross_power_spectra[which_slice, :, :] = (
                np.fft.fftshift(cross_power_spectrum * fourier_mask, axes=0))
            spikes[which_slice, :, :] = np.fft.fftshift(spike)
    if debug:
        np_tif.array_to_tif(masked_stack, 'DEBUG_masked_stack.tif')
        np_tif.array_to_tif(np.log(np.abs(masked_stack_ft)),
                            'DEBUG_masked_stack_FT_log_magnitudes.tif')
        np_tif.array_to_tif(np.angle(masked_stack_ft),
                            'DEBUG_masked_stack_FT_phases.tif')
        np_tif.array_to_tif(np.angle(cross_power_spectra),
                            'DEBUG_cross_power_spectral_phases.tif')
        np_tif.array_to_tif(spikes, 'DEBUG_spikes.tif')
        if register_in_place:
            np_tif.array_to_tif(s, 'DEBUG_registered_stack.tif')
    return registration_shifts

mr_stacky = stack_registration #I like calling it this.

def bucket(x, bucket_size):
    x = np.ascontiguousarray(x)
    new_shape = np.concatenate((np.array(x.shape) // bucket_size, bucket_size))
    old_strides = np.array(x.strides)
    new_strides = np.concatenate((old_strides * bucket_size, old_strides))
    axis = tuple(range(x.ndim, 2*x.ndim))
    return np.lib.stride_tricks.as_strided(x, new_shape, new_strides).sum(axis)

if __name__ == '__main__':
    ## Simple debugging tests. Put a 2D TIF where python can find it.
    print("Loading test object...")
    obj = np_tif.tif_to_array('blobs.tif').astype(np.float64)
    print(" Done.")
    shifts = [
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
    print("Creating shifted stack from test object...")
    shifted_obj = np.zeros_like(obj)
    stack = np.zeros((len(shifts) + 1,
                      obj.shape[1]//bucket_size[1] - 2*crop,
                      obj.shape[2]//bucket_size[2] - 2*crop))
    stack[0, :, :] = bucket(obj, bucket_size)[0, crop:-crop, crop:-crop]
    for s, (y, x) in enumerate(shifts):
        top = max(0, y)
        lef = max(0, x)
        bot = min(obj.shape[-2], obj.shape[-2] + y)
        rig = min(obj.shape[-1], obj.shape[-1] + x)
        shifted_obj.fill(0)
        shifted_obj[0, top:bot, lef:rig] = obj[0, top-y:bot-y, lef-x:rig-x]
        stack[s+1, :, :] = bucket(shifted_obj, bucket_size
                                  )[0, crop:-crop, crop:-crop]
    np_tif.array_to_tif(stack, 'DEBUG_stack.tif')
    print(" Done.")
    print("Registering test stack...")
    shifts = stack_registration(
        stack,
        refinement='spike_interpolation',
        debug=True)
    print(" Done.")
    for s in shifts: print(s)
    np_tif.array_to_tif(stack, 'DEBUG_stack_registered.tif')
