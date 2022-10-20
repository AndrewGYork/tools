#!/usr/bin/python

import time
import numpy as np

def main():
    _test_sin_cos()
    _test_to_xyz()
    _test_polar_displacement()
    _test_propagators()
    _test_diffusive_step_speed()
    _test_safe_diffusive_step()
    try:
        _test_diffusive_step_accuracy()
    except KeyboardInterrupt:
        pass


def safe_diffusive_step(
    x, y, z,
    normalized_time_step,
    max_safe_step=0.5, # Don't count on this, could be wrong
    ):
    num_steps, remainder = np.divmod(normalized_time_step, max_safe_step)
    num_steps = num_steps.astype('uint64') # Always an integer
    num_steps_min, num_steps_max = num_steps.min(), num_steps.max()
    if num_steps_min == num_steps_max: # Scalar time step
        for _ in range(num_steps_max):
            x, y, z = diffusive_step(x, y, z, max_safe_step)
    else: # Vector time step
        assert len(normalized_time_step) == len(x)
        t_is_sorted = np.all(np.diff(normalized_time_step) > 0)
        if not t_is_sorted: # Sorted xyz makes selecting unfinished stuff fast
            idx = np.argsort(num_steps)
            x, y, z = x[idx], y[idx], z[idx]
            num_steps = num_steps[idx]
        which_step = 1
        while True:
            first_unfinished = np.searchsorted(num_steps, which_step)
            if first_unfinished == len(num_steps): # We're done taking steps
                break
            s = slice(first_unfinished, None)
            x[s], y[s], z[s] = diffusive_step(x[s], y[s], z[s], max_safe_step)
            which_step += 1
        if not t_is_sorted: # Undo our sorting
            idx_rev = np.empty_like(idx)
            idx_rev[idx] = np.arange(len(idx), dtype=idx.dtype)
            x, y, z = x[idx_rev], y[idx_rev], z[idx_rev]
    # Finally, take our 'remainder' step:
    if remainder.max() > 0:
        x, y, z = diffusive_step(x, y, z, remainder)
    return x, y, z

def _test_safe_diffusive_step(n=int(1e5)):
    x, y, z = np.zeros(n), np.zeros(n), np.ones(n)
    print()
    for tstep in (np.array(5),
                  np.random.exponential(size=n, scale=5),
                  ):
        t = []
        for i in range(10):
            start = time.perf_counter()
            safe_diffusive_step(x, y, z, normalized_time_step=tstep)
            end = time.perf_counter()
            t.append(end - start)
        print('%0.1f nanoseconds per'%(1e9*np.mean(t) / n),
              "safe_diffusive_step(normalized_time_step=%.1f +/- %.1f)"%(
                  tstep.mean(), tstep.std()))

def diffusive_step(x, y, z, normalized_time_step, propagator='ghosh'):
    assert len(x) == len(y) == len(z)
    angle_step = np.sqrt(2*normalized_time_step)
    assert angle_step.shape in ((), (1,), x.shape)
    angle_step = np.broadcast_to(angle_step, x.shape)
    assert propagator in ('ghosh', 'gaussian')
    prop = ghosh_propagator if propagator == 'ghosh' else gaussian_propagator
    theta_d = prop(angle_step)
    phi_d = np.random.uniform(0, 2*np.pi, len(angle_step))
    return polar_displacement(x, y, z, theta_d, phi_d)

def _test_diffusive_step_speed(n=int(1e5)):
    x, y, z = np.zeros(n), np.zeros(n), np.ones(n)
    t = {}
    print()
    for prop in ('gaussian', 'ghosh'):
        t[prop] = []
        for i in range(10):
            start = time.perf_counter()
            diffusive_step(x, y, z, normalized_time_step=0.1, propagator=prop)
            end = time.perf_counter()
            t[prop].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[prop]) / n),
              "nanoseconds per diffusive_step('%s')"%(prop))

def _test_diffusive_step_accuracy(n=int(1e5)):
    x, y, z = np.zeros(n), np.zeros(n), np.ones(n)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib import failed; no graphical test of diffusive_step()")
        return None

    def from_the_pole(propagator, n_steps):
        xp, yp, zp = x, y, z
        normalized_time_interval = 0.5
        dt = normalized_time_interval / n_steps
        for _ in range(n_steps):
            xp, yp, zp = diffusive_step(xp, yp, zp, dt, propagator)
        hist, bin_edges = np.histogram(np.arccos(zp), bins=30, range=(0, np.pi))
        zc = (bin_edges[:-1] + bin_edges[1:]) / 2
        return zc, hist

    results = {'gaussian': {}, 'ghosh': {}}
    propagators = results.keys()
    step_numbers = (1, 2, 3, 10, 40, 200)
    num_molecules = 0

    # Save a figure on disk to show the relative accuracy of the propagators
    # This takes a while to converge to good accuracy, so keep saving
    # intermediate figures as we make progress through the calculation.
    print("\nTesting diffusive_step() accuracy vs. step size.")
    print("Use a KeyboardInterrupt (Ctrl-C) to abort")
    print("Saving results in test_diffusive_step.png...", end='')
    for rep in range(20000): # A long, long time
        for prop in propagators:
            for num_steps in step_numbers:
                if num_steps not in results[prop]:
                    results[prop][num_steps] = np.zeros(30)
                zc, hist = from_the_pole(prop, num_steps)
                results[prop][num_steps] += hist
        num_molecules += n
        ref = results['ghosh'][max(step_numbers)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for prop in propagators:
            for i, num_steps in enumerate(step_numbers):
                fmt = '-C%i'%i if prop == 'gaussian' else '--C%i'%i
                label='%s %i steps'%(prop, num_steps)
                ax1.plot(zc, results[prop][num_steps],       fmt, label=label)
                ax2.plot(zc, results[prop][num_steps] - ref, fmt, label=label)
        fig.suptitle("Number of molecules: %0.2e"%num_molecules)
        ax1.set_title("Propagator result")
        ax2.set_title("Result minus ref. (Ghosh %i steps)"%(max(step_numbers)))
        for ax in (ax1, ax2):
            ax.set_xlabel("Angle (radians)")
            ax.legend()
            ax.grid('on')
        plt.savefig('test_diffusive_step.png')
        plt.close(fig)
        print('.', end='', sep='')
    print("done.")
    return None

def ghosh_propagator(step_sizes):
    """Draw random angular displacements from the "new" propagator for
    diffusion on a sphere, as described by Ghosh et al. in arXiv:1303.1278.

    The new propagator, hopefully accurate for larger time steps:
        q_new = ((2 / sigma**2) * np.exp(-(beta/sigma)**2) *
                 np.sqrt(beta * np.sin(beta)) *
                 norm)

                 ...where 'norm' is chosen to normalize the distribution.

    'step_sizes' is a 1d numpy array of nonnegative floating point
    numbers ('sigma' in the equation above).

    Return value is a 1d numpy array the same shape as 'step_sizes',
    with each entry drawn from a distribution determined by the
    corresponding entry of 'step_sizes'.
    """
    min_step_sizes = np.min(step_sizes)
    assert min_step_sizes >= 0
    if min_step_sizes == 0:
        step_sizes = np.clip(step_sizes, a_min=1e-12, a_max=None)
    # Iteratively populate the result vector:
    first_iteration = True
    while True:
        # Which step sizes do we still need to draw random numbers for?
        steps = step_sizes if first_iteration else step_sizes[tbd]
        # Draw from a truncated non-normalized version of the Gaussian
        # propagator as an upper bound for rejection sampling. Don't
        # bother drawing values that will exceed pi.
        will_draw_pi = np.exp(-(np.pi / steps)**2)
        candidates = steps * np.sqrt(-np.log(
            np.random.uniform(will_draw_pi, 1, len(steps))))
        # To convert draws from our upper bound distribution to our desired
        # distribution, reject samples stochastically by the ratio of the
        # desired distribution to the upper bound distribution,
        # which is sqrt(sin(x)/x).
        rejected = (np.random.uniform(0, 1, candidates.shape) >
                    np.sqrt(np.sin(candidates) / candidates))
        # Update results
        if first_iteration:
            result = candidates
            tbd = np.nonzero(rejected)[0] # Coordinates of unset results
            first_iteration = False
        else:
            result[tbd] = candidates
            tbd = tbd[rejected]
        if len(tbd) == 0: break # We've set every element of the result
    return result

def gaussian_propagator(step_sizes):
    """Draw random angular displacements from the Gaussian propagator for
    diffusion on a sphere, as described by Ghosh et al. in arXiv:1303.1278.

    The Gaussian propagator, accurate for small time steps:
        q_gauss = ((2 / sigma**2) * np.exp(-(beta/sigma)**2) *
                   beta)

    'step_sizes' is a 1d numpy array of nonnegative floating point
    numbers ('sigma' in the equation above).

    Return value is a 1d numpy array the same shape as 'step_sizes',
    with each entry drawn from a distribution determined by the
    corresponding entry of 'step_sizes'.

    This is mostly useful for verifying that the Ghosh propagator works,
    yielding equivalent results with fewer, larger steps.
    """
    # Calculate draws via inverse transform sampling.
    result = step_sizes * np.sqrt(-np.log(
        np.random.uniform(0, 1, len(step_sizes))))
    return result

def _test_propagators(n=int(1e6)):
    step_sizes = 0.1 * np.ones(n, dtype='float64')
    print()
    t = {}
    for propagator, method in ((gaussian_propagator, 'gaussian'),
                               (ghosh_propagator,    'ghosh'   )):
        t[method] = []
        for i in range(10):
            start = time.perf_counter()
            propagator(step_sizes)
            end = time.perf_counter()
            t[method].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[method]) / n),
              "nanoseconds per %s_propagator()"%(method))

def polar_displacement(x, y, z, theta_d, phi_d, method='accurate', norm=True):
    """Take a Cartesian positions x, y, z and update them by
    spherical displacements (theta_d, phi_d). Theta is how much you
    moved and phi is which way.

    Note that this returns gibberish for theta_d > pi
    """
    assert method in ('naive', 'accurate')
    x_d, y_d, z_d = to_xyz(theta_d, phi_d)
    # Since the particles aren't (generally) at the north pole, we
    # have to rotate back to each particle's actual position. We'll
    # do this via a rotation matrix calculated as described in:
    #  doi.org/10.1080/10867651.1999.10487509
    #  "Efficiently Building a Matrix to Rotate One Vector to Another",
    with np.errstate(divide='ignore'): # In case z = -1
        ovr_1pz = 1 / (1+z)
    if method == 'naive': # The obvious way
        with np.errstate(invalid='ignore'):
            x_f = x_d*(z + y*y*ovr_1pz) + y_d*(   -x*y*ovr_1pz) + z_d*(x)
            y_f = x_d*(   -x*y*ovr_1pz) + y_d*(z + x*x*ovr_1pz) + z_d*(y)
            z_f = x_d*(     -x        ) + y_d*(     -y        ) + z_d*(z)
        isnan = (z == -1) # We divided by zero above, we have to fix it now
        x_f[isnan] = -x_d[isnan]
        y_f[isnan] =  y_d[isnan]
        z_f[isnan] = -z_d[isnan]
    elif method == 'accurate': # More complicated, but numerically stable?
        # Precompute a few intermediates:
        with np.errstate(invalid='ignore'):
            y_ovr_1pz =    y*ovr_1pz #  y / (1+z)
            xy_ovr_1pz = x*y_ovr_1pz # xy / (1+z)
            yy_ovr_1pz = y*y_ovr_1pz # yy / (1+z)
            xx_ovr_1pz = x*x*ovr_1pz # xx / (1+z)
        # We divided by (1+z), which is unstable for z ~= -1
        # We'll substitute slower stable versions:
        # x^2/(1+z) = (1-z) * cos(phi)^2
        # y^2/(1+z) = (1-z) * sin(phi)^2
        # x*y/(1+z) = (1-z) * sin(phi)*cos(phi)
        unstable = z < (-1 + 5e-2) # Not sure where instability kicks in...
        x_u, y_u, z_u = x[unstable], y[unstable], z[unstable]
        phi_u = np.arctan2(y_u, x_u)
        sin_ph_u, cos_ph_u = sin_cos(phi_u)
        xy_ovr_1pz[unstable] = (1 - z_u) * sin_ph_u * cos_ph_u
        yy_ovr_1pz[unstable] = (1 - z_u) * sin_ph_u * sin_ph_u
        xx_ovr_1pz[unstable] = (1 - z_u) * cos_ph_u * cos_ph_u
        # Now we're ready for the matrix multiply:
        x_f = x_d*(z + yy_ovr_1pz) + y_d*(   -xy_ovr_1pz) + z_d*(x)
        y_f = x_d*(   -xy_ovr_1pz) + y_d*(z + xx_ovr_1pz) + z_d*(y)
        z_f = x_d*(    -x        ) + y_d*(    -y        ) + z_d*(z)
    if norm:
        r = np.sqrt(x_f*x_f + y_f*y_f + z_f*z_f)
        x_f /= r
        y_f /= r
        z_f /= r
    return x_f, y_f, z_f

def _test_polar_displacement(n=int(1e6), dtype='float64', norm=True):
    theta_d =    np.abs(np.random.normal(  0,   np.pi, size=n)).astype(dtype)
    phi_d   =           np.random.uniform( 0, 2*np.pi, size=n ).astype(dtype)

    theta_i = np.arccos(np.random.uniform(-1,       1, size=n)).astype(dtype)
    phi_i   =           np.random.uniform( 0, 2*np.pi, size=n ).astype(dtype)
    theta_i[0] = np.pi
    theta_i[1] = np.pi - 1e-3
    x, y, z = to_xyz(theta_i, phi_i)
    print()    
    t = {}
    for method in ('naive', 'accurate'):
        t[method] = []
        for i in range(10):
            start = time.perf_counter()
            polar_displacement(x, y, z, theta_d, phi_d, method, norm)
            end = time.perf_counter()
            t[method].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[method]) / n),
              "nanoseconds per polar_displacement(method='%s')"%(method))
    for method in ('naive', 'accurate'):
        xf, yf, zf = polar_displacement(x, y, z, theta_d, phi_d, method, norm)
        angular_error = np.abs(np.cos(theta_d) - (x *xf + y *yf + z *zf))
        print("%0.3e maximum angular error"%(angular_error.max()),
              "for polar_displacement(method='%s')"%(method))
        radial_error = np.abs(1                - (xf*xf + yf*yf + zf*zf))
        print("%0.3e maximum  radial error"%(radial_error.max()),
              "for polar_displacement(method='%s')"%(method))        
    return None

def to_xyz(theta, phi, method='ugly'):
    """Convert spherical polar angles to unit-length Cartesian coordinates
    """
    assert method in ('ugly', 'direct')
    sin_th, cos_th = sin_cos(theta, method='0,pi')
    sin_ph, cos_ph = sin_cos(phi,   method='0,2pi')
    if method == 'direct': # The obvious way
        x = sin_th * cos_ph
        y = sin_th * sin_ph
        z = cos_th
    if method == 'ugly': # An uglier method with less memory allocation
        np.multiply(sin_th, cos_ph, out=cos_ph); x = cos_ph
        np.multiply(sin_th, sin_ph, out=sin_ph); y = sin_ph
        z = cos_th
    return x, y, z

def _test_to_xyz(n=int(1e6), dtype='float64'):
    theta = np.random.uniform(0,   np.pi, size=n).astype(dtype)
    phi   = np.random.uniform(0, 2*np.pi, size=n).astype(dtype)

    assert np.allclose(to_xyz(theta, phi, method='direct'),
                       to_xyz(theta, phi, method='ugly'))
    print()
    t = {}
    for method in ('direct', 'ugly'):
        t[method] = []
        for i in range(10):
            start = time.perf_counter()
            to_xyz(theta, phi, method)
            end = time.perf_counter()
            t[method].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[method]) / n),
              'nanoseconds per %6s to_xyz()'%(method))
    return None

def sin_cos(radians, method='sqrt'):
    """We often want both the sine and cosine of an array of angles. We
    can do this slightly faster with a sqrt, especially in the common
    cases where the angles are between 0 and pi, or 0 and 2pi.

    Since the whole point of this code is to be fast, there's no
    checking for validity, i.e. 0 < radians < pi, 2pi. Make sure you
    don't use out-of-range arguments.
    """
    radians = np.atleast_1d(radians)
    assert method in ('direct', 'sqrt', '0,2pi', '0,pi')
    cos = np.cos(radians)
    
    if method == 'direct': # Simple and obvious
        sin = np.sin(radians)
    else: # |sin| = np.sqrt(1 - cos*cos)
        sin = np.sqrt(1 - cos*cos)
        
    if method == 'sqrt': # Handle arbitrary values of 'radians'
        sin[np.pi - (radians % (2*np.pi)) < 0] *= -1
    elif method == '0,2pi': # Assume 0 < radians < 2pi, no mod
        sin[np.pi - (radians            ) < 0] *= -1
    elif method == '0,pi': # Assume 0 < radians < pi, no negation
        pass
    return sin, cos

def _test_sin_cos(n=int(1e6), dtype='float64'):
    """This test doesn't pass for float32, but neither does sin^2 + cos^2 == 1
    """
    theta = np.random.uniform(0,   np.pi, size=n).astype(dtype)
    phi   = np.random.uniform(0, 2*np.pi, size=n).astype(dtype)
    x     = np.random.uniform(0, 8*np.pi, size=n).astype(dtype)

    assert np.allclose(sin_cos(x,     method='direct'),
                       sin_cos(x,     method='sqrt'))
    
    assert np.allclose(sin_cos(phi,   method='direct'),
                       sin_cos(phi,   method='0,2pi'))

    assert np.allclose(sin_cos(theta, method='direct'),
                       sin_cos(theta, method='0,pi'))
    print()
    t = {}
    for method in ('direct', 'sqrt', '0,2pi', '0,pi'):
        t[method] = []
        for i in range(10):
            start = time.perf_counter()
            sin_cos(theta, method)
            end = time.perf_counter()
            t[method].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[method]) / n),
              'nanoseconds per %6s sin_cos()'%(method))
    return None


# An object that only knows how to rotate


### An object that only knows photophysics


if __name__ == '__main__':
    main()
