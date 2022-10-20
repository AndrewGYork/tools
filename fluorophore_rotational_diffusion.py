#!/usr/bin/python

import time
import numpy as np

def main():
##    _test_sin_cos()
##    _test_to_xyz()
##    _test_polar_displacement()
    _test_propagators()

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

def _test_propagators(n=int(1e7)):
##    step_sizes = 0.1 * np.ones(n, dtype='float64')
##    print()
##    t = {}
##    for propagator, method in ((gaussian_propagator, 'gaussian'),
##                               (ghosh_propagator,    'ghosh'   )):
##        t[method] = []
##        for i in range(10):
##            start = time.perf_counter()
##            propagator(step_sizes)
##            end = time.perf_counter()
##            t[method].append(end - start)
##        print('%0.1f'%(1e9*np.mean(t[method]) / n),
##              "nanoseconds per %s_propagator()"%(method))

    def from_the_pole(n, propagator, time_step, n_steps):
        dt = time_step / n_steps
        step_size = np.sqrt(2*dt)
        step_sizes = np.full(n, step_size)
        x, y, z = np.zeros(n), np.zeros(n), np.ones(n)
        for i in range(n_steps):
            theta_d = propagator(step_sizes)
            phi_d   = np.random.uniform(0, 2*np.pi, size=n)
            assert np.all(theta_d < np.pi)
            x, y, z = polar_displacement(x, y, z, theta_d, phi_d)
        return x, y, z

    gaussian, ghosh = [], []
    for n_steps in (160, 80, 40, 20, 10, 5):
        gaussian.append(from_the_pole(n, gaussian_propagator, 1, n_steps)[2])
        ghosh.append(   from_the_pole(n,    ghosh_propagator, 1, n_steps)[2])
    import matplotlib.pyplot as plt
    for i, z in enumerate(gaussian):
        hist, bin_edges = np.histogram(np.arccos(z), bins=30, range=(0, np.pi))
        if i == 0:
            ref = hist
        plt.plot((bin_edges[1:] + bin_edges[:-1])/2, hist-ref, '-',  c='C%i'%i,
                 label='Gauss %i'%i)
    for i, z in enumerate(ghosh):
        hist, bin_edges = np.histogram(np.arccos(z), bins=30, range=(0, np.pi))
        plt.plot((bin_edges[1:] + bin_edges[:-1])/2, hist-ref, '.-', c='C%i'%i,
                 label='Ghosh %i'%i)
    plt.legend()
    plt.show()


    # Run many small Gaussians to generate a 'ground truth'
    # Run one big Ghosh
    # Run one big Gaussian

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
