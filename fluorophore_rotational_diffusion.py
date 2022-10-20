#!/usr/bin/python

import time
import numpy as np

def main():
    n = int(1e7)
    theta = np.random.uniform(0,   np.pi, size=n).astype('float64')
    phi   = np.random.uniform(0, 2*np.pi, size=n).astype('float64')
    temp  = np.random.uniform(0, 8*np.pi, size=n).astype('float64')

##    phi_i   =           np.random.uniform( 0, 2*np.pi, n).astype('float32')
##    theta_i = np.arccos(np.random.uniform(-1      , 1, n)).astype('float32')
##    theta_i[0] = np.pi - 1e-3
##    x, y, z = to_xyz(theta_i, phi_i)
##
##    t1, t2 = [], []
##    for i in range(10):
##        start = time.perf_counter()
##        xf, yf, zf = polar_displacement(x, y, z, theta, phi, method='naive')
##        end = time.perf_counter()
##        t1.append(end - start)
##        dot = x*xf + y*yf + z*zf
##        cos = np.cos(theta)
##        print(xf[0], yf[0], zf[0])
##        print((cos-dot).min(), (cos-dot).max())
##
##        start = time.perf_counter()
##        xf, yf, zf = polar_displacement(x, y, z, theta, phi, method='accurate')
##        end = time.perf_counter()
##        t2.append(end - start)
##        dot = x*xf + y*yf + z*zf
##        cos = np.cos(theta)
##        print(xf[0], yf[0], zf[0])
##        print((cos-dot).min(), (cos-dot).max())
##        print()
##    print('%0.2f'%(1e9*np.mean(t1) / n), "ns per displacement")
##    print('%0.2f'%(1e9*np.mean(t2) / n), "ns per displacement")



##    t1, t2 = [], []
##    for i in range(10):
##        start = time.perf_counter()
##        small = theta < 0.01
##        theta[small] = 1
##        end = time.perf_counter()
##        t1.append(end - start)
##
##        start = time.perf_counter()
##        theta = np.sqrt(theta)
##        end = time.perf_counter()
##        t2.append(end - start)
##
##    print()
##    print('%0.2f'%(1e9*np.mean(t1) / n), "ns per")
##    print('%0.2f'%(1e9*np.mean(t2) / n), "ns per")


##    # Check to_xyz() performance
##    t1, t2 = [], []
##    for i in range(10):
##        start = time.perf_counter()
##        x1, y1, z1 = to_xyz(theta, phi, method='direct')
##        end = time.perf_counter()
##        t1.append(end - start)
##
##        start = time.perf_counter()
##        x2, y2, z2 = to_xyz(theta, phi)
##        end = time.perf_counter()
##        t2.append(end - start)
##        assert np.allclose(x1, x2)
##        assert np.allclose(y1, y2)
##        assert np.allclose(z1, z2)
##    print()
##    print('%0.2f'%(1e9*np.mean(t1) / n), "ns per direct to_xyz()")
##    print('%0.2f'%(1e9*np.mean(t2) / n), "ns per ugly   to_xyz()")

##    # Check sin_cos() performance    
##    assert np.allclose(sin_cos(temp,  method='direct'),
##                       sin_cos(temp,  method='sqrt'))
##    
##    assert np.allclose(sin_cos(phi,   method='direct'),
##                       sin_cos(phi,   method='0,2pi'))
##
##    assert np.allclose(sin_cos(theta, method='direct'),
##                       sin_cos(theta, method='0,pi'))
##    t1, t2, t3, t4 = [], [], [], []
##    for i in range(10):
##        start = time.perf_counter()
##        sin_cos(theta, method='direct')
##        end = time.perf_counter()
##        t1.append(end - start)
##
##        start = time.perf_counter()
##        sin_cos(theta, method='sqrt')
##        end = time.perf_counter()
##        t2.append(end - start)
##
##        start = time.perf_counter()
##        sin_cos(theta, method='0,2pi')
##        end = time.perf_counter()
##        t3.append(end - start)
##
##        start = time.perf_counter()
##        sin_cos(theta, method='0,pi')
##        end = time.perf_counter()
##        t4.append(end - start)
##
##    print()
##    print('%0.1f'%(1e9*np.mean(t1) / n), 'nanoseconds per direct sin_cos()')
##    print('%0.1f'%(1e9*np.mean(t2) / n), 'nanoseconds per sqrt   sin_cos()')
##    print('%0.1f'%(1e9*np.mean(t3) / n), 'nanoseconds per 0,2pi  sin_cos()')
##    print('%0.1f'%(1e9*np.mean(t4) / n), 'nanoseconds per 0,pi   sin_cos()')


def polar_displacement(x, y, z, theta_d, phi_d, method='accurate'):
    """Take a Cartesian positions x, y, z and update them by
    spherical displacements (theta_d, phi_d). Theta is how much you
    moved and phi is which way.
    """
    assert method in ('naive', 'accurate')
    x_d, y_d, z_d = to_xyz(theta_d, phi_d)
    # Since the particles aren't (generally) at the north pole, we
    # have to rotate back to each particle's actual position. We'll
    # do this via a rotation matrix calculated as described in:
    #  doi.org/10.1080/10867651.1999.10487509
    #  "Efficiently Building a Matrix to Rotate One Vector to Another",
    if method == 'naive': # The obvious way
        x_f = x_d*(z + y*y/(1+z)) + y_d*(   -x*y/(1+z)) + z_d*(x)
        y_f = x_d*(   -x*y/(1+z)) + y_d*(z + x*x/(1+z)) + z_d*(y)
        z_f = x_d*(     -x      ) + y_d*(     -y      ) + z_d*(z)
    elif method == 'accurate': # More complicated, but numerically stable?
        # Precompute a few intermediates:
        with np.errstate(divide='ignore'): # In case z = -1
            ovr_1pz = 1 / (1+z)
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
        unstable = z < (-1 + 1e-3) # Not sure where instability kicks in...
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
    return x_f, y_f, z_f

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

def sin_cos(radians, method='sqrt'):
    """We often want both the sine and cosine of an array of angles. We
    can do this slightly faster with a sqrt, especially in the common
    cases where the angles are between 0 and pi, or 0 and 2pi.

    Since the whole point of this code is to be fast, there's no
    checking for validity, i.e. 0 < radians < pi, 2pi. Make sure you
    don't use out-of-range arguments.
    """
    assert method in ('direct', 'sqrt', '0,2pi', '0,pi')
    cos = np.cos(radians)
    
    if method == 'direct': # Simple and obvious
        sin = np.sin(radians)
    else: # |sin| = np.sqrt(1 - cos*cos)
        s = cos*cos
        np.subtract(1, s, out=s)
        np.sqrt(s, out=s)
        sin = s
        
    if method == 'sqrt': # Handle arbitrary values of 'radians'
        r = radians % (2*np.pi)
        np.subtract(np.pi, r, out=r)
        np.copysign(sin, r, out=sin)
    elif method == '0,2pi': # Assume 0 < radians < 2pi, no mod
        sin[np.pi - radians < 0] *= -1
##        np.copysign(sin, np.pi - radians, out=sin) 
    elif method == '0,pi': # Assume 0 < radians < pi, no copysign
        pass
    return sin, cos


# An object that only knows how to rotate


##def polar_displacement(xyz, theta_d, phi_d):
##    """Take a Cartesian positions xyz and update them by
##    spherical displacements (theta_d, phi_d). Theta is how much you
##    moved and phi is which way.
##    """
##    sin_th_d, cos_th_d = np.sin(theta_d), np.cos(theta_d); del theta_d
##    sin_ph_d, cos_ph_d = np.sin(phi_d), np.cos(phi_d);     del phi_d
##    # Convert to Cartesian:
##    x_d = sin_th_d * cos_ph_d; del cos_ph_d
##    y_d = sin_th_d * sin_ph_d; del sin_ph_d, sin_th_d
##    z_d = cos_th_d;            del cos_th_d
##    # Since the particles aren't (generally) at the north pole, we
##    # have to rotate back to each particle's actual position. We'll
##    # do this via a rotation about the y-axis by theta, followed by
##    # a rotation about the z-axis by phi:
##    theta = np.arccos(z)
##    sin_th = np.sin(theta);                        del theta
##    z_f = x_d *  -sin_th                + z_d * z; del sin_th
##    phi = np.arctan2(y, x)
##    cos_ph, sin_ph = np.cos(phi), np.sin(phi);     del phi
##    y_f = x_d * z*sin_ph + y_d * cos_ph + z_d * y; del y
##    x_f = x_d * z*cos_ph + y_d *-sin_ph + z_d * x
##    return x_f, y_f, z_f


##def gaussian_propagator(step_sizes):
##    """Draw random angular displacements from the Gaussian propagator for
##    diffusion on a sphere, as described by Ghosh et al. in arXiv:1303.1278.
##
##    The Gaussian propagator, accurate for small time steps:
##        q_gauss = ((2 / sigma**2) * np.exp(-(beta/sigma)**2) *
##                   beta)
##
##    'step_sizes' is a 1d numpy array of nonnegative floating point
##    numbers ('sigma' in the equation above).
##
##    Return value is a 1d numpy array the same shape as 'step_sizes',
##    with each entry drawn from a distribution determined by the
##    corresponding entry of 'step_sizes'.
##
##    This is mostly useful for verifying that the Ghosh propagator works,
##    yielding equivalent results with fewer, larger steps.
##    """
##    # Calculate draws via inverse transform sampling.
##    result = step_sizes * np.sqrt(-np.log(
##        np.random.uniform(0, 1, len(step_sizes))))
##    return result
##
##
##def ghosh_propagator(step_sizes):
##    """Draw random angular displacements from the "new" propagator for
##    diffusion on a sphere, as described by Ghosh et al. in arXiv:1303.1278.
##
##    The new propagator, hopefully accurate for larger time steps:
##        q_new = ((2 / sigma**2) * np.exp(-(beta/sigma)**2) *
##                 np.sqrt(beta * np.sin(beta)) *
##                 norm)
##
##                 ...where 'norm' is chosen to normalize the distribution.
##
##    'step_sizes' is a 1d numpy array of nonnegative floating point
##    numbers ('sigma' in the equation above).
##
##    Return value is a 1d numpy array the same shape as 'step_sizes',
##    with each entry drawn from a distribution determined by the
##    corresponding entry of 'step_sizes'.
##    """
##    # Iteratively populate the result vector:
##    first_iteration = True
##    while True:
##        # Which step sizes do we still need to draw random numbers for?
##        steps = step_sizes if first_iteration else step_sizes[tbd]
##        # Draw from a truncated non-normalized version of the Gaussian
##        # propagator as an upper bound for rejection sampling. Don't
##        # bother drawing values that will exceed pi.
##        will_draw_pi = np.exp(-(np.pi / steps)**2)
##        candidates = steps * np.sqrt(-np.log(
##            np.random.uniform(will_draw_pi, 1, len(steps))))
##        # To convert draws from our upper bound distribution to our desired
##        # distribution, reject samples stochastically by the ratio of the
##        # desired distribution to the upper bound distribution,
##        # which is sqrt(sin(x)/x).
##        rejected = (np.random.uniform(0, 1, candidates.shape) >
##                    np.sqrt(np.sin(candidates) / candidates))
##        # Update results
##        if first_iteration:
##            result = candidates
##            tbd = np.nonzero(rejected)[0] # Coordinates of unset results
##            first_iteration = False
##        else:
##            result[tbd] = candidates
##            tbd = tbd[rejected]
##        if len(tbd) == 0: break # We've set every element of the result
##    return result
##
##
### An object that only knows photophysics


if __name__ == '__main__':
    main()
