import warnings
import numpy as np
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 
except ImportError:
    torch = None
    dtype = np.float64
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

""" Created by Nathaniel H. Thayer and Andrew G. York

A simple ray tracer for optical design.
"""

class NumpyOrTorch:
    """
    NumpyOrTorch is (roughly) an alias for torch on the gpu if it's
    available. If not, it's an alias to torch on the cpu if it's
    available. If not, it's an alias to numpy, but you can't do
    gradient-based optimization.
    This isn't a general-purpose torch/numpy fusion, but it's good
    enough to simplify the rest of this module. We special-case a few
    numpy/pytorch functions as-needed.
    """
    def tensor(self, a):
        if torch:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.tensor(a, device=device, dtype=dtype)
        return np.array(a, dtype=dtype)

    def as_tensor(self, a):
        if torch:
            if hasattr(a, 'flags') and not a.flags.writeable:
                a = np.array(a) 
            return torch.as_tensor(a, device=device, dtype=dtype)
        return np.asarray(a, dtype=dtype)

    def asarray(self, a):
        a = a.detach() if hasattr(a, 'detach') else a
        a = a.cpu() if hasattr(a, 'cpu') else a
        return np.asarray(a)

    def clone(self, a):
        return torch.clone(a) if torch else np.copy(a)

    def __getattr__(self, attr): # This does most of the work
        return getattr(torch, attr, getattr(np, attr))

npt = NumpyOrTorch()

class RayBundle:
    """A set of 3D positions (xyz) and directions (k_xyz). 
    Often a single point source or a collimated beam.
    """
    def __init__(
        self,
        xyz,
        k_xyz,
        wavelength_um,
        enforce_normalized_k=True,
        ):
        assert xyz.ndim == 2
        assert xyz.shape[0] == 3
        self.num_rays = xyz.shape[1]
        self.xyz = npt.as_tensor(xyz)
        self.x, self.y, self.z = self.xyz
        assert k_xyz.shape == xyz.shape
        self.k_xyz = npt.as_tensor(k_xyz)
        self.kx, self.ky, self.kz = self.k_xyz
        assert wavelength_um > 0
        self.wavelength_um = float(wavelength_um)

        if enforce_normalized_k:
            assert all_unit_vectors(self.k_xyz, ignore_nan=False)
        return None

    def normalize_k(self):
        make_unit_vectors_inplace(self.k_xyz)

    def set_target(self, xyz=None, k_xyz=None):
        if xyz is not None:
            xyz = npt.tensor(xyz)
            assert xyz.shape[0] == 3
            if xyz.ndim == 1:
                xyz = xyz.reshape(3, 1)
            assert xyz.ndim == 2
            assert xyz.shape[1] in (1, self.xyz.shape[1])
            self.target_xyz = xyz

        if k_xyz is not None:
            k_xyz = npt.tensor(k_xyz)
            assert k_xyz.shape[0] == 3
            if k_xyz.ndim == 1:
                k_xyz = k_xyz.reshape(3, 1)
            assert k_xyz.ndim == 2
            assert k_xyz.shape[1] in (1, self.k_xyz.shape[1])
            self.target_k_xyz = k_xyz
        
    def __str__(self):
        return str(self.__dict__)

class RayBundleSequence:
    """A sequence of ray bundles, presumably continuous.
    """
    def __init__(self, initial_ray_bundle):
        assert isinstance(initial_ray_bundle, RayBundle)
        self.initial_ray_bundle = initial_ray_bundle
        self.seq = [initial_ray_bundle]
        return None

    def append(self, ray_bundle):
        assert isinstance(ray_bundle, RayBundle)
        self.seq.append(ray_bundle)
        return None

    def clear(self):
        self.seq = [self.initial_ray_bundle]
        return None
    
    def __getitem__(self, idx):
        return self.seq[idx]

class OpticalSystem:
    def __init__(self, index_list, surface_list):
        assert len(index_list) == 1 + len(surface_list)
        assert all(callable(n) or (n>0) for n in index_list)
        assert all(hasattr(srf, 'trace') for srf in surface_list)
        self.index_list = index_list
        self.surface_list = surface_list
        return None

    def trace_sequence(self, initial_ray_bundle):
        """Trace a ray bundle sequence through all the surfaces
        """
        rbs = RayBundleSequence(initial_ray_bundle)
        for ni, nf, srf in zip(self.index_list[:-1],
                               self.index_list[1:],
                               self.surface_list):
            rbs.append(srf.trace(rbs[-1], ni, nf))
        return rbs

    def trace_all_sequences(self, initial_ray_bundle_list):
        return [self.trace_sequence(irb) for irb in initial_ray_bundle_list]

    def __add__(self, optical_system_2):
        assert isinstance(optical_system_2, OpticalSystem)
        if self.index_list[-1] != optical_system_2.index_list[0]:
            raise ValueError(
                "The last index of the first optical system must"+
                " match the first index of the last optical system.")
        return OpticalSystem(
            index_list=  self.index_list   + optical_system_2.index_list[1:],
            surface_list=self.surface_list + optical_system_2.surface_list)
    
class SphericalSurface:
    """Arguably the most useful surface shape in optics.
    """
    def __init__(
        self,
        xc, yc, zc,
        r, # r>0 is convex, r<0 is concave,
        allow_backwards=False,
        ):
        xc, yc, zc, r = (npt.as_tensor(a).reshape(1) for a in (xc, yc, zc, r)) 
        self.xyz = npt.stack((xc, yc, zc))
        self.r = r
        assert self.xyz.shape == (3, 1), self.xyz.shape
        assert allow_backwards in (True, False)
        self.allow_backwards = allow_backwards
        return None

    def trace(self, ray_bundle, ni, nf):
        """Traces one ray bundle.
        """
        # Ray direction vector should already be normalized:
        assert all_unit_vectors(ray_bundle.k_xyz)

        # Find line-sphere intersection via these formulae:
        # https://doi.org/10.1007/978-1-4842-4427-2_7 p92
        r, f, d = self.r, (ray_bundle.xyz - self.xyz), ray_bundle.k_xyz

        b_prime = dot(-f, d)
        c = sq_mag(f) - r**2
        Delta = r**2 - sq_mag(f + b_prime*d)
        with np.errstate(invalid='ignore'): # Delta<0 implies no intersection
            sqrt_Delta = npt.sqrt(Delta)

        t1 = b_prime + npt.copysign(sqrt_Delta, b_prime)
        with np.errstate(divide='ignore'): # t1=0 implies no second intersection
            t0 = c/t1
        
        # Pick the appropriate intersection based on the sphere's concavity:
        concavity = npt.sign(r)
        t = npt.minimum(concavity*t0, concavity*t1) / concavity

        if not self.allow_backwards:
            t[is_negative(t)] = float('nan')
        xyz_f = ray_bundle.xyz + t*ray_bundle.k_xyz

        surface_normal = (self.xyz-xyz_f) / r # Surface normal unit vector
        k_xyz_f = deflection(ray_bundle, surface_normal, ni, nf)

        # Make sure we landed on the sphere
        not_on_sphere = (~is_unit_vector(surface_normal) &
                         npt.isfinite(surface_normal))
        if npt.any(not_on_sphere):
            print('Warning: False intersections due to numerical error!')
            print(f'  Clipping {not_on_sphere.sum()} rays.')
            xyz_f[not_on_sphere] = float('nan')
            k_xyz_f[not_on_sphere] = float('nan')

        return RayBundle(xyz_f, k_xyz_f, wavelength_um=ray_bundle.wavelength_um,
                         enforce_normalized_k=False)

class PlanarSurface:
    """Arguably the second most useful surface shape in optics.
    """
    def __init__(
        self,
        x, y, z,
        kx, ky, kz,
        allow_backwards=False,
        ):
        """We define a plane by xyz (a point in the plane) and k_xyz (a
        unit vector normal to the plane).
        """
        x, y, z, kx, ky, kz = (npt.as_tensor(a).reshape(1)
                               for a in (x, y, z, kx, ky, kz))
        self.xyz = npt.stack((x, y, z))
        self.k_xyz = make_unit_vectors(npt.stack((kx, ky, kz)))
        assert allow_backwards in (True, False)
        self.allow_backwards = allow_backwards
        return None

    def trace(self, ray_bundle, ni, nf):
        """Traces one ray bundle.
        """
        # Find line-plane intersection via formulae from
        # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

        # Ray direction vector should already be normalized:
        assert all_unit_vectors(ray_bundle.k_xyz)

        numerator = dot((self.xyz - ray_bundle.xyz), self.k_xyz)
        denominator = dot(ray_bundle.k_xyz, self.k_xyz)
        with np.errstate(divide='ignore'): # denominator=0 means no intersection
            t = numerator / denominator
        
        if not self.allow_backwards:
            t[is_negative(t)] = float('nan')
        xyz_f = ray_bundle.xyz + t*ray_bundle.k_xyz

        surface_normal = self.k_xyz
        k_xyz_f = deflection(ray_bundle, surface_normal, ni, nf)

        # Did we land in the plane, up to floating-point accuracy?
        not_in_plane = (~is_orthogonal(xyz_f-self.xyz, self.k_xyz) &
                        npt.isfinite(xyz_f))

        if npt.any(not_in_plane):
            print('Warning: False intersections due to numerical error!')
            print(f'  Clipping {not_in_plane.sum()} rays.')
            xyz_f[not_in_plane] = float('nan')
            k_xyz_f[not_in_plane] = float('nan')

        return RayBundle(xyz_f, k_xyz_f, wavelength_um=ray_bundle.wavelength_um,
                         enforce_normalized_k=False)

class CylindricalSurface:
    """Arguably the third most useful surface shape in optics.
    """
    def __init__(
        self,
        x, y, z,
        kx, ky, kz,
        r, # r>0 is convex, r<0 is concave
        allow_backwards=False,
        ):
        """We define a cylinder by xyz (a point on its axis), k_xyz (a
        unit vector along its axis) and a radius r.
        """
        x, y, z, kx, ky, kz, r = (npt.as_tensor(a).reshape(1)
                                  for a in (x, y, z, kx, ky, kz, r))
        self.xyz = npt.stack((x, y, z))
        self.k_xyz = make_unit_vectors(npt.stack((kx, ky, kz)))
        self.r = r
        assert allow_backwards in (True, False)
        self.allow_backwards = allow_backwards
        return None

    def trace(self, ray_bundle, ni, nf, allow_backwards=False):
        """Traces one ray bundle.
        """
        # Ray direction vector should already be normalized:
        assert all_unit_vectors(ray_bundle.k_xyz)

        # Find line-sphere intersection via these formulae:
        # https://doi.org/10.1007/978-1-4842-4427-2_7 p92
        r, f, d = self.r, (ray_bundle.xyz - self.xyz), ray_bundle.k_xyz
        # Since we're dealing with a cylinder and not a sphere, subtract
        # off the components of 'f' and 'd' that are parallel to the
        # cylindrical axis:
        f = perpendicular_component(f, direction=self.k_xyz)
        d = perpendicular_component(d, direction=self.k_xyz)

        # Note that a!=1, so we can't ignore it like we did for the sphere:
        a = sq_mag(d)
        b_prime = dot(-f, d)
        c = sq_mag(f) - r**2
        with np.errstate(divide='ignore'): # a=0 implies no intersection
            one_over_a = 1/a
        Delta = r**2 - sq_mag(f + (b_prime * one_over_a)*d)
        with np.errstate(invalid='ignore'): # Delta<0 implies no intersection
            sqrt_a_Delta = npt.sqrt(a*Delta)

        q = b_prime + npt.copysign(sqrt_a_Delta, b_prime)
        t1 = q*one_over_a
        with np.errstate(divide='ignore'): # q=0 implies no second intersection
            t0 = c/q
        
        # Pick the appropriate intersection based on the cylinder's concavity:
        concavity = npt.sign(r)
        t = npt.minimum(concavity*t0, concavity*t1) / concavity

        if not self.allow_backwards:
            t[is_negative(t)] = float('nan')
        xyz_f = ray_bundle.xyz + t*ray_bundle.k_xyz

        surface_normal = perpendicular_component(self.xyz-xyz_f, self.k_xyz) / r
        k_xyz_f = deflection(ray_bundle, surface_normal, ni, nf)

        # Make sure we landed on the cylinder
        not_on_cylinder = (~is_unit_vector(surface_normal) &
                           npt.isfinite(surface_normal))
        if npt.any(not_on_cylinder):
            print('Warning: False intersections due to numerical error!')
            print(f'  Clipping {not_on_cylinder.sum()} rays.')
            xyz_f[not_on_cylinder] = float('nan')
            k_xyz_f[not_on_cylinder] = float('nan')

        return RayBundle(xyz_f, k_xyz_f, wavelength_um=ray_bundle.wavelength_um,
                         enforce_normalized_k=False)

class PlanarRelay(PlanarSurface):
    """A pair of image planes surrounding a PlanarSurface. The first
    image plane is imaged perfectly onto the second.

    Unlike the previous surfaces, this surface is defined by its
    (idealized) behavior, rather than simply its geometry.
    """
    def __init__(
        self,
        x, y, z,
        kx, ky, kz,
        input_distance=0,
        output_distance=0,
        output_index=1,
        magnification=None,
        allow_backwards=False,
        ):
        self.input_distance  = npt.as_tensor(input_distance ).reshape(1)
        self.output_distance = npt.as_tensor(output_distance).reshape(1)
        self.output_index = output_index
        if magnification is None:
            self.magnification = None
        else:
            self.magnification = npt.as_tensor(magnification).reshape(1)
        super().__init__(x, y, z, kx, ky, kz, allow_backwards)

    def trace(self, ray_bundle, ni, nf):
        """Traces one ray bundle.
        """
        # Trace back to the 'input image plane' of the optic:
        input_image_plane = PlanarSurface(
            *(self.xyz + self.k_xyz*self.input_distance),
            *self.k_xyz,
            allow_backwards=True)
        ray_bundle = input_image_plane.trace(ray_bundle, ni, ni)

        # Optionally, magnify the image:
        if self.magnification is not None:
            # xyz magnification is easy: scale by M, centered on
            # input_image_plane.xyz:
            M = self.magnification
            ray_bundle.xyz = ray_bundle.xyz*M + input_image_plane.xyz*(1-M)
            # k_xyz magnification is equivalent to scaling the
            # refractive index by M:
            ray_bundle.k_xyz = deflection(
                ray_bundle, self.k_xyz, ni, M*self.output_index)

        # Shift to the 'output image plane' of the optic:
        ray_bundle.xyz = ray_bundle.xyz + self.k_xyz * (self.output_distance -
                                                        self.input_distance)

        # Trace back to the output surface:
        output_surface = PlanarSurface(
            *self.xyz,
            *self.k_xyz,
            allow_backwards=True)
        return output_surface.trace(ray_bundle, self.output_index, nf)


def deflection(ray_bundle, surface_normal, ni, nf, reflect=False):
    """Given an array of incident k-vectors and surface normal vectors,
    return the refracted (or reflected) k-vector, using Snell's Law as
    described in Chapter 6 of http://hdl.handle.net/10754/674879
    """
    k_in, wavelength_um = ray_bundle.k_xyz, ray_bundle.wavelength_um
    assert all_unit_vectors(k_in)
    assert all_unit_vectors(surface_normal)
    if reflect: # Mathematically conceivable, but physically impossible
        assert ni == nf
    elif ni == nf: # No refraction or reflection, leave the rays undeviated
        return npt.clone(k_in)
    ni, nf = index2float(ni, wavelength_um), index2float(nf, wavelength_um)

    cos_phi = dot(k_in, surface_normal)
    assert not npt.any(is_negative(cos_phi)), 'Surface normal is backwards!'
    eta = ni / nf # Ratio of indices of refraction
    with np.errstate(invalid='ignore'): # Transmitted direction unit vector
        k_normal     = surface_normal * npt.sqrt(1 - (1 - cos_phi**2)*eta**2)
        k_transverse = eta * (k_in - surface_normal * cos_phi)
        if reflect:
            k_out = -k_normal + k_transverse
        else:
            k_out =  k_normal + k_transverse
    assert all_unit_vectors(k_out) # Probably overkill, delete later?
    return k_out

# Simple utility functions for arrays of vectors:

def dot(a, b): # The dot products of two arrays of vectors
    return npt.sum(a*b, axis=0)

def sq_mag(a): # The squared magnitudes of an array of vectors
    return dot(a, a)

def mag(a): # The magnitudes of an array of vectors
    return npt.sqrt(sq_mag(a))

def parallel_component(a, direction):
    assert all_unit_vectors(direction)
    return dot(a, direction) * direction

def perpendicular_component(a, direction):
    return a - parallel_component(a, direction)

def is_orthogonal(a, b):
    with np.errstate(invalid='ignore'):
        return npt.abs(dot(a, b)) < 100*npt.finfo(dtype).resolution

def is_negative(a):
    with np.errstate(invalid='ignore'):
        return (a < 0) & npt.isfinite(a)

def is_unit_vector(x): # Where are the magnitudes of our vectors ~1?
    with np.errstate(invalid='ignore'):
        return npt.abs((mag(x) - 1)) < 100*npt.finfo(dtype).resolution

def all_unit_vectors(x, ignore_nan=True):
    is_unit = is_unit_vector(x)
    if ignore_nan:
        is_unit = is_unit[~npt.isnan(x.sum(axis=0))]
    return npt.all(is_unit)

def make_unit_vectors(x):
    return x / mag(x)

def make_unit_vectors_inplace(x):
    x /= mag(x)
    return None

def rotation_matrix(theta, rotation_axis):
    assert rotation_axis in ('x', 'y', 'z')
    theta, zero, one = (npt.as_tensor(x).reshape(()) for x in (theta, 0, 1))
    cos, sin = npt.cos(theta), npt.sin(theta)
    if rotation_axis == 'x':
        m = (( one, zero, zero),
             (zero,  cos, -sin),
             (zero,  sin,  cos))
    elif rotation_axis == 'y':
        m = (( cos, zero,  sin),
             (zero,  one, zero),
             (-sin, zero,  cos))
    elif rotation_axis == 'z':
        m = (( cos, -sin, zero),
             ( sin,  cos, zero),
             (zero, zero,  one))
    return npt.stack(tuple(npt.stack(m[i]) for i in range(3)))
        
# Short nicknames:
def rot_x(theta):
    return rotation_matrix(theta, 'x')

def rot_y(theta):
    return rotation_matrix(theta, 'y')

def rot_z(theta):
    return rotation_matrix(theta, 'z')


# RayBundle creation convenience functions:

def conical_bundle(
    x =0,  y=0,  z=0,
    kx=0, ky=0, kz=1,
    cone_angle=np.pi/32,
    num_cones=7, points_per_cone=51,
    wavelength_um=0.488,
    inner_cone_angle=0,
    ):
    """A pointlike emitter at (x, y, z), emitting in the (kx, ky, kz) direction.

    Note that 'cone_angle' is the full angle of the cone, not the half-angle.
    """
    num_rays = num_cones * points_per_cone
    theta = np.linspace(inner_cone_angle/2, cone_angle/2, num_cones)
    phi   = np.linspace(0,      2*np.pi, points_per_cone)
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)

    k_xyz = np.empty((3, num_cones, points_per_cone), dtype='float64')
    np.outer(sin_th, cos_ph,                   out=k_xyz[0, :])
    np.outer(sin_th, sin_ph,                   out=k_xyz[1, :])
    np.outer(cos_th, np.ones(points_per_cone), out=k_xyz[2, :])
    k_xyz = k_xyz.reshape(3, num_rays)
    k_xyz = _new_z_axis(k_xyz, kx, ky, kz)

    xyz = np.broadcast_to(np.array((x, y, z)).reshape(3, 1), (3, num_rays))
    
    return RayBundle(xyz, k_xyz, wavelength_um, enforce_normalized_k=False)

def random_conical_bundle(
    x =0,  y=0,  z=0,
    kx=0, ky=0, kz=1,
    cone_angle=np.pi/32,
    num_rays=100,
    wavelength_um=0.488,
    ):
    """A pointlike emitter at (x, y, z), emitting in the (kx, ky, kz) direction.

    Note that 'cone_angle' is the full angle of the cone, not the half-angle.
    """
    theta = np.arccos(np.random.uniform(np.cos(cone_angle/2), 1, num_rays))
    phi   =           np.random.uniform(0,              2*np.pi, num_rays)

    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)

    k_xyz = np.empty((3, num_rays), dtype='float64')
    k_xyz[0, :] = sin_th * cos_ph
    k_xyz[1, :] = sin_th * sin_ph
    k_xyz[2, :] = cos_th
    k_xyz = _new_z_axis(k_xyz, kx, ky, kz)
    
    xyz = np.broadcast_to(np.array((x, y, z)).reshape(3, 1), (3, num_rays))

    return RayBundle(xyz, k_xyz, wavelength_um, enforce_normalized_k=False)

def collimated_bundle(
    x =0,  y=0,  z=0,
    kx=0, ky=0, kz=1,
    radius=1,
    num_rings=7, points_per_ring=51,
    wavelength_um=0.488,
    ):
    num_rays = num_rings * points_per_ring
    r   = np.linspace(0,  radius,       num_rings)
    phi = np.linspace(0, 2*np.pi, points_per_ring)

    xyz = np.empty((3, num_rings, points_per_ring), dtype='float64')
    xyz[0, :] = np.outer(r, np.cos(phi))
    xyz[1, :] = np.outer(r, np.sin(phi))
    xyz[2, :] = 0
    xyz = xyz.reshape(3, num_rays)
    xyz = _new_z_axis(xyz, kx, ky, kz)
    xyz += np.array((x, y, z)).reshape((3, 1))
   
    k_xyz = np.array((kx, ky, kz), dtype='float64').reshape(3, 1)
    k_xyz /= np.linalg.norm(k_xyz, axis=0)
    k_xyz = np.broadcast_to(k_xyz, (3, num_rays))

    return RayBundle(xyz, k_xyz, wavelength_um, enforce_normalized_k=False)

def random_collimated_bundle(
    x =0,  y=0,  z=0,
    kx=0, ky=0, kz=1,
    radius=1,
    num_rays=100,
    wavelength_um=0.488,
    ):
    r = np.sqrt(np.random.uniform(0, radius**2, num_rays))
    phi =       np.random.uniform(0,   2*np.pi, num_rays)

    xyz = np.empty((3, num_rays), dtype='float64')
    xyz[0, :] =  r*np.cos(phi)
    xyz[1, :] =  r*np.sin(phi)
    xyz[2, :] =  0
    xyz = _new_z_axis(xyz, kx, ky, kz)
    xyz += np.array((x, y, z)).reshape((3, 1))

    k_xyz = np.array((kx, ky, kz), dtype='float64').reshape(3, 1)
    k_xyz /= np.linalg.norm(k_xyz, axis=0)
    k_xyz = np.broadcast_to(k_xyz, (3, num_rays))

    return RayBundle(xyz, k_xyz, wavelength_um, enforce_normalized_k=False)

def _new_z_axis(xyz, kx, ky, kz):
    """Rotate the vectors xyz so their old z-axis now points along (kx, ky, kz)
    """
    mag = np.sqrt(kx**2 + ky**2 + kz**2)
    kx, ky, kz = kx/mag, ky/mag, kz/mag
    theta = np.arccos(kz)
    phi = np.arctan2(ky, kx)
    return npt.asarray(rot_z(phi) @ rot_y(theta)) @ xyz

# Convenience functions for creating materials with wavelength-dependend
# index of refraction. We either specify index of refraction as a simple
# float (which ignores wavelength), or as a callable function that takes
# a wavelength in microns and returns a floating-point index of
# refraction. We include a few example materials based on Sellmeier
# coefficients.

def index2float(n, wavelength_um):
    return n(wavelength_um) if callable(n) else float(n)
    
class SellmeierIndex:
    def __init__(self, b1, c1, b2, c2, b3, c3):
        self.b1, self.b2, self.b3 = b1, b2, b3
        self.c1, self.c2, self.c3 = c1, c2, c3
        
    def __call__(self, wavelength_um):
        x = wavelength_um**2 # Short nicknames
        b1, b2, b3 = self.b1, self.b2, self.b3
        c1, c2, c3 = self.c1, self.c2, self.c3
        return npt.sqrt(
            npt.as_tensor(1 + b1*x/(x-c1) + b2*x/(x-c2) + b3*x/(x-c3)))

N_BK7 = SellmeierIndex( #SCHOTT N-BK7® 517642.251 
    b1=1.03961212,
    b2=0.231792344,
    b3=1.010469450,
    c1=0.00600069867,
    c2=0.0200179144,
    c3=103.5606530)

N_SF5 = SellmeierIndex( #SCHOTT N-SF5 673323.286
    b1=1.52481889,
    b2=0.187085527,
    b3=1.427290150,
    c1=0.01125475600,
    c2=0.0588995392,
    c3=129.1416750)

SF2 = SellmeierIndex( #SCHOTT SF2 648339.386
    b1=1.40301821,
    b2=0.231767504,
    b3=0.939056586,
    c1=0.01057954660,
    c2=0.0493226978,
    c3=112.4059550)

LAFN7 = SellmeierIndex( #SCHOTT LAFN7 750350.438
    b1=1.66842615,
    b2=0.298512803,
    b3=1.077437600,
    c1=0.01031599990,
    c2=0.0469216348,
    c3=82.5078509)

# Plotting convenience functions:

def create_3d_ax():
    fig = plt.figure()
    ax = fig.add_axes((.1, .1, .8, .8), projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def format_3d_ray_plot(ax):
    ax.set_ylim(-25, 25)
    ax.set_xlim(-25, 25)
    ax.set_zlim(-25, 25)
    ax.grid('on')
    
def set_aspect_equal_3d(ax):
    """ 
    Fix the 3D graph to have similar scale on all the axes.
    Call this after you do all the plot3D, but before show
    """
    X = ax.get_xlim3d()
    Y = ax.get_ylim3d()
    Z = ax.get_zlim3d()
    a = [X[1]-X[0],Y[1]-Y[0],Z[1]-Z[0]]
    b = np.amax(a)
    ax.set_xlim3d(X[0]-(b-a[0])/2,X[1]+(b-a[0])/2)
    ax.set_ylim3d(Y[0]-(b-a[1])/2,Y[1]+(b-a[1])/2)
    ax.set_zlim3d(Z[0]-(b-a[2])/2,Z[1]+(b-a[2])/2)
    ax.set_box_aspect(aspect = (1,1,1))

def plot_ray_bundle_sequence_3d(ray_bundle_sequence, ax, color='C0'):
    for bundle_i, bundle_f in zip(ray_bundle_sequence.seq[:-1],
                                  ray_bundle_sequence.seq[1:]):
        xi, yi, zi = bundle_i.xyz
        xf, yf, zf = bundle_f.xyz
        for i in range(len(xi)):
            ax.plot(
                xs=[xi[i], xf[i]],
                ys=[yi[i], yf[i]],
                zs=[zi[i], zf[i]],
                color=color,
                alpha=.3
            )
    xi, yi, zi = bundle_f.xyz
    xf, yf, zf = bundle_f.xyz + bundle_f.k_xyz
    for i in range(len(xi)):
        ax.plot(
            xs=[xi[i], xf[i]],
            ys=[yi[i], yf[i]],
            zs=[zi[i], zf[i]],
            color=color,
            alpha=.3
        )

def plot_surface_3d(surface, ax, surface_diameter):
    bundle1 = collimated_bundle(radius=surface_diameter/2)
    bundle2 = surface.trace(bundle1, 1, 1)
    valid = np.all(~np.isnan(bundle2.xyz), axis=0)
    ax.plot_trisurf(*bundle2.xyz[:, valid], color='k', alpha=.3)

def plot_all_surfaces_3d(optical_system, ax, surface_diameter=10):
    for surface in optical_system.surface_list:
        plot_surface_3d(surface, ax, surface_diameter=surface_diameter)


def plot_ray_diagram_3d(opt_sys, ray_bundle_sequences, ax=None, surface_diameter=10):
    if ax is None:
        ax = create_3d_ax()
    plot_all_surfaces_3d(opt_sys, ax, surface_diameter)
    for idx, rbs in enumerate(ray_bundle_sequences):
        plot_ray_bundle_sequence_3d(rbs, ax, color=f'C{idx%8}')
    set_aspect_equal_3d(ax)
    # format_3d_ray_plot(ax)

def plot_ray_diagram_2d(opt_sys, ray_bundle_sequence_list, ax, surface_diameter=10):
    for i, bundle in enumerate(ray_bundle_sequence_list):
        plot_ray_bundle_sequence_2d(bundle, ax, color=f'C{i}')
    plot_surfaces_2d(opt_sys, ax, surface_diameter=surface_diameter)
    # Set Aspect ratio equal
    ax.grid()
    x_range = np.abs(np.subtract(*ax.get_xlim()))
    ax.set_ylim(-x_range/2, x_range/2)
    ax.set_ylabel('X-dim')
    ax.set_xlabel('Z-dim')
    ax.set_aspect('equal')

def plot_ray_bundle_sequence_2d(ray_bundle_sequence, ax, color='C0'):
    # Plot each ray n the bundle
    for bundle_i, bundle_f in zip(ray_bundle_sequence.seq[:-1],
                                  ray_bundle_sequence.seq[1:]):

        xi, yi, zi = npt.asarray(bundle_i.xyz)
        xf, yf, zf = npt.asarray(bundle_f.xyz)

        for i in range(len(xi)):
            if (abs(yi[i]) < .1) and (abs(yf[i]) < .1):
                ax.plot(
                    [zi[i], zf[i]],
                    [xi[i], xf[i]],
                    color=color,
                    alpha=.3
                )
    # # Extend each ray by 1 from final surface
    # xi, yi, zi = npt.asarray(bundle_f.xyz)
    # xf, yf, zf = npt.asarray(bundle_f.xyz) + npt.asarray(bundle_f.k_xyz)
    # for i in range(len(xi)):
    #     if (abs(yi[i]) < .1) and (abs(yf[i]) < .1):
    #         ax.plot(
    #             [zi[i], zf[i]],
    #             [xi[i], xf[i]],
    #             color=color,
    #             alpha=.3
    #         )

def plot_surfaces_2d(optical_system, ax, surface_diameter=25.4):
    for surface in optical_system.surface_list:
        plot_surface_2d(surface, ax, surface_diameter=surface_diameter)

def plot_surface_2d(surface, ax, surface_diameter=25.4):
    bundle1 = collimated_bundle(z=-1, radius=surface_diameter/2)
    bundle2 = surface.trace(bundle1, 1, 1)
    xyz = npt.asarray(bundle2.xyz)
    valid = np.all(~np.isnan(xyz), axis=0)
    x, y, z = xyz[:, valid]
    _x = np.array([x[i] for i in range(len(x)) if abs(y[i])<.000001])
    _z = np.array([z[i] for i in range(len(z)) if abs(y[i])<.000001])
    idxs = np.argsort(_x)
    _x = _x[idxs]
    _z = _z[idxs]
    ax.plot(_z, _x, color='k', alpha=1)


def plot_spot_diagram_2d(ray_bundle_sequences, ax, n=-1, bundle_idxs=None):
    if bundle_idxs is None:
        bundle_idxs = slice(None, None)
    idxs = range(len(ray_bundle_sequences))

    # Create new axes to plot onto
    axes = []
    fig = plt.gcf()
    ax.axis("off")
    left, bottom, width, height = ax.get_position().bounds

    nrows = ncols = int(np.ceil(np.sqrt(len(idxs[bundle_idxs]))))

    ax_width = width/ncols
    ax_height = height/nrows
    for i in idxs:
        ax_left = left + i%ncols*ax_width
        ax_bottom = bottom + i//ncols*ax_height
        axes.append(fig.add_axes((ax_left, ax_bottom, ax_width*.7, ax_height*.7)))


    # Plot
    ranges = []
    for i, (bundle, idx) in enumerate(zip(ray_bundle_sequences[bundle_idxs], idxs[bundle_idxs])):
        row, col = i//ncols, i%ncols
        ax = axes[i]
        x, y, z = npt.asarray(bundle.seq[n].xyz) 
        ax.plot(y, x, linestyle='none', marker='.',
                alpha=.3, color=f'C{idx%8}', markeredgecolor='none')
        if hasattr(bundle.initial_ray_bundle, 'target_xyz'):
            ax.plot(
                npt.asarray(bundle.initial_ray_bundle.target_xyz[1, 0]),
                npt.asarray(bundle.initial_ray_bundle.target_xyz[0, 0]),
                marker='x', color=f'C{idx%8}'
            )
        # Formatting 
        ax.grid()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ranges.append(np.abs(np.subtract(xmin, xmax)))
        ranges.append(np.abs(np.subtract(ymin, ymax)))
        if row == 0:
            ax.set_xlabel('Y-dim')
        if col == 0:
            ax.set_ylabel('X-dim')
        ax.yaxis.tick_right()
        ax.xaxis.tick_top()
    ax_range = max(ranges)
    for ax in axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_cen = np.mean((xmin, xmax))
        y_cen = np.mean((ymin, ymax))
        ax.set_xlim(x_cen-ax_range/2, x_cen+ax_range/2)
        ax.set_ylim(y_cen-ax_range/2, y_cen+ax_range/2)


    return axes


def test(plot=True):
    index_list = [
        1,
        N_SF5,
        N_BK7,
        1.3,
        1
    ]

    r0 = 77.4
    f0 = 10    
    x0, y0, z0 = 0, 0, f0+r0

    t0 = 16
    r1 = -87.6
    x1, y1, z1 = 0, 0, z0-r0+t0+r1

    
    t1 = 30

    r2 = -73.2
    x2, y2, z2 = 0, 0, z1-r1+t1+r2

    fb = 20
    x3, y3, z3 = 0, 0, z2-r2+fb

    s0 = SphericalSurface(x0, y0, z0, r0)
    s1 = SphericalSurface(x1, y1, z1, r1)
    s2 = CylindricalSurface(x2, y2, z2, .8, .05, .2, r=r2)
    s3 = PlanarSurface(x3, y3, z3, 0, 0, 1)

    surface_list = [s0, s1, s2, s3]

    o = OpticalSystem(index_list, surface_list)


    initial_ray_bundle_list = [
        random_conical_bundle(num_rays=100, cone_angle=np.pi/8, kx=.1),
        conical_bundle(cone_angle=np.pi/8, kx=.1),
        collimated_bundle(kx=.3),
        random_collimated_bundle(kx=.3, num_rays=100)
    ]

    ray_bundle_sequences = o.trace_all_sequences(initial_ray_bundle_list)

    if plot:
        plot_ray_diagram_3d(o, ray_bundle_sequences, surface_diameter=60)
        plt.show()

        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
        plot_ray_diagram_2d(o, ray_bundle_sequences, axes[0], surface_diameter=60)
        plot_spot_diagram_2d(ray_bundle_sequences, axes[1], n=-1, bundle_idxs=slice(None, None))
        plt.show()


if __name__ == '__main__':
    import torch
    dtype = torch.float32
    test(False)

    import torch
    dtype = torch.float64
    test(False)

    torch = None
    dtype = np.float32

    test(False)

    torch = None
    dtype = np.float64
    test(True)
            


    
