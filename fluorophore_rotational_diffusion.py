#!/usr/bin/python
import time
import numpy as np
from numpy.random import uniform, exponential, normal


class Fluorophores:
    """
    An example usage, showing photoactivation and photoexcitation:

    state_info = FluorophoreStates()
    state_info.add('inactive')
    state_info.add('active')
    state_info.add('excited', lifetime=2, final_states='active')

    f = Fluorophores(
        number_of_molecules=10,
        diffusion_time=20,
        state_info=state_info)
    f.phototransition('inactive', 'active',
                      intensity=10, polarization_xyz=(0, 0, 1))
    f.phototransition('active', 'excited',
                      intensity=10, polarization_xyz=(0, 0, 1))
    f.time_evolve(5)
    """
    def __init__(
        self,
        number_of_molecules,
        diffusion_time,
        initial_orientations='uniform',
        state_info=None,
        initial_state=0,
        ):        
        self.orientations = Orientations(number_of_molecules,
                                         diffusion_time,
                                         initial_orientations)
        if state_info is None: # Default to the simplest photophysics
            state_info = FluorophoreStates()
            state_info.add('ground')
            state_info.add('excited', lifetime=1, final_states='ground')
        assert isinstance(state_info, FluorophoreStates)
        self.state_info = state_info

        assert initial_state in state_info
        self.states = np.full(self.orientations.n, initial_state, dtype='uint')
        self.transition_times = exponential(
            self.state_info[initial_state].lifetime, self.orientations.n)

        self.id = np.arange(self.orientations.n, dtype='int')
        return None

    def phototransition(
        self,
        initial_state, # Integer or string
        final_states,  # Integer/string or iterable of integers/strings
        state_probabilities=None, # None, or arraylike of floats
        intensity=1,                # Saturation units
        polarization_xyz=(0, 0, 1), # Only the direction matters
        ):
        # Input sanitization
        assert initial_state in self.state_info
        initial_state = self.state_info[initial_state].n # Ensure int
        final_states, lifetimes = self.state_info.n_and_lifetime(final_states)
        if final_states.shape == (1,):
            assert state_probabilities is None
        else:
            state_probabilities = np.asarray(state_probabilities, 'float')
            assert state_probabilities.shape == final_states.shape
            assert np.all(state_probabilities > 0)
            state_probabilities /= state_probabilities.sum() # Sums to 1
        assert intensity > 0
        polarization_xyz = np.asarray(polarization_xyz, dtype='float')
        assert polarization_xyz.shape == (3,)
        polarization_xyz /= np.linalg.norm(polarization_xyz) # Unit vector
        # A linearly polarized pulse of light, oriented in an arbitrary
        # direction, drives molecules to change their state. The
        # 'effective intensity' for each molecule varies like the square
        # of the cosine of the angle between the light's polarization
        # direction and the molecular orientation.
        i = (self.states == initial_state) # Who's in the initial state?
        px, py, pz, = np.sqrt(intensity) * polarization_xyz
        o = self.orientations # Temporary short nickname
        effective_intensity = (px*o.x[i] + py*o.y[i] + pz*o.z[i])**2 # Dot prod.
        selection_prob = 1 - 2**(-effective_intensity) # Saturation units
        selected = uniform(0, 1, len(selection_prob)) <= selection_prob
        # Every photoselected molecule now changes to a new state. If
        # multiple 'final_states' are specified, the new state is
        # randomly selected according to 'state_probabilities'. New
        # 'transition_times' are randomly drawn for each new state from
        # an exponential distribution given by 'lifetimes'.
        t = o.t[i][selected] # The current time
        tr_t = self.transition_times[i] # A copy of relevant transition times
        if state_probabilities is None:
            self.states[i] = np.where(selected, final_states, self.states[i])
            tr_t[selected] = t + exponential(lifetimes, t.shape)
        else:
            which_state = np.random.choice(
                np.arange(len(final_states), dtype='int'),
                size=t.shape, p=state_probabilities)
            ss = self.states[i] # A copy of the relevant states
            ss[selected] = final_states[which_state]
            self.states[i] = ss
            tr_t[selected] = t + exponential(lifetimes[which_state])
        self.transition_times[i] = tr_t
        return None

    def time_evolve(self, delta_t):
        assert delta_t > 0
        o = self.orientations # Local nickname
        assert np.isclose(o.t.min(), o.t.max()) # Orientations are synchronized
        target_time = o.t[0] + delta_t
        while np.any(o.t < target_time):
            # How much shall we step each molecule in time?
            dt = np.minimum(target_time, self.transition_times) - o.t
            idx = self._sort_by(dt)
            dt = dt if idx is None else dt[idx] # Skip if dt is already sorted
            s = slice(np.searchsorted(dt, 0, 'right'), None) # Skip dt == 0
            # Update the orientations
            o.x[s], o.y[s], o.z[s] = safe_diffusive_step(
                o.x[s], o.y[s], o.z[s], (dt/o.diffusion_time)[s])
            o.t[s] += dt[s]
            # Calculate spontaneous transitions
            transitioning = (o.t >= self.transition_times)
            states = self.states[transitioning] # Copy of states that change
            idx = np.argsort(states)
            states = states[idx] # A sorted copy of the states that change
            transition_times = np.empty(len(states), dtype='float')
            t = o.t[transitioning][idx]
            for initial_state in range(len(self.state_info)):
                s = slice(np.searchsorted(states, initial_state, 'left'),
                          np.searchsorted(states, initial_state, 'right'))
                if s.start == s.stop: # No population in this initial state
                    continue
                fs = self.state_info[initial_state].final_states
                final_states, lifetimes = self.state_info.n_and_lifetime(fs)
                probabilities = self.state_info[initial_state].probabilities
                which_final = np.random.choice(
                    np.arange(len(final_states), dtype='int'),
                    size=(s.stop-s.start), p=probabilities)
                states[s] = final_states[which_final]
                transition_times[s] = t[s] + exponential(lifetimes[which_final])
            # Undo our sorting of states and transition times, update originals
            idx_rev = np.empty_like(idx)
            idx_rev[idx] = np.arange(len(idx), dtype=idx.dtype)
            self.states[          transitioning] = states[          idx_rev]
            self.transition_times[transitioning] = transition_times[idx_rev]
        return None

    def _sort_by(self, x):
        x_is_sorted = np.all(np.diff(x) >= 0)
        if x_is_sorted:
            return None
        idx = np.argsort(x)
        self.states = self.states[idx]
        self.transition_times = self.transition_times[idx]
        o = self.orientations # Local nickname
        o.x, o.y, o.z, o.t = o.x[idx], o.y[idx], o.z[idx], o.t[idx]
        if o.diffusion_time.size == o.n:
            o.diffusion_time = o.diffusion_time[idx]
        self.id = self.id[idx]
        return idx

def _test_fluorophores_speed(n=int(1e6)):
    t = []
    for i in range(10):
        start = time.perf_counter()
        f = Fluorophores(number_of_molecules=n, diffusion_time=1)
        end = time.perf_counter()
        t.append(end-start)
    print('\n%0.1f nanoseconds per'%(1e9*np.mean(t) / f.orientations.n),
          "fluorophore created by Fluorophores()")

    f = Fluorophores(number_of_molecules=n, diffusion_time=1)
    start = time.perf_counter()
    f.phototransition('ground', 'excited')
    end = time.perf_counter()
    t = end - start
    print('%0.1f nanoseconds per'%(1e9*t / f.orientations.n),
          "fluorophore phototransitioning")
    
    dt = 5
    f = Fluorophores(number_of_molecules=n, diffusion_time=1)
    start = time.perf_counter()
    f.time_evolve(dt)
    end = time.perf_counter()
    t = end - start
    print('%0.1f nanoseconds per'%(1e9*t / f.orientations.n),
          "fluorophore time evolving for %0.1f diffusion times"%(
              dt/f.orientations.diffusion_time))
    return None

class FluorophoreStates:
    def __init__(self):
        self.clear()

    def clear(self):
        self.list = []
        self.dict = {}
        self.validate()

    def add(self, name, lifetime=np.inf, final_states=None, probabilities=None):
        state = FluorophoreState(name, lifetime, final_states, probabilities)
        assert name not in self.dict
        state.n = len(self.list)
        self.list.append(state)
        self.dict[state.name] = state
        self.validate()
        return None

    def validate(self):
        self.valid = True # Optimistic!
        self.num = {}
        for n, state in enumerate(self.list):
            self.num[state.name] = n
            for final_state in state.final_states:
                if not final_state in self.dict:
                    self.orphan_state = (state.name, final_state)
                    self.valid = False
        return None

    def n_and_lifetime(self, states):
        if isinstance(states, int) or isinstance(states, str):
            states = [states]
        n        = np.asarray([self[s].n        for s in states], 'int')
        lifetime = np.asarray([self[s].lifetime for s in states], 'float')
        return n, lifetime

    def __getitem__(self, x):
        if not self.valid:
            raise LookupError(
                "\nFluorophoreStates is not usable yet.\n" +
                "State '%s'"%(self.orphan_state[0]) +
                " has a final_state '%s'"%(self.orphan_state[1]) +
                ", which has not been set with .add().")
        if isinstance(x, str):
            return self.dict[x]
        elif isinstance(x, int):
            return self.list[x]
        else:
            raise TypeError("FluorophoreStates indices must be"+
                            " integers or strings")

    def __contains__(self, x):
        try:
            self[x]
            return True
        except (KeyError, IndexError):
            return False

    def __len__(self):
        return len(self.list)

class FluorophoreState:
    # This object exists to sanitize input and to hold a namespace
    def __init__(
        self,
        name,
        lifetime=np.inf,
        final_states=None,
        probabilities=None,
        ):
        # 'name' is a string:
        assert isinstance(name, str)
        self.name = name
        # 'lifetime' is a positive float:
        assert lifetime > 0
        self.lifetime = lifetime
        # 'final_states' is a single string, or a list/tuple of strings:
        if final_states is None:
            assert probabilities is None
            final_states = [name]
        if isinstance(final_states, str):
            final_states = [final_states]
        for final_state in final_states:
            assert isinstance(final_state, str)
        self.final_states = list(final_states)
        # 'probabilities' is a 1D array-like of positive floats that
        # sums to 1, with each entry corresponding to an entry in
        # 'final_states':
        if len(final_states) == 1 and probabilities is None:
            probabilities = [1]
        probabilities = np.asarray(probabilities, 'float64')
        assert probabilities.shape == (len(final_states),)
        assert np.all(probabilities > 0)
        probabilities /= probabilities.sum()
        self.probabilities = probabilities
        return None

class Orientations:
    def __init__(
        self,
        number_of_molecules,
        diffusion_time,
        initial_orientations='uniform',
        ):
        """A class to simulate the orientations of an ensemble of freely
        rotating molecules, effectively a randomn walk on a sphere.

        Time evolution consists of rotational diffusion of orientation.
        You get to choose the "diffusion_time" (roughly, how long it
        takes the molecules to scramble their orientations).
        """
        assert number_of_molecules >= 1
        self.n = int(number_of_molecules)
        self.t = np.zeros(self.n, 'float64')
        diffusion_time = np.asarray(diffusion_time)
        assert diffusion_time.shape in ((), (1,), (self.n,))
        assert np.all(diffusion_time > 0)
        self.diffusion_time = diffusion_time
        assert initial_orientations in ('uniform', 'polar')
        if initial_orientations == 'uniform':
            # Generate random points on a sphere:
            sin_ph, cos_ph = sin_cos(uniform(0, 2*np.pi, self.n), '0,2pi')
            cos_th = uniform(-1, 1, self.n)
            sin_th = np.sqrt(1 - cos_th*cos_th)
            self.x = sin_th * cos_ph
            self.y = sin_th * sin_ph
            self.z = cos_th
        elif initial_orientations == 'polar':
            # Everybody starts at the north pole:
            self.x = np.zeros(self.n)
            self.y = np.zeros(self.n)
            self.z = np.ones( self.n)
        return None

    def time_evolve(self, delta_t):
        delta_t = np.asarray(delta_t)
        assert delta_t.shape in ((), (1,), (self.n,))
        assert np.all(delta_t > 0)        
        self.x, self.y, self.z = safe_diffusive_step(
            x=self.x, y=self.y, z=self.z,
            normalized_time_step=delta_t/self.diffusion_time)
        self.t += delta_t
        return None

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
        t_is_sorted = np.all(np.diff(normalized_time_step) >= 0)
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
    # Finally, take our 'remainder' step:
    if remainder.max() > 0:
        x, y, z = diffusive_step(x, y, z, remainder)
    return x, y, z

def _test_safe_diffusive_step(n=int(1e5)):
    x, y, z = np.zeros(n), np.zeros(n), np.ones(n)
    print()
    for tstep in (np.array(5),
                  exponential(size=n, scale=5),
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
    phi_d = uniform(0, 2*np.pi, len(angle_step))
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
            uniform(will_draw_pi, 1, len(steps))))
        # To convert draws from our upper bound distribution to our desired
        # distribution, reject samples stochastically by the ratio of the
        # desired distribution to the upper bound distribution,
        # which is sqrt(sin(x)/x).
        rejected = (uniform(0, 1, candidates.shape) >
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
    result = step_sizes * np.sqrt(-np.log(uniform(0, 1, len(step_sizes))))
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
    theta_d =    np.abs(normal(  0,   np.pi, size=n)).astype(dtype)
    phi_d   =           uniform( 0, 2*np.pi, size=n ).astype(dtype)

    theta_i = np.arccos(uniform(-1,       1, size=n)).astype(dtype)
    phi_i   =           uniform( 0, 2*np.pi, size=n ).astype(dtype)
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
    theta = uniform(0,   np.pi, size=n).astype(dtype)
    phi   = uniform(0, 2*np.pi, size=n).astype(dtype)

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
    theta = uniform(0,   np.pi, size=n).astype(dtype)
    phi   = uniform(0, 2*np.pi, size=n).astype(dtype)
    x     = uniform(0, 8*np.pi, size=n).astype(dtype)

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

if __name__ == '__main__':
    print("Simulating simple photoactivation/excitation photophysics...")
    state_info = FluorophoreStates()
    state_info.add('inactive')
    state_info.add('active')
    state_info.add('excited', lifetime=2, final_states='active')

    f = Fluorophores(
        number_of_molecules=6,
        diffusion_time=20,
        initial_orientations='uniform',
        state_info=state_info,
        initial_state=0)

    def show(f):
        print('\nid \tstate \tt_transtion \tt\tx\ty\tz')
        for i in range(len(f.id)):
            print('%d\t%d\t%0.2f\t\t%0.2f\t%0.2f\t%0.2f\t%0.2f'%(
                f.id[i],
                f.states[i],
                f.transition_times[i],
                f.orientations.t[i],
                f.orientations.x[i],
                f.orientations.y[i],
                f.orientations.z[i]))
    show(f)
    print("\nActivating...")
    f.phototransition('inactive', 'active',
                      intensity=10, polarization_xyz=(0, 0, 1))
    show(f)
    print("\nExciting...")
    f.phototransition('active', 'excited',
                      intensity=10, polarization_xyz=(0, 0, 1))
    show(f)
    print("\nTime evolving...")
    f.time_evolve(1)
    show(f)

    print("\nTesting performance...")
    _test_sin_cos()
    _test_to_xyz()
    _test_polar_displacement()
    _test_propagators()
    _test_diffusive_step_speed()
    _test_safe_diffusive_step()
    _test_fluorophores_speed()
##    try:
##        _test_diffusive_step_accuracy()
##    except KeyboardInterrupt:
##        pass
