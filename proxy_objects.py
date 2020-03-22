# Multiprocessing to spread CPU load, threading for concurrency:
import multiprocessing as mp
import threading
# Printing from a child process is tricky:
import io
from contextlib import redirect_stdout
# Showing exceptions from a child process is tricky:
import sys
import traceback
# Making sure a child process closes when the parent exits is tricky:
import atexit
import signal
# Sharing memory between child processes is tricky:
import ctypes as C
try:
    import numpy as np
except ImportError:
    np = None

"""
Sometimes we put a computationally demanding Python object in a
multiprocessing child process, but this usually leads to high mental
overhead. Using pipes and queues for calling methods, getting/setting
attributes, printing, handling exceptions, and cleanup can lean to ugly
code and confusion. Can we isolate most of this mental overhead to this
module? If we do it right, we'll be able to write code like:

from camera_module import Camera
cam = ProxyObject(Camera) # Camera object instance lives in a child process...
cam.set_fps(10000)        # ...but acts like it's in the parent process!
cam.set_trigger('external')
x = cam.record(num_images=10000) # Fast, demanding, performance-critical.
cam.close()

...without paying much attention to the fact that the instance of the
Camera object actually lives in a child process, and communicates over
pipes.

Note that the method calls to our proxied object still block the parent
process; the idea is, the parent process is now effectively IO-limited
rather than CPU-limited, so we can write clean(er)-looking threading
code in the parent if we want multiple things to happen at once in the
parent.

Also note that multiprocessing already has proxied objects via
"managers"; we're rolling our own to learn, and have complete control.
If at the end of the process, we don't like ours better, we'll switch to
multiprocessing proxies.

CURRENT LIMITATIONS:

If you use this module, you have to protect the "entry point" of your
program, like this:

from camera_module import Camera
def main():
    cam = ProxyObject(Camera)
    cam.set_fps(10000)
    cam.set_trigger('external')
    x = cam.record(num_images=10000)
    cam.close()

if __name__ == '__main__':
    main()
"""

class ProxyManager:
    """Allocates shared memory and spawns proxy objects

    If you want your ProxyObjects to share memory with the parent (and
    each other), allocate a ProxyManager first, and use it to spawn your
    ProxyObjects.
    """
    def __init__(self, shared_memory_sizes=tuple()):
        if np is None:
            raise ImportError("We failed to import Numpy," +
                              " so there's no point using a ProxyManager.")
        # Allocate shared memory as multiprocessing Arrays:
        self.shared_mp_arrays = tuple(mp.Array(C.c_uint8, sz)
                                      for sz in shared_memory_sizes)

    def proxy_object(self, initializer, *initargs, **initkwargs):
        return ProxyObject(initializer, *initargs, **initkwargs,
                           shared_mp_arrays=self.shared_mp_arrays)

    def shared_numpy_array(self, which_mp_array, shape, dtype=np.uint8):
        assert which_mp_array in range(len(self.shared_mp_arrays))
        return _SharedNumpyArray(arrays=self.shared_mp_arrays, shape=shape,
                                 dtype=dtype, buffer=which_mp_array)

class ProxyObject:
    def __init__(
        self,
        initializer,
        *initargs,
        shared_mp_arrays=tuple(),
        **initkwargs,
        ):
        """Make an object in a child process, that acts like it isn't.

        initializer -- callable that returns an instance of a Python object.
        initargs, initkwargs --  Arguments to 'initializer'
        shared_mp_arrays -- Used by a ProxyManager to pass in shared memory.
        """
        # Put an instance of the Python object returned by 'initializer'
        # in a child process:
        initargs, initkwargs = _disconnect_shared_arrays(initargs, initkwargs)
        parent_pipe, child_pipe = mp.Pipe()
        child_process = mp.Process(target=_child_loop,
                                   name=initializer.__name__,
                                   args=(initializer, initargs, initkwargs,
                                         child_pipe, shared_mp_arrays))
        # Attribute-setting looks weird because we override __setattr__,
        # and because we use a dummy object's namespace to hold our
        # attributes so we shadow as little of the proxied object's
        # namespace as possible:
        super().__setattr__('_', _DummyClass()) # Weird, but for a reason.
        self._.parent_pipe = parent_pipe
        self._.child_pipe = child_pipe
        self._.child_process = child_process
        self._.shared_mp_arrays = shared_mp_arrays
        self._.fifo_lock = FIFOLock()
        # Make sure the child process initialized successfully:
        self._.child_process.start()
        assert _get_response(self) == 'Successfully initialized'
        # Try to ensure the child process closes when we exit:
        atexit.register(lambda: _close(self))
        signal.signal(signal.SIGTERM, lambda s, f: _close(self))

    def __getattr__(self, name):
        """Access attributes of the child-process object in the parent process.

        As much as possible, we want attribute access and method calls
        to *seem* like they're happening in the parent process, if
        possible, even though they actually involve asking the child
        process over a pipe.
        """
        self._.parent_pipe.send(('__getattribute__', (name,), {}))
        attr = _get_response(self)
        if callable(attr):
            def attr(*args, **kwargs):
                args, kwargs = _disconnect_shared_arrays(args, kwargs)
                self._.parent_pipe.send((name, args, kwargs))
                return _get_response(self)
        return attr

    def __setattr__(self, name, value):
        self._.parent_pipe.send(('__setattr__', (name, value), {}))
        return _get_response(self)

    def __enter__(self):
        return self._.fifo_lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._.fifo_lock.release()

def _get_response(proxy_object):
    """Effectively a method of ProxyObject, but defined externally to
    minimize shadowing of the proxied object's namespace"""
    resp, printed_output = proxy_object._.parent_pipe.recv()
    if len(printed_output) > 0:
        print(printed_output, end='')
    if isinstance(resp, Exception):
        raise resp
    if isinstance(resp, _SharedNumpyArrayStub):
        resp = resp._reconnect(proxy_object._.shared_mp_arrays)
    return resp

def _close(proxy_object):
    """Effectively a method of ProxyObject, but defined externally to
    minimize shadowing of the proxied object's namespace"""
    if not proxy_object._.child_process.is_alive():
        return
    proxy_object._.parent_pipe.send(None)
    proxy_object._.child_process.join()

def _child_loop(initializer, args, kwargs, child_pipe, shared_arrays):
    """The event loop of a ProxyObject's child process
    """
    # If any of the input arguments are _SharedNumpyArrays, we have to
    # show them where to find shared memory:
    args, kwargs = _reconnect_shared_arrays(args, kwargs, shared_arrays)
    # Initialization.
    printed_output = io.StringIO()
    try: # Create an instance of our object...
        with redirect_stdout(printed_output):
            obj = initializer(*args, **kwargs)
        child_pipe.send(('Successfully initialized', printed_output.getvalue()))
    except Exception as e: # If we fail to initialize, just give up.
        e.child_traceback_string = traceback.format_exc()
        child_pipe.send((e, printed_output.getvalue()))
        return None
    # Main loop:
    while True:
        printed_output = io.StringIO()
        try:
            cmd = child_pipe.recv()
        except EOFError: # This implies the parent is dead; exit.
            return None
        if cmd is None: # This is how the parent signals us to exit.
            return None
        attr_name, args, kwargs = cmd
        args, kwargs = _reconnect_shared_arrays(args, kwargs, shared_arrays)
        try:
            with redirect_stdout(printed_output):
                result = getattr(obj, attr_name)(*args, **kwargs)
            if callable(result):
                result = _dummy_function # Cheaper than sending a real callable
            if isinstance(result, _SharedNumpyArray):
                result = result._disconnect()
            child_pipe.send((result, printed_output.getvalue()))
        except Exception as e:
            e.child_traceback_string = traceback.format_exc()
            child_pipe.send((e, printed_output.getvalue()))

# If we're trying to return a (presumably worthless) "callable" to
# the parent, it might as well be small and simple:
def _dummy_function():
    return None

# A minimal class that we use just to get another namespace:
class _DummyClass:
    pass

class FIFOLock:
    """Like threading.Lock, but releases first-in, first-out.

    From the guy who made timsort! https://stackoverflow.com/a/19695878

    Suppose you have several shared resources that each proceed at their
    own speed (for example, camera, processing, gui, disk), and
    several threads executing a task like:

    def snap():
        raw_image = camera.record()
        image = processing(raw_image)
        gui.display(image)
        disk.save(raw_image)

    We want each thread to proceed at top speed, but we also want each
    thread to wait its turn before accessing a one-at-a-time shared
    resource:

    def snap():
        with camera_lock:
            raw_image = camera.record()
        with processing_lock:
            image = processing(raw_image)
        with display_lock:
            gui.display(image)
        disk.save(raw_image)

    Suppose the camera was much faster than processing, and we launch
    several snap() threads at once; we could easily end up with multiple
    snap() threads blocking on processing simultaneously. If our locks
    were instances of threading.Lock, image 3 will sometimes be
    processed and displayed before image 2. This is bad! If we use a
    FIFOLock instead, images will display in the same order that the
    camera records them. This is good.
    """
    def __init__(self):
        self.lock = threading.Lock() # A lock for your lock!
        self.waiters = [] # a collections.deque would be faster...
        self.count = 0

    def acquire(self):
        with self.lock:
            if self.count > 0: # Someone else has custody, get in line
                new_lock = threading.Lock()
                new_lock.acquire()
                self.waiters.append(new_lock)
                self.lock.release()
                new_lock.acquire() # Block here until another thread releases
                self.lock.acquire()
            self.count += 1
        return True

    def release(self):
        with self.lock:
            if not self.locked():
                raise RuntimeError("Can't release a lock that isn't locked")
            self.count -= 1
            if len(self.waiters) > 0:
                self.waiters.pop(0).release() # Call the next-in-line

    def locked(self):
        return self.count > 0

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.release()

class _SharedNumpyArray(np.ndarray):
    """A numpy array that lives in shared memory

    In general, don't create these directly, create _SharedNumpyArrays
    (and ProxyObjects) through a ProxyManager.

    Inputs and outputs to/from proxied objects are 'serialized', which
    is pretty fast - except for large in-memory objects. The only large
    in-memory objects we regularly deal with are numpy arrays, so it
    makes sense to provide a way to pass large numpy arrays to and from
    proxied objects via shared memory (which avoids slow serialization).
    
    There's a straightforward recipe to do this: declare an mp.Array in
    the parent process, pass it to the initializer of the child process,
    and view it as a numpy array with np.frombuffer in both parent and
    child. This works great, but breaks our ability to treat proxied
    objects as if they are in the parent process, and the resulting code
    can get pretty crazy looking.

    _SharedNumpyArray is a compromise: maybe you wanted to write code that
    looks like this:

        data_buf = np.zeros((400, 2000, 2000), dtype=np.uint16)
        display_buf = np.zeros((2000, 2000), dtype=np.uint8)

        camera = Camera()
        preprocessor = Preprocessor()
        display = Display()

        camera.record(num_images=400, out=data_buf)
        preprocessor.process(in=data_buf, out=display_buf)
        display.show(display_buf)

    ...but instead you write code that looks like this:
    
        pm = ProxyManager(shared_memory_sizes=(400*2000*2000*2, 2000*2000))
        data_buf = pm.shared_numpy_array(0, (400, 2000, 2000), dtype=np.uint16)
        display_buf = pm.shared_numpy_array(1, (2000, 2000), dtype=np.uint8)

        camera = pm.proxy_object(Camera)
        preprocessor = pm.proxy_object(Preprocessor)
        display = pm.proxy_object(Display)

        camera.record(num_images=400, out=data_buf)
        preprocessor.process(in=data_buf, out=display_buf)
        display.show(display_buf)

    ...and your payoff is, each object gets its own CPU core, AND passing
    large numpy arrays between the processes is still really fast!
    """
    def __new__(cls, arrays, shape=None, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        dtype = np.dtype(dtype)
        assert buffer in range(len(arrays)), f'Invalid buffer: <{buffer}>'
        requested_bytes = np.prod(shape) * dtype.itemsize
        if requested_bytes > len(arrays[buffer]):
            raise ValueError("Multiprocessing shared memory array is too "+
                             "small to hold the requested Numpy array.\n " +
                             f"{requested_bytes} > {len(arrays[buffer])}")
        obj = super(_SharedNumpyArray, cls).__new__(cls, shape, dtype,
                                                    arrays[buffer].get_obj(),
                                                    offset, strides, order)
        obj.buffer = buffer
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.offset = (
            self.__array_interface__['data'][0] -
            obj.__array_interface__['data'][0])
        self.buffer = getattr(obj, 'buffer', None)

    def _disconnect(self):
        """Return a tiny object describing ourselves

        You can pass this 'stub' down a pipe to a different process,
        giving it enough information to cheaply re-create an equivalent
        _SharedNumpyArray viewing the same shared memory.
        """
        return _SharedNumpyArrayStub(shape=self.shape, dtype=self.dtype,
                                     buffer=self.buffer,
                                     offset=getattr(self, 'offset', 0),
                                     strides=self.strides,
                                     order=None)

class _SharedNumpyArrayStub():
    """Suitable for cheaply passing through pipes to other processes"""
    def __init__(self, shape=None, dtype=float, buffer=None, offset=0,
                 strides=None, order=None):
        if shape is None:
            raise TypeError("Missing required argument 'shape'.")
        if buffer is None:
            raise TypeError("Missing required argument 'buffer'.")
        self.shape = shape
        self.dtype = dtype
        self.buffer = buffer
        self.offset = offset
        self.strides = strides
        self.order = order

    def _reconnect(self, arrays):
        """Reconstruct a 'full' version from this stub.
        """
        return _SharedNumpyArray(arrays, self.shape, self.dtype, self.buffer,
                                 self.offset, self.strides, self.order)

def _disconnect_shared_arrays(args, kwargs):
    """Replaces _SharedNumpyArrays in the args and kwargs with new stubs
    """
    args = tuple(a._disconnect()
                 if isinstance(a, _SharedNumpyArray) else a
                 for a in args)
    kwargs = {k: v._disconnect()
              if isinstance(v, _SharedNumpyArray) else v
              for k, v in kwargs.items()}
    return args, kwargs

def _reconnect_shared_arrays(args, kwargs, shared_arrays):
    """Replaces stubs in the args and kwargs with new _SharedNumpyArrays
    """
    args = tuple(a._reconnect(shared_arrays)
                 if isinstance(a, _SharedNumpyArrayStub) else a
                 for a in args)
    kwargs = {k: v._reconnect(shared_arrays)
              if isinstance(v, _SharedNumpyArrayStub) else v
              for k, v in kwargs.items()}
    return args, kwargs

# When an exception from the child process isn't handled by the parent
# process, we'd like the parent to print the child traceback. Overriding
# sys.excepthook seems to be the standard way to do this:
def _my_excepthook(t, v, tb):
    """Show a traceback when a child exception isn't handled by the parent.
    """
    if hasattr(v, 'child_traceback_string'):
        print(f'{" Child Process Traceback ":v^80s}\n',
              v.child_traceback_string,
              f'{" Child Process Traceback ":^^80s}\n',
              f'{" Main Process Traceback ":v^80s}')
    sys.__excepthook__(t, v, tb)

sys.excepthook = _my_excepthook

# Multiprocessing code works fairly differently depending whether you
# use 'spawn' or 'fork'. Since 'spawn' seems to be available on every
# platform we care about, and 'fork' is either missing or broken on some
# platforms, we'll always use 'spawn'. If your code calls
# mp.set_start_method() and sets it to anything other than 'spawn', this
# will crash with a RuntimeError. If you really need 'fork', or
# 'forkserver', then you probably know what you're doing better than us,
# and you shouldn't be using this module.
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

# Testing block.
class _Tests():
    '''
    Method names that start with `test_` will be run.

    Use: _test_... method names for tests that are called as part
    of a larger group.
    '''
    class TestClass:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            for i, a in enumerate(args):
                setattr(self, f'arg_{i}', a)

        def printing_method(self, *args, **kwargs):
            print(*args, **kwargs)

        def printing_method2(self):
            print('Hello world 2', end='', flush=False)
            print(end='', flush=True)

        def test_method(self, *args, **kwargs):
            return (args, kwargs)

        def test_shared_numpy_input(self, shared_numpy_array):
            return shared_numpy_array.shape

        def test_shared_numpy_return(self, shape=(5,5)):
            return _SharedNumpyArray(shape=shape)

        def test_modify_array(self, a):
            a.fill(1)
            return a

        def test_return_array(self, a):
            return a

        def nested_method(self, crash=False):
            self._nested_method(crash)

        def _nested_method(self, crash):
            if crash:
                raise ValueError('This error was supposed to be raised')

    def __init__(self):
        print(f'{"#":#^80s}')
        print(f'{" Running Tests ":#^80s}')
        print(f'{"#":#^80s}')
        self.tests = 0
        self.passed = 0

    def test_create_proxy_manager(self):
        pm = ProxyManager((100, 100))

    def test_create_proxy_object(self):
        pm = ProxyManager((100, 100))
        proxy_obj = pm.proxy_object(_Tests.TestClass)

    def test_reconnnecting_and_disconnecting_views(self):
        pm = ProxyManager((int(1e9),))
        for i in range(1000):
            self._trial_slicing_of_shared_array(pm)

    def _trial_slicing_of_shared_array(self, pm):
        ri = np.random.randint # Just to get short lines
        dtype = np.dtype(np.random.choice(
            [np.uint16, np.uint8, float, np.float32, np.float64]))
        original_dimensions = tuple(
            ri(2, 100) for d in range(ri(2, 5)))
        slicer = tuple(
            slice(
                ri(0, a//2),
                ri(0, a//2)*-1,
                ri(1, min(6, a))
                )
            for a in original_dimensions)
        a = pm.shared_numpy_array(0, shape=original_dimensions, dtype=dtype)
        a.fill(0)
        b = a[slicer] ## should be a view
        b.fill(1)
        expected_total = int(b.sum())
        reloaded_total = b._disconnect()._reconnect(pm.shared_mp_arrays).sum()
        assert expected_total == reloaded_total, \
            f'Failed {dtype.name}/{original_dimensions}/{slicer}'

    def test_passing_normal_numpy_array(self):
        shape = (10, 10)
        dtype = int
        sz = int(np.prod(shape)*np.dtype(int).itemsize)
        pm = ProxyManager((sz, sz))
        a = np.zeros(shape, dtype)
        object_with_shared_memory = pm.proxy_object(_Tests.TestClass)
        object_with_shared_memory.test_shared_numpy_input(a)

    def test_passing_retrieving_shared_array(self):
        shape = (10, 10)
        dtype = int
        sz = int(np.prod(shape)*np.dtype(int).itemsize)
        pm = ProxyManager((sz, sz))
        object_with_shared_memory = pm.proxy_object(_Tests.TestClass)
        a = pm.shared_numpy_array(which_mp_array=0, shape=shape, dtype=dtype)
        a.fill(0)
        a = object_with_shared_memory.test_modify_array(a)
        assert a.sum() == np.product(shape), 'Contents of array not correct!'

    def test_raise_attribute_error(self):
        a = ProxyObject(_Tests.TestClass, 'attribute', x=4,)
        try:
            a.z
        except AttributeError as e: # Get __this__ specific error
            print("Attribute error handled by parent process:\n ", e)

    def test_printing_in_child_process(self):
        a = ProxyObject(_Tests.TestClass, 'attribute', x=4,)
        b = ProxyObject(_Tests.TestClass, x=5)
        b.printing_method('Hello')
        a.printing_method('A')
        a.printing_method('Hello', 'world', end='', flush=True)
        a.printing_method('')
        a.printing_method(a.x, '... ', end='', flush=False)
        b.printing_method(b.x)
        expected_output = 'Hello\nA\nHello world\n4 ... 5\n'
        return expected_output

    def test_setting_attribute_of_proxy(self):
        a = ProxyObject(_Tests.TestClass, 'attribute', x=4,)
        a.z = 10
        assert a.z == 10
        setattr(a, 'z', 100)
        assert a.z == 100

    def test_getting_attribute_of_proxy(self):
        a = ProxyObject(_Tests.TestClass, 'attribute', x=4)
        assert a.x == 4
        assert getattr(a, 'x') == 4

    def test_overhead(self):
        n_loops = 10000
        a = ProxyObject(_Tests.TestClass, 'attribute', x=4,)
        t = self.time_it(
            n_loops, a.test_method, timeout_us=100, name='a.test_method')
        print(f" {t:.2f} \u03BCs per trivial method call.")
        t = self.time_it(
            n_loops, lambda: a.x, timeout_us=100, name='a.x')
        print(f" {t:.2f} \u03BCs per get-attribute.")
        a.x = 4 ## test set attribute with normal syntax
        t = self.time_it(n_loops, lambda: setattr(a, 'x', 5),
                         timeout_us=100, name='setattr(a, "x", 5)')
        print(f" {t:.2f} \u03BCs per set-attribute.")
        t = self.time_it(n_loops, lambda: a.z, fail=False, timeout_us=100,
                         name='a.z (raises AttributeError)')
        print(f" {t:.2f} \u03BCs per parent-handled exception.")
        self._test_passing_array_performance()

    def _test_passing_array_performance(self):
        from itertools import product
        shape = (1000, 1000)
        dtype = np.uint8
        pass_methods = ['reference', 'serialization']
        method_names = ['test_shared_numpy_input', 'test_modify_array']
        shapes = [(10, 10), (1000, 1000)]
        for s, f, m in product(shapes, method_names, pass_methods):
            self._test_array_passing(m, f, s, dtype, 1000)

    def _test_array_passing(self, pass_by, method_name, shape, dtype, n_loops):
        dtype = np.dtype(dtype)
        name = f'{pass_by} -- {method_name} -- {dtype.name}-{shape}'
        sz = int(np.prod(shape)*np.dtype(int).itemsize)
        pm = ProxyManager((sz, sz))
        object_with_shared_memory = pm.proxy_object(_Tests.TestClass)
        if pass_by == 'reference':
            a = pm.shared_numpy_array(0, shape, dtype=dtype)
            timeout_us = 5e3
        elif pass_by == 'serialization':
            a = np.zeros(shape=shape, dtype=dtype)
            timeout_us = 1e6
        func = getattr(object_with_shared_memory, method_name)
        t_per_loop = self.time_it(n_loops, func, (a,), timeout_us=timeout_us,
                                  name=name)
        print(f' {t_per_loop:.2f} \u03BCs per {name}')

    def test_FIFO(self):
        import time
        camera_queue = FIFOLock()
        processing_queue = FIFOLock()
        display_queue = FIFOLock()
        disk_queue = FIFOLock()
        acq_order = {'camera': [],
                     'processing': [],
                     'display': [],
                     'disk': []}
        def use_resource(thread_id):
            for resource, name, duration in (
                (camera_queue, 'camera', 0.05),
                (processing_queue, 'processing', 0.2),
                (display_queue, 'display', 0.05),
                (disk_queue, 'disk', 0.3)
                ):
                print("Thread %i waiting in line for %s"%(thread_id, name))
                with resource:
                    print("Thread %i acquired %s"%(thread_id, name))
                    acq_order[name].append(thread_id)
                    time.sleep(duration)
                print("Thread %i released %s"%(thread_id, name))
        threads = []
        n_threads = 4
        for i in range(n_threads):
            threads.append(threading.Thread(target=use_resource, args=(i,)))
            threads[-1].start()
        for th in threads:
            th.join()
        print("Acquisition order:")
        for k, v in acq_order.items():
            print(' ', (k + ':').ljust(12), v, sep='')
            assert v == list(range(n_threads))

    def test_proxy_fifo_lock(self):
        import time

        # Create proxy objects of resources to use.
        # Each has a method that sleep for some ammount of time and
        # returns `True`.
        cam = ProxyObject(_DummyCamera)
        proc = ProxyObject(_DummyProcessor)
        gui = ProxyObject(_DummyGUI)
        fout = ProxyObject(_DummyFileSaver)

        threads = []
        num_threads = 10 # Number of threads to start

        # Create container to hold results
        results = [[] for i in range(num_threads)]
        acq_order = {'camera': [],
                     'processing': [],
                     'display': [],
                     'disk': []}

        def acquisition(thread_id, r):
            with cam as locked:
                r.append(cam.record(thread_id))
                acq_order['camera'].append(thread_id)
            with proc as locked:
                r.append(proc.process(thread_id))
                acq_order['processing'].append(thread_id)
            with gui as locked:
                r.append(gui.display(thread_id))
                acq_order['display'].append(thread_id)
            with fout as locked:
                r.append(fout.save(thread_id))
                acq_order['disk'].append(thread_id)

        NUM_STEPS = 4 # matches the number of steps for check results.

        for i in range(num_threads):
            threads.append(
                threading.Thread(
                    target=acquisition,
                    args=(i, results[i])))
            threads[-1].start()

        # Wait for threads
        for th in threads:
            th.join()

        # Check results
        for i, a in enumerate(results):
            assert sum(a) == NUM_STEPS*i, f'{i}-{a}'
        for r, th_o in acq_order.items():
            assert sorted(th_o) == th_o,\
                f'Resource `{r}` was used out of order! -- {th_o}'

    def run(self):
        tests = [i for i in dir(self) if i.startswith('test_')]
        self.tests = len(tests)
        for i, t in enumerate(tests):
            self._run_single_test(i, t)
        self._summarize_results()


    def _run_single_test(self, i, t):
        printed_output = io.StringIO()
        name = t[5:].replace('_', ' ')
        print(f'{f"     {i+1} of {self.tests} | Testing {name}    ":-^80s}')
        try:
            with redirect_stdout(printed_output):
                expected_output = getattr(self, t)()
            if expected_output is not None:
                o = printed_output.getvalue()
                assert expected_output == o, \
                    f'\n Returned result:\n'\
                    f'    `{repr(o)}`\n'\
                    f' Did not match expected output:\n'\
                    f'     "{repr(expected_output)}"\n'
        except Exception as e:
            print('v'*80)
            print(traceback.format_exc().strip('\n'))
            print('^'*80)
        else:
            self.passed += 1
            if printed_output.getvalue():
                for l in printed_output.getvalue().strip('\n').split('\n'):
                    print(f'   {l}')
            print(f'{f"> Success <":-^80s}')

    def _summarize_results(self):
        fill = '#' if self.passed == self.tests else '!'
        print(f'{fill}'*80)
        message = f"Completed Tests -- passed {self.passed} of {self.tests}"
        if fill == "#":
            print(f'{f"  {message}  ":#^80s}')
        else:
            print(f'{f"  {message}  ":!^80s}')
        print(f'{fill}'*80)


    def time_it(self, n_loops, func, args=None, kwargs=None, fail=True,
                timeout_us=None, name=None):
        import time
        start = time.perf_counter()
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        for i in range(n_loops):
            try:
                func(*args, **kwargs)
            except Exception as e:
                if fail:
                    raise e
                else:
                    pass
        end = time.perf_counter()
        time_per_loop_us = ((end-start) / n_loops)*1e6
        if timeout_us is not None:
            if time_per_loop_us > timeout_us:
                name = func.__name__ if name is None else name
                raise TimeoutError(
                    f'Timed out on {name}\n'
                    f'   args:{args}\n'
                    f'   kwargs: {kwargs}\n'
                    f' Each loop took {time_per_loop_us:.2f} \u03BCs'
                    f' (Allowed: {timeout_us:.2f} \u03BCs)')
        return time_per_loop_us

### TODO: Remove these if I can....
class _DummyCamera:
    def record(self, a):
        import time
        time.sleep(.05)
        return a

class _DummyProcessor:
    def process(self, a):
        import time
        time.sleep(.2)
        return a

class _DummyGUI:
    def display(self, a):
        import time
        time.sleep(.002)
        return a

class _DummyFileSaver:
    def save(self, a):
        import time
        time.sleep(.3)
        return a

if __name__ == '__main__':
    _Tests().run()

