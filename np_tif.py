import numpy as np

def tif_to_array(
    filename,
    image_descriptions=False,
    verbose=False,
    ):
    """Load a tif into memory and return it as a numpy array.

    This is primarily a tool we use to interact with ImageJ, so that's
    the only case it's really been debugged for. I bet somebody made
    nice python bindings for LibTIFF, if you want a more general purpose
    reader.
    """
    ifds, endian = parse_tif(filename, verbose)
    """
    Ensure that the various IFD's are consistent: same length, width,
    bit depth, data format, etc.
    Also check that our assumptions about other tags are true.
    """
    width = ifds[0]['ImageWidth']
    length = ifds[0]['ImageLength']
    bit_depth = ifds[0]['BitsPerSample']
    data_format = ifds[0].get('SampleFormat', 1) #Default to unsigned int
    for d in ifds:
        try:
            assert width == d['ImageWidth']
            assert length == d['ImageLength']
            assert bit_depth == d['BitsPerSample']
            assert data_format == d.get('SampleFormat', 1)
        except AssertionError:
            print("To load a TIF as a numpy array, the IFDs all have to match.")
            print("IFD A:", ifds[0])
            print("IFD B:", d)
            raise UserWarning("The TIF we're trying to load has mismatched IFD's")
        try:
            assert d.get('SamplesPerPixel', 1) == 1
            assert d.get('NewSubFileType', 0) == 0
            assert d.get('Compression', 1) == 1
            assert d.get('PhotometricInterpretation', 0) in (0, 1)
        except AssertionError:
            print("Offending IFD:", d)
            raise UserWarning(
                "The TIF we're trying to load" +
                " uses options that np_tif doesn't support.")
    """
    Collect the strip offsets and the strip byte counts
    """
    strip_offsets = []
    strip_byte_counts = []
    for d in ifds:
        try: #Just one strip per IFD
            strip_offsets.append(int(d['StripOffsets']))
            strip_byte_counts.append(int(d['StripByteCounts']))
        except TypeError: #Many strips per IFD
            strip_offsets.extend(int(x) for x in d['StripOffsets'])
            strip_byte_counts.extend(int(x) for x in d['StripByteCounts'])
    assert len(strip_offsets) == len(strip_byte_counts)
    """
    Allocate our numpy array, and load data into our array from disk,
    one strip at a time.
    """
    data = np.zeros(sum(strip_byte_counts), dtype=np.ubyte)
    data_offset = 0
    with open(filename, 'rb') as f:
        for i in range(len(strip_offsets)):
            file_offset = strip_offsets[i]
            num_bytes = strip_byte_counts[i]
            data[data_offset:data_offset + num_bytes] = np.frombuffer(
                get_bytes_from_file(f, file_offset, num_bytes),
                dtype=np.ubyte)
            data_offset += num_bytes
    """
    Determine the numpy data type from the TIF bit depth and data
    format, and reshape based on width, height, and number of ifd's:
    """
    data_type = {
        1: 'uint',
        2: 'int',
        3: 'float',
        4: 'undefined',
        }[data_format] + ascii(bit_depth)
    try:
        data_type = getattr(np, data_type)
    except AttributeError:
        raise UserWarning("Unsupported data format: " + data_type)
    data = data.view(data_type)
    if endian == 'big':
        data = data.byteswap()
    data = data.reshape(len(ifds), length, width)
    """
    Optionally, return the image descriptions.
    """
    if image_descriptions:
        image_descriptions = [d.get('ImageDescription', '') for d in ifds]
        for desc in image_descriptions:
            if desc != image_descriptions[0]:
                break
        else:
            image_descriptions = image_descriptions[0:1]
        return data, image_descriptions
    return data

def array_to_tif(
    x,
    filename,
    slices=None,
    channels=None,
    frames=None,
    verbose=False,
    coerce_64bit_to_32bit=True,
    backup_filename=None,
    ):
    """Save a numpy array as a TIF

    We'll structure our TIF the same way ImageJ does:
    *8-bit header
    *First image file directory (IFD, description of one 2D slice)
    *Image description
    *All image data
    *Remaining IFDs
    
    First, ensure a three dimensional input:
    """
    if len(x.shape) == 1:
        x = x.reshape((1, 1,) + x.shape)
    if len(x.shape) == 2:
        x = x.reshape((1,) + x.shape)
    assert len(x.shape) == 3
    """
    All our IFDs are very similar; reuse what we can:
    """
    ifd = Simple_IFD()
    ifd.width[0] = x.shape[2]
    ifd.length[0] = x.shape[1]
    ifd.rows_per_strip[0] = x.shape[1]
    if coerce_64bit_to_32bit and x.dtype in (np.float64, np.int64, np.uint64):
        if x.dtype == np.float64:
            dtype = np.dtype('float32')
        elif x.dtype == np.int64:
            dtype = np.dtype('int32')
        elif x.dtype == np.uint64:
            dtype = np.dtype('uint32')
    elif x.dtype == np.bool: # Coorce boolean arrays to uint8
        dtype = np.dtype('uint8')
    else:
        dtype = x.dtype
    ifd.set_dtype(dtype)
    ifd.strip_byte_counts[0] = (x.shape[1] *
                                x.shape[2] *
                                ifd.bits_per_sample[0] // 8)

    if slices is not None and channels is not None and frames is not None:
        assert slices * channels * frames == x.shape[0]
        image_description = bytes(''.join((
            'ImageJ=1.48e\nimages=%i\nchannels=%i\n'%(x.shape[0], channels),
            'slices=%i\nframes=%i\nhyperstack=true\n'%(slices, frames),
            'mode=grayscale\nloop=false\nmin=%0.3f\nmax=%0.3f\n\x00'%(
                x.min(), x.max()))), encoding='ascii')        
    elif slices is not None and channels is not None and frames is None:
        assert slices * channels == x.shape[0]
        image_description = bytes(''.join((
            'ImageJ=1.48e\nimages=%i\nchannels=%i\n'%(x.shape[0], channels),
            'slices=%i\nhyperstack=true\nmode=grayscale\n'%(slices),
            'loop=false\nmin=%0.3f\nmax=%0.3f\n\x00'%(x.min(), x.max()))),
                                  encoding='ascii')
    else:
        image_description = bytes(''.join((
            'ImageJ=1.48e\nimages=%i\nslices=%i\n'%(x.shape[0], x.shape[0]),
            'loop=false\nmin=%0.3f\nmax=%0.3f\n\x00'%(x.min(), x.max()))),
                                  encoding='ascii')
    ifd.num_chars_in_image_description[0] = len(image_description)
    ifd.offset_of_image_description[0] = 8 + ifd.bytes.nbytes
    ifd.strip_offsets[0] = 8 + ifd.bytes.nbytes + len(image_description)
    if x.shape[0] == 1:
        ifd.next_ifd_offset[0] = 0
    else:
        ifd.next_ifd_offset[0] = (
            ifd.strip_offsets[0] + x.size * ifd.bits_per_sample[0] // 8)
    """
    We have all our ducks in a row, time to actually write the TIF:
    """
    for fn in (filename, backup_filename):
        try:
            with open(fn, 'wb') as f:
                f.write(b'II*\x00\x08\x00\x00\x00') #Little tif, turn to page 8
                ifd.bytes.tofile(f)
                f.write(image_description)
                if dtype != x.dtype: # We have to coerce to a different dtype
                    for z in range(x.shape[0]): #Convert one at a time (memory)
                        x[z, :, :].astype(dtype).tofile(f)
                else:
                    x.tofile(f)
                for which_header in range(1, x.shape[0]):
                    if which_header == x.shape[0] - 1:
                        ifd.next_ifd_offset[0] = 0
                    else:
                        ifd.next_ifd_offset[0] += ifd.bytes.nbytes
                    ifd.strip_offsets[0] += ifd.strip_byte_counts[0]
                    ifd.bytes.tofile(f)
            break 
        except Exception as e:
            print("np_tif.array_to_tif failed to save:")
            print(fn)
            print(" with error:", repr(e))
            if backup_filename is not None and fn!=backup_filename:
                continue
            else:
                raise
    return None

def parse_tif(
    filename,
    verbose=False,
    ):
    """
    Open a file, determine that it's a TIF by parsing its header, and
    read through the TIF's Image File Directories (IFDs) one at a time
    to determine the structure of the TIF.
    See:
     partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
    for reference.
    """
    with open(filename, 'rb') as file:
        next_address, endian = parse_header(file, verbose)
        ifds = []
        while next_address != 0:
            """
            TODO: Check that the next address is sane (unique, in-bounds, etc.)
            """
            next_address, entries = parse_ifd(file, next_address, endian, verbose)
            ifds.append(entries)
            if verbose:
                print("Interpreted entries:")
                for k in entries.keys():
                    print(' ', k, ": ", ascii(entries[k]), sep='')
            if verbose: print("Next address:", next_address, '\n')
        if verbose: print('No more IFDs\n')
    return ifds, endian

def parse_header(file, verbose):
    """
    Read the 8 bytes at the start of a file to determine:
    1. Does the file seem to be a TIF?
    2. Is it little or big endian?
    3. What is the address of the first IFD?
    """
    header = get_bytes_from_file(file, offset=0, num_bytes=8)
    if verbose: print("Header:", header)
    if (header[0] == 73 and header[1] == 73 #Little-endian
        and header[2] == 42 and header[3] == 0): #Little-endian 42
        endian = 'little'
    elif (header[0] == 77 and header[1] == 77 #Big-endian
          and header[2] == 0 and header[3] == 42): #Big-endian 42
        endian = 'big'
    else:
        raise UserWarning("Not a TIF file")
    next_address = bytes_to_int(header[4:8], endian)
    if verbose: print(" (I'm a ", endian, "-endian tif, turn to page ",
                      next_address, ")\n", sep='')
    return next_address, endian
            
def parse_ifd(file, address, endian, verbose):
    """
    An IFD has:
     2-bytes to tell how many entries
     12 bytes per entry
     4 bytes to store the next address
    """
    num_entries = bytes_to_int(
        get_bytes_from_file(file, offset=address, num_bytes=2),
        endian)
    if verbose: print("IFD at address", address, "with", num_entries, "entries:")
    ifd_bytes = get_bytes_from_file(
        file, offset=address+2, num_bytes=12*num_entries + 4)
    entries = {}
    for t in range(num_entries):
        """
        TODO? TIF spec says entries must be stored in ascending
        numerical order, but I don't enforce that.
        """
        entry = ifd_bytes[12 * t:
                          12 *(t+1)]
        if verbose:
            print("   Entry ", '%02i'%t, ": ", sep='', end='')
            for e in entry:
                print('%03i,'%e, '', end='')
            print()
        tag, value = interpret_ifd_entry(file, entry, endian, verbose)
        entries[tag] = value
    next_address = bytes_to_int(ifd_bytes[12*num_entries:
                                          12*num_entries + 4],
                                endian)
    return next_address, entries

def interpret_ifd_entry(file, entry, endian, verbose):
    """
    Each IFD entry is stored in a binary format. Decode this to a python
    dict.
    """
    tag = bytes_to_int(entry[0:2], endian)
    tag_lookup = {
        254: 'NewSubFileType',
        256: 'ImageWidth',
        257: 'ImageLength',
        258: 'BitsPerSample',
        259: 'Compression',
        262: 'PhotometricInterpretation',
        270: 'ImageDescription',
        273: 'StripOffsets',
        277: 'SamplesPerPixel',
        278: 'RowsPerStrip',
        279: 'StripByteCounts',
        282: 'XResolution',
        283: 'YResolution',
        296: 'ResolutionUnit',
        339: 'SampleFormat',
        }
    try:
        tag = tag_lookup[tag]
    except KeyError:
        if verbose: print("Unknown tag in TIF:", tag)
    field_type = bytes_to_int(entry[2:4], endian)
    field_type_lookup = {
        1: ('BYTE', 1),
        2: ('ASCII', 1),
        3: ('SHORT', 2),
        4: ('LONG', 4),
        5: ('RATIONAL', 8), #It gets a little weird past here
        6: ('SBYTE', 1),
        7: ('UNDEFINED', 8),
        8: ('SSHORT', 2),
        9: ('SLONG', 4),
        10: ('SRATIONAL', 8),
        11: ('FLOAT', 4),
        12: ('DOUBLE', 8),
        }
    try:
        field_type, bytes_per_count = field_type_lookup[field_type]
    except KeyError:
        if verbose: print("Unknown field type in TIF:", field_type)
        return tag, entry[8:12] #Field type is unknown, value is just bytes
    num_values = bytes_to_int(entry[4:8], endian)
    value_size_bytes = num_values * bytes_per_count
    if value_size_bytes <= 4:
        """
        The bytes directly encode the value
        """
        value = entry[8:8+value_size_bytes]
    else:
        """
        The bytes encode a pointer to the value 
        """
        address = bytes_to_int(entry[8:12], endian)
        value = get_bytes_from_file(file, address, value_size_bytes)
    """
    We still haven't converted the value from bytes yet, but at least we
    got the correct bytes that encode the value.
    """
    if field_type in ('BYTE', 'SHORT', 'LONG'):
        if num_values == 1:
            value = bytes_to_int(value, endian)
        else:
            typestr = ({'big': '<', 'little': '>'}[endian] +
                       {'BYTE': 'u1','SHORT': 'u2', 'LONG': 'u4'}[field_type])
            value = np.fromstring(value, dtype=np.dtype(typestr))
##    elif field_type == 'ASCII':
##        value = str(value, encoding='ascii')
    else:
        pass #Just leave it as bytes. TODO: interpret more field types?
    return tag, value

class Simple_IFD:
    def __init__(self):
        """
        A very simple TIF IFD with 11 tags (2 + 11*12 + 4 = 138 bytes)
        """
        self.bytes = np.array([
            #Num. entries = 11
             11,   0, 
            #NewSubFileType = 0
            254,   0,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #Width = 0
              0,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #Length = 0
              1,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #BitsPerSample = 0
              2,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #PhotometricInterpretation = 1
              6,   1,   3,   0,   1,   0,   0,   0,   1,   0,   0,   0,
            #ImageDescription (num_chars = 0, pointer = 0)
             14,   1,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            #StripOffsets = 0
             17,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #SamplesPerPixel = 1
             21,   1,   3,   0,   1,   0,   0,   0,   1,   0,   0,   0,
            #RowsPerStrip = 0
             22,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #StripByteCounts = 0
             23,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #SampleFormat = 3
             83,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0,
            #Next IFD = 0
              0,   0,   0,   0,
            ], dtype=np.ubyte)
        self.width = self.bytes[22:26].view(dtype=np.uint32)
        self.length = self.bytes[34:38].view(dtype=np.uint32)
        self.bits_per_sample = self.bytes[46:50].view(dtype=np.uint32)
        self.num_chars_in_image_description = self.bytes[66:70].view(np.uint32)
        self.offset_of_image_description = self.bytes[70:74].view(np.uint32)
        self.strip_offsets = self.bytes[82:86].view(np.uint32)
        self.rows_per_strip = self.bytes[106:110].view(np.uint32)
        self.strip_byte_counts = self.bytes[118:122].view(np.uint32)
        self.data_format = self.bytes[130:134].view(np.uint32)
        self.next_ifd_offset = self.bytes[134:138].view(np.uint32)
        return None
    
    def set_dtype(self, dtype):
        allowed_dtypes = {
            np.dtype('uint8'): (1, 8),
            np.dtype('uint16'): (1, 16),
            np.dtype('uint32'): (1, 32),
            np.dtype('uint64'): (1, 64),
            np.dtype('int8'): (2, 8),
            np.dtype('int16'): (2, 16),
            np.dtype('int32'): (2, 32),
            np.dtype('int64'): (2, 64),
            ##np.dtype('float16'): (3, 16), #Not supported in older numpy?
            np.dtype('float32'): (3, 32),
            np.dtype('float64'): (3, 64),
            }
        try:
            self.data_format[0], self.bits_per_sample[0] = allowed_dtypes[dtype]
        except KeyError:
            warning_string = "Array datatype (%s) not allowed. Allowed types:"%(
                dtype)
            for i in sorted(allowed_dtypes.keys()):
                warning_string += '\n ' + repr(i)
            raise UserWarning(warning_string)
        return None

def get_bytes_from_file(file, offset, num_bytes):
    file.seek(offset)
    return file.read(num_bytes)

def bytes_to_int(x, endian): #Isn't there a builtin to do this...?
    if endian == 'little':
        return sum(c*256**i for i, c in enumerate(x))
    elif endian == 'big':
        return sum(c*256**(len(x) - 1 - i) for i, c in enumerate(x))
    else:
        raise UserWarning("'endian' must be either big or little")

if __name__ == '__main__':
    """
    Simple tests, not comprehensive.
    """
    import os
    
    for a in (
        np.random.random_sample((30, 432, 500)),
        np.random.random_sample((3, 432, 500)),
        np.random.random_sample((432, 500)),
        np.random.random_sample(500),
        np.random.random_integers(0, 1e6, (3, 437, 500)),
        np.random.random_integers(0, 1e6, (3, 437, 500)).astype(np.float32),
        np.random.random_integers(0, 1e6, (3, 437, 500)).astype(np.uint64),
        ):
        array_to_tif(a, 'test.tif')
        b = tif_to_array('test.tif')
        print("To disk:", a.shape, a.dtype, a.min(), a.max())
        print("From disk", b.shape, b.dtype, b.min(), b.max())
        assert np.all(np.isclose(a, b))
    array_to_tif(a, os.path.join(os.getcwd(),
                                 'this_folder_does_not_exist',
                                 'test.tif'),
                 backup_filename='test.tif')
        


