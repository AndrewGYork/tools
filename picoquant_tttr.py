import os
import struct
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

"""
Utility functions for loading picoquant TTTR .ptu files as numpy
arrays. So far, we only support version 1.0.00 of the PicoHarp T3 format,
because that's the only type of data our hardware produces.
"""

def parse_tttr_header(filename, verbose=True, max_tags=1000):
    """Reads the header of a PicoQuant TTTR .ptu file as a dict of tags.

    You'll need these tags as input if you want to run the other
    functions in this module.
    """
    # We're only trying to read .ptu files:
    assert os.path.splitext(filename)[1] == '.ptu'
    with open(filename, 'rb') as file:
        # We're only trying to read TTTR files:
        preamble_1 = file.read(8)
        assert preamble_1 == b'PQTTTR\x00\x00' 
        # We're only trying to support version 1.0.00:
        preamble_2 = file.read(8)
        assert preamble_2 == b'1.0.00\x00\x00'
        # Populate a dict of tags:
        tags = {}
        while True:
            # Read the tags. It seems like they're using little endian?
            tag_id = file.read(32).strip(b'\x00').decode('ascii')
            # If the tag has a positive index, we've seen it before:
            tag_idx = int.from_bytes(
                file.read(4), byteorder='little', signed=True)
            if tag_idx not in (-1, 0): assert tag_id in tags
            # Tags store their info in a few different formats:
            tag_typecode = {
                0xffff0008: 'Empty8',
                0x00000008: 'Bool8',
                0x10000008: 'Int8',
                0x11000008: 'BitSet64',
                0x12000008: 'Color8',
                0x20000008: 'Float8',
                0x21000008: 'TDateTime',
                0x2001ffff: 'Float8Array',
                0x4001ffff: 'ASCII-String',
                0x4002ffff: 'Wide-String',
                0xffffffff: 'BinaryBlob'}[int.from_bytes(
                    file.read(4), byteorder='little', signed=False)]
            # ...and this info doesn't always fit into 8 bytes:
            tag_value = file.read(8)
            if tag_typecode in ('Float8Array',
                                'ASCII-String',
                                'Wide-String',
                                'BinaryBlob'):
                num_extra_bytes = int.from_bytes(
                    tag_value, byteorder='little', signed=False)
                assert num_extra_bytes % 8 == 0
                tag_value = file.read(num_extra_bytes)
            else:
                # Some simple tags safely convert:
                if tag_typecode == 'Bool8':
                    tag_value = (tag_value == (b'\x00' * 8))
                if tag_typecode == 'Int8':
                    tag_value = int.from_bytes(
                        tag_value, byteorder='little', signed=True)
                if tag_typecode in ('Float8', 'TDateTime'):
                    tag_value = struct.unpack('<d', tag_value)[0]
            # Wanna print? Sometimes helpful for debugging.
            if verbose:
                print("\nTag ID:", tag_id)
                print("Tag index:", tag_idx)
                print("Tag typecode:", tag_typecode)
                print("Tag value:", tag_value)
            # Store the tag data in a python dict. Because tags might
            # appear more than once, the tag's values are stored as a
            # list, one list entry for each time the tag appears:
            if tag_idx in (-1, 0):
                tags[tag_id] = {'typecode': tag_typecode,
                                'values': [tag_value]}
            else:
                # This isn't our first rodeo with this tag; hopefully
                # the index isn't lying about how many tags of its kind
                # preceded it:
                assert tag_id in tags
                assert tag_idx == len(tags[tag_id]['values'])
                tags[tag_id]['values'].append(tag_value)
            # How do we tell if we're done? Hopefully the file tells us:
            if tag_id == 'Header_End':
                # Since we're about to close the file, make note of how
                # many bytes we've read, and how many remain:
                data_offset = file.tell()
                file.seek(0, 2)
                remaining_bytes = file.tell() - data_offset
                # ...before closing the file:
                break
            # In case the file forgot to declare the end of the header:
            assert len(tags) <= max_tags
    # These tags are mandatory:
    for t in ('File_GUID',
              'Measurement_Mode',
              'Measurement_SubMode',
              'Header_End',
              'MeasDesc_GlobalResolution',
              'MeasDesc_Resolution',
              'TTResult_SyncRate',
              'TTResult_NumberOfRecords',
              'TTResultFormat_TTTRRecType',
              'TTResultFormat_BitsPerRecord'):
        if not t in tags: raise KeyError("Mandatory TTTR tag missing: %s"%t)
    # Does the header's opinion about the number of data bytes match reality?
    num_data_bytes = int(tags['TTResultFormat_BitsPerRecord']['values'][0] / 8 *
                         tags['TTResult_NumberOfRecords']['values'][0])
    assert num_data_bytes == remaining_bytes
    return tags

def generate_picoharp_t3_frames(
    filename,
    tags,
    records_per_chunk=2000000,
    verbose=True,
    ):
    """
    Returns a generator which loads one frame at a time from PicoHarp T3
    formatted .ptu files.

    I don't think there's an upper bound on how big .ptu files can be,
    and sometimes they're too big to comfortably hold in memory, so I
    want to be able to load one frame at a time.

    The hitch is we don't know how many bytes to load to get one frame,
    so we load bytes in chunks, search the chunks for frame markers, and
    'yield' one frame at a time.
    """
    # We're going to use this symbol to identify frames:
    assert 3 == tags['ImgHdr_Frame']['values'][0]
    # This tag is mandatory, so we know it's present:
    num_records = tags['TTResult_NumberOfRecords']['values'][0]
    loaded_records = []
    with open(filename, 'rb') as f:
        f.seek(-4*num_records, 2) # 4 bytes per record
        while True:
            records = np.frombuffer(f.read(4*records_per_chunk),
                                    dtype=np.uint32)
            if len(records) == 0: # End of file
                break
            # Inspect the records for frame markers, which have
            # channel = 15 and dtime >= 3.
            is_frame_marker = records.view(dtype=np.uint16)[1::2] >= 0xf004
            frame_marker_indices = np.nonzero(is_frame_marker)[0]
            while len(frame_marker_indices) > 0:
                # We have zero or more previously loaded chunks of
                # records that don't contain frame markers, and a
                # freshly loaded chunk of records that has one or more
                # frame markers.
                fmi = frame_marker_indices[0]
                loaded_records.append(records[:fmi])
                if len(loaded_records) > 1: # Remember to use the old records!
                    yield np.concatenate(loaded_records)
                else: # Don't bother with a slow 'concatenate'
                    yield loaded_records[-1]
                # Now we chew one frame marker worth of records off the
                # freshly loaded records, pop one frame marker index,
                # and re-align any remaining frame marker indices with
                # the new record length:
                loaded_records = []
                frame_marker_indices = frame_marker_indices[1:] - (fmi+1)
                records = records[fmi+1:]
            # We're out of frame markers; store any leftover records for
            # the next pass through the loop:
            loaded_records.append(records)                   
        
def parse_picoharp_t3_frame(
    records,
    tags,
    verbose=False,
    show_plot=False,
    sinusoid_correction=True,
    ):
    """Converts one frame worth of records to a dict of 1D numpy arrays.

    The output is suitable for binning into a 4D numpy histogram showing
    photoelectrons vs. x/y position, arrival time relative to the
    excitation pulse, and channel.
    """
    # This function only supports the PicoHarp T3 (with 32-bit records):
    assert 0x00010303 == tags['TTResultFormat_TTTRRecType']['values'][0]
    assert 32 == tags['TTResultFormat_BitsPerRecord']['values'][0]
    overflow_period = 65536 # 2**16
    # We're going to use these symbols to identify lines:
    assert 1 == tags['ImgHdr_LineStart']['values'][0]
    assert 2 == tags['ImgHdr_LineStop']['values'][0]
    # This function assumes that 'records' is from a single frame,
    # meaning it doesn't contain any frame markers.
    num_records = len(records)
    # First, let's convert 'overflows' (records which indicate the clock
    # overflowed) and 'nsync' (how many laser pulses have passed since
    # the last clock overflow?) to 'num_pulses' (how many laser pulses
    # have passed since the overflow immediately preceding this frame?):
    overflows = (records == 0xf0000000)
    num_overflows = np.cumsum(overflows, dtype=np.uint64)
    # Now we can clean out overflow records, which can be the majority
    # record for dim samples:
    records = records[~overflows]
    num_overflows = num_overflows[~overflows]
    del overflows
    # The least significant 16 bits of each record is an 'nsync';
    # we'll use overflows and nsync to calculate absolute times for
    # each record:
    nsync = (records & 0x0000ffff)
    num_pulses = num_overflows * overflow_period + nsync
    del num_overflows, nsync
    # The 4 most-significant bits of each record is a 'channel':
    channel = (records >> 28).astype(np.uint8)
    # The next 12 bits of each record is a 'dtime':
    dtime = ((records & 0x0fff0000) >> 16).astype(np.uint16)
    # Extract line-start and line-end markers to calculate x and y
    # positions. All have channel=15, line starts have dtime=1, line
    # ends have dtime=2
    y_pos = np.cumsum((records & 0xf0010000) == 0xf0010000).astype(np.uint32)
    mark_indices = ((records & 0xf0030000) > 0xf0000000) # Starts AND ends
    # TODO: Interpolation is maybe a bottleneck; I should profile this if we
    # end up slow.
    x_pos = np.interp(
        x=num_pulses, # The times when we want to know our scan pos.       
        xp=num_pulses[mark_indices], # The times when we actually know pos.
        fp=((records[mark_indices] & 0x00020000) >> 17), # 0: start, 1: end
        left=-0.01,
        right=-0.01)
    x_pos[1:][np.diff(x_pos) < 0] = -0.01 # Ignore x during flyback
    del records, mark_indices
    if 'ImgHdr_SinCorrection' in tags and sinusoid_correction:
        # The scan is sinusoidal, not constant velocity. We'd better
        # correct for it! This is a mess, but it seems to work.
        # TODO: clean up this code a bit.
        sinusoidal_fraction = tags['ImgHdr_SinCorrection']['values'][0] / 100
        x_pos[x_pos >= 0] = (
            0.5 - 0.5 * (np.cos(np.pi * ((1 - sinusoidal_fraction) / 2 +
                                         (sinusoidal_fraction *
                                          x_pos[x_pos >= 0])
                                         )) /
                         np.cos(np.pi * (1 - sinusoidal_fraction) / 2)))
    # Convert from "fraction" units to pixel units:
    x_pos *=  tags['ImgHdr_PixX']['values'][0]
    # Now that we've calculated positions, we can remove reports that
    # aren't photoelectrons. Allocating new memory like this is slow,
    # but this prevents accidentally using extra memory to hold the
    # records we're ignoring:
    electrons = (channel != 15)
    y_pos   = y_pos[electrons].copy()
    x_pos   = x_pos[electrons].copy()
    channel = channel[electrons].copy()
    dtime   = dtime[electrons].copy()
    num_pulses = num_pulses[electrons].copy()
    if show_plot:
        if plt is None:
            raise UserWarning(
                "Failed to import matplotlib; you won't be able to plot.")
        t = num_pulses * tags['MeasDesc_GlobalResolution']['values'][0]
        fig = plt.figure()
        plt.plot(t, y_pos / y_pos.max(), label='Y pos.')
        plt.plot(t, x_pos / x_pos.max(), label='X pos.')
        for ch in range(1, 5):
            plt.plot(t[channel == ch],
                     dtime[channel == ch]/dtime.max(),
                     'x', label='Chan. %d'%ch)
        plt.grid('on')
        plt.legend()
        plt.title('Parsed records')
        fig.show()
    if verbose:
        print("done.")
        print(num_records, "records")
        print('', len(channel), "electrons")
        for ch in range(1, 5):
            print(' ', (channel == ch).sum(), "electrons in channel", ch)
    return {'tags': tags, 'num_pulses': num_pulses,
            'dtime':   dtime, 'channel': channel,
            'y_pixel': y_pos, 'x_pixel': x_pos}

def parsed_frame_to_histogram(
    parsed_frame,
    x_pix_per_bin=1,
    y_pix_per_bin=1,
    t_pix_per_bin=40,
    ):
    """Convert the output of parse_picoharp_t3 frame to a 4D numpy array. 

    If you know what you're doing, you can probably make better use of
    np.histogramdd. If you don't, this function is a convenient way to
    get a histogram without thinking too hard.

    '*_pix_per_bin' lets you choose how many pixels per bin. The parsed
    frame data already has opinions about the x/y pixel size. Since dtime
    for the Picoharp T3 format is 12-bit, we'll use 2**12 time bins to
    define the native time pixel size.

    TODO: make it easier to discover the pixel size in nanometers and
    picoseconds.
    """
    x_pix = parsed_frame['tags']['ImgHdr_PixX']['values'][0]
    y_pix = parsed_frame['tags']['ImgHdr_PixY']['values'][0]
    im, bins = np.histogramdd(
        sample=np.stack((parsed_frame['dtime'],
                         parsed_frame['channel'],
                         parsed_frame['y_pixel'],
                         parsed_frame['x_pixel']), axis=1),
        bins=(np.arange(0, 2**12+1, t_pix_per_bin),
              np.arange(1, 6, 1), # Four channels, right?
              np.arange(0, y_pix+1, y_pix_per_bin),
              np.arange(0, x_pix+1, x_pix_per_bin)))
    return im

# TODO: write a function that generates one histogram for every frame of
# a file, and saves them as 3D tifs.

if __name__ == '__main__':
    # Example code for dealing with parts of a large file:
    filename = 'LSM_1.ptu'
    print("Reading header...", end='')
    tags = parse_tttr_header(filename, verbose=False)
    print("done")
    print("Num tags:", sum(len(tags[t]['values']) for t in tags))
    print("Num unique tags:", len(tags))
    print()
    # We loop over the frames one at a time:
    frames = generate_picoharp_t3_frames(filename, tags, verbose=True)
    for i, f in enumerate(frames):
        # We don't have to parse every frame:
        if i == 10:
            print("Parsing frame ", i, '... ', sep='', end='')
            parsed_frame = parse_picoharp_t3_frame(
                records=f,
                tags=tags,
                verbose=True,
                show_plot=True)
            print("done.")
    # We can visualize just the ONE frame we parsed:
    im = parsed_frame_to_histogram(parsed_frame)
    print(im.shape, im.dtype)
    if plt is not None:
        fig = plt.figure()
        for ch in range(4):
            plt.plot(im[:, ch, :, :].sum(axis=-1).sum(axis=-1),
                     '.-', label="Channel %d"%(ch+1))
        plt.grid('on')
        plt.title("Lifetime histograms")
        plt.legend()
        fig.show()

        fig = plt.figure()
        for ch in range(4):
            plt.subplot(2, 2, ch+1)
            plt.imshow(im[:, ch, :, :].sum(axis=0), cmap=plt.cm.gray)
            plt.title("Channel %d"%(ch+1))
        fig.show()
