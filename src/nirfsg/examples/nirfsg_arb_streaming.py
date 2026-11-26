'''Streams a continuously computed waveform that is longer than the device memory.

Based on the "Streaming", "Streaming Waveform Data in NI-RFSG", and "Improving NI-RFSG Streaming
Performance" topics of the NI-RFSG user manual. Python counterpart of the NI-RFSG .NET
"ArbitraryWaveforms/ArbitraryWaveformStreaming" example.

Background:
    Streaming generates waveforms that are too large for onboard memory, or that are computed
    while the device is generating. Part of the onboard memory is used as a circular buffer
    (the streaming waveform), which the program keeps refilling while the device plays it:
      1. Set power_level_type to PEAK. Streaming is not possible with AVERAGE power, because
         the driver cannot know the average power of data that has not been written yet.
      2. Set streaming_waveform_name and streaming_enabled. The driver then tracks which part
         of the buffer has been generated, so fresh data never overwrites samples that are
         still waiting to be generated.
      3. Reserve the buffer with allocate_arb_waveform(). Write in blocks of a fixed size that
         divides the buffer size evenly, so that no write has to wrap around the end of the
         buffer.
      4. Fill the buffer with the first data, then call initiate().
      5. While generating, read streaming_space_available_in_waveform and write the next block
         when there is room for it. If a block is larger than the free space, write_arb_waveform()
         waits up to streaming_write_timeout for the space to become available.
    If the program does not keep up, the device runs out of fresh data (an underflow) and
    NI-RFSG reports an error. To sustain high rates:
      - Use interleaved int16 data. It needs 4 bytes per sample, compared with 8 for complex64.
      - Use large blocks, so there are fewer, longer transfers.
      - Prepare the data ahead of time, or in a separate thread, not in the write loop.
    The required data rate is iq_rate * 4 bytes/s with int16 data, for example 400 MB/s at
    100 MS/s.

    This example streams a sawtooth frequency sweep across the signal bandwidth. Its phase is
    carried from block to block, so the signal is continuous and never repeats within a sweep
    period, no matter how the sweep lines up with the buffer. The stream runs for the
    requested time, then the generation is aborted. Data that is still in the buffer at that
    point is not generated.

Steps:
    1. Configure the RF settings, ARB mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Enable streaming, set the write timeout, and allocate the streaming buffer.
    3. Fill the buffer, then start the generation.
    4. Write each new block when there is space for it, and check the generation status.
    5. Stop after the requested time and report the sustained data rate.
'''

import argparse
import datetime
import nirfsg
import numpy as np
import sys
import time


class _SweepSource(object):
    '''Produces consecutive blocks of a phase-continuous sawtooth frequency sweep as interleaved int16 data.'''

    def __init__(self, iq_rate, sweep_bandwidth, sweep_period):
        self._iq_rate = iq_rate
        self._sweep_bandwidth = sweep_bandwidth
        self._sweep_period = sweep_period
        self._next_sample = 0
        self._phase = 0.0

    def next_block(self, number_of_samples):
        n = np.arange(self._next_sample, self._next_sample + number_of_samples)
        # Instantaneous frequency ramps from -bandwidth/2 to +bandwidth/2 once per sweep period.
        position_in_sweep = np.mod(n / self._iq_rate, self._sweep_period) / self._sweep_period
        instantaneous_frequency = (position_in_sweep - 0.5) * self._sweep_bandwidth
        phase = self._phase + 2 * np.pi * np.cumsum(instantaneous_frequency) / self._iq_rate
        self._phase = phase[-1] % (2 * np.pi)
        self._next_sample += number_of_samples

        interleaved = np.empty(2 * number_of_samples, dtype=np.int16)
        interleaved[0::2] = np.round(np.cos(phase) * 32767)
        interleaved[1::2] = np.round(np.sin(phase) * 32767)
        return interleaved


def _wait_for_space(session, number_of_samples, timeout):
    # Wait until the device has generated enough of the buffer to make room for the next block.
    # The status check reports an underflow or any other generation error while waiting.
    deadline = time.monotonic() + timeout
    while session.streaming_space_available_in_waveform < number_of_samples:
        session.check_generation_status()
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def example(resource_name, options, frequency, power_level, iq_rate, sweep_bandwidth, sweep_period, buffer_size, block_size, write_timeout, stream_time, stall_timeout):
    waveform_name = 'streamingWaveform'
    if buffer_size % block_size != 0:
        raise ValueError('The buffer size must be a multiple of the block size.')

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        if sweep_bandwidth > 0.8 * actual_iq_rate:
            raise ValueError(f'The sweep bandwidth must be at most 0.8 * I/Q rate ({0.8 * actual_iq_rate / 1e6} MHz).')
        session.signal_bandwidth = sweep_bandwidth

        # Identify the streaming waveform, enable streaming, and reserve the buffer.
        session.streaming_waveform_name = waveform_name
        session.streaming_enabled = True
        session.streaming_write_timeout = datetime.timedelta(seconds=write_timeout)
        session.allocate_arb_waveform(waveform_name, buffer_size)

        source = _SweepSource(actual_iq_rate, sweep_bandwidth, sweep_period)

        # Fill the whole buffer before the generation starts.
        for _ in range(buffer_size // block_size):
            session.write_arb_waveform(waveform_name, source.next_block(block_size), True)
        samples_written = buffer_size

        print(f'Streaming at {actual_iq_rate / 1e6} MS/s ({actual_iq_rate * 4 / 1e6:.1f} MB/s of int16 data) through a {buffer_size}-sample buffer '
              f'({buffer_size / actual_iq_rate * 1e3:.2f} ms) in {block_size}-sample blocks, for {stream_time} s...')
        with session.initiate():
            start_time = time.monotonic()
            while time.monotonic() - start_time < stream_time:
                # Compute the next block before there is room for it, so it is ready to write.
                block = source.next_block(block_size)
                if not _wait_for_space(session, block_size, stall_timeout):
                    print(f'No buffer space became available for {stall_timeout} s; stopping the stream.')
                    break
                session.write_arb_waveform(waveform_name, block, True)
                samples_written += block_size

            elapsed = time.monotonic() - start_time
            session.check_generation_status()

        streamed = samples_written - buffer_size
        print(f'Wrote {samples_written} samples ({samples_written * 4 / 1e6:.1f} MB), {streamed} of them while generating.')
        if elapsed > 0:
            print(f'Sustained {streamed / elapsed / 1e6:.2f} MS/s while generating (the device consumes {actual_iq_rate / 1e6} MS/s).')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Streams a continuously computed frequency sweep that is longer than the device memory, using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-b', '--sweep-bandwidth', default=6e6, type=float, help='Bandwidth, in Hz, covered by the frequency sweep.')
    parser.add_argument('-sp', '--sweep-period', default=0.1, type=float, help='Duration, in seconds, of one sweep.')
    parser.add_argument('-bs', '--buffer-size', default=4000000, type=int, help='Size, in samples, of the streaming buffer in device memory.')
    parser.add_argument('-blk', '--block-size', default=500000, type=int, help='Number of samples per write. Must divide the buffer size evenly.')
    parser.add_argument('-wt', '--write-timeout', default=10.0, type=float, help='Maximum time, in seconds, that a streaming write waits for buffer space.')
    parser.add_argument('-t', '--stream-time', default=10.0, type=float, help='Time, in seconds, to stream.')
    parser.add_argument('-st', '--stall-timeout', default=2.0, type=float, help='Stop if no buffer space becomes available for this many seconds.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.sweep_bandwidth, args.sweep_period, args.buffer_size,
            args.block_size, args.write_timeout, args.stream_time, args.stall_timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 6e6, 0.01, 40000, 10000, 1.0, 0.2, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--buffer-size', '100000', '--block-size', '25000', '--stream-time', '0.2', '--stall-timeout', '0.1',
                '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
