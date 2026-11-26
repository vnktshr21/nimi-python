'''Writes a large arbitrary waveform to device memory in blocks.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/LargeArbitraryWaveformGeneration" and
"ArbitraryWaveforms/WaveformFromBinaryFile" examples.

Background:
    Long waveforms, such as a wideband chirp or a waveform read from a large file, can use
    more host memory than you want to hold at once. Instead you can:
      1. Reserve the full waveform in device memory with allocate_arb_waveform().
      2. Write it in consecutive blocks with write_arb_waveform(). Each write continues
         where the previous one stopped. Set more_data_pending to True for every block
         except the last.
    Only one block is held in host memory at a time. The waveform must be complete before the
    generation starts; to keep writing while generating, use streaming instead.

    To write a waveform in more than one block, set power_level_type to PEAK. With AVERAGE
    power, the driver needs the whole waveform to compute its average power before it can
    scale the data. With PEAK, a sample magnitude of 1.0 corresponds to power_level.

    This example computes a linear frequency chirp block by block and writes each block as it
    is computed.

Steps:
    1. Configure ARB mode, PEAK power, the I/Q rate, and the signal bandwidth.
    2. Check the waveform size against the device capabilities and allocate the waveform.
    3. Compute and write the chirp one block at a time.
    4. Start the generation and monitor it for the requested time.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def _chirp(start_sample, number_of_samples, total_samples, iq_rate, chirp_bandwidth):
    # Linear chirp from -chirp_bandwidth/2 to +chirp_bandwidth/2 across total_samples. Any
    # sub-range of it can be computed on its own, which keeps the memory per block small.
    t = np.arange(start_sample, start_sample + number_of_samples) / iq_rate
    duration = total_samples / iq_rate
    phase = 2 * np.pi * (-chirp_bandwidth / 2 * t + chirp_bandwidth / (2 * duration) * t ** 2)
    return np.exp(1j * phase).astype(np.complex64)


def example(resource_name, options, frequency, power_level, iq_rate, chirp_bandwidth, number_of_samples, block_size, generation_time):
    waveform_name = 'largeWaveform'

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        # Required to write the waveform in several blocks.
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = chirp_bandwidth

        _, waveform_quantum, min_waveform_size, max_waveform_size = session.query_arb_waveform_capabilities()
        if number_of_samples % waveform_quantum != 0 or not min_waveform_size <= number_of_samples <= max_waveform_size:
            raise ValueError(f'The waveform size must be a multiple of {waveform_quantum} between {min_waveform_size} and {max_waveform_size} samples.')

        # Reserve the full waveform in device memory.
        session.allocate_arb_waveform(waveform_name, number_of_samples)

        for start in range(0, number_of_samples, block_size):
            samples_in_block = min(block_size, number_of_samples - start)
            block = _chirp(start, samples_in_block, number_of_samples, actual_iq_rate, chirp_bandwidth)
            more_data_pending = start + samples_in_block < number_of_samples
            session.write_arb_waveform(waveform_name, block, more_data_pending)
        print(f'Wrote {number_of_samples} samples in blocks of {block_size} samples ({block_size * 8 / 1e6:.1f} MB of host memory per block).')

        print(f'Generating a {chirp_bandwidth / 1e6} MHz chirp lasting {number_of_samples / actual_iq_rate * 1e3:.3f} ms per repetition...')
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Writes a large chirp waveform to device memory in blocks using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=100e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-b', '--chirp-bandwidth', default=40e6, type=float, help='Bandwidth, in Hz, swept by the chirp.')
    parser.add_argument('-s', '--number-of-samples', default=10000000, type=int, help='Total number of samples in the waveform.')
    parser.add_argument('-bs', '--block-size', default=1000000, type=int, help='Number of samples written per call to write_arb_waveform().')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.chirp_bandwidth, args.number_of_samples, args.block_size, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 100e6, 40e6, 100000, 30000, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-samples', '200000', '--block-size', '50000', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
