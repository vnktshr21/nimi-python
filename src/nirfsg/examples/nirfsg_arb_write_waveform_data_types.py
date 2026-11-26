'''Writes the same waveform in every data format that write_arb_waveform() accepts.

Background:
    write_arb_waveform(waveform_name, waveform_data_array, more_data_pending) chooses the driver
    call from the numpy dtype of waveform_data_array. Three formats are supported:

      numpy.complex128  Complex double-precision samples (16 bytes per I/Q sample).
                        Each sample is I + jQ, with |I + jQ| <= 1.0.
      numpy.complex64   Complex single-precision samples (8 bytes per I/Q sample). Same
                        scaling as complex128 with half the host memory. Use it for long
                        waveforms when double precision is not needed.
      numpy.int16       Interleaved 16-bit integers (4 bytes per I/Q sample):
                        [I0, Q0, I1, Q1, ...], so the array has 2 * N elements for N samples.
                        Full scale is +/-32767. This is the usual format of waveforms
                        stored in binary files.

    Any other dtype, or data that is not a numpy array (such as a Python list), raises
    TypeError.

    int16 data requires power_level_type to be PEAK, so this example uses PEAK for every format:
    power_level is the power of a full-scale sample (magnitude 1.0 for complex data, 32767 for
    int16).

    This example writes the same tone at a frequency offset from the carrier once in each
    requested format, as separate named waveforms. It then generates each one in turn. Because
    the data and the power level are the same, the RF output is the same for every format.

Steps:
    1. Configure the RF settings, ARB mode, PEAK power level type, I/Q rate, and signal bandwidth.
    2. Build the tone as complex samples and convert it to each requested format.
    3. Write each version under its own name.
    4. Select and generate each waveform for the requested time.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

DATA_TYPES = ['COMPLEX128', 'COMPLEX64', 'INT16']


def _convert(iq_samples, data_type):
    # Convert complex samples with a magnitude of at most 1.0 to the requested format.
    if data_type == 'COMPLEX128':
        return iq_samples.astype(np.complex128)
    if data_type == 'COMPLEX64':
        return iq_samples.astype(np.complex64)
    # Interleave I and Q, and scale to the int16 full-scale value.
    interleaved = np.empty(2 * len(iq_samples), dtype=np.int16)
    interleaved[0::2] = np.round(iq_samples.real * 32767)
    interleaved[1::2] = np.round(iq_samples.imag * 32767)
    return interleaved


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, data_types, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        # Required for int16 data.
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        # One tone with a whole number of cycles in the waveform, so it repeats seamlessly.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        tone = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples)

        waveform_names = []
        for data_type in data_types:
            data = _convert(tone, data_type)
            name = f'tone{data_type.capitalize()}'
            # The dtype of the array selects the format. For int16, len(data) is 2 * number_of_samples.
            session.write_arb_waveform(name, data, False)
            waveform_names.append(name)
            print(f'Wrote "{name}": dtype {data.dtype}, {len(data)} array elements, {data.nbytes} bytes')

        for name in waveform_names:
            session.select_arb_waveform(name)
            print(f'Generating "{name}" at {power_level} dBm peak power for {generation_time} s...')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Writes the same waveform as complex128, complex64, and interleaved int16 data using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of I/Q samples in each waveform.')
    parser.add_argument('-dt', '--data-types', default=DATA_TYPES, nargs='+', choices=DATA_TYPES, type=str.upper, help='Data formats to write.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.data_types, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, DATA_TYPES, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
