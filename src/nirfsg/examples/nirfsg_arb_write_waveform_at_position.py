'''Overwrites part of a waveform in device memory by moving the write position.

Background:
    write_arb_waveform() writes at the waveform's current write position, which normally
    advances past the data just written. set_arb_waveform_next_write_position() moves that
    position before the next write, so you can update part of a waveform that is already in
    memory without writing all of it again. The position is given by:
      RelativeTo.START_OF_WAVEFORM  - offset, in samples, from the first sample.
      RelativeTo.CURRENT_POSITION   - offset from the end of the previous write.
    This saves time when only a small region of a long waveform changes, for example a
    payload or a gap inside a frame.

    Rules:
      - Set power_level_type to PEAK, which is required to write the same waveform in more
        than one call.
      - Do not rewrite a waveform while it is generating; the output may be undefined.

    This example writes a continuous tone, then cuts evenly spaced gaps into it: the first gap
    is placed relative to the start of the waveform, and every following gap relative to the
    end of the previous one.

Steps:
    1. Configure ARB mode, PEAK power, the I/Q rate, and the signal bandwidth.
    2. Allocate and write the full tone.
    3. Overwrite regions with zeros using START_OF_WAVEFORM, then CURRENT_POSITION.
    4. Generate the modified waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, number_of_samples, first_gap_start, gap_length, gap_spacing, generation_time):
    waveform_name = 'gatedTone'
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # A tone at one tenth of the I/Q rate, with a whole number of cycles in the waveform.
        cycles = round(number_of_samples / 10)
        session.signal_bandwidth = 2 * cycles * actual_iq_rate / number_of_samples
        tone = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        session.allocate_arb_waveform(waveform_name, number_of_samples)
        session.write_arb_waveform(waveform_name, tone, False)

        gap = np.zeros(gap_length, dtype=np.complex64)
        # The first gap is positioned from the start of the waveform.
        session.set_arb_waveform_next_write_position(waveform_name, nirfsg.RelativeTo.START_OF_WAVEFORM, first_gap_start)
        session.write_arb_waveform(waveform_name, gap, False)
        gaps = [first_gap_start]
        # Each following gap starts gap_spacing samples after the end of the previous one.
        while gaps[-1] + gap_length + gap_spacing + gap_length <= number_of_samples:
            session.set_arb_waveform_next_write_position(waveform_name, nirfsg.RelativeTo.CURRENT_POSITION, gap_spacing)
            session.write_arb_waveform(waveform_name, gap, False)
            gaps.append(gaps[-1] + gap_length + gap_spacing)
        print(f'Wrote {len(gaps)} gaps of {gap_length} samples starting at samples {gaps}.')

        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Overwrites parts of a stored waveform by moving the NI-RFSG write position.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-s', '--number-of-samples', default=100000, type=int, help='Number of samples in the waveform.')
    parser.add_argument('-gs', '--first-gap-start', default=10000, type=int, help='Sample at which the first gap starts.')
    parser.add_argument('-gl', '--gap-length', default=5000, type=int, help='Length of each gap in samples.')
    parser.add_argument('-gp', '--gap-spacing', default=20000, type=int, help='Number of samples between the end of one gap and the start of the next.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_samples, args.first_gap_start, args.gap_length, args.gap_spacing, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 20000, 1000, 1000, 4000, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
