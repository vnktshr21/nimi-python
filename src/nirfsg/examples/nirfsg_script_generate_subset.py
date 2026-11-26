'''Generates parts of a stored waveform with the script instruction "generate <waveform> subset(<start>, <length>)".

Based on the "generate" topic of the NI-RFSG user manual.

Background:
    The generate instruction plays a stored waveform. With subset(start, length), it plays only
    length samples of the waveform, starting at sample start:

        generate myWfm subset(40, 12)

    One stored waveform can then hold several segments that the script plays in any order,
    which saves device memory and the number of waveforms. Markers placed on a subset are
    relative to the start of the subset. The start position and the length must be multiples
    of 4 samples.

    This example stores one waveform whose first half is a tone above the carrier and whose
    second half is a tone below it. The script plays the first half three times, then the
    second half once, and repeats; marker0 marks each start of the second half.

Steps:
    1. Configure the RF settings, SCRIPT mode, I/Q rate, and signal bandwidth.
    2. Write the two-segment waveform and the script.
    3. Generate the script.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, segment_samples, marker_terminal, generation_time):
    if segment_samples % 4 != 0:
        raise ValueError('The segment length must be a multiple of 4 samples.')
    script = f'''script subsets
    repeat forever
        repeat 3
            generate segments subset(0, {segment_samples})
        end repeat
        generate segments subset({segment_samples}, {segment_samples}) marker0(0)
    end repeat
end script'''

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        # Each segment holds a whole number of tone cycles, so segments join without a phase jump.
        cycles = round(tone_offset * segment_samples / actual_iq_rate)
        phase = 2 * np.pi * cycles * np.arange(segment_samples) / segment_samples
        segments = np.concatenate([np.exp(1j * phase), np.exp(-1j * phase)]).astype(np.complex64)
        session.write_arb_waveform('segments', segments, False)
        session.write_script(script)
        session.markers[0].exported_marker_event_output_terminal = marker_terminal

        print(script)
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates parts of a stored waveform with the NI-RFSG script instruction "generate ... subset".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-s', '--segment-samples', default=4000, type=int, help='Length of each segment in samples. Must be a multiple of 4.')
    parser.add_argument('-m', '--marker-terminal', default='', help='Terminal to export marker0 to, for example PFI0. Leave empty to not export.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.segment_samples, args.marker_terminal, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 4000, 'PXI_Trig0', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
