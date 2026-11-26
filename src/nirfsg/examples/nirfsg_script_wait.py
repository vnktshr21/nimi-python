'''Pauses a script for a fixed number of samples, with the script instruction "wait <samples>".

Based on the "wait" topic of the NI-RFSG user manual.

Background:
    "wait <N>" pauses the script for N samples before the next instruction. During the wait, the
    device holds the last generated sample at the output. A finite wait can also emit markers:

        wait 256 marker1(0, 128) marker0(64)

    Waits are useful for gaps of a known length, because they need no device memory: a long
    off time between pulses does not need a long waveform of zeros. To make the output silent
    during the wait, end the preceding waveform with a zero sample; otherwise the held sample
    (a constant I/Q value) appears as an unmodulated carrier.

    This example generates a pulse train: a short tone burst followed by a wait, repeated
    forever. marker0 marks the start of each pulse and marker1 the start of each off time.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Write the pulse waveform (ending in zeros) and the script.
    3. Export the markers and generate.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, pulse_samples, off_samples, marker0_terminal, marker1_terminal, generation_time):
    script = f'''script pulseTrain
    repeat forever
        generate pulse marker0(0)
        wait {off_samples} marker1(0)
    end repeat
end script'''

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        # power_level is the power during the pulse.
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 0.8 * actual_iq_rate

        # An unmodulated burst whose last 4 samples are zero, so the held sample is zero during the wait.
        pulse = np.ones(pulse_samples, dtype=np.complex64)
        pulse[-4:] = 0
        session.write_arb_waveform('pulse', pulse, False)
        session.write_script(script)
        session.markers[0].exported_marker_event_output_terminal = marker0_terminal
        session.markers[1].exported_marker_event_output_terminal = marker1_terminal

        period = (pulse_samples + off_samples) / actual_iq_rate
        print(script)
        print(f'Pulse {pulse_samples / actual_iq_rate * 1e6:.1f} us every {period * 1e6:.1f} us ({100 * pulse_samples / (pulse_samples + off_samples):.1f}% duty cycle)')
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Pauses an NI-RFSG script for a fixed number of samples with "wait", to generate a pulse train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, during the pulse.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-ps', '--pulse-samples', default=1000, type=int, help='Length of the pulse in samples.')
    parser.add_argument('-os', '--off-samples', default=9000, type=int, help='Length of the wait between pulses, in samples.')
    parser.add_argument('-m0', '--marker0-terminal', default='', help='Terminal to export the pulse-start marker (marker0) to. Leave empty to not export.')
    parser.add_argument('-m1', '--marker1-terminal', default='', help='Terminal to export the off-time marker (marker1) to. Leave empty to not export.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.pulse_samples, args.off_samples, args.marker0_terminal, args.marker1_terminal, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1000, 9000, 'PXI_Trig0', 'PXI_Trig1', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
