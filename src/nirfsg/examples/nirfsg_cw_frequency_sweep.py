'''Sweeps the frequency of a continuous wave (CW) tone by stopping, reconfiguring, and restarting the generation at each point.

Python counterpart of the NI-RFSG .NET "Sweeps/FrequencySweep" example.

Background:
    The regular way to change a setting is to make the change in the Configuration state: abort
    the generation, set the new value, and initiate again. initiate() applies the new
    configuration and waits for the output to settle before it returns. This works on every
    device, including devices that cannot change the frequency while generating (for example
    the PXIe-5820). Each point takes longer than with on-the-fly (OTF) reconfiguration, and
    the output is off briefly between points. For the OTF version of this sweep, refer to
    nirfsg_cw_frequency_sweep_otf.py.

Steps:
    1. Open a session, select CW mode, and configure the start frequency and the power level.
    2. For each frequency point: set the frequency, start the generation, check its status,
       dwell for the requested time, and abort.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, start_frequency, stop_frequency, number_of_points, power_level, dwell_time):
    frequencies = np.linspace(start_frequency, stop_frequency, number_of_points)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequencies[0], power_level)

        for point, frequency in enumerate(frequencies):
            # Configuration state: the new value is applied by the next initiate().
            session.frequency = frequency
            # initiate() applies the configuration and waits for the output to settle.
            # Leaving the "with" block aborts the generation before the next point.
            with session.initiate():
                session.check_generation_status()
                print(f'Point {point + 1}/{number_of_points}: generating {session.frequency / 1e6:.3f} MHz at {power_level} dBm')
                time.sleep(dwell_time)

        print('Sweep complete.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Sweeps the frequency of a CW tone by aborting, reconfiguring, and re-initiating NI-RFSG at each point.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-start', '--start-frequency', default=1e9, type=float, help='Start frequency in Hz.')
    parser.add_argument('-stop', '--stop-frequency', default=2e9, type=float, help='Stop frequency in Hz.')
    parser.add_argument('-pts', '--number-of-points', default=11, type=int, help='Number of frequency points in the sweep.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-d', '--dwell-time', default=0.5, type=float, help='Time, in seconds, to dwell at each frequency point.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.start_frequency, args.stop_frequency, args.number_of_points, args.power_level, args.dwell_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, 2e9, 5, -10.0, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-points', '3', '--dwell-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
