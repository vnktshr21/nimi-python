'''Sweeps the power level of a continuous wave (CW) tone by stopping, reconfiguring, and restarting the generation at each point.

Python counterpart of the NI-RFSG .NET "Sweeps/PowerSweep" example.

Background:
    The regular way to change a setting is to make the change in the Configuration state: abort
    the generation, set the new value, and initiate again. initiate() applies the new
    configuration and waits for the output to settle before it returns. This works on every
    device, including devices that cannot change power_level while generating (for example
    the PXIe-5820). Each point takes longer than with on-the-fly (OTF) reconfiguration, and
    the output is off briefly between points. For the OTF version of this sweep, refer to
    nirfsg_cw_power_sweep_otf.py.

    Before sweeping, the example calls get_max_settable_power(), which returns the highest
    power level the device can generate at the current frequency and configuration. Points
    above that limit are clipped, so the sweep does not fail partway through.

Steps:
    1. Open a session, select CW mode, and configure the frequency and the start power level.
    2. Query the maximum settable power and clip the sweep points to it.
    3. For each power point: set power_level, start the generation, check its status, dwell
       for the requested time, and abort.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, start_power_level, stop_power_level, number_of_points, dwell_time):
    power_levels = np.linspace(start_power_level, stop_power_level, number_of_points)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequency, power_levels[0])

        max_power_level = session.get_max_settable_power()
        print(f'Maximum settable power level at {frequency / 1e6} MHz: {max_power_level:.2f} dBm')
        if np.any(power_levels > max_power_level):
            print('Some sweep points exceed the maximum settable power level and will be clipped.')
            power_levels = np.minimum(power_levels, max_power_level)

        for point, power_level in enumerate(power_levels):
            # Configuration state: the new value is applied by the next initiate().
            session.power_level = power_level
            # initiate() applies the configuration and waits for the output to settle.
            # Leaving the "with" block aborts the generation before the next point.
            with session.initiate():
                session.check_generation_status()
                print(f'Point {point + 1}/{number_of_points}: generating {session.power_level:.2f} dBm at {frequency / 1e6} MHz')
                time.sleep(dwell_time)

        print('Sweep complete.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Sweeps the power level of a CW tone by aborting, reconfiguring, and re-initiating NI-RFSG at each point.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-start', '--start-power-level', default=-40.0, type=float, help='Start power level in dBm.')
    parser.add_argument('-stop', '--stop-power-level', default=0.0, type=float, help='Stop power level in dBm.')
    parser.add_argument('-pts', '--number-of-points', default=21, type=int, help='Number of power points in the sweep.')
    parser.add_argument('-d', '--dwell-time', default=0.5, type=float, help='Time, in seconds, to dwell at each power point.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.start_power_level, args.stop_power_level, args.number_of_points, args.dwell_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -40.0, 0.0, 5, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-points', '3', '--dwell-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
