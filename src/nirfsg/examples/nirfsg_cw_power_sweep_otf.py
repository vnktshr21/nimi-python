'''Sweeps the power level of a continuous wave (CW) tone with on-the-fly (OTF) reconfiguration, without stopping the generation.

Python counterpart of the NI-RFSG .NET "Sweeps/PowerSweep(565xand5673)" example.

Background:
    On-the-fly (OTF) reconfiguration, also called a dynamic change, means writing a property
    while the device is in the Generation state. The new value is applied without aborting
    and re-initiating, so each sweep point is faster and the output is never switched off
    between points. power_level supports OTF reconfiguration on devices such as the
    PXIe-5644/5645/5646, PXIe-5650/5651/5652/5654, PXIe-5673/5673E, and
    PXIe-5830/5831/5832/5840/5841/5842. After each change, wait_until_settled() blocks until the
    output has settled at the new level. Devices without OTF support (for example the PXIe-5820)
    must use the regular sweep in nirfsg_cw_power_sweep.py, which aborts and re-initiates the
    generation at every point.

    A power sweep is commonly used to measure amplifier compression or receiver sensitivity.

    Before sweeping, the example calls get_max_settable_power(), which returns the highest
    power level the device can generate at the current frequency and configuration. Points
    above that limit are clipped, so the sweep does not fail partway through.

Steps:
    1. Open a session, select CW mode, and configure the frequency and the start power level.
    2. Query the maximum settable power and clip the sweep points to it.
    3. Start the generation once.
    4. For each power point, while generating: write power_level (OTF), wait for the output to
       settle, check the generation status, and dwell for the requested time.
'''

import argparse
import datetime
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, start_power_level, stop_power_level, number_of_points, dwell_time):
    power_levels = np.linspace(start_power_level, stop_power_level, number_of_points)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequency, power_levels[0])

        # Ask the driver for the highest power level that this configuration can reach.
        max_power_level = session.get_max_settable_power()
        print(f'Maximum settable power level at {frequency / 1e6} MHz: {max_power_level:.2f} dBm')
        if np.any(power_levels > max_power_level):
            print('Some sweep points exceed the maximum settable power level and will be clipped.')
            power_levels = np.minimum(power_levels, max_power_level)

        with session.initiate():
            for point, power_level in enumerate(power_levels):
                # OTF reconfiguration: write the new power level while the device is generating.
                session.power_level = power_level

                # Block until the output power has settled, or raise an error after 10 seconds.
                session.wait_until_settled(datetime.timedelta(seconds=10))

                # Report any error that occurred during the generation.
                session.check_generation_status()

                print(f'Point {point + 1}/{number_of_points}: generating {session.power_level:.2f} dBm at {frequency / 1e6} MHz')

                # Dwell at this power level. Replace this with your measurement, if needed.
                time.sleep(dwell_time)

        print('Sweep complete.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Sweeps the power level of a CW tone with NI-RFSG on-the-fly (OTF) reconfiguration, without stopping the generation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
