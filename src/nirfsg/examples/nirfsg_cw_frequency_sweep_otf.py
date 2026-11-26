'''Sweeps the frequency of a continuous wave (CW) tone with on-the-fly (OTF) reconfiguration, without stopping the generation.

Python counterpart of the NI-RFSG .NET "Sweeps/FrequencySweep(565xand5673)" example.

Background:
    On-the-fly (OTF) reconfiguration, also called a dynamic change, means writing a property
    while the device is in the Generation state. On devices such as the PXIe-5644/5645/5646,
    PXIe-5650/5651/5652/5653/5654, PXIe-5673/5673E, and PXIe-5830/5831/5832/5840/5841/5842, the
    frequency and power_level properties support it. This lets you step a tone across a band
    without stopping and restarting the generation for every point. After each change, call
    wait_until_settled() so that the output has settled before the next part of your test,
    such as measuring the DUT, runs.

    How long the device waits for the frequency to settle is controlled by:
      - frequency_settling_units: how to interpret frequency_settling.
          TIME_AFTER_LOCK - seconds to wait after the LO PLL locks.
          TIME_AFTER_IO   - seconds to wait after the frequency is written to the hardware.
          PPM             - wait until the frequency is within this many parts per million.
      - frequency_settling: the settling value in those units.

    Devices without OTF support (for example the PXIe-5820) must instead abort the
    generation, change the frequency, and initiate again, as in nirfsg_cw_frequency_sweep.py.

Steps:
    1. Open a session, select CW mode, and configure the start frequency and the power level.
    2. Configure the frequency settling criteria.
    3. Start the generation once.
    4. For each frequency point, while generating: write the frequency property (OTF), wait for
       the output to settle, check the generation status, and dwell for the requested time.
'''

import argparse
import datetime
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, start_frequency, stop_frequency, number_of_points, power_level, dwell_time, settling_time):
    # The frequency points to visit, evenly spaced and including both end points.
    frequencies = np.linspace(start_frequency, stop_frequency, number_of_points)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequencies[0], power_level)

        # Wait settling_time seconds after the LO locks before the device reports that it has settled.
        session.frequency_settling_units = nirfsg.FrequencySettlingUnits.TIME_AFTER_LOCK
        session.frequency_settling = settling_time

        with session.initiate():
            for point, frequency in enumerate(frequencies):
                # OTF reconfiguration: write the new frequency while the device is generating.
                session.frequency = frequency

                # Block until the output has settled, or raise an error after 10 seconds.
                session.wait_until_settled(datetime.timedelta(seconds=10))

                # Report any error that occurred during the generation.
                session.check_generation_status()

                print(f'Point {point + 1}/{number_of_points}: generating {session.frequency / 1e6:.3f} MHz at {power_level} dBm')

                # Dwell at this frequency. Replace this with your measurement, if needed.
                time.sleep(dwell_time)

        print('Sweep complete.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Sweeps the frequency of a CW tone with NI-RFSG on-the-fly (OTF) reconfiguration, without stopping the generation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-start', '--start-frequency', default=1e9, type=float, help='Start frequency in Hz.')
    parser.add_argument('-stop', '--stop-frequency', default=2e9, type=float, help='Stop frequency in Hz.')
    parser.add_argument('-pts', '--number-of-points', default=11, type=int, help='Number of frequency points in the sweep.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-d', '--dwell-time', default=0.5, type=float, help='Time, in seconds, to dwell at each frequency point.')
    parser.add_argument('-st', '--settling-time', default=100e-6, type=float, help='Time, in seconds, to wait after the LO locks before the output is considered settled.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.start_frequency, args.stop_frequency, args.number_of_points, args.power_level, args.dwell_time, args.settling_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, 2e9, 5, -10.0, 0.01, 100e-6)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-points', '3', '--dwell-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
