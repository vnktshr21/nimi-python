'''Performs thermal correction during a long generation when the device temperature drifts.

Python counterpart of the NI-RFSG .NET "Utilities/ThermalCorrection" example.

Background:
    The gain and the I/Q impairments of an RF signal generator drift with temperature. NI-RFSG
    corrects for temperature automatically whenever settings change. When the same signal is
    generated for a long time without any settings change, the device temperature can drift and
    nothing triggers a correction. In that case, call perform_thermal_correction() while the
    device is in the Generation state. It re-applies the temperature-dependent corrections for
    the current temperature.

    This example reads device_temperature periodically and calls perform_thermal_correction()
    whenever the temperature has moved by more than a threshold since the last correction.
    Calling it only when needed avoids unnecessary small output transients.

Steps:
    1. Configure the RF settings and select CW mode.
    2. Start the generation.
    3. Periodically read the temperature and perform thermal correction when it has drifted.
'''

import argparse
import nirfsg
import sys
import time


def example(resource_name, options, frequency, power_level, temperature_threshold, check_interval, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.CW

        with session.initiate():
            last_correction_temperature = session.device_temperature
            print(f'Generation started at {last_correction_temperature:.2f} C. Correcting when the temperature drifts by more than {temperature_threshold} C.')
            corrections = 0
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                temperature = session.device_temperature
                drift = temperature - last_correction_temperature
                if abs(drift) > temperature_threshold:
                    # The device must be generating when this method is called.
                    session.perform_thermal_correction()
                    corrections += 1
                    print(f'Temperature {temperature:.2f} C (drift {drift:+.2f} C): thermal correction performed.')
                    last_correction_temperature = temperature
                time.sleep(check_interval)
            print(f'Generation finished after {corrections} thermal corrections.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Performs thermal correction during a long NI-RFSG generation when the device temperature drifts.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-th', '--temperature-threshold', default=1.0, type=float, help='Temperature drift, in degrees Celsius, that triggers a thermal correction.')
    parser.add_argument('-i', '--check-interval', default=5.0, type=float, help='Time, in seconds, between temperature checks.')
    parser.add_argument('-t', '--generation-time', default=60.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.temperature_threshold, args.check_interval, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    # A negative threshold forces a thermal correction at every check.
    example('simulated5841', options, 1e9, -10.0, -1.0, 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--check-interval', '0.01', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
