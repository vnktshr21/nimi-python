'''Performs a full self-calibration of an NI RF signal generator.

Background:
    Self-calibration uses the device's onboard references to compensate for drift since the
    last external calibration, for example after the device temperature has changed.
    self_cal() calibrates the device and its associated modules, and stores the new
    calibration data in the onboard nonvolatile memory, so it is used by every later session.
    It also updates the self-calibration date and temperature.

    Before calibrating:
      - Let the device warm up, so that the calibration is done at the operating temperature.
      - If an NI-RFSA session is open on the same device, it cannot access the hardware while
        the calibration runs. Close and reopen it afterwards to use the new calibration data.
    Self-calibration can take several minutes. To calibrate only the frequencies and power
    levels that you use, which is faster, use self_calibrate_range() instead.

Steps:
    1. Print the date and temperature of the last self-calibration.
    2. Run self_cal().
    3. Print the new date and temperature.
'''

import argparse
import nirfsg
import sys


def _read_date(read):
    # Simulated devices do not return a valid calibration date, which raises ValueError.
    try:
        return read()
    except ValueError:
        return None


def example(resource_name, options):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        print(f'Before: last self-calibration {_read_date(session.get_self_cal_last_date_and_time) or "not available"}, at {session.get_self_calibration_temperature():.2f} C')
        print('Running self_cal(). This can take several minutes...')
        session.self_cal()
        print(f'After : last self-calibration {_read_date(session.get_self_cal_last_date_and_time) or "not available"}, at {session.get_self_calibration_temperature():.2f} C')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Performs a full self-calibration of an NI-RFSG device.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
