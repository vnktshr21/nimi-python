'''Reports the external calibration and self-calibration status of an NI RF signal generator.

Background:
    NI RF signal generators are calibrated in two ways:
      - External calibration is done by a metrology lab at a recommended interval.
        get_ext_cal_last_date_and_time() returns the date of the last external calibration, and
        external_calibration_recommended_interval the interval, as a timedelta.
      - Self-calibration is done by the device itself, with its onboard references, to
        compensate for drift since the external calibration.
        get_self_cal_last_date_and_time() and get_self_calibration_temperature() return the
        date and the device temperature of the last self-calibration.
    Self-calibration should be repeated when the device temperature (device_temperature) has
    moved away from the temperature at the last self-calibration. Refer to the specifications
    of your device for the allowed temperature difference.

    For multi-module devices, the module parameter (PRIMARY_MODULE, AWG, or LO) selects which
    module to query.

Steps:
    1. Read and print the external calibration date and the recommended interval, and when the
       next external calibration is due.
    2. Read and print the self-calibration date and temperature, and compare the temperature
       with the current device temperature.
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


def example(resource_name, options, max_temperature_difference):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        last_external_calibration = _read_date(session.get_ext_cal_last_date_and_time)
        interval = session.external_calibration_recommended_interval
        # NI-RFSG counts 30.4167 days per month when it converts the interval to a timedelta.
        print(f'Last external calibration        : {last_external_calibration or "not available"}')
        print(f'Recommended calibration interval : {round(interval.days / 30.4167)} months')
        if last_external_calibration is not None:
            print(f'Next external calibration due    : {last_external_calibration + interval}')

        self_cal_temperature = session.get_self_calibration_temperature(nirfsg.Module.PRIMARY_MODULE)
        current_temperature = session.device_temperature
        difference = current_temperature - self_cal_temperature
        last_self_calibration = _read_date(lambda: session.get_self_cal_last_date_and_time(nirfsg.Module.PRIMARY_MODULE))
        print(f'Last self-calibration            : {last_self_calibration or "not available"}')
        print(f'Self-calibration temperature     : {self_cal_temperature:.2f} C')
        print(f'Current device temperature       : {current_temperature:.2f} C (difference {difference:+.2f} C)')
        if abs(difference) > max_temperature_difference:
            print(f'The temperature has changed by more than {max_temperature_difference} C since the last self-calibration; consider running a self-calibration.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Reports the external calibration and self-calibration status of an NI-RFSG device.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-d', '--max-temperature-difference', default=5.0, type=float, help='Temperature change, in degrees Celsius, since the last self-calibration above which a new self-calibration is suggested.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.max_temperature_difference)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 5.0)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
