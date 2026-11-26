'''Self-calibrates only a range of frequencies and power levels, which is faster than a full self-calibration.

Background:
    self_calibrate_range() self-calibrates every configuration within the given frequency and
    peak power level limits. Because it covers only the settings that your test uses, it is
    faster than self_cal(). Its results differ from those of self_cal():
      - They are kept only until the system restarts or clear_self_calibrate_range() is called.
      - They do not update the stored self-calibration date and temperature.
    Some steps can be skipped to save more time, at the cost of the performance that the step
    calibrates. Combine SelfCalibrateRangeStepsToOmit flags with the | operator:
      LO_SELF_CAL           - LO power level
      POWER_LEVEL_ACCURACY  - power level accuracy
      RESIDUAL_LO_POWER     - residual LO power (carrier leakage)
      IMAGE_SUPPRESSION     - residual sideband image
      SYNTHESIZER_ALIGNMENT - LO PLL (VCO) alignment
    OMIT_NONE runs every step, which NI recommends for the best results.

    As for self_cal(), disconnect any signal from the RF IN and IQ IN ports, and note that an
    NI-RFSA session on the same device cannot access the hardware while the calibration runs.

Steps:
    1. Combine the requested steps to omit.
    2. Run self_calibrate_range() for the requested frequency and power ranges.
    3. Optionally discard the range calibration with clear_self_calibrate_range().
'''

import argparse
import functools
import nirfsg
import operator
import sys

OMIT_STEPS = [name for name in nirfsg.SelfCalibrateRangeStepsToOmit.__members__ if name != 'OMIT_NONE']


def example(resource_name, options, min_frequency, max_frequency, min_power_level, max_power_level, steps_to_omit, clear_range):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # OMIT_NONE (0) is the starting value, so an empty list runs every step.
        omit = functools.reduce(operator.or_, (nirfsg.SelfCalibrateRangeStepsToOmit[step] for step in steps_to_omit), nirfsg.SelfCalibrateRangeStepsToOmit.OMIT_NONE)
        print(f'Self-calibrating {min_frequency / 1e6} MHz to {max_frequency / 1e6} MHz, {min_power_level} dBm to {max_power_level} dBm, omitting {omit!r}...')
        session.self_calibrate_range(omit, min_frequency, max_frequency, min_power_level, max_power_level)
        print('Range self-calibration complete. It applies until the system restarts or the data is cleared.')
        if clear_range:
            session.clear_self_calibrate_range()
            print('Range self-calibration data cleared; the stored self-calibration is used again.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Self-calibrates a range of frequencies and power levels of an NI-RFSG device.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-fmin', '--min-frequency', default=1e9, type=float, help='Minimum frequency, in Hz.')
    parser.add_argument('-fmax', '--max-frequency', default=2e9, type=float, help='Maximum frequency, in Hz.')
    parser.add_argument('-pmin', '--min-power-level', default=-30.0, type=float, help='Minimum peak power level, in dBm.')
    parser.add_argument('-pmax', '--max-power-level', default=0.0, type=float, help='Maximum peak power level, in dBm.')
    parser.add_argument('-o', '--omit', default=[], nargs='*', choices=OMIT_STEPS, type=str.upper, help='Calibration steps to skip.')
    parser.add_argument('-cl', '--clear-range', action='store_true', help='Clear the range self-calibration data afterwards.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.min_frequency, args.max_frequency, args.min_power_level, args.max_power_level, args.omit, args.clear_range)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, 2e9, -30.0, 0.0, ['LO_SELF_CAL', 'IMAGE_SUPPRESSION'], True)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--clear-range', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
