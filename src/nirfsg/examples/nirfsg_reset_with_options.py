'''Resets the session to its default settings while keeping selected items, such as the waveforms.

Background:
    NI-RFSG offers several ways to return a session to its default state:
      reset()               - aborts the generation, resets every property to its default value,
                              deletes the de-embedding tables, and clears all signal routes.
      reset_with_options()  - the same, except for the items selected with the
                              ResetWithOptionsStepsToOmit flags. Combine flags with |:
                                WAVEFORMS          keep the waveforms in device memory
                                SCRIPTS            keep the scripts
                                ROUTES             keep the signal routes
                                DEEMBEDDING_TABLES keep the de-embedding tables
                              NONE omits no steps.
    Keeping the waveforms avoids writing them again, which saves time when the waveforms are
    large. On devices shared with NI-RFSA, such as the PXIe-5830/5831/5832/5840/5841/5842,
    reset() also clears routes that an NI-RFSA session is using; NI recommends
    reset_with_options() with ROUTES in that case.

Steps:
    1. Change the power level from its default and write a waveform.
    2. Call reset_with_options() with the requested items to keep, and show that the power
       level is back to its default and whether the waveform still exists.
'''

import argparse
import functools
import nirfsg
import numpy as np
import operator
import sys

KEEP_CHOICES = [name for name in nirfsg.ResetWithOptionsStepsToOmit.__members__ if name != 'NONE']


def example(resource_name, options, keep):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        default_power_level = session.power_level
        session.power_level = default_power_level + 10.0
        session.write_arb_waveform('myWaveform', np.ones(1000, dtype=np.complex64), False)
        print(f'Before reset: power level {session.power_level} dBm, waveform exists: {session.check_if_waveform_exists("myWaveform")}')

        # NONE (0) is the starting value, so an empty list resets everything.
        steps_to_omit = functools.reduce(operator.or_, (nirfsg.ResetWithOptionsStepsToOmit[item] for item in keep), nirfsg.ResetWithOptionsStepsToOmit.NONE)
        session.reset_with_options(steps_to_omit)
        print(f'After reset_with_options({steps_to_omit!r}): power level {session.power_level} dBm (default {default_power_level} dBm), '
              f'waveform exists: {session.check_if_waveform_exists("myWaveform")}')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Resets an NI-RFSG session while keeping selected items.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-k', '--keep', default=['WAVEFORMS', 'ROUTES'], nargs='*', choices=KEEP_CHOICES, type=str.upper, help='Items to keep through the reset.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.keep)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, ['WAVEFORMS', 'ROUTES'])
    example('simulated5841', options, [])


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
