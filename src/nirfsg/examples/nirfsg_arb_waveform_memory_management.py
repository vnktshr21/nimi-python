'''Lists, checks, and clears the arbitrary waveforms stored in device memory.

Background:
    Waveforms written with write_arb_waveform() stay in the onboard memory of the device until
    they are cleared. To manage them:
      get_all_named_waveform_names()   the names of all stored waveforms
      check_if_waveform_exists(name)   whether a waveform is stored
      clear_arb_waveform(name)         removes one waveform and the properties assigned to it
      clear_all_arb_waveforms()        removes every waveform
    query_arb_waveform_capabilities() returns the limits that apply when writing waveforms:
    the maximum number of waveforms, the waveform quantum, and the minimum and maximum
    waveform sizes. Clear waveforms that are no longer needed to free memory for new ones.

Steps:
    1. Print the waveform capabilities of the device.
    2. Write several waveforms and list them.
    3. Clear one waveform, check that it no longer exists, then clear all waveforms.
'''

import argparse
import nirfsg
import numpy as np
import sys


def _print_waveforms(session, names):
    print(f'  stored waveforms: {session.get_all_named_waveform_names()}')
    print('  ' + ', '.join(f'{name} exists: {session.check_if_waveform_exists(name)}' for name in names))


def example(resource_name, options, number_of_waveforms, number_of_samples):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM

        max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size = session.query_arb_waveform_capabilities()
        print(f'Up to {max_number_waveforms} waveforms of {min_waveform_size} to {max_waveform_size} samples, in multiples of {waveform_quantum} samples.')

        names = [f'waveform{i}' for i in range(number_of_waveforms)]
        for i, name in enumerate(names):
            phase = 2 * np.pi * (i + 1) * np.arange(number_of_samples) / number_of_samples
            session.write_arb_waveform(name, np.exp(1j * phase).astype(np.complex64), False)
        print(f'After writing {number_of_waveforms} waveforms:')
        _print_waveforms(session, names)

        session.clear_arb_waveform(names[0])
        print(f'After clear_arb_waveform("{names[0]}"):')
        _print_waveforms(session, names)

        session.clear_all_arb_waveforms()
        print('After clear_all_arb_waveforms():')
        _print_waveforms(session, names)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Lists, checks, and clears the arbitrary waveforms stored in NI-RFSG device memory.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-w', '--number-of-waveforms', default=3, type=int, help='Number of waveforms to write.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.number_of_waveforms, args.number_of_samples)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 3, 1000)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
