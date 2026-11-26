'''Runs the self-test of an NI RF signal generator.

Background:
    self_test() runs a quick series of tests that check that the device is powered up and
    responding. It does not test external connections, and it is not a calibration or a full
    functional test. The device must be in the Configuration state, so run it before
    initiating a generation.

    self_test() returns nothing when the test passes. When it fails, it raises
    nirfsg.errors.SelfTestError, whose code and message attributes describe the failure.
    Running the self-test at the start of a test sequence catches a missing or faulty
    instrument before any measurement is made.

Steps:
    1. Open a session and run the self-test.
    2. Report the result.
'''

import argparse
import nirfsg
import sys


def example(resource_name, options):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        try:
            session.self_test()
            print(f'{session.instrument_model} ({resource_name}): self-test passed.')
        except nirfsg.errors.SelfTestError as e:
            print(f'{session.instrument_model} ({resource_name}): self-test failed with code {e.code}: {e.message}')
            raise


def _main(argsv):
    parser = argparse.ArgumentParser(description='Runs the self-test of an NI-RFSG device.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
