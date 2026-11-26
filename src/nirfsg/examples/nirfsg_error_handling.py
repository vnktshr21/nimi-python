'''Shows how NI-RFSG reports errors and warnings, and how to handle them in Python.

Background:
    The nirfsg module reports problems in these ways:
      - nirfsg.Error is the base class of every exception that the module raises. Catch it to
        handle any NI-RFSG failure.
      - nirfsg.errors.DriverError (a subclass of nirfsg.Error) is raised when the driver
        returns a negative status code. It has these attributes:
          code         the numeric status code, for example -1074135027
          description  the full error message from the driver
        error_message() converts any status code into its message text.
      - nirfsg.errors.SelfTestError is raised by self_test() when the self-test fails. It has
        code and message attributes.
      - nirfsg.DriverWarning is issued through the standard Python warnings module when the
        driver returns a positive (warning) status code. Warnings do not stop the program,
        but you can turn them into exceptions with warnings.simplefilter('error', ...).
      - Some mistakes are caught by the Python bindings before anything is sent to the
        driver, and raise standard Python exceptions:
          TypeError      wrong argument type, such as a waveform with an unsupported dtype
          ValueError     inconsistent arguments, such as mismatched array shapes
          AttributeError setting a property that does not exist (sessions reject unknown
                         attribute names, which catches typos)
    Errors that occur while the device is generating are reported the next time you call
    check_generation_status(), so call it periodically during long generations.

    This example triggers each kind of error on purpose and handles it. The invalid-session
    case always uses a simulated device, so it never affects your hardware.

Steps:
    1. Open a session with an invalid simulated model and catch the DriverError.
    2. Open a valid session, write a read-only property, catch the DriverError, and look up
       its message with error_message().
    3. Trigger TypeError, ValueError, and AttributeError in the Python bindings.
    4. Run the self-test and handle SelfTestError.
    5. Turn driver warnings into exceptions for a block of code.
'''

import argparse
import nirfsg
import numpy as np
import sys
import warnings


def _report(scenario, exception):
    if isinstance(exception, nirfsg.errors.DriverError):
        print(f'{scenario}: DriverError {exception.code}: {exception.description}')
    else:
        print(f'{scenario}: {type(exception).__name__}: {exception}')


def example(resource_name, options):
    # 1. Errors while opening a session, for example because of a bad resource name or option
    #    string, are raised by the Session constructor.
    try:
        with nirfsg.Session(resource_name='invalidModelSim', id_query=False, reset_device=False, options='Simulate=1, DriverSetup=Model:invalid_model'):
            print('Opening a session with an invalid model unexpectedly succeeded.')
    except nirfsg.Error as e:
        _report('Invalid model', e)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # 2. The driver rejects writes to read-only properties.
        try:
            session.instrument_model = ''
            print('Writing a read-only property unexpectedly succeeded.')
        except nirfsg.errors.DriverError as e:
            _report('Read-only property', e)
            # error_message() returns the message for any status code, for example a code
            # stored in a log file.
            print(f'  error_message({e.code}) = "{session.error_message(e.code)}"')

        # 3a. write_arb_waveform() accepts only numpy arrays of complex128, complex64, or int16.
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        for data in (np.ones(1000, dtype=np.int32), [1 + 0j] * 1000):
            try:
                session.write_arb_waveform('badWaveform', data, False)
            except TypeError as e:
                _report(f'Waveform of type {type(data).__name__}', e)

        # 3b. The bindings check array shapes before calling the driver. Here there are three
        #     frequencies but only two S-parameter matrices.
        try:
            session.create_deembedding_sparameter_table_array('', 'badTable', np.array([1e9, 2e9, 3e9]), np.zeros((2, 2, 2), dtype=np.complex128), nirfsg.SparameterOrientation.PORT1_TOWARDS_DUT)
        except ValueError as e:
            _report('Mismatched S-parameter table', e)

        # 3c. Misspelled property names raise AttributeError instead of silently creating a
        #     new Python attribute.
        try:
            session.frequncy = 1e9
        except AttributeError as e:
            _report('Misspelled property', e)

        # 4. self_test() returns nothing when the test passes and raises SelfTestError otherwise.
        try:
            session.self_test()
            print('Self-test: passed')
        except nirfsg.errors.SelfTestError as e:
            print(f'Self-test: failed with code {e.code}: {e.message}')

        # 5. Treat driver warnings as errors while generating, so that any warning stops the
        #    program instead of only being printed.
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(1e9, -10.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', nirfsg.DriverWarning)
            try:
                with session.initiate():
                    # Errors that happen during the generation are reported here.
                    session.check_generation_status()
                print('Generation: no errors or warnings')
            except nirfsg.DriverWarning as w:
                print(f'Generation: warning raised as an exception: {w}')
            except nirfsg.Error as e:
                _report('Generation', e)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Shows how to handle NI-RFSG errors and warnings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
