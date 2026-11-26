'''Generates a continuous wave (CW) tone.

Python counterpart of the NI-RFSG .NET "GettingStarted/SingleToneGeneration" example.

Background:
    A CW signal is an unmodulated carrier: a single sine wave at the configured frequency and
    power level. It is the simplest signal an RF signal generator produces, and is used, for
    example, to measure the gain or the frequency response of a device under test, or as an
    LO for a mixer.

    Every NI-RFSG program follows the same basic flow:
      1. Open a session to the device (nirfsg.Session).
      2. Configure the device. configure_rf() sets the frequency and the power level.
      3. Start the generation with initiate().
      4. Call check_generation_status() periodically. It reports any error that occurred
         during the generation.
      5. Close the session. Using the session as a context manager ("with") closes it
         automatically.
    Until initiate() is called, the session is in the Configuration state: settings are stored
    but not applied to the hardware. initiate() applies them and moves the device to the
    Generation state. Leaving the "with session.initiate():" block calls abort(), which stops
    the generation.

Steps:
    1. Open a session, configure the frequency and power level, and select CW mode.
    2. Start the generation and check its status.
'''

import argparse
import nirfsg
import sys


def example(resource_name, options, frequency, power_level):
    # id_query=False skips checking the instrument identity, and reset_device=False opens the
    # session without resetting the device.
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # Configure RF settings
        session.configure_rf(
            frequency,  # Frequency in Hz
            power_level  # Power level in dBm
        )
        # CW mode generates an unmodulated tone; no waveform needs to be written.
        session.generation_mode = nirfsg.GenerationMode.CW

        # Start generation. Leaving the "with" block aborts the generation.
        with session.initiate():
            # Raises an nirfsg.Error if an error occurred during the generation.
            session.check_generation_status()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a continuous wave (CW) signal using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--frequency', '1e9', '--power-level', '-10', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
