'''Generates a continuous wave (CW) tone that is locked to a selectable Reference Clock.

Python counterpart of the NI-RFSG .NET "GettingStarted/ReferenceClock" example.

Background:
    The Reference Clock is the frequency standard that every clock inside the signal
    generator (the LO synthesizers and the arbitrary waveform sample clock) is
    phase-locked to. By default the device uses its own onboard oscillator. Locking to an
    external reference, such as the 10 MHz PXI backplane clock (PXI_CLK) or a lab
    standard connected to REF IN, makes the output frequency exactly as accurate as that
    reference and gives several instruments a common timebase. The device can also
    export its Reference Clock (for example on REF OUT) so other instruments can lock to it.

Steps:
    1. Open an NI-RFSG session.
    2. Set the RF frequency and power level with configure_rf().
    3. Select CW generation mode.
    4. Select the Reference Clock source and rate with configure_ref_clock().
    5. Optionally export the Reference Clock to an output terminal.
    6. Enable the RF output, start the generation, and poll check_generation_status() so that
       errors raised during generation (for example losing lock to the reference) are reported.
    7. Stop the generation and disable the RF output.

Refer to your device documentation for the Reference Clock sources and export terminals
that it supports.
'''

import argparse
import nirfsg
import sys
import time


def example(resource_name, options, frequency, power_level, ref_clock_source, ref_clock_rate, ref_clock_export_terminal, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # Configure the carrier frequency (Hz) and the output power level (dBm).
        session.configure_rf(frequency, power_level)

        # CW mode generates an unmodulated tone at the configured frequency.
        session.generation_mode = nirfsg.GenerationMode.CW

        # configure_ref_clock() takes the source as a string. The ReferenceClockSource enum lists
        # the valid strings, so pass the value of the enum member. The rate is used only when
        # the source is an external connector (ClkIn, RefIn, or RefIn2); the onboard clock and
        # PXI_CLK always run at 10 MHz.
        session.configure_ref_clock(nirfsg.ReferenceClockSource[ref_clock_source].value, ref_clock_rate)

        # Exporting the Reference Clock lets other instruments, such as an NI-RFSA analyzer,
        # share this generator's timebase. DO_NOT_EXPORT leaves the output terminal unused.
        session.exported_ref_clock_output_terminal = nirfsg.ReferenceClockExportOutputTerminal[ref_clock_export_terminal]

        print(f'Reference Clock source : {session.ref_clock_source}')
        print(f'Reference Clock rate   : {session.ref_clock_rate / 1e6} MHz')
        # get_terminal_name() returns the fully qualified terminal name of a signal, which is the
        # string another device uses to route that signal (for example '/PXI1Slot2/RefClock').
        print(f'Reference Clock signal : {session.get_terminal_name(nirfsg.Signal.REF_CLOCK, "")}')

        session.output_enabled = True

        # initiate() returns a context manager that calls abort() when the block exits, so the
        # generation always stops, even if an exception is raised.
        with session.initiate():
            print(f'Generating {frequency / 1e6} MHz at {power_level} dBm for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                # check_generation_status() raises an nirfsg.Error if a generation error has
                # occurred. For continuous CW generation it always returns False (not done).
                session.check_generation_status()
                time.sleep(0.1)

        # Disabling the output reduces the noise floor at the RF OUT connector when the
        # generator is idle.
        session.output_enabled = False


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a CW tone locked to the selected Reference Clock source using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-s', '--ref-clock-source', default='ONBOARD_CLOCK', choices=nirfsg.ReferenceClockSource.__members__.keys(), type=str.upper, help='Reference Clock source.')
    parser.add_argument('-r', '--ref-clock-rate', default=10e6, type=float, help='Reference Clock rate in Hz. Used only for external sources.')
    parser.add_argument('-e', '--ref-clock-export-terminal', default='DO_NOT_EXPORT', choices=nirfsg.ReferenceClockExportOutputTerminal.__members__.keys(), type=str.upper, help='Terminal on which to export the Reference Clock.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.ref_clock_source, args.ref_clock_rate, args.ref_clock_export_terminal, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 'PXI_CLK', 10e6, 'REF_OUT', 0.2)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.2', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
