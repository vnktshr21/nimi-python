'''Exports the Done Event, which signals that a finite generation has completed.

Based on the "NI-RFSG Event Types" topic of the NI-RFSG user manual.

Background:
    The Done Event indicates that the generation of the last waveform is complete. It occurs
    only for finite generations, such as an arbitrary waveform with a finite repeat count or a
    script that ends without "repeat forever". Another instrument can use it to stop an
    acquisition, or to start the next step of a test sequence, without polling the generator.
    Route it with exported_done_event_output_terminal; done_event_terminal_name and
    get_terminal_name(Signal.DONE_EVENT, '') return its fully qualified name. In the program,
    check_generation_status() returns True at the same point.

Steps:
    1. Configure a finite ARB generation (a tone repeated a fixed number of times).
    2. Export the Done Event.
    3. Initiate and poll check_generation_status() until the generation is done.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, repeat_count, done_event_terminal, timeout):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('tone', np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)

        # A finite generation, so that the Done Event occurs.
        session.arb_waveform_repeat_count_is_finite = True
        session.arb_waveform_repeat_count = repeat_count

        session.exported_done_event_output_terminal = done_event_terminal
        print(f'Done Event exported to {done_event_terminal or "(not exported)"}')
        print(f'done_event_terminal_name: {session.done_event_terminal_name}')
        print(f'get_terminal_name()     : {session.get_terminal_name(nirfsg.Signal.DONE_EVENT, "")}')

        expected = repeat_count * number_of_samples / actual_iq_rate
        with session.initiate():
            print(f'Generating for about {expected:.3f} s...')
            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('Generation complete; the Done Event was asserted.' if is_done else f'Generation did not complete within {timeout} s.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Exports the NI-RFSG Done Event at the end of a finite generation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-c', '--repeat-count', default=1000, type=int, help='Number of times to generate the 10000-sample tone.')
    parser.add_argument('-e', '--done-event-terminal', default='PXI_Trig1', help='Terminal to export the Done Event to. Leave empty to not export.')
    parser.add_argument('-t', '--timeout', default=10.0, type=float, help='Maximum time, in seconds, to wait for the generation to complete.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.repeat_count, args.done_event_terminal, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10, 'PXI_Trig1', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--repeat-count', '10', '--timeout', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
