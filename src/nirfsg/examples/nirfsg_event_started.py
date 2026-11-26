'''Exports the Started Event, which marks the moment the generation actually begins.

Based on the "NI-RFSG Event Types" topic of the NI-RFSG user manual.

Background:
    The Started Event is emitted when the generator has received its Start Trigger and begins
    generating, so another instrument that starts on the Started Event is aligned with the start
    of the signal rather than with the trigger. Route it with exported_started_event_output_terminal ('PFI0',
    'PXI_Trig0', ...; an empty string means not exported). The fully qualified name, returned by
    started_event_terminal_name or get_terminal_name(Signal.STARTED_EVENT, ''), can be used as a
    trigger source by another device.

    This example arms the generation with a software Start Trigger, so that the Started Event
    occurs when the program sends the trigger.

Steps:
    1. Configure the RF settings and ARB mode, and write a tone.
    2. Configure a software Start Trigger and export the Started Event.
    3. Initiate, send the Start Trigger, and monitor the generation.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, started_event_terminal, trigger_delay, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('tone', np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)

        session.configure_software_start_trigger()
        session.exported_started_event_output_terminal = started_event_terminal
        print(f'Started Event exported to {started_event_terminal or "(not exported)"}')
        print(f'started_event_terminal_name: {session.started_event_terminal_name}')
        print(f'get_terminal_name()        : {session.get_terminal_name(nirfsg.Signal.STARTED_EVENT, "")}')

        with session.initiate():
            time.sleep(trigger_delay)
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
            print('Start Trigger sent; the Started Event is asserted when the generation begins.')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Exports the NI-RFSG Started Event.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-e', '--started-event-terminal', default='PXI_Trig0', help='Terminal to export the Started Event to. Leave empty to not export.')
    parser.add_argument('-d', '--trigger-delay', default=1.0, type=float, help='Time, in seconds, between arming and sending the Start Trigger.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.started_event_terminal, args.trigger_delay, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 'PXI_Trig0', 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--trigger-delay', '0.01', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
