'''Exports the Start Trigger and a Script Trigger to terminals, so that other instruments receive the same triggers.

Python counterpart of the NI-RFSG .NET "Triggers/ExportStartTrigger" example. Based on the
"Signal Routing in NI-RFSG" and "Syntax for Terminal Names" topics of the NI-RFSG user manual.

Background:
    A trigger received by the generator can be routed to an output terminal, so another
    instrument (for example an NI-RFSA analyzer) starts or steps at the same moment:
      exported_start_trigger_output_terminal                  - the Start Trigger
      script_triggers[N].exported_script_trigger_output_terminal - Script Trigger N
    Valid terminals include front-panel PFI lines and the PXI trigger lines ('PXI_Trig0' to
    'PXI_Trig7'). An empty string means the trigger is not exported. A signal exported to a PXI
    trigger line reaches only the instruments in the same PXI chassis segment, unless a route
    across segments is reserved.

    Instead of a physical terminal, the other instrument can also use the fully qualified name
    of the trigger, returned by get_terminal_name() and by the start_trigger_terminal_name and
    script_trigger_terminal_name properties (for example '/PXI1Slot2/ao/0/StartTrigger'). The
    driver then chooses the route automatically, which NI recommends whenever possible.

    This example uses software triggers so it can run without external signals: a software Start
    Trigger starts the generation, and each software Script Trigger toggles between two tones.
    Both triggers appear on the export terminals.

Steps:
    1. Configure the RF settings, SCRIPT mode, I/Q rate, and signal bandwidth, and write the
       waveforms and a toggling script.
    2. Configure software Start and Script Triggers and export them.
    3. Print the fully qualified trigger names.
    4. Initiate, send the Start Trigger, then send Script Triggers at a fixed interval.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script toggleOnTrigger
    repeat forever
        repeat until scriptTrigger0
            generate toneUp
        end repeat
        repeat until scriptTrigger0
            generate toneDown
        end repeat
    end repeat
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, start_trigger_terminal, script_trigger_terminal, number_of_triggers, trigger_interval):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        phase = 2 * np.pi * cycles * np.arange(number_of_samples) / number_of_samples
        session.write_arb_waveform('toneUp', np.exp(1j * phase).astype(np.complex64), False)
        session.write_arb_waveform('toneDown', np.exp(-1j * phase).astype(np.complex64), False)
        session.write_script(SCRIPT)

        session.configure_software_start_trigger()
        session.script_triggers[0].configure_software_script_trigger()
        session.exported_start_trigger_output_terminal = start_trigger_terminal
        session.script_triggers[0].exported_script_trigger_output_terminal = script_trigger_terminal

        print(f'Start Trigger  : exported to {start_trigger_terminal or "(not exported)"}, name {session.get_terminal_name(nirfsg.Signal.START_TRIGGER, "")}')
        print(f'scriptTrigger0 : exported to {script_trigger_terminal or "(not exported)"}, name {session.get_terminal_name(nirfsg.Signal.SCRIPT_TRIGGER, "scriptTrigger0")}')
        print(f'start_trigger_terminal_name                    : {session.start_trigger_terminal_name}')
        print(f'script_triggers[0].script_trigger_terminal_name : {session.script_triggers[0].script_trigger_terminal_name}')

        with session.initiate():
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
            print('Start Trigger sent.')
            for trigger in range(number_of_triggers):
                time.sleep(trigger_interval)
                session.check_generation_status()
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
                print(f'scriptTrigger0 sent ({trigger + 1}/{number_of_triggers}).')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Exports the NI-RFSG Start Trigger and a Script Trigger to terminals.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-st', '--start-trigger-terminal', default='PXI_Trig0', help='Terminal to export the Start Trigger to. Leave empty to not export.')
    parser.add_argument('-sc', '--script-trigger-terminal', default='PXI_Trig1', help='Terminal to export scriptTrigger0 to. Leave empty to not export.')
    parser.add_argument('-c', '--number-of-triggers', default=5, type=int, help='Number of Script Triggers to send.')
    parser.add_argument('-i', '--trigger-interval', default=1.0, type=float, help='Time, in seconds, between Script Triggers.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.start_trigger_terminal, args.script_trigger_terminal, args.number_of_triggers, args.trigger_interval)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 'PXI_Trig0', 'PXI_Trig1', 2, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-triggers', '2', '--trigger-interval', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
