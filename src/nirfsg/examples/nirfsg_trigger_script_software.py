'''Generates one burst each time the program sends a software Script Trigger.

Python counterpart of the NI-RFSG .NET "Triggers/ScriptTrigger-SoftwareSource" and
"Scripts/FiniteGeneration-Re-Triggered" examples. Based on the "Script Trigger Details" and
"Software Edge" topics of the NI-RFSG user manual.

Background:
    A Script Trigger is a general-purpose trigger whose meaning is defined by the script that
    tests it. There are four, scriptTrigger0 to scriptTrigger3, configured through the
    script_triggers[] repeated capability. configure_software_script_trigger() sets the type to
    SOFTWARE, and send_software_edge_trigger(SoftwareTriggerType.SCRIPT,
    TriggerIdentifier.SCRIPT_TRIGGER<N>) sends it.

    This example produces a retriggerable finite generation: the script waits for
    scriptTrigger0, generates one burst, and waits again. The burst ends with zero samples,
    so the output is silent while the script waits.

        repeat forever
            clear scriptTrigger0
            wait until scriptTrigger0
            generate burst
        end repeat

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 as a software trigger, and write the burst and the script.
    3. Initiate and send a software Script Trigger at a fixed interval.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script burstOnTrigger
    repeat forever
        clear scriptTrigger0
        wait until scriptTrigger0
        generate burst
    end repeat
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, burst_samples, number_of_triggers, trigger_interval):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        session.script_triggers[0].configure_software_script_trigger()
        print(f'scriptTrigger0 type: {session.script_triggers[0].script_trigger_type}')

        cycles = round(tone_offset * burst_samples / actual_iq_rate)
        burst = np.exp(2j * np.pi * cycles * np.arange(burst_samples) / burst_samples)
        burst[-4:] = 0
        session.write_arb_waveform('burst', burst.astype(np.complex64), False)
        session.write_script(SCRIPT)

        with session.initiate():
            for trigger in range(number_of_triggers):
                time.sleep(trigger_interval)
                session.check_generation_status()
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
                print(f'Trigger {trigger + 1}/{number_of_triggers}: one {burst_samples / actual_iq_rate * 1e3:.2f} ms burst generated')
            time.sleep(trigger_interval)
            session.check_generation_status()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates one burst for each software Script Trigger using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the burst tone from the carrier.')
    parser.add_argument('-s', '--burst-samples', default=100000, type=int, help='Length of the burst in samples.')
    parser.add_argument('-c', '--number-of-triggers', default=10, type=int, help='Number of Script Triggers to send.')
    parser.add_argument('-i', '--trigger-interval', default=1.0, type=float, help='Time, in seconds, between Script Triggers.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.burst_samples, args.number_of_triggers, args.trigger_interval)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 3, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-triggers', '2', '--trigger-interval', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
