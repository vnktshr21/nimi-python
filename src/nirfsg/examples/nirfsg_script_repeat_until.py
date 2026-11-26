'''Repeats part of a script until a Script Trigger arrives, with the script instruction "repeat until scriptTrigger<N>".

Based on the "repeat/end repeat" topic of the NI-RFSG user manual.

Background:
    "repeat until scriptTrigger0 ... end repeat" runs the enclosed instructions again and again
    until Script Trigger 0 is received. The trigger does not interrupt the loop immediately: the
    instructions of the current pass are completed first, then the script continues after
    "end repeat". This keeps every waveform intact, so the loop length sets how quickly the script
    can react.

    This example runs a finite script:

        generate toneUp
        repeat until scriptTrigger0
            generate toneDown
        end repeat
        generate toneUp

    The program sends a software Script Trigger after a delay; the script then finishes the
    current toneDown, plays toneUp once more, and the generation is done.

Steps:
    1. Configure the RF settings, SCRIPT mode, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 as a software trigger, and write the waveforms and the script.
    3. Initiate, send the Script Trigger after the delay, and wait for the script to complete.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script repeatUntilTrigger
    generate toneUp
    repeat until scriptTrigger0
        generate toneDown
    end repeat
    generate toneUp
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, trigger_delay, timeout):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        session.script_triggers[0].configure_software_script_trigger()

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        phase = 2 * np.pi * cycles * np.arange(number_of_samples) / number_of_samples
        session.write_arb_waveform('toneUp', np.exp(1j * phase).astype(np.complex64), False)
        session.write_arb_waveform('toneDown', np.exp(-1j * phase).astype(np.complex64), False)
        session.write_script(SCRIPT)
        print(SCRIPT)

        with session.initiate():
            print(f'Repeating toneDown; sending scriptTrigger0 in {trigger_delay} s...')
            time.sleep(trigger_delay)
            session.check_generation_status()
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)

            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('The loop ended and the script completed.' if is_done else f'The script did not complete within {timeout} s.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Repeats part of an NI-RFSG script until a Script Trigger arrives, with "repeat until".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in each waveform.')
    parser.add_argument('-d', '--trigger-delay', default=3.0, type=float, help='Time, in seconds, before the Script Trigger is sent.')
    parser.add_argument('-t', '--timeout', default=5.0, type=float, help='Maximum time, in seconds, to wait for the script to complete after the trigger.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.trigger_delay, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--trigger-delay', '0.01', '--timeout', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
