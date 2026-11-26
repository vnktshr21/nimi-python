'''Ignores Script Triggers that arrive too early, with the script instruction "clear scriptTrigger<N>".

Based on the "clear" topic of the NI-RFSG user manual.

Background:
    A Script Trigger is remembered from the moment it is received until an instruction consumes
    it ("wait until", "repeat until"). A trigger that arrives while the script is still busy with
    earlier instructions therefore makes the next "wait until" continue at once. When only
    triggers that arrive after a certain point should count, place "clear scriptTrigger0" just
    before the "wait until":

        repeat 100
            generate preamble
        end repeat
        clear scriptTrigger0
        wait until scriptTrigger0
        generate payload

    Triggers that are consumed are cleared automatically, so two consecutive
    "wait until scriptTrigger0" instructions wait for two separate triggers.

    This example sends one trigger while the preamble is still being generated (too early), and
    a second one afterwards. With "clear", the early trigger is ignored and the payload starts
    only after the second trigger. Run it with --no-clear to see the payload start right after
    the preamble instead.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 as a software trigger, and write the waveforms and the script.
    3. Initiate, send an early trigger during the preamble and a second trigger later, and wait
       for the script to complete.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def _tone_with_zero_tail(cycles, number_of_samples):
    tone = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples)
    tone[-4:] = 0
    return tone.astype(np.complex64)


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, preamble_repeats, use_clear, second_trigger_delay, timeout):
    clear_line = '    clear scriptTrigger0\n' if use_clear else ''
    script = f'''script ignoreEarlyTriggers
    repeat {preamble_repeats}
        generate preamble
    end repeat
{clear_line}    wait until scriptTrigger0
    generate payload
end script'''
    number_of_samples = 10000

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        session.script_triggers[0].configure_software_script_trigger()

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('preamble', _tone_with_zero_tail(cycles, number_of_samples), False)
        session.write_arb_waveform('payload', _tone_with_zero_tail(-cycles, number_of_samples), False)
        session.write_script(script)
        print(script)
        preamble_time = preamble_repeats * number_of_samples / actual_iq_rate

        with session.initiate():
            # Too early: the preamble is still being generated.
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
            print(f'Early trigger sent during the {preamble_time:.2f} s preamble' + (' (ignored because of "clear").' if use_clear else ' (remembered, so the payload follows the preamble at once).'))

            time.sleep(second_trigger_delay)
            session.check_generation_status()
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
            print('Second trigger sent.')

            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('Script completed.' if is_done else f'The script did not complete within {timeout} s.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Ignores early Script Triggers with the NI-RFSG script instruction "clear".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-pr', '--preamble-repeats', default=1000, type=int, help='Number of times the 10000-sample preamble is repeated.')
    parser.add_argument('-nc', '--no-clear', dest='use_clear', action='store_false', help='Leave out the clear instruction, for comparison.')
    parser.add_argument('-d', '--second-trigger-delay', default=3.0, type=float, help='Time, in seconds, after initiating before the second trigger is sent. Should be longer than the preamble.')
    parser.add_argument('-t', '--timeout', default=5.0, type=float, help='Maximum time, in seconds, to wait for the script to complete after the second trigger.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.preamble_repeats, args.use_clear, args.second_trigger_delay, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    for use_clear in (True, False):
        example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10, use_clear, 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--second-trigger-delay', '0.01', '--timeout', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
