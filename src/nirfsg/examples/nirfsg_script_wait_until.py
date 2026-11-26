'''Pauses a script until a Script Trigger arrives, with the script instruction "wait until scriptTrigger<N>".

Based on the "wait" topic of the NI-RFSG user manual.

Background:
    "wait until scriptTrigger0" pauses the script until Script Trigger 0 is received, then
    continues with the next instruction. While waiting, the device holds the last generated
    sample at the output, so end the preceding waveform with a zero sample to keep the output
    silent. If the trigger was already received before the wait, the script continues after the
    smallest possible delay; use the "clear" instruction to ignore such early triggers (refer to
    nirfsg_script_clear.py). On vector signal transceivers, markers are not allowed in a
    "wait until" instruction.

    This example steps through a sequence on demand:

        generate toneUp
        wait until scriptTrigger0
        generate toneDown
        wait until scriptTrigger0
        generate toneUp

    The program sends a software Script Trigger at each step, for example after a measurement
    of the device under test has finished.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 as a software trigger, and write the waveforms and the script.
    3. Initiate, send a Script Trigger for each step, and wait for the script to complete.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script stepOnTrigger
    generate toneUp
    wait until scriptTrigger0
    generate toneDown
    wait until scriptTrigger0
    generate toneUp
end script'''


def _tone_with_zero_tail(cycles, number_of_samples):
    tone = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples)
    tone[-4:] = 0
    return tone.astype(np.complex64)


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, step_interval, timeout):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        session.script_triggers[0].configure_software_script_trigger()

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('toneUp', _tone_with_zero_tail(cycles, number_of_samples), False)
        session.write_arb_waveform('toneDown', _tone_with_zero_tail(-cycles, number_of_samples), False)
        session.write_script(SCRIPT)
        print(SCRIPT)

        with session.initiate():
            for step in (1, 2):
                time.sleep(step_interval)
                session.check_generation_status()
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
                print(f'Step {step}: scriptTrigger0 sent')

            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('Script completed.' if is_done else f'The script did not complete within {timeout} s.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Pauses an NI-RFSG script until a Script Trigger arrives, with "wait until".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in each waveform.')
    parser.add_argument('-i', '--step-interval', default=2.0, type=float, help='Time, in seconds, between Script Triggers.')
    parser.add_argument('-t', '--timeout', default=5.0, type=float, help='Maximum time, in seconds, to wait for the script to complete after the last trigger.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.step_interval, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--step-interval', '0.01', '--timeout', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
