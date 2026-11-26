'''Repeats parts of a script a fixed number of times, or forever, with the script instructions "repeat <N>" and "repeat forever".

Python counterpart of the NI-RFSG .NET "Scripts/FiniteGeneration" example. Based on the
"repeat/end repeat" topic of the NI-RFSG user manual.

Background:
    repeat/end repeat loops over the enclosed instructions:
      repeat <N> ... end repeat       runs the instructions N times
      repeat forever ... end repeat   runs the instructions until the generation is aborted
    A script with no "repeat forever" is finite: when its last instruction completes, the
    generation is done, the last sample is held at the output, and check_generation_status()
    returns True.

    On vector signal transceivers such as the PXIe-5840/5841/5842, repeats can be nested only one
    level deep, inside a "repeat forever" that is the only top-level instruction of the script.
    (The "repeat until scriptTriggerN" form is shown in nirfsg_script_repeat_until.py.)

    This example offers two scripts built from two tone waveforms (above and below the carrier):
      FINITE - toneUp once, toneDown N times, toneUp once, then the generation is done.
      NESTED - repeats "toneUp once, toneDown N times" forever.

Steps:
    1. Configure the RF settings, SCRIPT mode, I/Q rate, and signal bandwidth.
    2. Write the waveforms and the selected script.
    3. Generate. A FINITE script is monitored until it completes or the timeout expires.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT_TYPES = ['FINITE', 'NESTED']


def _script(script_type, repeat_count):
    if script_type == 'FINITE':
        return f'''script finiteRepeat
    generate toneUp
    repeat {repeat_count}
        generate toneDown
    end repeat
    generate toneUp
end script'''
    return f'''script nestedRepeat
    repeat forever
        generate toneUp
        repeat {repeat_count}
            generate toneDown
        end repeat
    end repeat
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, script_type, repeat_count, timeout):
    script = _script(script_type, repeat_count)
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        # A whole number of tone cycles per waveform, so consecutive waveforms join without a phase jump.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        phase = 2 * np.pi * cycles * np.arange(number_of_samples) / number_of_samples
        session.write_arb_waveform('toneUp', np.exp(1j * phase).astype(np.complex64), False)
        session.write_arb_waveform('toneDown', np.exp(-1j * phase).astype(np.complex64), False)
        session.write_script(script)
        print(script)

        with session.initiate():
            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                # Only the FINITE script finishes on its own.
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print(f'Script completed after {time.monotonic() - start_time:.3f} s.' if is_done else f'Stopped after {timeout} s.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Repeats parts of an NI-RFSG script with "repeat <N>" and "repeat forever".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=100000, type=int, help='Number of samples in each waveform.')
    parser.add_argument('-sc', '--script-type', default='FINITE', choices=SCRIPT_TYPES, type=str.upper, help='Script to run.')
    parser.add_argument('-c', '--repeat-count', default=50, type=int, help='Number of times toneDown is repeated.')
    parser.add_argument('-t', '--timeout', default=10.0, type=float, help='Maximum time, in seconds, to generate.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.script_type, args.repeat_count, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    for script_type in SCRIPT_TYPES:
        example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, script_type, 5, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--timeout', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
