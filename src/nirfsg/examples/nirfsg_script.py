'''Generates a signal whose sequence of waveforms is defined by a script.

Python counterpart of the NI-RFSG .NET "Scripts/SimpleScript" example.

Background:
    In SCRIPT generation mode, waveforms are first written to device memory, and then a
    script describes in which order and how many times to generate them. Scripts can also
    react to script triggers and emit markers. A script starts with "script <name>" and ends
    with "end script". This example uses the simplest useful script, which repeats one
    waveform until the generation is aborted:

        script continuousWaveform
            repeat forever
                generate wfm
            end repeat
        end script

    Each of the other instructions has its own example:
      generate <waveform> subset(<start>, <length>)   nirfsg_script_generate_subset.py
      repeat <N> / repeat forever                     nirfsg_script_repeat.py
      repeat until scriptTrigger<N>                   nirfsg_script_repeat_until.py
      wait <samples>                                  nirfsg_script_wait.py
      wait until scriptTrigger<N>                     nirfsg_script_wait_until.py
      clear scriptTrigger<N>                          nirfsg_script_clear.py
      generate <waveform> marker<N>(<positions>)      nirfsg_event_marker.py

    Vector signal transceivers (VSTs) such as the PXIe-5840/5841/5842 do not support the
    if/else, break, and stream instructions. On VSTs, repeats can be nested only one level deep
    inside a "repeat forever" that is the only top-level instruction, markers are not allowed in
    "wait until", and a compiled script can have at most 6,100 instructions (about one per
    instruction, plus one per marker).

Steps:
    1. Open a session, configure the frequency and power level, and select SCRIPT mode.
    2. Write the waveform, then the script that refers to it by name.
    3. Start the generation and check its status.
'''

import argparse
import nirfsg
import numpy as np
import sys

SAMPLE_SCRIPT = '''
script continuousWaveform
    repeat forever
        generate wfm
    end repeat
end script
'''


def example(resource_name, options, frequency, power_level, number_of_samples):
    # A constant I/Q value of 1 + 0j produces an unmodulated carrier at the configured frequency.
    waveform_data = np.full(number_of_samples, 1 + 0j, dtype=np.complex64)
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # Carrier frequency in Hz and average power level in dBm.
        session.configure_rf(
            frequency,
            power_level
        )
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        # Write the waveform before the script, because the script refers to it by name.
        session.write_arb_waveform('wfm', waveform_data, False)
        session.write_script(SAMPLE_SCRIPT)
        # The script repeats forever, so the generation runs until the "with" block aborts it.
        with session.initiate():
            session.check_generation_status()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a signal based on the script provided.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-s', '--number-of-samples', default=1000, type=int, help='Number of samples.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.number_of_samples)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 1000)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--frequency', '1e9', '--power-level', '-10', '--number-of-samples', '1000', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
