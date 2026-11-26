'''Stores several scripts on the device, selects the one to run, and lists, reads back, and deletes scripts.

Python counterpart of the script handling in the NI-RFSG .NET "Scripts/SimpleScript" example.

Background:
    Several scripts can be stored on the device at the same time. Each one is named by the
    "script <name>" line at its start. The driver provides these methods to manage them:
      write_script(script)              compiles and stores a script
      selected_script                   the name of the script that initiate() runs
      get_all_script_names()            the names of all stored scripts
      check_if_script_exists(name)      whether a script is stored
      get_script(name)                  the text of a stored script
      delete_script(name)               removes a script from the device
    Storing several scripts and switching selected_script is faster than writing a new script
    before every generation.

    Simulated devices do not compile scripts, so on a simulated device the lookup methods may
    report that no script is stored. This example checks for that case.

Steps:
    1. Configure the RF settings and SCRIPT mode, and write a waveform.
    2. Write two scripts and list the stored scripts.
    3. Select each script, read it back, and generate it for a short time.
    4. Delete one script and list the stored scripts again.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPTS = {
    'continuousTone': '''script continuousTone
    repeat forever
        generate tone
    end repeat
end script''',
    'gatedTone': '''script gatedTone
    repeat forever
        generate tone
        wait 10000
    end repeat
end script''',
}


def example(resource_name, options, frequency, power_level, iq_rate, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # A tone at one tenth of the I/Q rate, ending with zeros so that "wait" holds a zero sample.
        number_of_samples = 10000
        tone = np.exp(2j * np.pi * 1000 * np.arange(number_of_samples) / number_of_samples)
        tone[-100:] = 0
        session.signal_bandwidth = 2 * actual_iq_rate / 10
        session.write_arb_waveform('tone', tone.astype(np.complex64), False)

        for script in SCRIPTS.values():
            session.write_script(script)
        print(f'Stored scripts: {session.get_all_script_names()}')

        for name in SCRIPTS:
            session.selected_script = name
            if session.check_if_script_exists(name):
                print(f'--- {session.selected_script}, read back from the device ---')
                print(session.get_script(name))
            else:
                print(f'--- {session.selected_script}: not reported as stored (scripts are not compiled on simulated devices) ---')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)

        if session.check_if_script_exists('gatedTone'):
            session.delete_script('gatedTone')
            print(f'Deleted gatedTone. Stored scripts: {session.get_all_script_names()}')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Stores, selects, lists, reads back, and deletes NI-RFSG scripts.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-t', '--generation-time', default=3.0, type=float, help='Time, in seconds, to generate each script.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
