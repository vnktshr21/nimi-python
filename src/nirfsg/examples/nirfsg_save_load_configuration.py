'''Saves the session configuration to a file and restores it later.

Background:
    save_configurations_to_file() writes the current configuration of the session (RF
    settings, ARB settings, waveforms, scripts, and so on) to a file. load_configurations_from_file()
    restores it later, for example to switch quickly between test setups or to share a
    setup between test stations. Two properties control the load:
      load_configurations_from_file_load_options  - what to skip when loading from the file:
          NONE      load everything
          WAVEFORMS do not load the waveforms stored in the file
          SCRIPTS   do not load the scripts stored in the file
      load_configurations_from_file_reset_options - what to keep (not reset) in the session
                                                    before loading:
          NONE               reset everything
          WAVEFORMS          keep the waveforms that are already in the session
          SCRIPTS            keep the scripts that are already in the session
          DEEMBEDDING_TABLES keep the de-embedding tables that are already in the session

Steps:
    1. Configure an ARB generation and write a waveform, then save the configuration.
    2. Change the settings and write a second waveform.
    3. Load the saved configuration and show which settings and waveforms are present.
    4. Generate with the restored configuration.
'''

import argparse
import nirfsg
import numpy as np
import os
import sys
import tempfile
import time


def _print_state(session, label):
    print(f'{label}: frequency {session.frequency / 1e6} MHz, power level {session.power_level} dBm, '
          f'generation mode {session.generation_mode}, I/Q rate {session.iq_rate / 1e6} MS/s, '
          f'savedWaveform exists: {session.check_if_waveform_exists("savedWaveform")}, '
          f'otherWaveform exists: {session.check_if_waveform_exists("otherWaveform")}')


def example(resource_name, options, file_path, load_skip, reset_skip, generation_time):
    waveform = np.exp(2j * np.pi * np.arange(1000) / 100).astype(np.complex64)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # 1. Configure and save.
        session.configure_rf(2e9, -5.0)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = 1e6
        session.write_arb_waveform('savedWaveform', waveform, False)
        session.save_configurations_to_file(file_path)
        _print_state(session, 'Saved   ')

        # 2. Change the configuration.
        session.configure_rf(3e9, -15.0)
        session.iq_rate = 2e6
        session.write_arb_waveform('otherWaveform', waveform, False)
        _print_state(session, 'Changed ')

        # 3. Restore the saved configuration.
        session.load_configurations_from_file_load_options = nirfsg.LoadOptions[load_skip]
        session.load_configurations_from_file_reset_options = nirfsg.LoadConfigurationResetOptions[reset_skip]
        session.load_configurations_from_file(file_path)
        _print_state(session, 'Restored')

        # 4. Generate with the restored configuration.
        if session.check_if_waveform_exists('savedWaveform'):
            session.select_arb_waveform('savedWaveform')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Saves an NI-RFSG configuration to a file and restores it.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-fp', '--file-path', default='', help='Path of the configuration file. By default, a temporary file is used and then deleted.')
    parser.add_argument('-ls', '--load-skip', default='NONE', choices=nirfsg.LoadOptions.__members__.keys(), type=str.upper, help='Items in the file to skip when loading.')
    parser.add_argument('-rs', '--reset-skip', default='NONE', choices=nirfsg.LoadConfigurationResetOptions.__members__.keys(), type=str.upper, help='Items in the session to keep (not reset) when loading.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate with the restored configuration.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    if args.file_path:
        example(args.resource_name, args.option_string, os.path.abspath(args.file_path), args.load_skip, args.reset_skip, args.generation_time)
    else:
        with tempfile.TemporaryDirectory() as directory:
            example(args.resource_name, args.option_string, os.path.join(directory, 'nirfsg_configuration.json'), args.load_skip, args.reset_skip, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    with tempfile.TemporaryDirectory() as directory:
        example('simulated5841', options, os.path.join(directory, 'nirfsg_configuration.json'), 'NONE', 'WAVEFORMS', 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
