'''Reads waveforms from a TDMS file, downloads them to the device, and generates them.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/GenFromFileSingle" and
"ArbitraryWaveforms/GenFromFileMultiple" examples.

Background:
    Waveforms created with NI tools such as the RFmx waveform creator or the NI modulation
    toolkits are often saved as TDMS files. read_and_download_waveform_from_file_tdms() reads one
    waveform (selected by its index) from the file and writes it to device memory. It also
    applies the metadata stored with the waveform to that waveform in the session:
      - I/Q rate (waveform_iq_rate)
      - peak-to-average power ratio (waveform_papr)
      - runtime scaling (waveform_runtime_scaling)
      - signal bandwidth (waveform_signal_bandwidth)
      - waveform size (waveform_waveform_size)
      - burst start and stop locations, RF blanking settings, and markers
    Because these settings are stored as attributes of the waveform (refer to
    nirfsg_arb_waveform_attributes.py), the session does not need to be configured with them,
    and a file with several waveforms can be loaded under different names and switched with
    select_arb_waveform(). The file path must be absolute.

    This folder contains SampleWaveform.tdms, a Bluetooth waveform with one waveform at
    index 0. Pass --file-path and --waveform-indices to load your own waveforms.

Steps:
    1. Configure the RF settings and ARB generation mode.
    2. Read and download each requested waveform from the TDMS file.
    3. Print the I/Q rate and size that were loaded with each waveform.
    4. Select each waveform in turn and generate it.
'''

import argparse
import nirfsg
import os
import sys
import time

DEFAULT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SampleWaveform.tdms')


def example(resource_name, options, frequency, power_level, file_path, waveform_indices, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM

        waveform_names = []
        for index in waveform_indices:
            name = f'waveform{index}'
            # The driver needs an absolute path to the TDMS file.
            session.read_and_download_waveform_from_file_tdms(name, os.path.abspath(file_path), index)
            waveform_names.append(name)

            waveform = session.waveforms[name]
            print(f'Loaded index {index} of {os.path.basename(file_path)} as "{name}": '
                  f'{waveform.waveform_waveform_size} samples at {waveform.waveform_iq_rate / 1e6} MS/s')

        for name in waveform_names:
            session.select_arb_waveform(name)
            print(f'Generating "{name}" at {frequency / 1e6} MHz, {power_level} dBm for {generation_time} s...')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Reads waveforms from a TDMS file, downloads them to the NI-RFSG device, and generates them.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=2.402e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-fp', '--file-path', default=DEFAULT_FILE_PATH, help='Path to the TDMS waveform file.')
    parser.add_argument('-i', '--waveform-indices', default=[0], type=int, nargs='+', help='Indices of the waveforms to load from the file.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.file_path, args.waveform_indices, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 2.402e9, -10.0, DEFAULT_FILE_PATH, [0], 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
