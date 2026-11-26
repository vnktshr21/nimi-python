'''Lets NI-RFSG detect the burst start and stop locations when a waveform is written.

Background:
    Bursted waveforms need burst start and stop locations, for example for RF blanking. Instead
    of computing and setting them yourself, you can let the driver detect them while
    write_arb_waveform() writes the data. The detection settings apply to the next writes:
      write_waveform_burst_detection
          ENABLE turns burst detection on (default: DISABLE).
      write_waveform_burst_detection_mode
          AUTO   - the driver analyzes the waveform to find the bursts.
          MANUAL - a burst starts where the signal rises above, and stops where it falls below,
                   write_waveform_burst_detection_power_threshold, in dB relative to the peak
                   power of the waveform.
    After the write, get_waveform_burst_start_locations() and get_waveform_burst_stop_locations()
    return the detected locations through the waveforms[] repeated capability.

    When you load a waveform with read_and_download_waveform_from_file_tdms() and RF blanking is
    enabled for it, burst detection must be disabled.

Steps:
    1. Configure the RF settings, ARB mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Build a waveform with several tone bursts of different lengths.
    3. Enable burst detection in the requested mode, write the waveform, and print the
       detected burst locations next to the true ones.
    4. Generate the waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

DETECTION_MODES = ['AUTO', 'THRESHOLD']


def example(resource_name, options, frequency, power_level, iq_rate, detection_mode, threshold_db, generation_time):
    number_of_samples = 20000
    bursts = [(1000, 3000), (6000, 10000), (13000, 18000)]
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # Tone bursts at one tenth of the I/Q rate, zeros between them.
        tone_offset = actual_iq_rate / 10
        n = np.arange(number_of_samples)
        waveform = np.zeros(number_of_samples, dtype=np.complex64)
        for start, stop in bursts:
            waveform[start:stop] = np.exp(2j * np.pi * tone_offset * n[start:stop] / actual_iq_rate)
        session.signal_bandwidth = 2 * tone_offset

        session.write_waveform_burst_detection = nirfsg.WriteWaveformBurstDetection.ENABLE
        if detection_mode == 'AUTO':
            session.write_waveform_burst_detection_mode = nirfsg.WriteWaveformBurstDetectionMode.AUTO
        else:
            session.write_waveform_burst_detection_mode = nirfsg.WriteWaveformBurstDetectionMode.MANUAL
            session.write_waveform_burst_detection_power_threshold = threshold_db
        session.write_arb_waveform('bursts', waveform, False)

        print(f'Burst detection mode: {session.write_waveform_burst_detection_mode}')
        print(f'true burst starts     : {[start for start, _ in bursts]}')
        print(f'detected burst starts : {session.waveforms["bursts"].get_waveform_burst_start_locations()}')
        print(f'true burst stops      : {[stop for _, stop in bursts]}')
        print(f'detected burst stops  : {session.waveforms["bursts"].get_waveform_burst_stop_locations()}')

        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Lets NI-RFSG detect the burst locations of a waveform when it is written.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, during the bursts.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-m', '--detection-mode', default='AUTO', choices=DETECTION_MODES, type=str.upper, help='Detect the bursts automatically, or with a power threshold.')
    parser.add_argument('-th', '--threshold-db', default=-20.0, type=float, help='Detection threshold, in dB relative to the peak power. Used only with THRESHOLD.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.detection_mode, args.threshold_db, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    for detection_mode in DETECTION_MODES:
        example('simulated5841', options, 1e9, -10.0, 10e6, detection_mode, -20.0, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
