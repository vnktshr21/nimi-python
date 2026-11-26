'''Blanks the RF output outside the bursts of a bursted waveform (RF blanking).

Background:
    A bursted signal, such as a TDD frame, is on for part of each period and off for the rest.
    Zero-valued samples alone leave some residual output (carrier leakage and noise) during the
    off time. RF blanking strongly attenuates the RF output outside the bursts, which improves
    the on/off ratio. It is configured per waveform, through the waveforms[] repeated capability:
      set_waveform_burst_start_locations() / set_waveform_burst_stop_locations()
          the samples at which each burst starts and stops
      waveform_rf_blanking
          ENABLE blanks the RF output outside the bursts. When rf_blanking_source is empty (the
          default), the driver derives the blanking from the burst locations.
    The burst locations are set after the waveform is written. The matching get_* methods
    read them back. On the PXIe-5840/5841, RF blanking does not occur below 120 MHz.

    Use PEAK power for bursted signals, so that power_level is the power during the burst.
    With AVERAGE, the power would be averaged over the on and off times.

Steps:
    1. Configure the RF settings, ARB mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Write a waveform with one tone burst and zeros elsewhere.
    3. Set the burst start and stop locations and enable RF blanking.
    4. Generate the waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, number_of_samples, burst_start, burst_length, rf_blanking, generation_time):
    waveform_name = 'burstWaveform'
    burst_stop = burst_start + burst_length
    if burst_stop > number_of_samples:
        raise ValueError('The burst must end within the waveform.')

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # A tone at one tenth of the I/Q rate during the burst, zeros elsewhere.
        tone_offset = actual_iq_rate / 10
        n = np.arange(number_of_samples)
        waveform = np.zeros(number_of_samples, dtype=np.complex64)
        waveform[burst_start:burst_stop] = np.exp(2j * np.pi * tone_offset * n[burst_start:burst_stop] / actual_iq_rate)
        session.signal_bandwidth = 2 * tone_offset
        session.write_arb_waveform(waveform_name, waveform, False)

        waveform_properties = session.waveforms[waveform_name]
        waveform_properties.set_waveform_burst_start_locations([burst_start])
        waveform_properties.set_waveform_burst_stop_locations([burst_stop])
        waveform_properties.waveform_rf_blanking = nirfsg.RFBlanking.ENABLE if rf_blanking else nirfsg.RFBlanking.DISABLE

        print(f'burst starts : {waveform_properties.get_waveform_burst_start_locations()}')
        print(f'burst stops  : {waveform_properties.get_waveform_burst_stop_locations()}')
        print(f'RF blanking  : {waveform_properties.waveform_rf_blanking}')

        duty_cycle = 100 * burst_length / number_of_samples
        print(f'Generating a {burst_length / actual_iq_rate * 1e6:.1f} us burst every {number_of_samples / actual_iq_rate * 1e6:.1f} us ({duty_cycle:.1f}% duty cycle)...')
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Blanks the RF output outside the bursts of a bursted waveform using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz. RF blanking does not occur below 120 MHz on the PXIe-5840/5841.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, during the burst.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in the waveform (one burst period).')
    parser.add_argument('-bs', '--burst-start', default=1000, type=int, help='Sample at which the burst starts.')
    parser.add_argument('-bl', '--burst-length', default=4000, type=int, help='Length of the burst in samples.')
    parser.add_argument('-nb', '--no-rf-blanking', dest='rf_blanking', action='store_false', help='Do not blank the RF output outside the burst, for comparison.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_samples, args.burst_start, args.burst_length, args.rf_blanking, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 10000, 1000, 4000, True, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
