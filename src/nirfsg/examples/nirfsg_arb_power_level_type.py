'''Shows how the power level type (average or peak) sets the output power of an arbitrary waveform.

Background:
    In ARB and SCRIPT generation modes, power_level_type decides what power_level means:
      AVERAGE - power_level is the average power of the waveform. The driver computes the
                waveform's peak-to-average power ratio (PAPR) and scales the data, so the peak
                power is power_level + PAPR. Use it when the signal is specified by its average
                (RMS) power, as most communication standards are.
      PEAK    - power_level is the power of a full-scale sample (magnitude 1.0, or 32767 for
                int16 data). The data is not rescaled, so a waveform whose largest sample is
                below 1.0 has a lower peak power. PEAK is required for interleaved int16 data,
                for streaming, and for writing a waveform in several blocks.
    peak_envelope_power returns the resulting peak power of the generated signal.

    This example writes a two-tone waveform (PAPR = 3.01 dB, normalized to a peak magnitude of
    1.0), and generates it once with each power level type at the same power_level.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write the
       two-tone waveform.
    2. For each power level type: set it, generate, and read the peak envelope power.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_spacing, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = tone_spacing

        # Two equal tones at +/- tone_spacing / 2, normalized so the largest sample is 1.0.
        cycles = round(tone_spacing / 2 * number_of_samples / actual_iq_rate)
        waveform = np.cos(2 * np.pi * cycles * np.arange(number_of_samples) / number_of_samples) + 0j
        waveform /= np.max(np.abs(waveform))
        papr_db = 10 * np.log10(np.max(np.abs(waveform) ** 2) / np.mean(np.abs(waveform) ** 2))
        session.write_arb_waveform('twoTone', waveform.astype(np.complex64), False)
        print(f'Two-tone waveform PAPR: {papr_db:.2f} dB')

        for power_level_type in (nirfsg.PowerLevelType.AVERAGE, nirfsg.PowerLevelType.PEAK):
            session.power_level_type = power_level_type
            with session.initiate():
                print(f'{power_level_type}: power level {power_level} dBm, peak envelope power {session.peak_envelope_power:.2f} dBm')
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates the same waveform with the AVERAGE and the PEAK power level types using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-s', '--tone-spacing', default=1e6, type=float, help='Spacing between the two tones, in Hz.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate with each power level type.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_spacing, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
