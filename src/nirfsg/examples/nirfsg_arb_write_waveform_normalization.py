'''Normalizes waveform data to full scale as it is written, with the write waveform normalization attribute.

Background:
    With the PEAK power level type, power_level is the power of a full-scale sample (magnitude
    1.0). A waveform whose largest sample is smaller than 1.0, for example a waveform file that
    was saved with some headroom, is therefore generated below power_level, and its
    signal-to-noise ratio is lower than it could be.

    write_waveform_normalization = ENABLE makes the driver scale the data of the following
    write_arb_waveform() calls so that its maximum is 1.0 and its minimum is -1.0. The
    waveform then uses the full range of the DAC, and its peak power equals power_level,
    without scaling the data in your program. The default is DISABLE.

    This example writes the same tone, with a peak magnitude of 0.25 (-12 dBFS), once without
    and once with normalization, and generates both. On a spectrum analyzer, the normalized
    tone is about 12 dB stronger.

Steps:
    1. Configure the RF settings, ARB mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Write the tone with normalization disabled, then again with normalization enabled.
    3. Select and generate each waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, peak_magnitude, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        tone = (peak_magnitude * np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples)).astype(np.complex64)

        # The normalization setting applies to the data written after it is set.
        session.write_waveform_normalization = nirfsg.WriteWaveformNormalization.DISABLE
        session.write_arb_waveform('asWritten', tone, False)
        session.write_waveform_normalization = nirfsg.WriteWaveformNormalization.ENABLE
        session.write_arb_waveform('normalized', tone, False)

        expected = {
            'asWritten': power_level + 20 * np.log10(peak_magnitude),
            'normalized': power_level,
        }
        for name, peak_power in expected.items():
            session.select_arb_waveform(name)
            print(f'Generating "{name}": peak power about {peak_power:.1f} dBm')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Normalizes waveform data to full scale as NI-RFSG writes it.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, of a full-scale sample.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-m', '--peak-magnitude', default=0.25, type=float, help='Peak magnitude of the tone as written (greater than 0 and at most 1).')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.peak_magnitude, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 0.25, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
