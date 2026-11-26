'''Scales waveform data as it is written, with the arbitrary waveform software scaling factor.

Background:
    arb_waveform_software_scaling_factor multiplies the waveform data by a factor before the data
    is written with write_arb_waveform(). It lets you reduce the level of a waveform without
    modifying the data in your program, for example to leave headroom in a waveform file
    that is already normalized to full scale. The factor applies to the waveforms written
    after it is set; waveforms already in memory keep their scaling.

    The property is supported only when power_level_type is PEAK. The scaled waveform must stay
    smaller than 1.0 in complex magnitude. Because power_level is the power of a full-scale
    sample, a factor of 0.5 lowers the peak power of a full-scale waveform by about 6 dB
    (20 * log10(0.5)).

Steps:
    1. Configure the RF settings, ARB mode, PEAK power, I/Q rate, and signal bandwidth.
    2. For each scaling factor, set the factor and write the same tone under a new name.
    3. Select and generate each waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, scaling_factors, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        # A full-scale tone. Its magnitude is just below 1.0, so a factor of 1.0 is also valid.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        tone = (0.999 * np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples)).astype(np.complex64)

        names = []
        for index, factor in enumerate(scaling_factors):
            # The factor is applied to the data written after it is set.
            session.arb_waveform_software_scaling_factor = factor
            name = f'scaledTone{index}'
            session.write_arb_waveform(name, tone, False)
            names.append((name, factor))

        for name, factor in names:
            session.select_arb_waveform(name)
            print(f'"{name}": scaling factor {factor}, peak power about {power_level + 20 * np.log10(factor):.1f} dBm')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Scales waveform data as it is written with the NI-RFSG software scaling factor.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, of a full-scale sample.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-sf', '--scaling-factors', default=[1.0, 0.5, 0.25], type=float, nargs='+', help='Scaling factors (greater than 0 and at most 1) to apply in turn.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.scaling_factors, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, [1.0, 0.5], 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
