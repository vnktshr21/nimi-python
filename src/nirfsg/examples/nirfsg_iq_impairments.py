'''Adds I/Q impairments (DC offsets, gain imbalance, and skew) to a generated signal.

Python counterpart of the I/Q impairment settings in the NI-RFSG .NET "Utilities/ExternalAWG(5611)" example.

Background:
    A real I/Q modulator has imperfections that a receiver must tolerate. NI-RFSG can add
    controlled impairments to the generated signal, for example to test how robust a
    receiver's I/Q correction is. They take effect only when iq_impairment_enabled is True:
      iq_i_offset, iq_q_offset - DC offsets on I and Q, which appear as carrier leakage (a tone
                                 at the carrier frequency). Their units are set by
                                 iq_offset_units (PERCENT of the peak level, or VOLTS).
      iq_gain_imbalance        - gain of I relative to Q, in dB (-6 dB to 6 dB). Creates an
                                 image of the signal mirrored around the carrier.
      iq_skew                  - deviation from 90 degrees between I and Q (-30 to 30 degrees).
                                 Also creates an image.

    This example generates a single tone above the carrier. With the impairments enabled, a
    spectrum analyzer shows the tone, an image at the mirrored frequency below the carrier, and
    carrier leakage at the carrier frequency.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write a
       tone at a frequency offset.
    2. Enable the I/Q impairments and set the offsets, gain imbalance, and skew.
    3. Generate the impaired signal.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, i_offset, q_offset, gain_imbalance, skew, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # A single tone above the carrier.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        waveform = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        session.signal_bandwidth = 2 * abs(tone_offset)
        session.write_arb_waveform('tone', waveform, False)

        # The impairment properties are ignored unless iq_impairment_enabled is True.
        session.iq_impairment_enabled = True
        session.iq_offset_units = nirfsg.OffsetUnits.PERCENT
        session.iq_i_offset = i_offset
        session.iq_q_offset = q_offset
        session.iq_gain_imbalance = gain_imbalance
        session.iq_skew = skew

        print(f'I offset: {session.iq_i_offset}%, Q offset: {session.iq_q_offset}%, gain imbalance: {session.iq_gain_imbalance} dB, skew: {session.iq_skew} deg')
        print(f'Expect the tone at {(frequency + tone_offset) / 1e6} MHz, an image at {(frequency - tone_offset) / 1e6} MHz, and carrier leakage at {frequency / 1e6} MHz.')
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Adds I/Q DC offsets, gain imbalance, and skew to a generated signal using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-io', '--i-offset', default=1.0, type=float, help='I offset in percent.')
    parser.add_argument('-qo', '--q-offset', default=0.0, type=float, help='Q offset in percent.')
    parser.add_argument('-g', '--gain-imbalance', default=0.5, type=float, help='I/Q gain imbalance in dB.')
    parser.add_argument('-sk', '--skew', default=2.0, type=float, help='I/Q skew in degrees.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.i_offset, args.q_offset, args.gain_imbalance, args.skew, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 1.0, 0.5, 0.5, 2.0, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
