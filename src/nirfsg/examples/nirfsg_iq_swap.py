'''Swaps the I and Q signals, which mirrors the generated spectrum around the carrier.

Background:
    iq_swap_enabled exchanges the I and Q data before upconversion. Swapping I and Q is the same
    as conjugating the baseband signal and rotating it by 90 degrees, so every frequency
    component at carrier + f moves to carrier - f. Use it, for example, to test a receiver
    with an inverted spectrum, or to correct a waveform file whose I and Q columns were
    exchanged.

    This example writes a single tone above the carrier, generates it, then enables the swap
    and generates it again. A spectrum analyzer shows the tone move to the mirrored frequency.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write a tone.
    2. Generate with iq_swap_enabled False, then with iq_swap_enabled True.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        waveform = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        session.signal_bandwidth = 2 * abs(tone_offset)
        session.write_arb_waveform('tone', waveform, False)

        for swap in (False, True):
            session.iq_swap_enabled = swap
            expected = frequency - tone_offset if swap else frequency + tone_offset
            print(f'I/Q swap {"enabled" if session.iq_swap_enabled else "disabled"}: tone at {expected / 1e6} MHz')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Swaps the I and Q signals to mirror the NI-RFSG output spectrum around the carrier.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate with and without the swap.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.generation_time)


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
