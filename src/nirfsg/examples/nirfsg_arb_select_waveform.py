'''Stores several arbitrary waveforms on the device and selects which one to generate.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/MultipleArbitraryWaveforms" example.

Background:
    The device can store many named waveforms in its onboard memory. In ARB generation mode,
    the waveform that plays when you call initiate() is the one selected with
    select_arb_waveform() or the arb_selected_waveform property. Selecting a waveform requires
    the Configuration state, so the generation is aborted and started again for every switch.
    Because the waveforms are already in memory, switching is much faster than writing a new
    waveform each time.

    This example writes three waveforms, each containing tones at a frequency offset from
    the carrier:
      upperSideband:  exp(+j*2*pi*f*t)  -> one tone at carrier + f
      lowerSideband:  exp(-j*2*pi*f*t)  -> one tone at carrier - f
      doubleSideband: cos(2*pi*f*t)     -> two tones at carrier +/- f

Steps:
    1. Configure the frequency, power level, ARB mode, I/Q rate, and signal bandwidth.
    2. Write the three named waveforms.
    3. For each waveform: select it, generate it for a while, and abort.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # Every waveform contains tones at most tone_offset away from the carrier.
        session.signal_bandwidth = 2 * abs(tone_offset)

        # Snap the tone to an integer number of cycles over the waveform so that it repeats
        # seamlessly.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        t = np.arange(number_of_samples) / actual_iq_rate
        phase = 2 * np.pi * (cycles * actual_iq_rate / number_of_samples) * t
        waveforms = {
            'upperSideband': np.exp(1j * phase),
            'lowerSideband': np.exp(-1j * phase),
            'doubleSideband': np.cos(phase) + 0j,
        }

        # Write each waveform under its own name. Names are case-insensitive and must be
        # alphanumeric.
        for name, data in waveforms.items():
            session.write_arb_waveform(name, data.astype(np.complex128), False)

        for name in waveforms:
            # The device must be in the Configuration state to select a waveform. Leaving the
            # initiate() context below aborts the previous generation.
            session.select_arb_waveform(name)
            print(f'Generating "{session.arb_selected_waveform}" for {generation_time} s...')
            with session.initiate():
                end_time = time.monotonic() + generation_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Stores several named waveforms and selects which one NI-RFSG generates.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tones from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in each waveform.')
    parser.add_argument('-t', '--generation-time', default=2.0, type=float, help='Time, in seconds, to generate each waveform.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
