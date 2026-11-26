'''Shows how the I/Q rate limits the frequency offsets that a waveform can contain, and what aliasing does.

Based on the "Sample Rate" and "Nyquist and Shannon's Sampling Theorems" topics of the NI-RFSG
user manual.

Background:
    The I/Q rate limits the signal that a waveform can represent:
      - Shannon's sampling theorem: complex samples at iq_rate can only represent frequencies
        within +/- iq_rate / 2 of the carrier. A tone computed at an offset beyond that
        produces exactly the same samples as its alias, a tone at (offset - k * iq_rate), so
        the alias is what is generated.
      - NI-RFSG interpolates the waveform with a filter that is flat up to 0.4 * iq_rate. Keep
        signal content within +/- 0.4 * iq_rate (a bandwidth of 0.8 * iq_rate); content
        between 0.4 and 0.5 * iq_rate may be attenuated.
    To generate a wider signal, increase the I/Q rate.

    This example computes a tone at the requested offset, finds the frequency that the samples
    really contain, warns about aliasing or filter attenuation, and generates the tone.

Steps:
    1. Configure the RF settings, ARB mode, and the I/Q rate, and read back the actual rate.
    2. Compute the tone and find the frequency that its samples contain.
    3. Write the tone and generate it.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def _apparent_offset(samples, iq_rate):
    # The strongest frequency bin of the samples is the offset that is actually generated.
    spectrum = np.abs(np.fft.fft(samples))
    frequencies = np.fft.fftfreq(len(samples), 1 / iq_rate)
    return frequencies[np.argmax(spectrum)]


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        tone = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        computed_offset = cycles * actual_iq_rate / number_of_samples
        generated_offset = _apparent_offset(tone, actual_iq_rate)

        print(f'I/Q rate {actual_iq_rate / 1e6} MS/s: representable offsets +/-{actual_iq_rate / 2e6} MHz, flat response +/-{0.4 * actual_iq_rate / 1e6} MHz')
        print(f'Tone computed at {computed_offset / 1e6:+.3f} MHz; the samples contain {generated_offset / 1e6:+.3f} MHz.')
        if abs(computed_offset) > actual_iq_rate / 2:
            print(f'  The offset exceeds iq_rate / 2, so it aliases to {generated_offset / 1e6:+.3f} MHz.')
        elif abs(computed_offset) > 0.4 * actual_iq_rate:
            print('  The offset exceeds 0.4 * iq_rate, so the interpolation filter may attenuate it.')

        session.signal_bandwidth = max(2 * abs(generated_offset), 1.0)
        session.write_arb_waveform('tone', tone, False)
        with session.initiate():
            print(f'Generating at {(frequency + generated_offset) / 1e6:.3f} MHz for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Shows how the NI-RFSG I/Q rate limits the frequency offsets that a waveform can contain, and what aliasing does.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=7e6, type=float, help='Frequency offset of the tone from the carrier, in Hz. Values above half the I/Q rate alias.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate the tone.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    # 7 MHz at 10 MS/s aliases to -3 MHz; 4.5 MHz is outside the flat part of the interpolation filter.
    for tone_offset in (7e6, 4.5e6, 1e6):
        example('simulated5841', options, 1e9, -10.0, 10e6, tone_offset, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
