'''Generates a multitone signal with uniformly spaced tones.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/MultitoneUniformSpacing" example.

Background:
    A multitone signal is a set of tones with equal amplitude, spaced evenly around the
    carrier. It is often used to test amplifier intermodulation distortion and the
    flatness of a receiver. The peak-to-average power ratio (PAPR) of a multitone depends
    heavily on the tone phases. When every tone starts at the same phase, the tones add up
    coherently and the PAPR is about 10*log10(N) dB for N tones. Schroeder phases
    (phi_k = -pi * k^2 / N) spread the peaks out in time and give a much lower PAPR. A lower
    PAPR lets the device drive the DAC harder, which improves the dynamic range.

    The waveform holds a whole number of periods of the multitone, and its length is a multiple
    of the waveform quantum, so it repeats seamlessly. power_level is the average power of the
    whole multitone, so each of the N tones is about 10*log10(N) dB below power_level.

Steps:
    1. Configure ARB mode and the I/Q rate.
    2. Build one period of the multitone with numpy.
    3. Set the carrier frequency, power level, and signal bandwidth (the span of the tones).
    4. Write the waveform and generate it.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

PHASE_TYPES = ['SCHROEDER', 'ZERO', 'RANDOM']


def _create_multitone(number_of_tones, tone_spacing, iq_rate, waveform_quantum, min_waveform_size, phase_type):
    # Snap the tone spacing so that 1 / tone_spacing is a whole number of samples.
    samples_per_period = int(round(iq_rate / tone_spacing))
    actual_tone_spacing = iq_rate / samples_per_period
    # With an even number of tones, the offsets are odd multiples of half the spacing
    # (for example +/-0.5 * spacing), so the signal only repeats after 2 / tone_spacing seconds.
    if number_of_tones % 2 == 0:
        samples_per_period *= 2

    # Repeat whole periods until the length is a multiple of the waveform quantum and at
    # least the minimum waveform size.
    number_of_samples = samples_per_period
    while number_of_samples % waveform_quantum != 0 or number_of_samples < min_waveform_size:
        number_of_samples += samples_per_period

    # Tone offsets from the carrier, centered on 0 Hz: for example -1.5, -0.5, 0.5, 1.5 times the spacing.
    k = np.arange(number_of_tones)
    offsets = (k - (number_of_tones - 1) / 2) * actual_tone_spacing

    if phase_type == 'SCHROEDER':
        phases = -np.pi * k ** 2 / number_of_tones
    elif phase_type == 'RANDOM':
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, number_of_tones)
    else:
        phases = np.zeros(number_of_tones)

    t = np.arange(number_of_samples) / iq_rate
    waveform = np.sum(np.exp(1j * (2 * np.pi * np.outer(offsets, t) + phases[:, np.newaxis])), axis=0)

    # Normalize so that the largest sample has a magnitude of 1.0.
    waveform /= np.max(np.abs(waveform))
    papr_db = 10 * np.log10(np.max(np.abs(waveform) ** 2) / np.mean(np.abs(waveform) ** 2))
    return waveform.astype(np.complex64), actual_tone_spacing, papr_db


def example(resource_name, options, frequency, power_level, iq_rate, number_of_tones, tone_spacing, phase_type, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # Leave 2 dB of headroom for overshoot in the interpolation filters.
        session.arb_pre_filter_gain = -2.0

        _, waveform_quantum, min_waveform_size, _ = session.query_arb_waveform_capabilities()
        waveform, actual_tone_spacing, papr_db = _create_multitone(number_of_tones, tone_spacing, actual_iq_rate, waveform_quantum, min_waveform_size, phase_type)

        span = number_of_tones * actual_tone_spacing
        if span > actual_iq_rate:
            raise ValueError(f'The tones span {span / 1e6} MHz, which exceeds the I/Q rate of {actual_iq_rate / 1e6} MS/s.')

        session.configure_rf(frequency, power_level)
        # The tones occupy number_of_tones * spacing of bandwidth around the carrier.
        session.signal_bandwidth = span
        session.write_arb_waveform('multitone', waveform, False)

        print(f'{number_of_tones} tones, {actual_tone_spacing / 1e3:.3f} kHz spacing, {phase_type.lower()} phases, {len(waveform)} samples')
        print(f'PAPR of the waveform: {papr_db:.2f} dB; each tone is about {power_level - 10 * np.log10(number_of_tones):.2f} dBm')

        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a multitone signal with uniformly spaced tones using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level of the multitone, in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-nt', '--number-of-tones', default=10, type=int, help='Number of tones.')
    parser.add_argument('-s', '--tone-spacing', default=100e3, type=float, help='Frequency spacing between adjacent tones in Hz.')
    parser.add_argument('-ph', '--phase-type', default='SCHROEDER', choices=PHASE_TYPES, type=str.upper, help='Initial phases of the tones.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_tones, args.tone_spacing, args.phase_type, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 8, 250e3, 'RANDOM', 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
