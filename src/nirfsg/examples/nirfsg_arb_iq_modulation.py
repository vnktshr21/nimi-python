'''Generates analog (AM, FM, PM) and digital (BPSK, QPSK, 16-QAM) modulation by computing I/Q samples.

Based on the "RF Fundamentals" topics of the NI-RFSG user manual (Carrier Wave, Message Signal,
Modulation, and I/Q Modulation). Also the Python counterpart of the NI-RFSG .NET
"565xModulation/Analog" and "565xModulation/Digital" examples: vector signal generators
such as the PXIe-5840/5841/5842 produce these modulations from arbitrary I/Q waveforms
instead of from dedicated modulation hardware.

Background:
    A carrier wave cos(2*pi*fc*t) carries information when its amplitude or phase is varied
    by a message signal. An I/Q modulator varies both at once:

        s(t) = I(t) * cos(2*pi*fc*t) - Q(t) * sin(2*pi*fc*t)
             = R(t) * cos(2*pi*fc*t + phi(t)),   with I + jQ = R * exp(j*phi)

    The rectangular form (I, Q) and the polar form (R, phi) describe the same signal. NI-RFSG
    generates the carrier at the frequency property, so the program only provides the complex
    baseband samples I + jQ:
      AM    - R(t) = 1 + m*x(t), phi(t) = 0. Only the amplitude changes (I varies, Q = 0).
              m is the modulation index (0 to 1).
      FM    - R(t) = 1, phi(t) = 2*pi*deviation*integral(x). The instantaneous frequency
              follows the message. Occupied bandwidth (Carson's rule): 2*(deviation + fm).
      PM    - R(t) = 1, phi(t) = deviation_rad*x(t). The phase follows the message.
      BPSK, QPSK, 16-QAM
            - Digital symbols are points in the I/Q plane. BPSK uses 2 phases, QPSK 4 phases,
              and 16-QAM 16 combinations of amplitude and phase. Each symbol carries
              1, 2, or 4 bits. The symbols are shaped with a root-raised-cosine (RRC) filter,
              which limits the occupied bandwidth to symbol_rate * (1 + rolloff).
    Angle-modulated signals (FM, PM, PSK before filtering) have a constant envelope, which
    gives a low peak-to-average power ratio (PAPR). AM and QAM vary the envelope and have a
    higher PAPR.

    Sample rate: NI-RFSG filters the waveform with an FIR filter that is flat up to
    0.4 * iq_rate, so the occupied bandwidth must be at most 0.8 * iq_rate. Higher I/Q rates
    represent the signal more accurately but use more memory per second of signal.

    The waveform holds a whole number of message periods (analog) or a whole number of symbols
    with circular filtering (digital), so it repeats without discontinuities. With the AVERAGE
    power level type, power_level is the average power of the modulated signal.

Steps:
    1. Compute the baseband I/Q waveform for the selected modulation with numpy.
    2. Configure the RF settings, ARB mode, the I/Q rate, and the occupied signal bandwidth.
    3. Write the waveform and generate it.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

ANALOG = ['AM', 'FM', 'PM']
DIGITAL = {'BPSK': 1, 'QPSK': 2, '16QAM': 4}


def _analog_waveform(modulation, iq_rate, message_frequency, depth, number_of_periods):
    # A whole number of message periods, so the waveform repeats seamlessly.
    samples_per_period = int(round(iq_rate / message_frequency))
    actual_message_frequency = iq_rate / samples_per_period
    t = np.arange(samples_per_period * number_of_periods) / iq_rate
    message = np.cos(2 * np.pi * actual_message_frequency * t)

    if modulation == 'AM':
        # depth is the modulation index m (0 to 1). The envelope swings between 1 - m and 1 + m.
        iq = (1 + depth * message) + 0j
        bandwidth = 2 * actual_message_frequency
    elif modulation == 'FM':
        # depth is the peak frequency deviation in Hz. The phase is the integral of the
        # instantaneous frequency deviation * cos(2*pi*fm*t).
        iq = np.exp(1j * (depth / actual_message_frequency) * np.sin(2 * np.pi * actual_message_frequency * t))
        bandwidth = 2 * (depth + actual_message_frequency)
    else:
        # depth is the peak phase deviation in radians.
        iq = np.exp(1j * depth * message)
        bandwidth = 2 * (depth + 1) * actual_message_frequency
    return iq, bandwidth, f'message {actual_message_frequency / 1e3:.3f} kHz'


def _rrc_taps(samples_per_symbol, rolloff, span_symbols):
    t = np.arange(-span_symbols * samples_per_symbol // 2, span_symbols * samples_per_symbol // 2 + 1) / samples_per_symbol
    taps = np.empty(len(t))
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1 - rolloff + 4 * rolloff / np.pi
        elif rolloff > 0 and np.isclose(abs(ti), 1 / (4 * rolloff)):
            taps[i] = rolloff / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            taps[i] = (np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))) / (np.pi * ti * (1 - (4 * rolloff * ti) ** 2))
    return taps / np.sqrt(np.sum(taps ** 2))


def _digital_waveform(modulation, iq_rate, symbol_rate, rolloff, number_of_symbols):
    samples_per_symbol = max(int(round(iq_rate / symbol_rate)), 2)
    actual_symbol_rate = iq_rate / samples_per_symbol
    bits_per_symbol = DIGITAL[modulation]
    bits = np.random.default_rng(0).integers(0, 2, (number_of_symbols, bits_per_symbol))

    if modulation == 'BPSK':
        symbols = (2 * bits[:, 0] - 1) + 0j
    elif modulation == 'QPSK':
        symbols = ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)
    else:
        # 16-QAM: two Gray-coded bits per axis select one of the levels -3, -1, 1, 3.
        gray_levels = np.array([-3, -1, 3, 1])
        symbols = (gray_levels[2 * bits[:, 0] + bits[:, 1]] + 1j * gray_levels[2 * bits[:, 2] + bits[:, 3]]) / np.sqrt(10)

    # Upsample, then apply the RRC filter as a circular convolution, so that the end of the
    # waveform joins smoothly to its start when it repeats.
    upsampled = np.zeros(number_of_symbols * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols
    taps = _rrc_taps(samples_per_symbol, rolloff, span_symbols=8)
    kernel = np.zeros(len(upsampled))
    kernel[:len(taps)] = taps
    kernel = np.roll(kernel, -(len(taps) // 2))
    iq = np.fft.ifft(np.fft.fft(upsampled) * np.fft.fft(kernel))
    bandwidth = actual_symbol_rate * (1 + rolloff)
    return iq, bandwidth, f'{actual_symbol_rate / 1e3:.3f} ksym/s, {bits_per_symbol} bit(s)/symbol, rolloff {rolloff}'


def example(resource_name, options, frequency, power_level, iq_rate, modulation, message_frequency, depth, symbol_rate, rolloff, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.power_level_type = nirfsg.PowerLevelType.AVERAGE
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        if modulation in ANALOG:
            iq, bandwidth, description = _analog_waveform(modulation, actual_iq_rate, message_frequency, depth, number_of_periods=10)
        else:
            iq, bandwidth, description = _digital_waveform(modulation, actual_iq_rate, symbol_rate, rolloff, number_of_symbols=1024)

        # Content beyond 0.4 * iq_rate on either side of the carrier is attenuated by the
        # interpolation filter (and aliases if it exceeds 0.5 * iq_rate).
        if bandwidth > 0.8 * actual_iq_rate:
            raise ValueError(f'The occupied bandwidth ({bandwidth / 1e6:.3f} MHz) exceeds 0.8 * I/Q rate ({0.8 * actual_iq_rate / 1e6:.3f} MHz). Increase the I/Q rate.')
        session.signal_bandwidth = bandwidth

        # Normalize the largest sample to a magnitude of 1.0.
        iq = (iq / np.max(np.abs(iq))).astype(np.complex64)
        papr_db = 10 * np.log10(np.max(np.abs(iq) ** 2) / np.mean(np.abs(iq) ** 2))
        session.write_arb_waveform('modulated', iq, False)

        print(f'{modulation}: {description}, occupied bandwidth {bandwidth / 1e3:.1f} kHz, PAPR {papr_db:.2f} dB, {len(iq)} samples at {actual_iq_rate / 1e6} MS/s')
        with session.initiate():
            print(f'Generating {power_level} dBm average power at {frequency / 1e6} MHz for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates AM, FM, PM, BPSK, QPSK, or 16-QAM modulation from computed I/Q samples using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-m', '--modulation', default='QPSK', choices=ANALOG + list(DIGITAL), type=str.upper, help='Modulation type.')
    parser.add_argument('-mf', '--message-frequency', default=10e3, type=float, help='Message tone frequency, in Hz, for AM, FM, and PM.')
    parser.add_argument('-d', '--depth', default=0.5, type=float, help='Modulation depth: index (0 to 1) for AM, peak deviation in Hz for FM, peak deviation in radians for PM.')
    parser.add_argument('-sr', '--symbol-rate', default=1e6, type=float, help='Symbol rate, in symbols per second, for BPSK, QPSK, and 16QAM.')
    parser.add_argument('-a', '--rolloff', default=0.35, type=float, help='Root-raised-cosine filter rolloff (0 to 1) for BPSK, QPSK, and 16QAM.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.modulation, args.message_frequency, args.depth, args.symbol_rate, args.rolloff, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    depths = {'AM': 0.5, 'FM': 50e3, 'PM': 1.0}
    for modulation in ANALOG + list(DIGITAL):
        example('simulated5841', options, 1e9, -10.0, 10e6, modulation, 10e3, depths.get(modulation, 0.5), 1e6, 0.35, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
