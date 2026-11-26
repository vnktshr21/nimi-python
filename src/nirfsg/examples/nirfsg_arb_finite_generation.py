'''Generates an arbitrary waveform a finite number of times and waits for the generation to complete.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/FiniteGeneration" example.

Background:
    In arbitrary waveform (ARB) mode the device plays complex baseband I/Q samples from its
    onboard memory, upconverted to the RF carrier set by the frequency property. By default
    the selected waveform repeats until abort() is called. If you set
    arb_waveform_repeat_count_is_finite to True, the device plays the waveform exactly
    arb_waveform_repeat_count times and then stops. check_generation_status() returns True
    once the generation is done.

    Waveform sizes must respect the device capabilities returned by
    query_arb_waveform_capabilities(): the number of samples must be at least
    min_waveform_size, at most max_waveform_size, and a multiple of waveform_quantum.

    This example plays a single tone at a frequency offset from the carrier. The offset is
    snapped so that a whole number of tone cycles fits in the waveform. Then consecutive
    repetitions join without a phase discontinuity, which would otherwise spread energy
    into spurious tones.

Steps:
    1. Configure the frequency, power level, ARB generation mode, and I/Q rate.
    2. Query the waveform capabilities and choose a valid waveform size.
    3. Build the tone with numpy and set the signal bandwidth.
    4. Set the repeat count to a finite value.
    5. Write the waveform, start the generation, and poll until it completes or times out.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def _coerce_waveform_size(requested_size, waveform_quantum, min_waveform_size):
    # Round up to the next multiple of the waveform quantum and enforce the minimum size.
    size = max(requested_size, min_waveform_size)
    return int(np.ceil(size / waveform_quantum) * waveform_quantum)


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, waveform_duration, repeat_count, timeout):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM

        # The driver may coerce the I/Q rate to a value that the hardware supports, so read it
        # back and use the actual rate to compute the waveform.
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # Returns (max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size).
        _, waveform_quantum, min_waveform_size, max_waveform_size = session.query_arb_waveform_capabilities()
        number_of_samples = _coerce_waveform_size(int(round(actual_iq_rate * waveform_duration)), waveform_quantum, min_waveform_size)
        if number_of_samples > max_waveform_size:
            raise ValueError(f'The waveform needs {number_of_samples} samples, but the device supports at most {max_waveform_size}.')

        # Snap the tone to an integer number of cycles over the waveform so that it repeats seamlessly.
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        actual_tone_offset = cycles * actual_iq_rate / number_of_samples
        t = np.arange(number_of_samples) / actual_iq_rate
        waveform = np.exp(2j * np.pi * actual_tone_offset * t).astype(np.complex64)

        # The signal bandwidth tells the driver how much of the spectrum around the carrier the
        # waveform occupies. A tone at +/-offset needs at least 2 * |offset| of bandwidth. The
        # 1 Hz floor keeps the value valid when the tone sits exactly on the carrier.
        session.signal_bandwidth = max(2 * abs(actual_tone_offset), 1.0)

        # Play the waveform repeat_count times and then stop.
        session.arb_waveform_repeat_count_is_finite = True
        session.arb_waveform_repeat_count = repeat_count

        session.write_arb_waveform('finiteWaveform', waveform, False)

        expected_duration = repeat_count * number_of_samples / actual_iq_rate
        print(f'I/Q rate: {actual_iq_rate / 1e6} MS/s, waveform: {number_of_samples} samples, tone offset: {actual_tone_offset / 1e3:.3f} kHz')
        print(f'Generating the waveform {repeat_count} times (about {expected_duration:.3f} s)...')

        with session.initiate():
            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                # Returns True after the last repetition has been generated. Raises an
                # nirfsg.Error if an error occurred during the generation.
                is_done = session.check_generation_status()
                time.sleep(0.01)

            if is_done:
                print(f'Generation complete after {time.monotonic() - start_time:.3f} s.')
            else:
                print(f'Generation did not complete within {timeout} s; aborting.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates an arbitrary waveform a finite number of times using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-d', '--waveform-duration', default=1e-3, type=float, help='Duration, in seconds, of one repetition of the waveform.')
    parser.add_argument('-c', '--repeat-count', default=1000, type=int, help='Number of times to generate the waveform.')
    parser.add_argument('-t', '--timeout', default=10.0, type=float, help='Maximum time, in seconds, to wait for the generation to complete.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.waveform_duration, args.repeat_count, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 1e-3, 10, 0.5)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--repeat-count', '10', '--timeout', '0.5', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
