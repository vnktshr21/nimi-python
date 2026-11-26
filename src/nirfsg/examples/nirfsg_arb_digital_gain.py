'''Changes the output power with the arbitrary waveform digital gain while the device is generating.

Background:
    arb_digital_gain multiplies the waveform data by a gain, in dB, after the interpolation
    filters. It changes the output power without changing the analog signal path (attenuators
    and amplifiers), and it can be changed while the device is in the Generation state. This
    is useful for small, fast level changes, for example to step a signal through a receiver's
    dynamic range.

    The digital gain does not change the power_level property. The actual output power is
    approximately power_level + arb_digital_gain. Positive gains can clip the waveform or cause
    overflows; keep the gain at or below 0 dB unless the waveform has headroom.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write a tone.
    2. Start the generation.
    3. Step arb_digital_gain through the requested values while generating.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, digital_gain_steps, step_time):
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

        with session.initiate():
            for digital_gain in digital_gain_steps:
                # The digital gain can be changed while the device is generating.
                session.arb_digital_gain = digital_gain
                session.check_generation_status()
                print(f'Digital gain {session.arb_digital_gain:+.1f} dB: output power about {power_level + digital_gain:.1f} dBm')
                time.sleep(step_time)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Changes the output power with the arbitrary waveform digital gain while NI-RFSG is generating.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-dg', '--digital-gain-steps', default=[0.0, -3.0, -6.0, -9.0], type=float, nargs='+', help='Digital gain values, in dB, to apply in turn while generating.')
    parser.add_argument('-t', '--step-time', default=2.0, type=float, help='Time, in seconds, to generate at each digital gain value.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.digital_gain_steps, args.step_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, [0.0, -3.0], 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--step-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
