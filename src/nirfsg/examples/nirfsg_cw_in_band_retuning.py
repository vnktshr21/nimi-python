'''Performs a fast forward frequency sweep using in-band retuning.

Python counterpart of the NI-RFSG .NET "Sweeps/ForwardFrequencySweep(5672In-bandRetuning)" and
"Sweeps/ForwardFrequencySweep(5673In-bandRetuning)" examples.

Background:
    A vector signal generator upconverts its baseband signal with an LO. The band that can be
    reached without moving the LO is the instantaneous bandwidth of the device
    (device_instantaneous_bandwidth), centered on upconverter_center_frequency. When you change
    frequency normally, the driver retunes the LO for every point, which requires the LO
    synthesizer to lock and settle.

    With in-band retuning, you fix the upconverter center frequency yourself. Frequencies that
    fall inside the passband around it are then reached by shifting the signal digitally,
    without moving the LO, which is much faster. Writing upconverter_center_frequency enables
    in-band retuning on devices such as the PXIe-5644/5645/5646, PXIe-5672/5673/5673E, and
    PXIe-5820/5840/5841. The sweep uses on-the-fly (OTF) reconfiguration, as in
    nirfsg_cw_frequency_sweep_otf.py: properties are written while the device is generating.
    While generating, a new upconverter center frequency takes effect only when a dynamic
    property, such as frequency, is set next.

    This example sweeps upward. It places the upconverter center frequency so that the
    passband starts at the first point. When the next point falls outside the usable part of
    the passband, it moves the upconverter center frequency forward by one passband.

Steps:
    1. Configure CW mode, the start frequency, and the power level, and read the instantaneous
       bandwidth.
    2. Set the first upconverter center frequency and start the generation.
    3. For each point: move the upconverter center frequency if needed, write the frequency,
       and wait for the output to settle.
'''

import argparse
import datetime
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, start_frequency, stop_frequency, frequency_step, power_level, usable_bandwidth_fraction, dwell_time):
    frequencies = np.arange(start_frequency, stop_frequency + frequency_step / 2, frequency_step)

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequencies[0], power_level)

        # Use only part of the instantaneous bandwidth, which leaves some margin at the edges
        # of the passband.
        instantaneous_bandwidth = session.device_instantaneous_bandwidth
        half_span = instantaneous_bandwidth * usable_bandwidth_fraction / 2
        print(f'Instantaneous bandwidth: {instantaneous_bandwidth / 1e6} MHz, usable passband: +/-{half_span / 1e6} MHz')

        # Center the first passband so that its lower edge is the start frequency.
        upconverter_center_frequency = frequencies[0] + half_span
        session.upconverter_center_frequency = upconverter_center_frequency

        lo_retunes = 0
        with session.initiate():
            for frequency in frequencies:
                if abs(frequency - upconverter_center_frequency) > half_span:
                    # Out of the current passband: move the LO forward. The new value takes
                    # effect when the frequency is written below.
                    upconverter_center_frequency = frequency + half_span
                    session.upconverter_center_frequency = upconverter_center_frequency
                    lo_retunes += 1

                session.frequency = frequency
                session.wait_until_settled(datetime.timedelta(seconds=10))
                session.check_generation_status()
                print(f'{frequency / 1e6:10.3f} MHz (upconverter center {session.upconverter_center_frequency / 1e6:10.3f} MHz)')
                time.sleep(dwell_time)

        print(f'Swept {len(frequencies)} points with {lo_retunes} LO retunes.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Performs a fast forward frequency sweep using in-band retuning in NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-start', '--start-frequency', default=1e9, type=float, help='Start frequency in Hz.')
    parser.add_argument('-stop', '--stop-frequency', default=2e9, type=float, help='Stop frequency in Hz.')
    parser.add_argument('-step', '--frequency-step', default=50e6, type=float, help='Frequency step in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-u', '--usable-bandwidth-fraction', default=0.8, type=float, help='Fraction of the instantaneous bandwidth to use before moving the LO.')
    parser.add_argument('-d', '--dwell-time', default=0.1, type=float, help='Time, in seconds, to dwell at each frequency point.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.start_frequency, args.stop_frequency, args.frequency_step, args.power_level, args.usable_bandwidth_fraction, args.dwell_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, 1.2e9, 50e6, -10.0, 0.8, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--frequency-step', '250e6', '--dwell-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
