'''Delays the Marker, Started, and Done Events relative to the RF output, with the events delay property.

Based on the "Marker Event Details" topic of the NI-RFSG user manual.

Background:
    By default, NI-RFSG delays its events so that they line up with the waveform data at the RF
    output. events_delay adds an extra delay, in seconds, to the Started Event, the Done Event,
    and all Marker Events. Use it to compensate for delays outside the generator, for example
    a long cable to the device under test or the trigger latency of the instrument that receives
    the events. On the PXIe-5820/5830/5831/5832/5840/5841/5842 the valid range is 0 to 3.276 us.
    events_delay can be set in the Configuration or the Generation state; decreasing it while
    generating can drop markers.

    This example generates a tone with a marker at the start of every repetition, exported to a
    terminal, and steps events_delay while generating. On an oscilloscope, the marker moves
    relative to the RF burst edges.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write a tone.
    2. Place marker0 at sample 0 and export it.
    3. Start the generation and step events_delay through increasing values.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, marker_terminal, event_delays, step_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        cycles = round(number_of_samples / 10)
        session.signal_bandwidth = 2 * cycles * actual_iq_rate / number_of_samples
        session.write_arb_waveform('tone', np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)
        session.waveforms['tone'].markers[0].set_waveform_marker_event_locations([0])
        session.markers[0].exported_marker_event_output_terminal = marker_terminal

        with session.initiate():
            # Increase the delay step by step; decreasing it while generating can drop markers.
            for delay in sorted(event_delays):
                session.events_delay = delay
                session.check_generation_status()
                print(f'events_delay: {session.events_delay.total_seconds() * 1e6:.3f} us')
                time.sleep(step_time)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Delays the NI-RFSG Marker, Started, and Done Events relative to the RF output.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-m', '--marker-terminal', default='PFI0', help='Terminal to export marker0 to.')
    parser.add_argument('-d', '--event-delays', default=[0.0, 1e-6, 2e-6, 3e-6], type=float, nargs='+', help='Event delays to apply in turn, in seconds.')
    parser.add_argument('-t', '--step-time', default=2.0, type=float, help='Time, in seconds, to generate at each delay.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.marker_terminal, args.event_delays, args.step_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 'PXI_Trig0', [0.0, 1e-6], 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--step-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
