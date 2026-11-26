'''Places Marker Events at sample positions of an arbitrary waveform, in ARB generation mode.

Python counterpart of the NI-RFSG .NET "ArbitraryWaveforms/ConfigureMarkerAtSample0" example.

Background:
    In SCRIPT mode, markers are placed in the script ("generate wfm marker0(0)"). In ARB mode
    there is no script, so marker positions are stored with the waveform itself, through the
    waveforms[] and markers[] repeated capabilities:
        session.waveforms[name].markers[N].set_waveform_marker_event_locations(locations)
    Marker Event N then fires at those sample offsets every time the waveform plays. A marker
    at sample 0 marks the start of every repetition, which is useful to trigger an analyzer or
    an oscilloscope on the waveform period. get_waveform_marker_event_locations() reads the
    positions back.

    Set the marker locations after the waveform has been written. Route the Marker Event to a
    terminal with session.markers[N].exported_marker_event_output_terminal.

Steps:
    1. Configure the RF settings, ARB mode, I/Q rate, and signal bandwidth, and write a tone.
    2. Set the marker0 locations on the waveform and read them back.
    3. Export marker0 to the requested terminal and generate the waveform.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, number_of_samples, marker_locations, marker_terminal, generation_time):
    waveform_name = 'markedWaveform'
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # A tone at one tenth of the I/Q rate, with a whole number of cycles in the waveform.
        cycles = round(number_of_samples / 10)
        session.signal_bandwidth = 2 * cycles * actual_iq_rate / number_of_samples
        waveform = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        session.write_arb_waveform(waveform_name, waveform, False)

        # The marker locations belong to the waveform, so they are set after it is written.
        marker = session.waveforms[waveform_name].markers[0]
        marker.set_waveform_marker_event_locations(marker_locations)
        print(f'marker0 locations: {marker.get_waveform_marker_event_locations()}')

        session.markers[0].exported_marker_event_output_terminal = marker_terminal
        print(f'marker0 exported to {marker_terminal or "(not exported)"}; one marker period is {number_of_samples / actual_iq_rate * 1e6:.1f} us')

        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Places Marker Events at sample positions of an arbitrary waveform using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in the waveform.')
    parser.add_argument('-m', '--marker-locations', default=[0], type=int, nargs='+', help='Sample positions of marker0 within the waveform.')
    parser.add_argument('-e', '--marker-terminal', default='', help='Terminal to export marker0 to, for example PFI0 or PXI_Trig0. Leave empty to not export.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_samples, args.marker_locations, args.marker_terminal, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 10000, [0, 5000], 'PXI_Trig0', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
