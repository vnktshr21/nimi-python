'''Emits Marker Events at chosen samples of a script, configures their pulse or toggle output, and exports them.

Python counterpart of the NI-RFSG .NET "Scripts/MarkerEvents" example. Based on the "Marker Event
Details" topic of the NI-RFSG user manual.

Background:
    A Marker Event is a digital pulse that the device emits when generation reaches a given
    sample of a waveform. Markers are used to trigger other instruments (for example an
    analyzer or an oscilloscope) at an exact point in the generated signal. There are four
    markers, marker0 to marker3. In a script, you place a marker by naming it after the
    generate instruction, with one or more sample offsets into that waveform:

        generate waveformWithMarkers marker0(0, 5000)

    (In ARB generation mode, marker positions are stored with the waveform instead; refer to
    nirfsg_arb_marker_events.py.)

    Each marker is a repeated capability, so its properties are set through session.markers[N]:
      - exported_marker_event_output_terminal: where the pulse is routed, such as 'PFI0' or
        'PXI_Trig0'. An empty string means the marker is not exported.
      - marker_event_output_behavior: PULSE emits a pulse of marker_event_pulse_width (in
        marker_event_pulse_width_units) at each marker position. TOGGLE flips the output
        level at each marker position, starting from marker_event_toggle_initial_state.
    get_terminal_name() and the marker_event_terminal_name property return the fully qualified
    name of the Marker Event, which another device can use as its trigger source.

    With PULSE, every marker pulse has the same width. With TOGGLE, pairs of marker positions
    create a gate of any length: for example marker0(1000, 3000) drives the output high from
    sample 1000 to sample 3000, which can gate an external amplifier or define a measurement
    window. The marker reaches the output terminal slightly after the marked sample reaches
    RF OUT; refer to nirfsg_event_delay.py to adjust this alignment.

Steps:
    1. Configure the RF settings, SCRIPT mode, I/Q rate, and signal bandwidth.
    2. Configure the routing and output behavior of marker0.
    3. Write a tone waveform and a script that places marker0 at the requested samples.
    4. Generate for the requested time.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, number_of_samples, marker_positions, output_terminal, output_behavior, pulse_width, generation_time):
    positions = ', '.join(str(p) for p in marker_positions)
    script = f'''script generateWaveformWithMarkers
    repeat forever
        generate waveformWithMarkers marker0({positions})
    end repeat
end script'''

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate

        # Tone at one tenth of the I/Q rate, with a whole number of cycles per waveform.
        tone_offset = actual_iq_rate / 10
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        waveform = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
        session.signal_bandwidth = 2 * cycles * actual_iq_rate / number_of_samples

        # Configure marker0 through the markers repeated capability.
        marker = session.markers[0]
        marker.exported_marker_event_output_terminal = output_terminal
        marker.marker_event_output_behavior = nirfsg.MarkerEventOutputBehavior[output_behavior]
        if output_behavior == 'PULSE':
            marker.marker_event_pulse_width_units = nirfsg.MarkerEventPulseWidthUnits.SECONDS
            marker.marker_event_pulse_width = pulse_width
        else:
            marker.marker_event_toggle_initial_state = nirfsg.MarkerEventToggleInitialState.LOW

        session.write_arb_waveform('waveformWithMarkers', waveform, False)
        session.write_script(script)

        print(f'marker0 at samples [{positions}] of each {number_of_samples}-sample waveform, behavior: {output_behavior.lower()}')
        print(f'marker0 exported to: {output_terminal or "(not exported)"}')
        print(f'marker0 signal name: {session.get_terminal_name(nirfsg.Signal.MARKER_EVENT, "marker0")}')
        print(f'marker0 terminal name property: {marker.marker_event_terminal_name}')

        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Emits NI-RFSG Marker Events from a script, configures their output behavior, and exports them.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in the waveform.')
    parser.add_argument('-m', '--marker-positions', default=[0, 5000], type=int, nargs='+', help='Sample offsets within the waveform at which marker0 occurs.')
    parser.add_argument('-e', '--output-terminal', default='', help='Terminal to export marker0 to, for example PFI0 or PXI_Trig0. Leave empty to not export.')
    parser.add_argument('-b', '--output-behavior', default='PULSE', choices=nirfsg.MarkerEventOutputBehavior.__members__.keys(), type=str.upper, help='Whether the marker emits a pulse or toggles its output.')
    parser.add_argument('-w', '--pulse-width', default=200e-9, type=float, help='Marker pulse width in seconds. Used only with the PULSE behavior.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_samples, args.marker_positions, args.output_terminal, args.output_behavior, args.pulse_width, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 10000, [0, 2500, 5000], 'PXI_Trig0', 'TOGGLE', 200e-9, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--output-terminal', 'PXI_Trig0', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
