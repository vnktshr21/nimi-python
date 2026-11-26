'''Generates a frequency-hopping signal from a script, with per-hop power, hop markers, and trigger control.

This advanced example builds a frequency-hopping signal without retuning the LO. Each hop
channel is a short tone waveform at a different frequency offset inside the I/Q bandwidth, and
a script plays the hops in a pseudo-random order. Because the hops are switched digitally, the
hop time is set by the waveform length, not by LO settling. The example combines:

  Arb - waveform attributes (session.waveforms[name]):
      waveform_runtime_scaling   a different level for each hop channel, for example to
                                 emulate a frequency-dependent path loss (a power tilt)
  Arb - script mode:
      The hop pattern is written as a script. "repeat until scriptTrigger1" keeps hopping
      until a stop request arrives. The trigger is checked once per pass through the
      pattern, so the pattern always completes before the generation stops. The script then
      ends, the generation completes, and the Done Event is asserted.
  Triggers:
      Start Trigger   - starts hopping (software or digital edge).
      scriptTrigger1  - stops hopping (software or digital edge).
  Events:
      marker0 at sample 0 of every hop gives a hop clock that a receiver can follow.
      The Done Event signals the end of the hopping. Both can be exported to terminals.
  RF:
      signal_bandwidth covers the whole hop band. phase_continuity_enabled is DISABLE, because
      each hop starts a new tone.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and phase continuity.
    2. Write one tone waveform per hop channel and set its runtime scaling.
    3. Generate a pseudo-random hop pattern and write it as a script.
    4. Configure the Start Trigger, the stop trigger (scriptTrigger1), and the exports.
    5. Initiate and start hopping, request a stop after the hopping time, and wait for the
       Done state.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

TRIGGER_TYPES = ['SOFTWARE', 'DIGITAL_EDGE']


def example(resource_name, options, frequency, power_level, iq_rate, number_of_channels, channel_spacing, hop_duration, hops_per_pattern, power_tilt_db,
            start_trigger_type, start_trigger_source, stop_trigger_type, stop_trigger_source, marker_terminal, done_event_terminal, hopping_time, timeout):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.phase_continuity_enabled = nirfsg.PhaseContinuityEnabled.DISABLE

        hop_band = number_of_channels * channel_spacing
        if hop_band > 0.8 * actual_iq_rate:
            raise ValueError(f'The hop band ({hop_band / 1e6} MHz) must fit within 80% of the I/Q rate ({actual_iq_rate / 1e6} MS/s).')
        session.signal_bandwidth = hop_band

        # One waveform per hop channel, with a length that is a multiple of the waveform quantum.
        _, waveform_quantum, min_waveform_size, _ = session.query_arb_waveform_capabilities()
        hop_samples = max(int(round(actual_iq_rate * hop_duration / waveform_quantum)) * waveform_quantum, min_waveform_size, waveform_quantum)
        t = np.arange(hop_samples) / actual_iq_rate

        # Channel offsets centered on the carrier, and a linear power tilt across the band. The
        # scaling only attenuates (0 dB down to -|power_tilt_db|), so that with PEAK power no hop
        # exceeds power_level.
        offsets = (np.arange(number_of_channels) - (number_of_channels - 1) / 2) * channel_spacing
        tilts = np.linspace(0.0, -abs(power_tilt_db), number_of_channels)
        for channel, (offset, tilt) in enumerate(zip(offsets, tilts)):
            name = f'hop{channel}'
            session.write_arb_waveform(name, np.exp(2j * np.pi * offset * t).astype(np.complex64), False)
            hop = session.waveforms[name]
            hop.waveform_runtime_scaling = tilt
            print(f'{name:6}: {(frequency + offset) / 1e6:10.3f} MHz, runtime scaling {hop.waveform_runtime_scaling:+.2f} dB')

        # Pseudo-random hop pattern in which consecutive hops use different channels.
        rng = np.random.default_rng(7)
        pattern = [int(rng.integers(number_of_channels))]
        while len(pattern) < hops_per_pattern:
            channel = int(rng.integers(number_of_channels))
            if channel != pattern[-1] or number_of_channels == 1:
                pattern.append(channel)
        hops = '\n'.join(f'        generate hop{channel} marker0(0)' for channel in pattern)
        script = f'''script frequencyHopping
    repeat until scriptTrigger1
{hops}
    end repeat
end script'''
        session.write_script(script)
        session.selected_script = 'frequencyHopping'
        print(f'Hop pattern: {pattern} ({hop_samples / actual_iq_rate * 1e6:.1f} us per hop, {len(pattern) * hop_samples / actual_iq_rate * 1e3:.3f} ms per pattern)')

        # Triggers: the Start Trigger starts hopping, scriptTrigger1 stops it.
        if start_trigger_type == 'SOFTWARE':
            session.configure_software_start_trigger()
        else:
            session.configure_digital_edge_start_trigger(start_trigger_source, nirfsg.StartTriggerDigitalEdgeEdge.RISING)
        if stop_trigger_type == 'SOFTWARE':
            session.script_triggers[1].configure_software_script_trigger()
        else:
            session.script_triggers[1].configure_digital_edge_script_trigger(stop_trigger_source, nirfsg.ScriptTriggerDigitalEdgeEdge.RISING)

        # Events: hop clock on marker0, and the Done Event at the end of the hopping.
        session.markers[0].exported_marker_event_output_terminal = marker_terminal
        session.exported_done_event_output_terminal = done_event_terminal
        print(f'Hop clock (marker0) exported to {marker_terminal or "(not exported)"}; Done Event {session.done_event_terminal_name} exported to {done_event_terminal or "(not exported)"}')

        with session.initiate():
            if start_trigger_type == 'SOFTWARE':
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
                print('Hopping started.')
            else:
                print(f'Waiting for the Start Trigger on {start_trigger_source}.')

            end_time = time.monotonic() + hopping_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)

            if stop_trigger_type == 'SOFTWARE':
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER1)
                print('Stop requested; finishing the current hop pattern.')
            else:
                print(f'Waiting for the stop trigger on {stop_trigger_source}.')

            # The script ends after the current pass through the pattern; the generation is then done.
            stop_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - stop_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('Hopping complete.' if is_done else f'Hopping did not stop within {timeout} s; aborting.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a triggered frequency-hopping signal from a script, with per-hop power and hop markers, using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=2.44e9, type=float, help='Center frequency of the hop band, in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm (before the per-hop runtime scaling).')
    parser.add_argument('-r', '--iq-rate', default=100e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-nc', '--number-of-channels', default=16, type=int, help='Number of hop channels.')
    parser.add_argument('-cs', '--channel-spacing', default=2e6, type=float, help='Spacing between hop channels, in Hz.')
    parser.add_argument('-hd', '--hop-duration', default=100e-6, type=float, help='Duration of each hop, in seconds.')
    parser.add_argument('-hp', '--hops-per-pattern', default=32, type=int, help='Number of hops in the repeating hop pattern.')
    parser.add_argument('-pt', '--power-tilt-db', default=6.0, type=float, help='Attenuation, in dB, of the highest channel relative to the lowest channel, which is at 0 dB.')
    parser.add_argument('-stt', '--start-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='Start Trigger type.')
    parser.add_argument('-sts', '--start-trigger-source', default='PXI_Trig0', help='Source of a DIGITAL_EDGE Start Trigger.')
    parser.add_argument('-spt', '--stop-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='Type of the stop trigger (scriptTrigger1).')
    parser.add_argument('-sps', '--stop-trigger-source', default='PXI_Trig1', help='Source of a DIGITAL_EDGE stop trigger.')
    parser.add_argument('-m', '--marker-terminal', default='', help='Terminal to export the hop clock (marker0) to. Leave empty to not export.')
    parser.add_argument('-de', '--done-event-terminal', default='', help='Terminal to export the Done Event to. Leave empty to not export.')
    parser.add_argument('-t', '--hopping-time', default=5.0, type=float, help='Time, in seconds, to hop before requesting a stop.')
    parser.add_argument('-to', '--timeout', default=5.0, type=float, help='Maximum time, in seconds, to wait for the hopping to stop.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.number_of_channels, args.channel_spacing, args.hop_duration,
            args.hops_per_pattern, args.power_tilt_db, args.start_trigger_type, args.start_trigger_source, args.stop_trigger_type, args.stop_trigger_source,
            args.marker_terminal, args.done_event_terminal, args.hopping_time, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 2.44e9, -10.0, 100e6, 8, 2e6, 100e-6, 16, 6.0, 'SOFTWARE', 'PXI_Trig0', 'SOFTWARE', 'PXI_Trig1', 'PXI_Trig2', 'PXI_Trig3', 0.05, 0.2)
    example('simulated5841', options, 2.44e9, -10.0, 100e6, 4, 5e6, 50e-6, 8, 0.0, 'DIGITAL_EDGE', 'PXI_Trig0', 'DIGITAL_EDGE', 'PXI_Trig1', '', '', 0.05, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--hopping-time', '0.05', '--timeout', '0.2', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
