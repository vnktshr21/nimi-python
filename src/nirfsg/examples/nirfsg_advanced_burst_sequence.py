'''Generates a triggered burst sequence that combines waveform attributes, a script, triggers, and events.

This advanced example combines several groups of the NI-RFSG API in one realistic stimulus: a
packet-like signal in which each packet is a preamble followed by several payload bursts, sent
each time a trigger arrives. It uses:

  Arb - waveform attributes (per-waveform properties, set through session.waveforms[name]):
      waveform_signal_bandwidth   occupied bandwidth of each waveform
      waveform_runtime_scaling    gain, in dB, applied to one waveform when it is generated. Here
                                  it sets the payload level relative to the preamble without
                                  changing the data or the power level.
      set_waveform_burst_start_locations() / set_waveform_burst_stop_locations()
                                  where the RF burst is within the payload waveform
      waveform_rf_blanking        blanks the RF output outside those bursts for a better on/off ratio
      waveform_waveform_size      read back to confirm the stored size
  Arb - script mode:
      write_script() and selected_script define the packet structure. The script waits on
      scriptTrigger0 while generating an idle (all-zeros) waveform, then plays the preamble once
      and the payload N times, and repeats. Markers placed in the script mark the start of
      each preamble (marker0) and of each payload burst (marker1).
  Triggers:
      Start Trigger    - arms the generation and starts it on a software or digital-edge trigger.
      scriptTrigger0   - sends one packet per trigger, from software or a digital edge.
  Events:
      markers[0..1].exported_marker_event_output_terminal, exported_started_event_output_terminal,
      and get_terminal_name() route the timing signals to other instruments, such as an
      analyzer that captures each packet.

The waveforms:
  preamble - a Zadoff-Chu sequence. It has a constant envelope (0 dB PAPR) and a flat spectrum,
             which makes it a common synchronization sequence.
  payload  - random QPSK symbols shaped with a windowed-sinc filter, with zero-valued guard
             intervals before and after the burst.
  idle     - all zeros, generated while waiting for the next trigger.
power_level_type is PEAK, so power_level is the power of the largest sample of the preamble.
PEAK also suits the all-zeros idle waveform, which has no average power.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, and the I/Q rate.
    2. Build and write the preamble, payload, and idle waveforms.
    3. Set the per-waveform attributes and print them.
    4. Write the script.
    5. Configure the Start Trigger, scriptTrigger0, and the marker and event exports.
    6. Initiate, send the triggers (for software triggering), and monitor the generation.
'''

import argparse
import math
import nirfsg
import numpy as np
import sys
import time

TRIGGER_TYPES = ['SOFTWARE', 'DIGITAL_EDGE']


def _round_up(value, quantum):
    return int(np.ceil(value / quantum) * quantum)


def _zadoff_chu(length, root):
    # The root must be coprime with the length; fall back to the next coprime root if it is not.
    while math.gcd(root, length) != 1:
        root += 1
    # The formula differs for odd and even lengths.
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + length % 2) / length)


def _qpsk_burst(number_of_symbols, samples_per_symbol, seed):
    rng = np.random.default_rng(seed)
    symbols = (rng.choice([-1, 1], number_of_symbols) + 1j * rng.choice([-1, 1], number_of_symbols)) / np.sqrt(2)
    upsampled = np.zeros(number_of_symbols * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols
    # Windowed-sinc low-pass filter that limits the bandwidth to about the symbol rate.
    taps = np.arange(-4 * samples_per_symbol, 4 * samples_per_symbol + 1) / samples_per_symbol
    pulse = np.sinc(taps) * np.hamming(len(taps))
    shaped = np.convolve(upsampled, pulse, mode='same')
    return shaped / np.max(np.abs(shaped))


def _papr_db(samples):
    power = np.abs(samples) ** 2
    return 10 * np.log10(np.max(power) / np.mean(power))


def example(resource_name, options, frequency, power_level, iq_rate, symbol_rate, payload_symbols, payloads_per_packet, payload_offset_db,
            rf_blanking, start_trigger_type, start_trigger_source, script_trigger_type, script_trigger_source, marker0_terminal, marker1_terminal,
            started_event_terminal, number_of_packets, packet_interval):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # --- RF and Arb configuration ---
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        _, waveform_quantum, min_waveform_size, _ = session.query_arb_waveform_capabilities()

        # --- Waveforms ---
        samples_per_symbol = max(int(round(actual_iq_rate / symbol_rate)), 2)
        occupied_bandwidth = actual_iq_rate / samples_per_symbol
        # The session signal bandwidth must cover the widest waveform in the script (the preamble).
        session.signal_bandwidth = 0.8 * actual_iq_rate

        preamble_length = _round_up(max(127, min_waveform_size), waveform_quantum)
        preamble = _zadoff_chu(preamble_length, root=25)

        # The payload burst is centered in the waveform, with zero-valued guard intervals.
        burst = _qpsk_burst(payload_symbols, samples_per_symbol, seed=1)
        guard = 8 * samples_per_symbol
        payload_length = _round_up(len(burst) + 2 * guard, waveform_quantum)
        payload = np.zeros(payload_length, dtype=np.complex128)
        burst_start = guard
        burst_stop = guard + len(burst)
        payload[burst_start:burst_stop] = burst

        idle = np.zeros(_round_up(max(1000, min_waveform_size), waveform_quantum), dtype=np.complex64)

        session.write_arb_waveform('preamble', preamble.astype(np.complex64), False)
        session.write_arb_waveform('payload', payload.astype(np.complex64), False)
        session.write_arb_waveform('idle', idle, False)

        # --- Waveform attributes ---
        # The Zadoff-Chu sequence occupies the full I/Q bandwidth; the payload only its symbol bandwidth.
        session.waveforms['preamble'].waveform_signal_bandwidth = 0.8 * actual_iq_rate
        payload_properties = session.waveforms['payload']
        payload_properties.waveform_signal_bandwidth = occupied_bandwidth
        # Generate the payload payload_offset_db relative to the preamble.
        payload_properties.waveform_runtime_scaling = payload_offset_db
        payload_properties.set_waveform_burst_start_locations([burst_start])
        payload_properties.set_waveform_burst_stop_locations([burst_stop])
        payload_properties.waveform_rf_blanking = nirfsg.RFBlanking.ENABLE if rf_blanking else nirfsg.RFBlanking.DISABLE

        for name, data in (('preamble', preamble), ('payload', payload[burst_start:burst_stop])):
            waveform = session.waveforms[name]
            print(f'{name:8}: {waveform.waveform_waveform_size} samples, signal bandwidth {waveform.waveform_signal_bandwidth / 1e6:.3f} MHz, '
                  f'runtime scaling {waveform.waveform_runtime_scaling:+.1f} dB, PAPR {_papr_db(data):.2f} dB')
        print(f'payload : burst from sample {payload_properties.get_waveform_burst_start_locations()} to {payload_properties.get_waveform_burst_stop_locations()}, '
              f'RF blanking {payload_properties.waveform_rf_blanking}')

        # --- Script ---
        script = f'''script packetSequence
    repeat forever
        repeat until scriptTrigger0
            generate idle
        end repeat
        generate preamble marker0(0)
        repeat {payloads_per_packet}
            generate payload marker1({burst_start})
        end repeat
    end repeat
end script'''
        session.write_script(script)
        session.selected_script = 'packetSequence'

        # --- Triggers ---
        if start_trigger_type == 'SOFTWARE':
            session.configure_software_start_trigger()
        else:
            session.configure_digital_edge_start_trigger(start_trigger_source, nirfsg.StartTriggerDigitalEdgeEdge.RISING)
        if script_trigger_type == 'SOFTWARE':
            session.script_triggers[0].configure_software_script_trigger()
        else:
            session.script_triggers[0].configure_digital_edge_script_trigger(script_trigger_source, nirfsg.ScriptTriggerDigitalEdgeEdge.RISING)

        # --- Events ---
        session.markers[0].exported_marker_event_output_terminal = marker0_terminal
        session.markers[1].exported_marker_event_output_terminal = marker1_terminal
        session.exported_started_event_output_terminal = started_event_terminal
        print(f'Start Trigger : {session.start_trigger_type}, scriptTrigger0: {session.script_triggers[0].script_trigger_type}')
        print(f'Preamble marker (marker0) signal : {session.get_terminal_name(nirfsg.Signal.MARKER_EVENT, "marker0")}, exported to {marker0_terminal or "(not exported)"}')
        print(f'Burst marker (marker1) signal    : {session.get_terminal_name(nirfsg.Signal.MARKER_EVENT, "marker1")}, exported to {marker1_terminal or "(not exported)"}')

        # --- Generation ---
        packet_duration = (preamble_length + payloads_per_packet * payload_length) / actual_iq_rate
        print(f'Each packet lasts {packet_duration * 1e6:.1f} us ({payloads_per_packet} payload bursts).')
        with session.initiate():
            if start_trigger_type == 'SOFTWARE':
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
            else:
                print(f'Waiting for the Start Trigger on {start_trigger_source}.')

            for packet in range(number_of_packets):
                time.sleep(packet_interval)
                session.check_generation_status()
                if script_trigger_type == 'SOFTWARE':
                    session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
                    print(f'Packet {packet + 1}/{number_of_packets} triggered.')
            time.sleep(packet_interval)
            session.check_generation_status()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a triggered preamble/payload burst sequence using waveform attributes, a script, triggers, and markers in NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=2.4e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level of the preamble, in dBm.')
    parser.add_argument('-r', '--iq-rate', default=20e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-sr', '--symbol-rate', default=5e6, type=float, help='Payload symbol rate in symbols per second.')
    parser.add_argument('-ps', '--payload-symbols', default=500, type=int, help='Number of QPSK symbols in each payload burst.')
    parser.add_argument('-pp', '--payloads-per-packet', default=4, type=int, help='Number of payload bursts that follow each preamble.')
    parser.add_argument('-po', '--payload-offset-db', default=-3.0, type=float, help='Level of the payload relative to the preamble, in dB (waveform runtime scaling). Use 0 or a negative value to stay within the peak power level.')
    parser.add_argument('-nb', '--no-rf-blanking', dest='rf_blanking', action='store_false', help='Do not blank the RF output outside the payload bursts.')
    parser.add_argument('-stt', '--start-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='Start Trigger type.')
    parser.add_argument('-sts', '--start-trigger-source', default='PXI_Trig0', help='Source of a DIGITAL_EDGE Start Trigger.')
    parser.add_argument('-sct', '--script-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='scriptTrigger0 type. Each trigger sends one packet.')
    parser.add_argument('-scs', '--script-trigger-source', default='PXI_Trig1', help='Source of a DIGITAL_EDGE scriptTrigger0.')
    parser.add_argument('-m0', '--marker0-terminal', default='', help='Terminal to export the preamble marker (marker0) to. Leave empty to not export.')
    parser.add_argument('-m1', '--marker1-terminal', default='', help='Terminal to export the burst marker (marker1) to. Leave empty to not export.')
    parser.add_argument('-se', '--started-event-terminal', default='', help='Terminal to export the Started Event to. Leave empty to not export.')
    parser.add_argument('-c', '--number-of-packets', default=10, type=int, help='Number of packets to trigger (software script trigger), or number of monitoring intervals (hardware trigger).')
    parser.add_argument('-i', '--packet-interval', default=0.5, type=float, help='Time, in seconds, between packets.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.symbol_rate, args.payload_symbols, args.payloads_per_packet,
            args.payload_offset_db, args.rf_blanking, args.start_trigger_type, args.start_trigger_source, args.script_trigger_type, args.script_trigger_source,
            args.marker0_terminal, args.marker1_terminal, args.started_event_terminal, args.number_of_packets, args.packet_interval)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 2.4e9, -10.0, 20e6, 5e6, 200, 3, -3.0, True, 'SOFTWARE', 'PXI_Trig0', 'SOFTWARE', 'PXI_Trig1', 'PXI_Trig2', 'PXI_Trig3', 'PXI_Trig4', 3, 0.01)
    example('simulated5841', options, 2.4e9, -10.0, 20e6, 5e6, 200, 3, -6.0, False, 'DIGITAL_EDGE', 'PXI_Trig0', 'DIGITAL_EDGE', 'PXI_Trig1', '', '', '', 2, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-packets', '2', '--packet-interval', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
