'''Uses a script to generate pulsed RF data, such as a TDMA frame with one active time slot.

Python counterpart of the NI-RFSG .NET "Scripts/PulsedData" example.

Background:
    A pulsed signal can be built from two short waveforms instead of one long one:
      carrier - a constant-envelope burst (I = 1, Q = 0), which is a tone at the carrier frequency.
      offTime - all zeros, so no RF output.
    The script plays carrier for one time slot and offTime for the remaining slots of the
    frame, and repeats the frame forever. Changing the duty cycle only needs a new script,
    not new waveforms.

        script pulsedData
            repeat forever
                repeat <time slots - 1>
                    generate offTime
                end repeat
                generate carrier
            end repeat
        end script

    Notes:
      - power_level_type is set to PEAK so that power_level is the power during the pulse.
        With AVERAGE, the power would be averaged over the on and off times of each waveform.
      - phase_continuity_enabled is set to DISABLE. Phase continuity keeps the carrier phase
        continuous across waveform boundaries, which a gated tone does not need.
      - Each waveform must be a multiple of the waveform quantum, so the time slot length is
        coerced to the nearest valid number of samples.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and phase continuity.
    2. Compute the time slot size from the frame period, the number of slots, and the quantum.
    3. Write the carrier and offTime waveforms and the script.
    4. Generate for the requested time.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, frame_period, time_slots, generation_time):
    if time_slots < 2:
        raise ValueError('The frame must contain at least two time slots.')

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.arb_pre_filter_gain = -2.0
        session.phase_continuity_enabled = nirfsg.PhaseContinuityEnabled.DISABLE

        # The pulse edges are fast and spread energy over a wide band, so allow most of the
        # available bandwidth.
        session.signal_bandwidth = 0.8 * actual_iq_rate

        # One time slot, coerced to a valid waveform size.
        _, waveform_quantum, min_waveform_size, _ = session.query_arb_waveform_capabilities()
        slot_samples = int(round(actual_iq_rate * frame_period / time_slots / waveform_quantum)) * waveform_quantum
        slot_samples = max(slot_samples, min_waveform_size, waveform_quantum)
        actual_slot_period = slot_samples / actual_iq_rate

        session.write_arb_waveform('carrier', np.ones(slot_samples, dtype=np.complex64), False)
        session.write_arb_waveform('offTime', np.zeros(slot_samples, dtype=np.complex64), False)

        script = f'''script pulsedData
    repeat forever
        repeat {time_slots - 1}
            generate offTime
        end repeat
        generate carrier
    end repeat
end script'''
        session.write_script(script)

        print(f'Time slot: {actual_slot_period * 1e6:.3f} us ({slot_samples} samples), frame: {actual_slot_period * time_slots * 1e6:.3f} us, duty cycle: {100 / time_slots:.1f}%')
        with session.initiate():
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates a pulsed RF signal with a script using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level, in dBm, during the pulse.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-fp', '--frame-period', default=4.615e-3, type=float, help='Duration of one frame in seconds.')
    parser.add_argument('-ts', '--time-slots', default=8, type=int, help='Number of time slots per frame. The carrier is on for one slot.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.frame_period, args.time_slots, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e-3, 4, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
