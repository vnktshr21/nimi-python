'''Steps the power of a waveform loaded from a TDMS file through a script, advancing on triggers.

This advanced example builds a receiver-sensitivity style power ramp from a waveform file, for
example a Bluetooth packet created by an NI toolkit. Each power step is generated a fixed number
of times, then the script waits for a trigger, which typically comes from the test program
after the DUT has been measured at that step. It combines:

  Arb - waveforms loaded from a file:
      read_and_download_waveform_from_file_tdms() stores the waveform together with the
      attributes saved in the file: I/Q rate, signal bandwidth, PAPR, burst start and stop
      locations, and RF blanking. These per-waveform attributes travel with the waveform, so
      the session does not need to be configured with them.
  Arb - waveform attributes:
      The file is loaded once per power step, under a different name, and each copy gets its
      own waveform_runtime_scaling. The script then selects the power by naming the waveform,
      so no property changes are needed during the generation.
  Arb - script mode:
          for each step k: wait until scriptTrigger0, then generate step<k> <repetitions> times
      The script is finite, so the generation completes after the last step.
  Triggers:
      Start Trigger   - starts the ramp (software or digital edge).
      scriptTrigger0  - advances to the next power step (software or digital edge).
  Events:
      A Marker Event at the start of each repetition marks every generated packet. The
      file can already use a marker for RF blanking (rf_blanking_source), so the example picks
      a marker that is not in use. The Done Event marks the end of the ramp.

Steps:
    1. Configure the RF frequency and power level, SCRIPT mode, and PEAK power.
    2. Load the TDMS waveform once per power step and set the runtime scaling of each copy.
    3. Print the attributes loaded from the file.
    4. Write the ramp script, with a marker on a free marker line.
    5. Configure the triggers and event exports, initiate, and advance through the steps.
'''

import argparse
import nirfsg
import os
import sys
import time

TRIGGER_TYPES = ['SOFTWARE', 'DIGITAL_EDGE']
DEFAULT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nirfsg_arb_read_and_download_tdms_waveform', 'SampleWaveform.tdms')


def example(resource_name, options, frequency, power_level, file_path, waveform_index, power_steps_db, repetitions_per_step,
            start_trigger_type, start_trigger_source, step_trigger_type, step_trigger_source, marker_terminal, done_event_terminal, step_interval, timeout):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        # power_level is the peak power of the packet at a runtime scaling of 0 dB.
        session.power_level_type = nirfsg.PowerLevelType.PEAK

        # Load the file once per power step. Each copy keeps the attributes stored in the file.
        step_names = []
        for step, scaling in enumerate(power_steps_db):
            name = f'step{step}'
            session.read_and_download_waveform_from_file_tdms(name, os.path.abspath(file_path), waveform_index)
            session.waveforms[name].waveform_runtime_scaling = scaling
            step_names.append(name)

        first = session.waveforms[step_names[0]]
        print(f'Loaded {os.path.basename(file_path)} (index {waveform_index}) as {len(step_names)} waveforms:')
        print(f'  I/Q rate {first.waveform_iq_rate / 1e6} MS/s, signal bandwidth {first.waveform_signal_bandwidth / 1e6} MHz, PAPR {first.waveform_papr:.2f} dB, '
              f'{first.waveform_waveform_size} samples, RF blanking {first.waveform_rf_blanking}, burst starts {first.get_waveform_burst_start_locations()}')
        for name in step_names:
            print(f'  {name}: runtime scaling {session.waveforms[name].waveform_runtime_scaling:+.1f} dB -> peak power {power_level + session.waveforms[name].waveform_runtime_scaling:.1f} dBm')

        # Use a marker that the file does not already use for RF blanking.
        blanking_source = session.rf_blanking_source.lower()
        marker = next(m for m in range(4) if f'marker{m}' != blanking_source)
        print(f'RF blanking source: "{session.rf_blanking_source}"; packet marker: marker{marker}')

        steps = '\n'.join(f'''    wait until scriptTrigger0
    repeat {repetitions_per_step}
        generate {name} marker{marker}(0)
    end repeat''' for name in step_names)
        script = f'''script powerRamp
{steps}
end script'''
        session.write_script(script)
        session.selected_script = 'powerRamp'

        if start_trigger_type == 'SOFTWARE':
            session.configure_software_start_trigger()
        else:
            session.configure_digital_edge_start_trigger(start_trigger_source, nirfsg.StartTriggerDigitalEdgeEdge.RISING)
        if step_trigger_type == 'SOFTWARE':
            session.script_triggers[0].configure_software_script_trigger()
        else:
            session.script_triggers[0].configure_digital_edge_script_trigger(step_trigger_source, nirfsg.ScriptTriggerDigitalEdgeEdge.RISING)

        session.markers[marker].exported_marker_event_output_terminal = marker_terminal
        session.exported_done_event_output_terminal = done_event_terminal

        with session.initiate():
            if start_trigger_type == 'SOFTWARE':
                session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)

            if step_trigger_type == 'SOFTWARE':
                for step, name in enumerate(step_names):
                    # Measure the DUT at the previous step here, then advance to the next step.
                    session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, nirfsg.TriggerIdentifier.SCRIPT_TRIGGER0)
                    print(f'Step {step + 1}/{len(step_names)}: generating {name} {repetitions_per_step} times')
                    time.sleep(step_interval)
                    session.check_generation_status()
            else:
                print(f'Advance through the steps with rising edges on {step_trigger_source}.')

            start_time = time.monotonic()
            is_done = False
            while not is_done and time.monotonic() - start_time < timeout:
                is_done = session.check_generation_status()
                time.sleep(0.01)
            print('Power ramp complete.' if is_done else f'Power ramp did not complete within {timeout} s; aborting.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Steps the power of a TDMS waveform through a triggered script using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=2.402e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level, in dBm, at a runtime scaling of 0 dB.')
    parser.add_argument('-fp', '--file-path', default=DEFAULT_FILE_PATH, help='Path to the TDMS waveform file.')
    parser.add_argument('-i', '--waveform-index', default=0, type=int, help='Index of the waveform in the file.')
    parser.add_argument('-ps', '--power-steps-db', default=[0.0, -10.0, -20.0, -30.0], type=float, nargs='+', help='Runtime scaling, in dB, of each power step.')
    parser.add_argument('-rp', '--repetitions-per-step', default=100, type=int, help='Number of times the waveform is generated at each step.')
    parser.add_argument('-stt', '--start-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='Start Trigger type.')
    parser.add_argument('-sts', '--start-trigger-source', default='PXI_Trig0', help='Source of a DIGITAL_EDGE Start Trigger.')
    parser.add_argument('-tt', '--step-trigger-type', default='SOFTWARE', choices=TRIGGER_TYPES, type=str.upper, help='Type of the step trigger (scriptTrigger0).')
    parser.add_argument('-ts', '--step-trigger-source', default='PXI_Trig1', help='Source of a DIGITAL_EDGE step trigger.')
    parser.add_argument('-m', '--marker-terminal', default='', help='Terminal to export the packet marker to. Leave empty to not export.')
    parser.add_argument('-de', '--done-event-terminal', default='', help='Terminal to export the Done Event to. Leave empty to not export.')
    parser.add_argument('-si', '--step-interval', default=1.0, type=float, help='Time, in seconds, between software step triggers.')
    parser.add_argument('-to', '--timeout', default=10.0, type=float, help='Maximum time, in seconds, to wait for the ramp to complete after the last step.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.file_path, args.waveform_index, args.power_steps_db, args.repetitions_per_step,
            args.start_trigger_type, args.start_trigger_source, args.step_trigger_type, args.step_trigger_source, args.marker_terminal, args.done_event_terminal,
            args.step_interval, args.timeout)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 2.402e9, -10.0, DEFAULT_FILE_PATH, 0, [0.0, -10.0, -20.0], 10, 'SOFTWARE', 'PXI_Trig0', 'SOFTWARE', 'PXI_Trig1', 'PXI_Trig2', 'PXI_Trig3', 0.01, 0.2)
    example('simulated5841', options, 2.402e9, -10.0, DEFAULT_FILE_PATH, 0, [0.0, -6.0], 10, 'DIGITAL_EDGE', 'PXI_Trig0', 'DIGITAL_EDGE', 'PXI_Trig1', '', '', 0.01, 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--step-interval', '0.01', '--timeout', '0.2', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
