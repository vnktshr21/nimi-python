'''Generates one burst on each rising or falling edge of a hardware signal (digital edge Script Trigger).

Python counterpart of the digital edge case of the NI-RFSG .NET "Triggers/ScriptTrigger-HardwareSource"
example. Based on the "Script Trigger Details" and "Digital Edge" topics of the NI-RFSG user manual.

Background:
    script_triggers[N].configure_digital_edge_script_trigger(source, edge) makes Script Trigger N
    fire on an edge of a digital signal:
      source - a terminal such as 'PFI0' or a PXI trigger line such as 'PXI_Trig1'.
      edge   - ScriptTriggerDigitalEdgeEdge.RISING or FALLING.
    The settings can be read back with script_trigger_type, digital_edge_script_trigger_source,
    and digital_edge_script_trigger_edge. A software trigger sent with
    send_software_edge_trigger() also asserts the Script Trigger, which is useful for testing
    without the hardware signal, or as a timeout.

    The script waits for scriptTrigger0, generates one burst, and waits again, so each edge
    produces one burst. disable_script_trigger() returns the trigger to type NONE.

        repeat forever
            clear scriptTrigger0
            wait until scriptTrigger0
            generate burst
        end repeat

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 for a digital edge and read the settings back.
    3. Write the burst and the script, and generate for the requested time.
    4. Disable the Script Trigger.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script burstOnEdge
    repeat forever
        clear scriptTrigger0
        wait until scriptTrigger0
        generate burst
    end repeat
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, burst_samples, trigger_source, trigger_edge, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        trigger = session.script_triggers[0]
        trigger.configure_digital_edge_script_trigger(trigger_source, nirfsg.ScriptTriggerDigitalEdgeEdge[trigger_edge])
        print(f'scriptTrigger0: {trigger.script_trigger_type}, source {trigger.digital_edge_script_trigger_source}, edge {trigger.digital_edge_script_trigger_edge}')

        cycles = round(tone_offset * burst_samples / actual_iq_rate)
        burst = np.exp(2j * np.pi * cycles * np.arange(burst_samples) / burst_samples)
        burst[-4:] = 0
        session.write_arb_waveform('burst', burst.astype(np.complex64), False)
        session.write_script(SCRIPT)

        with session.initiate():
            print(f'Generating one burst per {trigger_edge.lower()} edge on {trigger_source} for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)

        trigger.disable_script_trigger()
        print(f'scriptTrigger0 type restored to {trigger.script_trigger_type}')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Generates one burst per digital edge with an NI-RFSG digital edge Script Trigger.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the burst tone from the carrier.')
    parser.add_argument('-s', '--burst-samples', default=100000, type=int, help='Length of the burst in samples.')
    parser.add_argument('-ts', '--trigger-source', default='PXI_Trig1', help='Terminal that receives the trigger edges.')
    parser.add_argument('-te', '--trigger-edge', default='RISING', choices=nirfsg.ScriptTriggerDigitalEdgeEdge.__members__.keys(), type=str.upper, help='Active edge.')
    parser.add_argument('-t', '--generation-time', default=30.0, type=float, help='Time, in seconds, to generate.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.burst_samples, args.trigger_source, args.trigger_edge, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 'PXI_Trig1', 'FALLING', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
