'''Gates bursts with the level of a hardware signal (digital level Script Trigger).

Python counterpart of the digital level case of the NI-RFSG .NET "Triggers/ScriptTrigger-HardwareSource"
example. Based on the "Script Trigger Details" and "Digital Level" topics of the NI-RFSG user manual.

Background:
    A digital level trigger is asserted for as long as its source is at the active level, not
    just at an edge. Only Script Triggers can be configured for a digital level, with
    script_triggers[N].configure_digital_level_script_trigger(source, level). The level is
    passed as an integer, so use the value of a ScriptTriggerDigitalLevelActiveLevel member
    (HIGH or LOW). The settings can be read back with script_trigger_type,
    digital_level_script_trigger_source, and digital_level_script_trigger_active_level.

    This example uses the level as a gate:

        repeat forever
            repeat until scriptTrigger0
                generate idle
            end repeat
            generate burst
        end repeat

    While the source is inactive, the script keeps generating the silent idle waveform. While it
    is active, every pass ends the "repeat until" loop, so bursts are generated back to back.

Steps:
    1. Configure the RF settings, SCRIPT mode, PEAK power, I/Q rate, and signal bandwidth.
    2. Configure scriptTrigger0 for a digital level and read the settings back.
    3. Write the idle and burst waveforms and the script, and generate for the requested time.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

SCRIPT = '''script gateOnLevel
    repeat forever
        repeat until scriptTrigger0
            generate idle
        end repeat
        generate burst
    end repeat
end script'''


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, burst_samples, trigger_source, active_level, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.SCRIPT
        session.power_level_type = nirfsg.PowerLevelType.PEAK
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)

        trigger = session.script_triggers[0]
        # This method takes the active level as an integer, so pass the value of the enum member.
        trigger.configure_digital_level_script_trigger(trigger_source, nirfsg.ScriptTriggerDigitalLevelActiveLevel[active_level].value)
        print(f'scriptTrigger0: {trigger.script_trigger_type}, source {trigger.digital_level_script_trigger_source}, active level {trigger.digital_level_script_trigger_active_level}')

        # A short idle waveform keeps the reaction to the level fast.
        session.write_arb_waveform('idle', np.zeros(1000, dtype=np.complex64), False)
        cycles = round(tone_offset * burst_samples / actual_iq_rate)
        session.write_arb_waveform('burst', np.exp(2j * np.pi * cycles * np.arange(burst_samples) / burst_samples).astype(np.complex64), False)
        session.write_script(SCRIPT)

        with session.initiate():
            print(f'Generating bursts while {trigger_source} is {active_level.lower()}, for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Gates bursts with the level of a hardware signal using an NI-RFSG digital level Script Trigger.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Peak power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the burst tone from the carrier.')
    parser.add_argument('-s', '--burst-samples', default=10000, type=int, help='Length of each burst in samples.')
    parser.add_argument('-ts', '--trigger-source', default='PXI_Trig1', help='Terminal that carries the gate signal.')
    parser.add_argument('-l', '--active-level', default='HIGH', choices=nirfsg.ScriptTriggerDigitalLevelActiveLevel.__members__.keys(), type=str.upper, help='Level at which bursts are generated.')
    parser.add_argument('-t', '--generation-time', default=30.0, type=float, help='Time, in seconds, to generate.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.burst_samples, args.trigger_source, args.active_level, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 10000, 'PXI_Trig1', 'LOW', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
