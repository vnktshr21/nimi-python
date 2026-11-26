'''Starts the generation from the program with a software Start Trigger.

Python counterpart of the NI-RFSG .NET "Triggers/StartTrigger-SoftwareSource" example. Based on
the "Start Trigger Details" and "Software Edge" topics of the NI-RFSG user manual.

Background:
    By default (Start Trigger type NONE) the device starts generating as soon as initiate() is
    called. With a Start Trigger configured, initiate() only arms the device, and the
    generation starts when the trigger arrives. This lets the program align the start of the
    signal with another step, for example right after an analyzer has been armed.

    configure_software_start_trigger() sets the Start Trigger type to SOFTWARE, and
    send_software_edge_trigger(SoftwareTriggerType.START, TriggerIdentifier.NONE) sends the
    trigger. Start Triggers that arrive after the generation has started are ignored.
    disable_start_trigger() returns to the default (NONE).

Steps:
    1. Configure the RF settings and ARB mode, and write a tone.
    2. Configure a software Start Trigger.
    3. Initiate (arm), wait, then send the software Start Trigger and monitor the generation.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, trigger_delay, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('tone', np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)

        session.configure_software_start_trigger()
        print(f'Start Trigger type: {session.start_trigger_type}')

        with session.initiate():
            print(f'Armed. Sending the software Start Trigger in {trigger_delay} s...')
            time.sleep(trigger_delay)
            # For the Start Trigger, the trigger identifier must be NONE.
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
            print('Start Trigger sent; the generation has started.')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Starts an NI-RFSG generation with a software Start Trigger.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-d', '--trigger-delay', default=2.0, type=float, help='Time, in seconds, between arming and sending the Start Trigger.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to generate after the trigger.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.trigger_delay, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--trigger-delay', '0.01', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
