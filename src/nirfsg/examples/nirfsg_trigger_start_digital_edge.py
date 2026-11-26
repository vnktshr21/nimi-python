'''Starts the generation on a rising or falling edge of a hardware signal (digital edge Start Trigger).

Python counterpart of the NI-RFSG .NET "Triggers/StartTrigger-HardwareSource" example. Based on
the "Start Trigger Details" and "Digital Edge" topics of the NI-RFSG user manual.

Background:
    configure_digital_edge_start_trigger(source, edge) arms the device to start generating on an
    edge of a digital signal, for example from another instrument:
      source - a terminal such as 'PFI0' or a PXI trigger line such as 'PXI_Trig0'. Another
               device's exported signal can also be given by its fully qualified name, for
               example '/PXI1Slot3/ao/0/StartTrigger'.
      edge   - StartTriggerDigitalEdgeEdge.RISING or FALLING.
    The configuration can be read back with start_trigger_type,
    digital_edge_start_trigger_source, and digital_edge_start_trigger_edge.

    A software Start Trigger also starts a generation that is configured for a digital edge, and
    Start Triggers after the start are ignored. This example uses that as a timeout: if no edge
    arrives within the timeout, it sends a software trigger so that the device does not wait
    forever.

Steps:
    1. Configure the RF settings and ARB mode, and write a tone.
    2. Configure the digital edge Start Trigger and read the settings back.
    3. Initiate (arm). After the timeout, send a software Start Trigger in case no edge arrived.
    4. Monitor the generation, then restore the default Start Trigger with disable_start_trigger().
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def example(resource_name, options, frequency, power_level, iq_rate, tone_offset, trigger_source, trigger_edge, edge_timeout, generation_time):
    number_of_samples = 10000
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        actual_iq_rate = session.iq_rate
        session.signal_bandwidth = 2 * abs(tone_offset)
        cycles = round(tone_offset * number_of_samples / actual_iq_rate)
        session.write_arb_waveform('tone', np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)

        session.configure_digital_edge_start_trigger(trigger_source, nirfsg.StartTriggerDigitalEdgeEdge[trigger_edge])
        print(f'Start Trigger: {session.start_trigger_type}, source {session.digital_edge_start_trigger_source}, edge {session.digital_edge_start_trigger_edge}')

        with session.initiate():
            print(f'Armed. Waiting up to {edge_timeout} s for a {trigger_edge.lower()} edge on {trigger_source}...')
            time.sleep(edge_timeout)
            # Starts the generation if the edge never arrived; ignored if it has already started.
            session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)
            print('Timeout reached; software Start Trigger sent. The generation is running either way.')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)

        session.disable_start_trigger()
        print(f'Start Trigger type restored to {session.start_trigger_type}')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Starts an NI-RFSG generation on a digital edge, with a software-trigger timeout.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-ts', '--trigger-source', default='PXI_Trig0', help='Terminal that receives the trigger edge.')
    parser.add_argument('-te', '--trigger-edge', default='RISING', choices=nirfsg.StartTriggerDigitalEdgeEdge.__members__.keys(), type=str.upper, help='Active edge.')
    parser.add_argument('-et', '--edge-timeout', default=10.0, type=float, help='Time, in seconds, to wait for the edge before sending a software Start Trigger.')
    parser.add_argument('-t', '--generation-time', default=5.0, type=float, help='Time, in seconds, to monitor the generation.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.trigger_source, args.trigger_edge, args.edge_timeout, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1e6, 'PXI_Trig0', 'FALLING', 0.01, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--edge-timeout', '0.01', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
