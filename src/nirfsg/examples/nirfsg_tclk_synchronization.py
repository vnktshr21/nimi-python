'''Synchronizes the generation of several NI-RFSG devices with NI-TClk.

Python counterpart of the NI-RFSG .NET "Synchronization/5673Synchronization(TClk,SharedLO)" and
"Synchronization/5673Synchronization(TClk,SharedLO,Script)" examples.

Background:
    NI-TClk aligns the sample clocks and triggers of several devices, so that their
    generations start on the same sample clock edge and stay aligned. This is used for
    MIMO, beamforming, and multi-channel test. Each nirfsg.Session can be passed directly
    to the nitclk functions:
      nitclk.configure_for_homogeneous_triggers(sessions)
          Configures the Start Triggers (and Script Triggers) so that the first session acts
          as the master and the others follow it.
      nitclk.synchronize(sessions, min_tclk_period)
          Aligns the TClk signals of all sessions.
      nitclk.initiate(sessions)
          Starts every session in the right order (slaves first, then the master).
    All devices must share a Reference Clock. In a PXI chassis, use the 10 MHz PXI backplane
    clock (PXI_CLK).

    With a software Start Trigger on the master, every device is armed by nitclk.initiate() and
    starts together when the trigger is sent to the master.

    For phase-coherent RF outputs, the devices must also share an LO. Refer to the
    synchronization topics in the NI-RFSG documentation for the LO sharing options of your
    device.

Steps:
    1. Open a session to each device, and configure the same RF settings, ARB waveform, and
       Reference Clock on each.
    2. Configure a software Start Trigger on the master (the first session).
    3. Configure the triggers for homogeneous devices, synchronize, and initiate.
    4. Send the software Start Trigger to the master, then monitor every device.
    5. Abort every session.
'''

import argparse
import contextlib
import nirfsg
import nitclk
import numpy as np
import sys
import time


def example(resource_names, options, frequency, power_level, iq_rate, tone_offset, number_of_samples, generation_time):
    with contextlib.ExitStack() as stack:
        sessions = [stack.enter_context(nirfsg.Session(resource_name=name, id_query=False, reset_device=False, options=options)) for name in resource_names]

        for session in sessions:
            session.configure_rf(frequency, power_level)
            session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
            session.iq_rate = iq_rate
            actual_iq_rate = session.iq_rate
            session.signal_bandwidth = 2 * abs(tone_offset)

            # Every device locks to the PXI backplane clock, so that all of them share one timebase.
            session.configure_ref_clock(nirfsg.ReferenceClockSource.PXI_CLK.value, 10e6)

            cycles = round(tone_offset * number_of_samples / actual_iq_rate)
            waveform = np.exp(2j * np.pi * cycles * np.arange(number_of_samples) / number_of_samples).astype(np.complex64)
            session.write_arb_waveform('tone', waveform, False)

        # The master waits for a software Start Trigger. configure_for_homogeneous_triggers()
        # makes the other devices follow the master's trigger.
        master = sessions[0]
        master.configure_software_start_trigger()

        nitclk.configure_for_homogeneous_triggers(sessions)
        # 0.0 lets NI-TClk use the smallest TClk period that it supports (200 ns in a single chassis).
        nitclk.synchronize(sessions, 0.0)
        try:
            nitclk.initiate(sessions)
            print(f'{len(sessions)} devices armed. Sending the software Start Trigger to the master ({resource_names[0]})...')
            master.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, nirfsg.TriggerIdentifier.NONE)

            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                for session in sessions:
                    session.check_generation_status()
                time.sleep(0.1)
            print('Synchronized generation complete.')
        finally:
            for session in sessions:
                session.abort()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Synchronizes several NI-RFSG devices with NI-TClk.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-names', default=['PXI1Slot2', 'PXI1Slot3'], nargs='+', help='Resource names of the NI RF signal generators. The first one is the master.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-o', '--tone-offset', default=1e6, type=float, help='Frequency offset, in Hz, of the tone from the carrier.')
    parser.add_argument('-s', '--number-of-samples', default=10000, type=int, help='Number of samples in the waveform.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the sessions.')
    args = parser.parse_args(argsv)
    example(args.resource_names, args.option_string, args.frequency, args.power_level, args.iq_rate, args.tone_offset, args.number_of_samples, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example(['simulated5841a', 'simulated5841b'], options, 1e9, -10.0, 10e6, 1e6, 10000, 0.1)


def test_main():
    cmd_line = ['--resource-names', 'simulated5841a', 'simulated5841b', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
