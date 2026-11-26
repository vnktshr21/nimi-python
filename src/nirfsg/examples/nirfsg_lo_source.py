'''Selects the source of each local oscillator (LO) on a device with several LOs, such as the PXIe-5830/5831/5832.

Background:
    A vector signal transceiver upconverts the baseband signal with one or more LOs. On the
    PXIe-5830/5831/5832, the LOs are addressed with the los[] repeated capability: los[1] is
    LO1 and los[2] is LO2. lo_source selects where each LO comes from:
      ONBOARD                - the device's own synthesizer (default)
      LO_IN                  - an external LO connected to LO IN
      SG_SA_SHARED           - one internal LO shared by NI-RFSG and NI-RFSA, for example for
                               loopback measurements with correlated phase noise
      AUTOMATIC_SG_SA_SHARED - NI-RFSG configures the sharing with NI-RFSA automatically
      SECONDARY              - a secondary LO on supported configurations
    Not every source is valid on every device; refer to the LO sharing topics of your device.
    With an onboard LO, lo_frequency is read-only and reports the frequency the LO is tuned to.

    On single-LO devices, such as the PXIe-5840/5841, the LO properties are set directly on the
    session.

Steps:
    1. Set the source of LO1 and LO2.
    2. Generate a CW tone and read the frequency of each LO.
'''

import argparse
import nirfsg
import sys
import time


def example(resource_name, options, frequency, power_level, lo1_source, lo2_source, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.los[1].lo_source = nirfsg.LoSource[lo1_source]
        session.los[2].lo_source = nirfsg.LoSource[lo2_source]
        for lo in (1, 2):
            print(f'LO{lo} source    : {session.los[lo].lo_source}')

        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequency, power_level)
        with session.initiate():
            for lo in (1, 2):
                print(f'LO{lo} frequency : {session.los[lo].lo_frequency / 1e9:.6f} GHz')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Selects the source of each LO of a multi-LO NI-RFSG device.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=6.5e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-lo1', '--lo1-source', default='ONBOARD', choices=nirfsg.LoSource.__members__.keys(), type=str.upper, help='Source of LO1.')
    parser.add_argument('-lo2', '--lo2-source', default='ONBOARD', choices=nirfsg.LoSource.__members__.keys(), type=str.upper, help='Source of LO2.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.lo1_source, args.lo2_source, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5831', }, }
    example('simulated5831', options, 6.5e9, -10.0, 'ONBOARD', 'ONBOARD', 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5831', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5831']
    _main(cmd_line)


if __name__ == '__main__':
    main()
