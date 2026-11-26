'''Adjusts the phase of the RF output with the phase offset property.

Background:
    phase_offset sets the phase, in degrees, of the RF output signal. Its main use is to align
    the RF outputs of several phase-coherent devices, for example devices that share an LO and
    are synchronized with NI-TClk, for beamforming or MIMO tests. Measure the phase difference
    between the outputs with an analyzer, then apply the correction to one of the devices.

    On a single device, changing the phase offset is visible on a vector signal analyzer that
    is locked to the same Reference Clock: the measured phase of the tone moves by the same
    number of degrees.

Steps:
    1. Configure the RF settings and select CW mode.
    2. For each requested phase offset: set phase_offset, then generate for the step time.
'''

import argparse
import nirfsg
import sys
import time


def example(resource_name, options, frequency, power_level, phase_offsets, step_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.CW

        for phase_offset in phase_offsets:
            # Set the phase offset in the Configuration state; initiate() applies it.
            session.phase_offset = phase_offset
            print(f'Phase offset: {session.phase_offset:.1f} degrees')
            with session.initiate():
                end_time = time.monotonic() + step_time
                while time.monotonic() < end_time:
                    session.check_generation_status()
                    time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Adjusts the phase of the NI-RFSG RF output with the phase offset property.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-ph', '--phase-offsets', default=[0.0, 45.0, 90.0, 180.0], type=float, nargs='+', help='Phase offsets, in degrees, to apply in turn.')
    parser.add_argument('-t', '--step-time', default=2.0, type=float, help='Time, in seconds, to generate at each phase offset.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.phase_offsets, args.step_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, [0.0, 90.0], 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--step-time', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
