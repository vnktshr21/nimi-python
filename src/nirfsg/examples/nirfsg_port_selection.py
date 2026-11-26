'''Selects the RF port to generate on, on a multi-port device such as the PXIe-5830/5831/5832.

Background:
    Some vector signal transceivers have several RF ports. available_ports lists the ports of
    the instrument configuration, and selected_ports chooses the port that the generation uses:
      PXIe-5830           if0, if1
      PXIe-5831/5832      if0, if1, and the mmRH-5582 ports, named rf<0-1>/port<x>
      PXIe-5840/5841/...  '' (empty string), because there is only one port
    Per-port properties, such as the de-embedding settings, are accessed through the ports[]
    repeated capability, for example session.ports['if1'].deembedding_selected_table.

Steps:
    1. Print the available ports and select the requested one.
    2. Generate a CW tone on that port.
'''

import argparse
import nirfsg
import sys
import time


def example(resource_name, options, port, frequency, power_level, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        print(f'Instrument      : {session.instrument_model}')
        print(f'Available ports : {session.available_ports}')

        session.selected_ports = port
        print(f'Selected port   : {session.selected_ports}')

        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequency, power_level)
        with session.initiate():
            print(f'Generating {frequency / 1e9} GHz at {power_level} dBm on {port} for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)


def _main(argsv):
    parser = argparse.ArgumentParser(description='Selects the RF port of a multi-port NI-RFSG device and generates on it.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-pt', '--port', default='if1', help='Port to generate on. Use the names listed in available_ports.')
    parser.add_argument('-f', '--frequency', default=6.5e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.port, args.frequency, args.power_level, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5831', }, }
    example('simulated5831', options, 'if1', 6.5e9, -10.0, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5831', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5831']
    _main(cmd_line)


if __name__ == '__main__':
    main()
