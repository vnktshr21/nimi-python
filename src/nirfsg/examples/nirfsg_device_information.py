'''Reports identity, capability, and status information about an NI RF signal generator.

Background:
    Many read-only properties describe the instrument and the driver. They are useful for
    logging the test setup, checking that the expected hardware is installed, or adapting a
    program to the capabilities of the device. This example reads:
      - Identity: manufacturer, model, serial number, module and firmware revisions.
      - Driver: driver revision and the instrument models that the driver supports.
      - Capabilities: available ports, onboard memory, instantaneous bandwidth, and the
        arbitrary waveform limits returned by query_arb_waveform_capabilities().
      - Status: device temperature and the maximum power level that can be set at a frequency.
    Some of these values depend on the configuration, so set the frequency and the generation
    mode first, then read them.

Steps:
    1. Open the session and print the identity and driver information.
    2. Configure a frequency and print the capability and status information.
'''

import argparse
import nirfsg
import sys


def example(resource_name, options, frequency):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        print('Identity')
        print(f'  manufacturer       : {session.instrument_manufacturer}')
        print(f'  model              : {session.instrument_model}')
        print(f'  serial number      : {session.serial_number}')
        print(f'  module revision    : {session.module_revision}')
        print(f'  firmware revision  : {session.instrument_firmware_revision}')
        print('Driver')
        print(f'  driver revision    : {session.specific_driver_revision}')
        print(f'  supported models   : {", ".join(session.supported_instrument_models)}')

        # Configure first, then read the values that depend on the configuration.
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.frequency = frequency
        max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size = session.query_arb_waveform_capabilities()

        print('Capabilities')
        print(f'  available ports    : {session.available_ports}')
        print(f'  onboard memory     : {session.memory_size / 2 ** 20:.0f} MiB')
        print(f'  instantaneous BW   : {session.device_instantaneous_bandwidth / 1e6} MHz')
        print(f'  max waveforms      : {max_number_waveforms}')
        print(f'  waveform quantum   : {waveform_quantum} samples')
        print(f'  waveform size      : {min_waveform_size} to {max_waveform_size} samples')
        print('Status')
        print(f'  temperature        : {session.device_temperature:.2f} C')
        print(f'  max power level    : {session.get_max_settable_power():.2f} dBm at {frequency / 1e6} MHz')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Reports identity, capability, and status information about an NI RF signal generator.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Frequency, in Hz, at which to query the maximum power level.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
