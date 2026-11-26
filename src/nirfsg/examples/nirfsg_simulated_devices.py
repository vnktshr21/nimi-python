'''Opens simulated NI-RFSG devices, and shows how to write the option string and the DriverSetup options.

Based on the "Simulating a Device Using NI-RFSG" and "NI-RFSG Driver Setup Options" topics of the
NI-RFSG user manual.

Background:
    A simulated device lets you develop and test a program without hardware. You enable it
    in the options of nirfsg.Session, which accepts either form:
      - a string:      'Simulate=1, DriverSetup=Model:5841'
      - a dictionary:  {'simulate': True, 'driver_setup': {'Model': '5841'}}
    The dictionary is converted to the string form; DriverSetup entries become "Tag:Value"
    pairs separated by semicolons. The resource name of a simulated device can be any name.

    Common DriverSetup tags:
      Model     - the simulated model, for example 5840, 5841, 5842, 5830, 5831, 5654, or 5673E.
      BoardType - PXI or PXIe. Needed only for PXI/PXIe-5650/5651/5652, for example
                  'Model:5650; BoardType:PXIe'.
      LO        - an associated LO module. The PXIe-5842 is simulated with 'Model:5842; LO:5655'.
      AWG       - an associated AWG module, for example 'Model:5611; AWG:5450; LO:5652E'.
      Bitfile   - an NI-RFSG instrument driver FPGA extensions bitfile.
    On real hardware, the AWG, LO, and AE tags override the module associations configured in
    Measurement & Automation Explorer (MAX).

    A simulated device accepts the same calls as real hardware, and returns plausible but
    not measured values. Some behaviors differ; for example, a simulated device can report a
    finite generation as done immediately.

Steps:
    For each requested model:
    1. Build the options in the requested form and open a simulated session.
    2. Print the identity and the capabilities reported by the simulated device.
    3. Generate a CW tone at the device's default frequency and check the status.
'''

import argparse
import nirfsg
import sys

# Extra DriverSetup tags that some models need when they are simulated.
EXTRA_DRIVER_SETUP = {
    '5842': {'LO': '5655'},
}


def _options(model, option_format):
    driver_setup = dict({'Model': model}, **EXTRA_DRIVER_SETUP.get(model, {}))
    if option_format == 'DICT':
        return {'simulate': True, 'driver_setup': driver_setup}
    return 'Simulate=1, DriverSetup=' + '; '.join(f'{tag}:{value}' for tag, value in driver_setup.items())


def example(models, option_format, power_level):
    for model in models:
        options = _options(model, option_format)
        print(f'--- Model {model}, options: {options!r}')
        with nirfsg.Session(resource_name=f'Simulated{model}', id_query=False, reset_device=False, options=options) as session:
            max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size = session.query_arb_waveform_capabilities()
            print(f'  instrument model : {session.instrument_model}')
            print(f'  serial number    : {session.serial_number}')
            print(f'  available ports  : {session.available_ports}')
            print(f'  waveforms        : up to {max_number_waveforms}, {min_waveform_size} to {max_waveform_size} samples, quantum {waveform_quantum}')

            # Generate at the default frequency of the model, which is always within its range.
            default_frequency = session.frequency
            session.generation_mode = nirfsg.GenerationMode.CW
            session.configure_rf(default_frequency, power_level)
            with session.initiate():
                session.check_generation_status()
            print(f'  generated a CW tone at {default_frequency / 1e6} MHz')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Opens simulated NI-RFSG devices using string or dictionary options.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--models', default=['5840', '5841', '5842', '5831'], nargs='+', help='Models to simulate.')
    parser.add_argument('-of', '--option-format', default='DICT', choices=['DICT', 'STRING'], type=str.upper, help='Pass the options as a dictionary or as a string.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level in dBm.')
    args = parser.parse_args(argsv)
    example(args.models, args.option_format, args.power_level)


def main():
    _main(sys.argv[1:])


def test_example():
    example(['5841', '5842'], 'STRING', -10.0)


def test_main():
    _main([])


if __name__ == '__main__':
    main()
