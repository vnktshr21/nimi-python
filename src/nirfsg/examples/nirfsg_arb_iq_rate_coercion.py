'''Shows how the requested I/Q rate is coerced to a rate that the Sample Clock can produce.

Based on the "Clocking Modes" and "Set Properties Before Reading Them" topics of the NI-RFSG
user manual.

Background:
    The I/Q rate (iq_rate) is the rate at which waveform samples are played. It is derived from
    the device Sample Clock, which is configured with:
      arb_sample_clock_source
          ONBOARD_CLOCK - the device's internal Sample Clock timebase.
          CLK_IN        - an external Sample Clock connected to the CLK IN terminal.
      arb_onboard_sample_clock_mode (onboard clock only)
          DIVIDE_DOWN     - divides the Sample Clock timebase. It has the lowest jitter, but
                            only certain I/Q rates are possible, so the requested rate is
                            coerced to an achievable rate.
          HIGH_RESOLUTION - synthesizes almost any I/Q rate, which can increase phase noise.
    Vector signal transceivers (PXIe-5644/5645/5646 and PXIe-5820/5830/5831/5832/5840/5841/5842/5860)
    support only ONBOARD_CLOCK and DIVIDE_DOWN.

    Always read iq_rate back after you set it, and compute waveforms with the actual rate.
    arb_sample_clock_rate returns the rate of the underlying Sample Clock. Read these values
    only after you have set every property they depend on: the driver computes coerced values
    from the current configuration, so a value read too early can change when you set another
    property later.

Steps:
    1. Configure ARB mode and the Sample Clock source and mode.
    2. For each requested I/Q rate, set iq_rate and read back the actual I/Q rate and the
       Sample Clock rate.
'''

import argparse
import nirfsg
import sys


def example(resource_name, options, sample_clock_source, sample_clock_mode, requested_iq_rates):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.arb_sample_clock_source = nirfsg.ArbSampleClockSource[sample_clock_source]
        if sample_clock_source == 'ONBOARD_CLOCK':
            session.arb_onboard_sample_clock_mode = nirfsg.ArbOnboardSampleClockMode[sample_clock_mode]

        print(f'Sample Clock source: {sample_clock_source}, mode: {sample_clock_mode}')
        print(f'{"requested I/Q rate":>20} {"actual I/Q rate":>20} {"difference":>12} {"Sample Clock rate":>20}')
        for requested in requested_iq_rates:
            # Set the property first, then read the coerced values that depend on it.
            session.iq_rate = requested
            actual = session.iq_rate
            difference_ppm = (actual - requested) / requested * 1e6
            print(f'{requested:>17.3f} Hz {actual:>17.3f} Hz {difference_ppm:>8.1f} ppm {session.arb_sample_clock_rate:>17.3f} Hz')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Shows how NI-RFSG coerces requested I/Q rates for the selected Sample Clock settings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-cs', '--sample-clock-source', default='ONBOARD_CLOCK', choices=nirfsg.ArbSampleClockSource.__members__.keys(), type=str.upper, help='Sample Clock source. Vector signal transceivers support only ONBOARD_CLOCK.')
    parser.add_argument('-cm', '--sample-clock-mode', default='DIVIDE_DOWN', choices=nirfsg.ArbOnboardSampleClockMode.__members__.keys(), type=str.upper, help='Onboard Sample Clock mode. Vector signal transceivers support only DIVIDE_DOWN.')
    parser.add_argument('-r', '--requested-iq-rates', default=[1e6, 3.3e6, 10e6, 12.345678e6, 100e6], type=float, nargs='+', help='I/Q rates to request, in samples per second.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.sample_clock_source, args.sample_clock_mode, args.requested_iq_rates)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 'ONBOARD_CLOCK', 'DIVIDE_DOWN', [1e6, 12.345678e6])


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
