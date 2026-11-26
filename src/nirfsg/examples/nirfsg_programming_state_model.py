'''Walks through the NI-RFSG programming state model: Configuration, Committed, and Generation.

Based on the "Programming State Model", "NI-RFSG Instrument Driver Programming Flow", and
"Set Properties Before Reading Them" topics of the NI-RFSG user manual.

Background:
    An NI-RFSG session is always in one of three states:
      Configuration - The initial state. Properties can be set, but they are only stored in the
                      session. The hardware is not changed and no signal is generated.
      Committed     - commit() validates the whole configuration, writes every setting to the
                      hardware, and writes the waveforms to device memory. The device is
                      configured but does not generate. Setting any property moves the session
                      back to the Configuration state; the hardware keeps the previously
                      committed settings until the next commit.
      Generation    - initiate() commits the configuration (if it is not already committed) and
                      starts the generation. abort() stops it. Some devices, including the
                      PXIe-5830/5831/5832/5840/5841/5842, accept changes to properties such as
                      frequency and power_level in this state.

    Why call commit() yourself?
      - Validation: configuration errors are reported at commit(), before the time-critical
        part of a test, instead of at initiate().
      - Speed: the hardware programming happens during commit(), so the following initiate()
        returns faster. This matters when initiate() must be quick, for example when a
        measurement waits on it.
      - Coerced values: after commit(), properties report the values the hardware actually uses.

    Set properties before reading them: reading a property makes the driver evaluate the
    current configuration, and a value read in the middle of a configuration can change when
    another property is set later. Make all settings first, then read the coerced values.

    This example measures how long initiate() takes when the configuration is already
    committed, and when initiate() has to commit it.

Steps:
    1. Configuration: make all settings and write the waveform, then read the coerced values.
    2. Committed: call commit() and time it, then initiate() and time it.
    3. Generation: make a dynamic frequency change while generating, then abort.
    4. Change a property (back to Configuration), then time initiate(), which now includes
       the commit.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time


def _timed(action):
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1e3


def example(resource_name, options, frequency, power_level, iq_rate, new_frequency, new_power_level, generation_time):
    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        # --- Configuration state: make every setting first. ---
        session.configure_rf(frequency, power_level)
        session.generation_mode = nirfsg.GenerationMode.ARB_WAVEFORM
        session.iq_rate = iq_rate
        session.signal_bandwidth = 0.4 * iq_rate
        number_of_samples = 100000
        session.write_arb_waveform('tone', np.exp(2j * np.pi * 1000 * np.arange(number_of_samples) / number_of_samples).astype(np.complex64), False)

        # Then read the values the driver coerced from the complete configuration.
        print('Configuration state:')
        print(f'  I/Q rate {session.iq_rate / 1e6} MS/s, frequency {session.frequency / 1e6} MHz, power level {session.power_level} dBm')

        # --- Committed state ---
        _, commit_ms = _timed(session.commit)
        print(f'Committed state: commit() took {commit_ms:.1f} ms')
        print(f'  upconverter center frequency {session.upconverter_center_frequency / 1e6:.3f} MHz')

        # --- Generation state ---
        generation, initiate_ms = _timed(session.initiate)
        with generation:
            print(f'Generation state: initiate() after commit() took {initiate_ms:.1f} ms')
            session.check_generation_status()
            # A dynamic change, which the device applies without leaving the Generation state.
            session.frequency = new_frequency
            session.wait_until_settled()
            print(f'  frequency changed to {session.frequency / 1e6} MHz while generating')
            time.sleep(generation_time)
            session.check_generation_status()
        print('abort() stopped the generation.')

        # Setting a property returns the session to the Configuration state, so the next
        # initiate() has to commit the change first.
        session.power_level = new_power_level
        generation, initiate_ms = _timed(session.initiate)
        with generation:
            print(f'initiate() without a prior commit() took {initiate_ms:.1f} ms (it includes the commit)')
            time.sleep(generation_time)
            session.check_generation_status()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Walks through the NI-RFSG Configuration, Committed, and Generation states and times commit() and initiate().', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=1e9, type=float, help='Initial carrier frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Initial average power level in dBm.')
    parser.add_argument('-r', '--iq-rate', default=10e6, type=float, help='I/Q rate in samples per second.')
    parser.add_argument('-nf', '--new-frequency', default=1.1e9, type=float, help='Frequency, in Hz, set while generating.')
    parser.add_argument('-np', '--new-power-level', default=-20.0, type=float, help='Power level, in dBm, set before the second generation.')
    parser.add_argument('-t', '--generation-time', default=2.0, type=float, help='Time, in seconds, to generate in each phase.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.iq_rate, args.new_frequency, args.new_power_level, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, -10.0, 10e6, 1.1e9, -20.0, 0.05)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.05', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
