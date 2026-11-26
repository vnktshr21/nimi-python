'''Compensates for the loss of an external network, such as a cable or fixture, with de-embedding.

Background:
    The signal reaching the device under test (DUT) passes through cables, fixtures, or
    attenuators that add frequency-dependent loss. De-embedding removes their effect, so
    that power_level is the power at the DUT reference plane instead of at the RF OUT
    connector. The external network is described by its 2-port S-parameters
    (S11, S12, S21, S22) at a list of frequencies:
      - create_deembedding_sparameter_table_array() creates a table from numpy arrays.
        frequencies has shape (N,) and sparameter_table has shape (N, 2, 2).
      - create_deembedding_sparameter_table_s2p_file() creates a table from a Touchstone .s2p
        file. It is not supported on Linux.
    The orientation tells the driver which port of the network faces the DUT.

    Between the table frequencies, the S-parameters are interpolated with one of:
      configure_deembedding_table_interpolation_nearest()  nearest table point
      configure_deembedding_table_interpolation_linear()   linear, in the selected Format
      configure_deembedding_table_interpolation_spline()   spline

    De-embedding is configured per port with the ports[] repeated capability. Use '' for
    single-port devices such as the PXIe-5840/5841/5842:
      deembedding_selected_table  - the table to apply
      deembedding_type            - SCALAR de-embeds using the gain term (S21) only. VECTOR
                                    also uses the reflection terms. Refer to your device
                                    documentation for the types that it supports.
    get_deembedding_sparameters() returns the interpolated 2x2 S-parameter matrix at the
    current frequency, and deembedding_compensation_gain returns the gain that the driver
    applies.

    This example models a cable whose loss increases with frequency, plus a small reflection
    at each end.

Steps:
    1. Build the S-parameter table of the cable with numpy.
    2. Create the de-embedding table, choose an interpolation method, and select it on the port.
    3. Configure the RF settings and generate a CW tone. power_level is now the power at the DUT.
    4. Print the interpolated S-parameters and the compensation gain.
    5. Delete the table, then all tables.
'''

import argparse
import nirfsg
import numpy as np
import sys
import time

INTERPOLATIONS = ['NEAREST', 'LINEAR', 'SPLINE']


def _cable_sparameters(frequencies, loss_db_at_1ghz, electrical_length, reflection):
    # Approximate cable loss in dB, which grows with the square root of the frequency.
    loss_db = loss_db_at_1ghz * np.sqrt(frequencies / 1e9)
    # Transmission: the magnitude from the loss, the phase from the delay through the cable.
    delay = electrical_length / 3e8
    s21 = 10 ** (-loss_db / 20) * np.exp(-2j * np.pi * frequencies * delay)
    table = np.empty((len(frequencies), 2, 2), dtype=np.complex128)
    table[:, 0, 0] = reflection
    table[:, 0, 1] = s21
    table[:, 1, 0] = s21
    table[:, 1, 1] = reflection
    return table


def example(resource_name, options, frequency, power_level, loss_db_at_1ghz, interpolation, deembedding_type, s2p_file_path, generation_time):
    port = ''
    table_name = 'cable'

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        if s2p_file_path:
            # Load the network from a Touchstone file. This is not supported on Linux.
            session.create_deembedding_sparameter_table_s2p_file(port, table_name, s2p_file_path, nirfsg.SparameterOrientation.PORT2_TOWARDS_DUT)
        else:
            frequencies = np.linspace(500e6, 6e9, 12)
            sparameters = _cable_sparameters(frequencies, loss_db_at_1ghz, 1.0, 0.05)
            # Port 2 of the cable faces the DUT; port 1 is connected to RF OUT.
            session.create_deembedding_sparameter_table_array(port, table_name, frequencies, sparameters, nirfsg.SparameterOrientation.PORT2_TOWARDS_DUT)

        if interpolation == 'LINEAR':
            # Interpolate linearly on the magnitude (in dB) and on the phase.
            session.configure_deembedding_table_interpolation_linear(port, table_name, nirfsg.Format.MAGNITUDE_DB_AND_PHASE)
        elif interpolation == 'SPLINE':
            session.configure_deembedding_table_interpolation_spline(port, table_name)
        else:
            session.configure_deembedding_table_interpolation_nearest(port, table_name)

        session.ports[port].deembedding_type = nirfsg.DeembeddingType[deembedding_type]
        session.ports[port].deembedding_selected_table = table_name

        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(frequency, power_level)

        sparameters = session.get_deembedding_sparameters()
        print(f'Interpolated S-parameters at {frequency / 1e6} MHz (port 1 towards DUT):')
        print(f'  S11 = {sparameters[0, 0]:.4f}   S12 = {sparameters[0, 1]:.4f}')
        print(f'  S21 = {sparameters[1, 0]:.4f}   S22 = {sparameters[1, 1]:.4f}')
        print(f'  |S21| = {20 * np.log10(max(abs(sparameters[1, 0]), 1e-12)):.2f} dB')

        with session.initiate():
            print(f'De-embedding gain applied: {session.ports[port].deembedding_compensation_gain:.2f} dB')
            print(f'Generating {power_level} dBm at the DUT for {generation_time} s...')
            end_time = time.monotonic() + generation_time
            while time.monotonic() < end_time:
                session.check_generation_status()
                time.sleep(0.1)

        # Deselect the table before deleting it.
        session.ports[port].deembedding_selected_table = ''
        session.delete_deembedding_table(port, table_name)
        # delete_all_deembedding_tables() removes the tables of every port.
        session.delete_all_deembedding_tables()


def _main(argsv):
    parser = argparse.ArgumentParser(description='Compensates for the loss of an external cable with de-embedding using NI-RFSG.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-f', '--frequency', default=2.4e9, type=float, help='Frequency in Hz.')
    parser.add_argument('-p', '--power-level', default=-10.0, type=float, help='Power level, in dBm, at the DUT.')
    parser.add_argument('-l', '--loss-db-at-1ghz', default=1.5, type=float, help='Loss of the modeled cable, in dB, at 1 GHz.')
    parser.add_argument('-i', '--interpolation', default='LINEAR', choices=INTERPOLATIONS, type=str.upper, help='Interpolation between the table frequencies.')
    parser.add_argument('-dt', '--deembedding-type', default='SCALAR', choices=nirfsg.DeembeddingType.__members__.keys(), type=str.upper, help='De-embedding type.')
    parser.add_argument('-s2p', '--s2p-file-path', default='', help='Optional Touchstone .s2p file to use instead of the modeled cable. Not supported on Linux.')
    parser.add_argument('-t', '--generation-time', default=10.0, type=float, help='Time, in seconds, to generate the signal.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.frequency, args.power_level, args.loss_db_at_1ghz, args.interpolation, args.deembedding_type, args.s2p_file_path, args.generation_time)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    for interpolation in INTERPOLATIONS:
        example('simulated5841', options, 2.4e9, -10.0, 1.5, interpolation, 'SCALAR', '', 0.1)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--generation-time', '0.1', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
