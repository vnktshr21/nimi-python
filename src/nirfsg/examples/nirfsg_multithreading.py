'''Shares one NI-RFSG session between threads safely, using session.lock().

Background:
    A session can be used from several Python threads, for example one thread that changes
    the signal while another monitors the generation. Every nirfsg method and property access
    locks the session for the duration of that single call, so individual calls never
    interleave. A sequence of calls is not protected, though: another thread can run between
    "set the frequency" and "set the power level", and observe or change a half-updated
    configuration.

    To make a sequence atomic, hold the session lock around it:
      with session.lock():          # preferred: always released, even on exceptions
          ...
    or call session.lock() and session.unlock() yourself, using try/finally so that the
    lock is always released. The lock is re-entrant, so the nirfsg calls inside the block
    (which lock the session again) do not deadlock. Other threads that use the session wait
    until the lock is released.

    This example runs two threads during a CW generation:
      monitor thread - every interval, takes the lock and reads the frequency, power level,
                       and generation status as one consistent snapshot.
      control thread - steps through (frequency, power level) pairs. For each step it takes
                       the lock, writes both values, and waits for the output to settle, so
                       the monitor never sees a frequency from one step with the power level
                       of another.
    Exceptions raised in a thread are collected and raised again in the main thread.

Steps:
    1. Configure CW generation and start it.
    2. Start the monitor and control threads.
    3. Wait for the control thread to finish its steps, then stop the monitor.
'''

import argparse
import datetime
import nirfsg
import numpy as np
import sys
import threading


def example(resource_name, options, start_frequency, stop_frequency, start_power_level, stop_power_level, number_of_steps, step_time, monitor_interval):
    steps = list(zip(np.linspace(start_frequency, stop_frequency, number_of_steps), np.linspace(start_power_level, stop_power_level, number_of_steps)))
    stop_monitor = threading.Event()
    thread_errors = []

    with nirfsg.Session(resource_name=resource_name, id_query=False, reset_device=False, options=options) as session:
        session.generation_mode = nirfsg.GenerationMode.CW
        session.configure_rf(*steps[0])

        def monitor():
            try:
                while not stop_monitor.is_set():
                    # Read the three values as one snapshot.
                    with session.lock():
                        frequency = session.frequency
                        power_level = session.power_level
                        session.check_generation_status()
                    print(f'[monitor] {frequency / 1e6:.3f} MHz, {power_level:.2f} dBm, generating')
                    stop_monitor.wait(monitor_interval)
            except Exception as e:
                thread_errors.append(e)

        def control():
            try:
                for frequency, power_level in steps:
                    # lock() and unlock() can also be called explicitly. Use try/finally so the
                    # lock is released even if a call fails.
                    session.lock()
                    try:
                        session.frequency = frequency
                        session.power_level = power_level
                        session.wait_until_settled(datetime.timedelta(seconds=10))
                    finally:
                        session.unlock()
                    print(f'[control] settled at {frequency / 1e6:.3f} MHz, {power_level:.2f} dBm')
                    stop_monitor.wait(step_time)
            except Exception as e:
                thread_errors.append(e)

        with session.initiate():
            monitor_thread = threading.Thread(target=monitor, name='monitor')
            control_thread = threading.Thread(target=control, name='control')
            monitor_thread.start()
            control_thread.start()

            control_thread.join()
            stop_monitor.set()
            monitor_thread.join()

        if thread_errors:
            raise thread_errors[0]
        print('All steps complete.')


def _main(argsv):
    parser = argparse.ArgumentParser(description='Shares one NI-RFSG session between a monitor thread and a control thread.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--resource-name', default='PXI1Slot2', help='Resource name of the NI RF signal generator.')
    parser.add_argument('-fstart', '--start-frequency', default=1e9, type=float, help='Frequency, in Hz, of the first step.')
    parser.add_argument('-fstop', '--stop-frequency', default=2e9, type=float, help='Frequency, in Hz, of the last step.')
    parser.add_argument('-pstart', '--start-power-level', default=-20.0, type=float, help='Power level, in dBm, of the first step.')
    parser.add_argument('-pstop', '--stop-power-level', default=-10.0, type=float, help='Power level, in dBm, of the last step.')
    parser.add_argument('-s', '--number-of-steps', default=5, type=int, help='Number of (frequency, power level) steps.')
    parser.add_argument('-t', '--step-time', default=1.0, type=float, help='Time, in seconds, to stay at each step.')
    parser.add_argument('-m', '--monitor-interval', default=0.25, type=float, help='Time, in seconds, between monitor readings.')
    parser.add_argument('-op', '--option-string', default='', type=str, help='Option string for the session.')
    args = parser.parse_args(argsv)
    example(args.resource_name, args.option_string, args.start_frequency, args.stop_frequency, args.start_power_level, args.stop_power_level, args.number_of_steps, args.step_time, args.monitor_interval)


def main():
    _main(sys.argv[1:])


def test_example():
    options = {'simulate': True, 'driver_setup': {'Model': '5841', }, }
    example('simulated5841', options, 1e9, 2e9, -20.0, -10.0, 3, 0.02, 0.01)


def test_main():
    cmd_line = ['--resource-name', 'simulated5841', '--number-of-steps', '2', '--step-time', '0.02', '--monitor-interval', '0.01', '--option-string', 'Simulate=1, DriverSetup=Model:5841']
    _main(cmd_line)


if __name__ == '__main__':
    main()
