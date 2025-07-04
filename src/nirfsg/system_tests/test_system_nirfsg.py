import os
import pytest
import hightime
import sys
import pathlib
import array
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'shared'))
import system_test_utilities  # noqa: E402

# Set up global information we need
test_files_base_dir = os.path.join(os.path.dirname(__file__))


def get_test_file_path(file_name):
    return os.path.join(test_files_base_dir, file_name)

import nirfsg
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'generated/nirfsg'))

class SystemTests:
    @pytest.fixture(scope='function')
    def simulated_5841_session(self, session_creation_kwargs):
        with Session(resource_name="5841", options="Simulate=1, DriverSetup=Model:5841", **session_creation_kwargs) as sim_5841_session:
            yield sim_5841_session

# Attribute set and get related tests
    def test_set_float_attribute(self, simulated_5841_session):
        simulated_5841_session.power_level = -3.0
        assert simulated_5841_session.power_level == -3.0

    def test_get_float_attribute(self, simulated_5841_session):
        value = simulated_5841_session.power_level
        assert isinstance(value, float)

    def test_set_string_attribute(self, simulated_5841_session):
        simulated_5841_session.selected_script = "myScript"
        assert simulated_5841_session.selected_script == "myScript"

    def test_get_string_attribute(self, simulated_5841_session):
        value = simulated_5841_session.selected_script
        assert isinstance(value, str)

    def test_set_int_attribute(self, simulated_5841_session):
        val = simulated_5841_session.script_trigger_type
        simulated_5841_session.script_trigger_type = val
        assert simulated_5841_session.script_trigger_type == val

    def test_get_int_attribute(self, simulated_5841_session):
        value = simulated_5841_session.script_trigger_type
        assert isinstance(value, int)

    def test_get_attribute_string(self, simulated_5841_session):
        model = simulated_5841_session.instrument_model
        assert model == "NI PXIe-5841"
    
    def test_set_invalid_attribute_raises(self, simulated_5841_session):
        with pytest.raises(AttributeError):
            simulated_5841_session.non_existent_attribute = 123

# Multi-threading related tests (picked from other driver tests)
    def test_multi_threading_lock_unlock(self, simulated_5841_session):
        system_test_utilities.impl_test_multi_threading_lock_unlock(simulated_5841_session)

    def test_multi_threading_ivi_synchronized_wrapper_releases_lock(self, simulated_5841_session):
        system_test_utilities.impl_test_multi_threading_ivi_synchronized_wrapper_releases_lock(
            simulated_5841_session.abort)

# Error handling related tests
    def test_error_message(self, session_creation_kwargs):
        try:
            with Session(resource_name="invalid_model", id_query=False, reset_device=False, options="Simulate=1, DriverSetup=Model:invalid_model", **session_creation_kwargs):
                assert False
        except nirfsg.Error as e:
            assert e.code is not None
            assert isinstance(e.description, str)

    def test_get_error(self, simulated_5841_session):
        try:
            simulated_5841_session.instrument_model = ''
            assert False
        except nirfsg.Error as e:
            assert e.code is not None
            assert 'read-only' in e.description.lower()

# Utility method tests
    def test_reset(self, simulated_5841_session):
        # Save the original value of an attribute, change it, reset, and check it returns to default
        default_power_level = simulated_5841_session.power_level
        simulated_5841_session.power_level = default_power_level + 1.0
        assert simulated_5841_session.power_level == default_power_level + 1.0
        simulated_5841_session.reset()
        # After reset, attribute should return to default
        assert simulated_5841_session.power_level == default_power_level

    def test_reset_device(self, simulated_5841_session):
        # Save the original value of an attribute, change it, reset_device, and check it returns to default
        default_output_enabled = simulated_5841_session.output_enabled
        simulated_5841_session.output_enabled = not default_output_enabled
        assert simulated_5841_session.output_enabled == (not default_output_enabled)
        simulated_5841_session.reset_device()
        # After reset_device, attribute should return to default
        assert simulated_5841_session.output_enabled == default_output_enabled

    def test_self_cal(self, simulated_5841_session):
        simulated_5841_session.self_cal()

    def test_get_external_calibration_last_date_and_time(self, simulated_5841_session):
        dt = simulated_5841_session.get_external_calibration_last_date_and_time()
        assert isinstance(dt, hightime.datetime)

    def test_get_self_calibration_last_date_and_time(self, simulated_5841_session):
        dt = simulated_5841_session.get_self_calibration_last_date_and_time()
        assert isinstance(dt, hightime.datetime)

    def test_get_terminal_name(self, simulated_5841_session):
        assert simulated_5841_session.get_terminal_name(nirfsg.Signal.MARKER_EVENT, 'marker3') == 'ao/MarkerEvent/marker3'

    def test_query_arb_waveform_capabilities(self, simulated_5841_session):
        max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size = simulated_5841_session.query_arb_waveform_capabilities()
        assert max_number_waveforms > 0 # TODO: Check the actual maximum number of waveforms supported
        assert waveform_quantum > 0 # TODO: Check the actual waveform quantum supported
        assert min_waveform_size > 0 # TODO: Check the actual minimum waveform size supported
        assert max_waveform_size > 0 # TODO: Check the actual maximum waveform size supported

# Repeated capability tests
    def test_markers_rep_cap(self, simulated_5841_session):
        marker = simulated_5841_session.markers[0]
        requested_terminal_name = '/Dev0/PXI_Trig0'
        marker.exported_marker_event_output_terminal = requested_terminal_name
        assert marker.exported_marker_event_output_terminal == requested_terminal_name

    def test_script_triggers_rep_cap(self, simulated_5841_session):
        trigger = simulated_5841_session.script_triggers[0]
        requested_terminal_name = '/Dev0/PXI_Trig0'
        trigger.exported_script_trigger_output_terminal = requested_terminal_name
        assert trigger.exported_script_trigger_output_terminal == requested_terminal_name

    def test_waveform_rep_cap(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform', waveform_data, False)
        requested_waveform_iq_rate = 1e6
        simulated_5841_session.waveform['my_waveform'].waveform_iq_rate = requested_waveform_iq_rate
        assert simulated_5841_session.waveform['my_waveform'].waveform_iq_rate == requested_waveform_iq_rate

    def test_deembedding_port_rep_cap(self, simulated_5841_session):
        port = simulated_5841_session.deembedding_port['']
        requested_compensation_gain = 10.0
        port.deembedding_compensation_gain = requested_compensation_gain
        assert port.deembedding_compensation_gain == requested_compensation_gain

# Configuration methods related tests
    def test_configure_power_and_frequency(self, simulated_5841_session):
        simulated_5841_session.configure_rf(2e9, -5.0)
        assert simulated_5841_session.power_level == -5.0
        assert simulated_5841_session.frequency == 2e9

    def test_configure_output_enabled(self, simulated_5841_session):
        simulated_5841_session.configure_output_enabled(True)
        assert simulated_5841_session.output_enabled is True
        simulated_5841_session.configure_output_enabled(False)
        assert simulated_5841_session.output_enabled is False

    def test_configure_generation_mode(self, simulated_5841_session):
        # Test CW mode
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.CW)
        assert simulated_5841_session.generation_mode == nirfsg.GenerationMode.CW
        # Test ARB_WAVEFORM mode
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        assert simulated_5841_session.generation_mode == nirfsg.GenerationMode.ARB_WAVEFORM
        # Test SCRIPT mode
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        assert simulated_5841_session.generation_mode == nirfsg.GenerationMode.SCRIPT

    def test_configure_signalbandwidth_and_powerleveltype(self, simulated_5841_session):
        simulated_5841_session.configure_signal_bandwidth(20e6)
        assert simulated_5841_session.signal_bandwidth == 20e6
        simulated_5841_session.configure_power_level_type(nirfsg.PowerLevelType.PEAK)
        assert simulated_5841_session.power_level_type == nirfsg.PowerLevelType.PEAK
    
    def test_write_arb_waveform_numpy_complex128(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data, False)
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == True
        assert simulated_5841_session.check_if_waveform_exists('my_waveform2') == False
    
    def test_write_arb_waveform_numpy_complex64(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1600, 1 + 0j, dtype=np.complex64)
        simulated_5841_session.write_arb_waveform('my_waveform2', waveform_data, False)
        assert simulated_5841_session.check_if_waveform_exists('my_waveform2') == True
        assert simulated_5841_session.check_if_waveform_exists('my_waveform3') == False

    def test_write_arb_waveform_numpy_interleaved_int16(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.array([1, 0] * 3000, dtype=np.int16)
        simulated_5841_session.write_arb_waveform('my_waveform3', waveform_data, False)
        assert simulated_5841_session.check_if_waveform_exists('my_waveform3') == True
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == False

    def test_write_arb_waveform_with_wrong_datatype(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data_wrong_numpy_type = np.array([1, 0] * 3000, dtype=np.int32)
        waveform_data_non_numpy_type = array.array('h', [1, 0] * 3000)
        try:
            simulated_5841_session.write_arb_waveform('my_waveform3', waveform_data_wrong_numpy_type, False)
        except TypeError:
            pass
        try:
            simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data_non_numpy_type, False)
        except TypeError:
            pass

    def test_clear_arb_waveform(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data, False)
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == True
        simulated_5841_session.clear_arb_waveform('my_waveform1')
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == False

    def test_clear_all_arb_waveforms(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data, False)
        simulated_5841_session.write_arb_waveform('my_waveform2', waveform_data, False)
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == True
        assert simulated_5841_session.check_if_waveform_exists('my_waveform2') == True
        simulated_5841_session.clear_all_arb_waveforms()
        assert simulated_5841_session.check_if_waveform_exists('my_waveform1') == False
        assert simulated_5841_session.check_if_waveform_exists('my_waveform2') == False

    def test_allocate_arb_waveform(self, simulated_5841_session):
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.allocate_arb_waveform('foo', len(waveform_data))
        simulated_5841_session.write_arb_waveform('foo', waveform_data, False)

    def test_set_arb_waveform_next_write_position(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data, True)
        simulated_5841_session.set_arb_waveform_next_write_position('my_waveform1', nirfsg.RelativeTo.START_OF_WAVEFORM, 500)
        waveform_data__new_second_half = np.full(500, 0 + 1j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data__new_second_half, False)

    def test_write_script(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(1000, 0.707 + 0.707j, dtype=np.complex64)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data, False)
        script = '''script myScript
        repeat forever
        generate my_waveform1
        end repeat
        end script'''
        simulated_5841_session.write_script('myScript1', script)
        assert simulated_5841_session.check_if_script_exists('myScript1') == True
        assert simulated_5841_session.check_if_script_exists('myScript2') == False
    
    def test_write_script_with_bad_script(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        script = '''script myScript
        repeat forever
        generate my_waveform1
        end repeat
        end script'''
        try:
            simulated_5841_session.write_script('myScript1', script)
        except nirfsg.Error as e:
            assert e.code == -1074135027  # Error : TODO
            assert e.description.find('Attribute is read-only.') != -1
    
    def test_configure_software_trigger(self, simulated_5841_session):
        simulated_5841_session.configure_software_start_trigger()
        assert simulated_5841_session.start_trigger_type == nirfsg.StartTrigType.SOFTWARE
        simulated_5841_session.configure_software_script_trigger()
        assert simulated_5841_session.script_trigger_type == nirfsg.ScriptTrigType.SOFTWARE

    def test_disable_trigger(self, simulated_5841_session):
        simulated_5841_session.configure_software_start_trigger()
        assert simulated_5841_session.start_trigger_type == nirfsg.StartTrigType.SOFTWARE
        simulated_5841_session.disable_start_trigger()
        assert simulated_5841_session.start_trigger_type == nirfsg.StartTrigType.NONE
        simulated_5841_session.configure_software_script_trigger()
        assert simulated_5841_session.script_trigger_type == nirfsg.ScriptTrigType.SOFTWARE
        simulated_5841_session.disable_script_trigger()
        assert simulated_5841_session.script_trigger_type == nirfsg.ScriptTrigType.NONE

    def test_export_signal(self, simulated_5841_session):
        simulated_5841_session.export_signal(nirfsg.Signal.START_TRIGGER, '', 'PXI_Trig0')
        assert simulated_5841_session.exported_start_trigger_output_terminal == 'PXI_Trig0'
        simulated_5841_session.export_signal(nirfsg.Signal.SCRIPT_TRIGGER, 'scriptTrigger2', 'PXI_Trig1')
        assert simulated_5841_session.script_triggers[2].exported_script_trigger_output_terminal == 'PXI_Trig1'
        simulated_5841_session.export_signal(nirfsg.Signal.MARKER_EVENT, 'marker1', 'PXI_Trig2')
        assert simulated_5841_session.markers[1].exported_marker_event_output_terminal == 'PXI_Trig2'
        simulated_5841_session.export_signal(nirfsg.Signal.REF_CLOCK, '', '')
        assert simulated_5841_session.exported_ref_clock_output_terminal == ''
        simulated_5841_session.export_signal(nirfsg.Signal.STARTED_EVENT, '', 'PXI_Trig3')
        assert simulated_5841_session.exported_started_event_output_terminal == 'PXI_Trig3'
        simulated_5841_session.export_signal(nirfsg.Signal.DONE_EVENT, '', 'PFI0')
        assert simulated_5841_session.exported_done_event_output_terminal == 'PFI0'

    def test_export_signal_with_invalid_signal(self, simulated_5841_session):
        with pytest.raises(nirfsg.Error) as exc_info:
            simulated_5841_session.export_signal(nirfsg.Signal.INVALID, '', 'PXI_Trig0')
        assert exc_info.value.code == -1074135027  # Error : TODO
        assert 'Invalid signal' in exc_info.value.description

    def test_export_signal_with_invalid_terminal(self, simulated_5841_session):
        with pytest.raises(nirfsg.Error) as exc_info:
            simulated_5841_session.export_signal(nirfsg.Signal.START_TRIGGER, '', 'InvalidTerminal')
        assert exc_info.value.code == -1074135027  # Error : TODO
        assert 'Invalid terminal' in exc_info.value.description
    
    def test_save_load_configuration(self, simulated_5841_session):
        simulated_5841_session.configure_rf(2e9, -5.0)
        simulated_5841_session.iq_rate = 1e6
        simulated_5841_session.save_configurations_to_file(get_test_file_path('tempConfiguration.json'))
        simulated_5841_session.configure_rf(3e9, -15.0)
        simulated_5841_session.iq_rate = 2e6
        assert simulated_5841_session.frequency == 3e9
        assert simulated_5841_session.power_level == -15.0
        assert simulated_5841_session.iq_rate == 2e6
        simulated_5841_session.load_configurations_from_file(get_test_file_path('tempConfiguration.json'))
        assert simulated_5841_session.frequency == 2e9
        assert simulated_5841_session.power_level == -5.0
        assert simulated_5841_session.iq_rate == 1e6
        os.remove(get_test_file_path('tempConfiguration.json'))

# Basic tests for generation
    def test_cw_generation(self, simulated_5841_session):
        simulated_5841_session.configure_rf(2e9, -5.0)
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False # IsDone will never be True in CW mode

    def test_abort(self, simulated_5841_session):
        simulated_5841_session.configure_rf(2e9, -5.0)
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False # IsDone will never be True in CW mode
        assert simulated_5841_session.check_generation_status() == True # IsDone should now be True after aborting

    def test_multiple_arb_waveform_generation(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data1 = np.full(1000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform1', waveform_data1, False)
        waveform_data2 = np.full(8000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform2', waveform_data2, False)
        simulated_5841_session.arb_selected_waveform = 'my_waveform1'
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False # IsDone will never be True since we have not authored waveform_repeat_count
        simulated_5841_session.arb_selected_waveform = 'my_waveform2'
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False # IsDone will never be True since we have not authored waveform_repeat_count

    def test_multiple_script_generation(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(2000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform', waveform_data, False)
        script0 = '''script myScript0
        repeat 1
        generate my_waveform
        end repeat
        end script'''
        script1 = '''script myScript1
        repeat forever
        generate my_waveform
        end repeat
        end script'''
        simulated_5841_session.write_script(script0)
        simulated_5841_session.write_script(script1)
        simulated_5841_session.selected_script = 'myScript0'
        with simulated_5841_session.initiate():
            hightime.sleep(2)
            assert simulated_5841_session.check_generation_status() == True # IsDone will be True since we are repeating only once
        simulated_5841_session.selected_script = 'myScript1'
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False # IsDone will never be True since we are repeating forever

    def test_send_software_edge_trigger(self, simulated_5841_session):
        simulated_5841_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(2000, 1 + 0j, dtype=np.complex128)
        simulated_5841_session.write_arb_waveform('my_waveform', waveform_data, False)
        script0 = '''script myScript0
        wait until scriptrigger0
        generate my_waveform
        end script'''
        simulated_5841_session.write_script(script0)
        simulated_5841_session.configure_software_start_trigger()
        simulated_5841_session.configure_software_script_trigger()
        with simulated_5841_session.initiate():
            simulated_5841_session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, '')
            simulated_5841_session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, 'scriptTrigger0')

    def test_deembedding_table_with_s2p_file(self, simulated_5841_session):
        simulated_5841_session.create_deembedding_sparameter_table_s2_p_file('', 'myTable1', get_test_file_path('samples2pfile.s2p'), nirfsg.SparameterOrientation.PORT2)
        simulated_5841_session.create_deembedding_sparameter_table_s2_p_file('', 'newTable2', get_test_file_path('samples2pfile.s2p'), nirfsg.SparameterOrientation.PORT1)
        simulated_5841_session.configure_deembedding_table_interpolation_linear('', 'myTable1', nirfsg.Format.MAGNITUDE_AND_PHASE)
        simulated_5841_session.deembedding_port[''].deembedding_selected_table = 'myTable1'
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False
        simulated_5841_session.delete_deembedding_table('', 'myTable1')
        simulated_5841_session.deembedding_port[''].deembedding_selected_table = 'myTable2'
        with simulated_5841_session.initiate():
            assert simulated_5841_session.check_generation_status() == False
        simulated_5841_session.delete_all_deembedding_tables()
        with pytest.raises(nirfsg.Error) as exc_info:
            simulated_5841_session.commit()
        assert exc_info.value.code == -1074135027  # Error : TODO
        assert 'Invalid table' in exc_info.value.description
        simulated_5841_session.deembedding_port[''].deembedding_selected_table = ''
        try:
            simulated_5841_session.commit()
        except Exception as e:
            assert False, f"simulated_5841_session.commit() raised an exception: {e}"

    def test_wait_until_settled(self, simulated_5841_session):
        simulated_5841_session.configure_rf(2e9, -5.0)
        with simulated_5841_session.initiate():
            simulated_5841_session.wait_until_settled(15000)

class TestLibrary(SystemTests):
    @pytest.fixture(scope='class')
    def session_creation_kwargs(self):
        return {}
