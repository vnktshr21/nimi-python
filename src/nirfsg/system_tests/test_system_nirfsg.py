import os
import pytest
import hightime
import time
import sys
import pathlib
import array
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'shared'))
import system_test_utilities  # noqa: E402

# Set up global information we need
test_files_base_dir = os.path.join(os.path.dirname(__file__))
use_simulated_session = True
real_hw_resource_name = '5841'

def get_test_file_path(file_name):
    return os.path.join(test_files_base_dir, file_name)

import nirfsg
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'generated/nirfsg'))

class SystemTests:
    @pytest.fixture(scope='function')
    def test_rfsg_device_session(self, session_creation_kwargs):
        if use_simulated_session:
            with nirfsg.Session("", options="Simulate=1, DriverSetup=Model:5841", **session_creation_kwargs) as sim_5841_session:
                yield sim_5841_session
        else:
            with nirfsg.Session(real_hw_resource_name, options="", **session_creation_kwargs) as real_rfsg_device_session:
                yield real_rfsg_device_session

# Attribute set and get related tests
    def test_get_float_attribute(self, test_rfsg_device_session):
        value = test_rfsg_device_session.power_level
        assert isinstance(value, float)
    
    def test_set_float_attribute(self, test_rfsg_device_session):
        test_rfsg_device_session.power_level = -3.0
        assert test_rfsg_device_session.power_level == -3.0

    def test_get_string_attribute(self, test_rfsg_device_session):
        model = test_rfsg_device_session.instrument_model
        assert model == "NI PXIe-5841"

    def test_set_string_attribute(self, test_rfsg_device_session):
        test_rfsg_device_session.selected_script = "myScript"
        assert test_rfsg_device_session.selected_script == "myScript"

    def test_get_int32_attribute(self, test_rfsg_device_session):
        value = test_rfsg_device_session.external_calibration_recommended_interval
        assert isinstance(value, int)

    def test_set_int32_enum_attribute(self, test_rfsg_device_session):
        test_rfsg_device_session.frequency_settling_units = nirfsg.FrequencySettlingUnits.TIME_AFTER_LOCK
        assert test_rfsg_device_session.frequency_settling_units == nirfsg.FrequencySettlingUnits.TIME_AFTER_LOCK

    def test_set_invalid_attribute_raises(self, test_rfsg_device_session):
        with pytest.raises(AttributeError):
            test_rfsg_device_session.non_existent_attribute = 123

# Multi-threading related tests (picked from other driver tests)
    def test_multi_threading_lock_unlock(self, test_rfsg_device_session):
        system_test_utilities.impl_test_multi_threading_lock_unlock(test_rfsg_device_session)

    def test_multi_threading_ivi_synchronized_wrapper_releases_lock(self, test_rfsg_device_session):
        system_test_utilities.impl_test_multi_threading_ivi_synchronized_wrapper_releases_lock(
            test_rfsg_device_session.abort)

# Error handling related tests
    def test_error_message(self, session_creation_kwargs):
        try:
            with nirfsg.Session(resource_name="invalid_model", id_query=False, reset_device=False, options="Simulate=1, DriverSetup=Model:invalid_model", **session_creation_kwargs):
                assert False
        except nirfsg.Error as e:
            print(e)
            assert e.code == -1074135025
            assert "Invalid model in DriverSetup string" in e.description

    def test_get_error(self, test_rfsg_device_session):
        try:
            test_rfsg_device_session.instrument_model = ''
            assert False
        except nirfsg.Error as e:
            print(e)
            assert e.code == -1074135027
            assert "Attribute is read-only" in e.description

# Utility method tests
    def test_reset(self, test_rfsg_device_session):
        # Save the original value of an attribute, change it, reset, and check it returns to default
        default_power_level = test_rfsg_device_session.power_level
        test_rfsg_device_session.power_level = default_power_level + 1.0
        assert test_rfsg_device_session.power_level == default_power_level + 1.0
        test_rfsg_device_session.reset()
        # After reset, attribute should return to default
        assert test_rfsg_device_session.power_level == default_power_level

    def test_self_cal(self, test_rfsg_device_session):
        test_rfsg_device_session.self_cal()

    def test_get_external_calibration_last_date_and_time(self, test_rfsg_device_session):
        year, month, day, hour, minute, second = test_rfsg_device_session._get_external_calibration_last_date_and_time()
        print(year, month, day, hour, minute, second)
        dt = test_rfsg_device_session.get_external_calibration_last_date_and_time()
        assert isinstance(dt, hightime.datetime)

    def test_get_self_calibration_last_date_and_time(self, test_rfsg_device_session):
        year, month, day, hour, minute, second = test_rfsg_device_session._get_self_calibration_date_and_time()
        print(year, month, day, hour, minute, second)
        dt = test_rfsg_device_session.get_self_calibration_last_date_and_time()
        assert isinstance(dt, hightime.datetime)

    def test_get_terminal_name(self, test_rfsg_device_session):
        assert test_rfsg_device_session.get_terminal_name(nirfsg.Signal.MARKER_EVENT, 'marker3') == '/5841/ao/0/Marker3Event'

    def test_query_arb_waveform_capabilities(self, test_rfsg_device_session):
        max_number_waveforms, waveform_quantum, min_waveform_size, max_waveform_size = test_rfsg_device_session.query_arb_waveform_capabilities()
        assert max_number_waveforms > 0 # TODO: Check the actual maximum number of waveforms supported
        assert waveform_quantum > 0 # TODO: Check the actual waveform quantum supported
        assert min_waveform_size > 0 # TODO: Check the actual minimum waveform size supported
        assert max_waveform_size > 0 # TODO: Check the actual maximum waveform size supported

# Repeated capability tests
    def test_markers_rep_cap(self, test_rfsg_device_session):
        marker = test_rfsg_device_session.markers[0]
        requested_terminal_name = '/Dev0/PXI_Trig0'
        marker.exported_marker_event_output_terminal = requested_terminal_name
        assert marker.exported_marker_event_output_terminal == requested_terminal_name

    def test_script_triggers_rep_cap(self, test_rfsg_device_session):
        trigger = test_rfsg_device_session.script_triggers[0]
        requested_terminal_name = '/Dev0/PXI_Trig0'
        trigger.exported_script_trigger_output_terminal = requested_terminal_name
        assert trigger.exported_script_trigger_output_terminal == requested_terminal_name

    def test_waveform_rep_cap(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform', waveform_data, False)
        requested_waveform_iq_rate = 1e6
        test_rfsg_device_session.waveform['my_waveform'].waveform_iq_rate = requested_waveform_iq_rate
        assert test_rfsg_device_session.waveform['my_waveform'].waveform_iq_rate == requested_waveform_iq_rate

    def test_deembedding_port_rep_cap(self, test_rfsg_device_session):
        port = test_rfsg_device_session.deembedding_port['']
        requested_deembedding_type = nirfsg.DeembeddingTypeAttrVals.SCALAR
        port.deembedding_type = requested_deembedding_type
        assert port.deembedding_type == requested_deembedding_type

# Configuration methods related tests
    def test_configure_power_and_frequency(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_rf(2e9, -5.0)
        assert test_rfsg_device_session.power_level == -5.0
        assert test_rfsg_device_session.frequency == 2e9

    def test_configure_output_enabled(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_output_enabled(True)
        assert test_rfsg_device_session.output_enabled is True
        test_rfsg_device_session.configure_output_enabled(False)
        assert test_rfsg_device_session.output_enabled is False

    def test_configure_generation_mode(self, test_rfsg_device_session):
        # Test CW mode
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.CW)
        assert test_rfsg_device_session.generation_mode == nirfsg.GenerationMode.CW
        # Test ARB_WAVEFORM mode
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        assert test_rfsg_device_session.generation_mode == nirfsg.GenerationMode.ARB_WAVEFORM
        # Test SCRIPT mode
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        assert test_rfsg_device_session.generation_mode == nirfsg.GenerationMode.SCRIPT

    def test_configure_signalbandwidth_and_powerleveltype(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_signal_bandwidth(8e5)
        assert test_rfsg_device_session.signal_bandwidth == 8e5
        test_rfsg_device_session.configure_power_level_type(nirfsg.PowerLevelType.PEAK)
        assert test_rfsg_device_session.power_level_type == nirfsg.PowerLevelType.PEAK
    
    def test_write_arb_waveform_numpy_complex128(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, False)
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == True
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform2')
        assert waveform_exists == False
    
    def test_write_arb_waveform_numpy_complex64(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1600, 1 + 0j, dtype=np.complex64)
        test_rfsg_device_session.write_arb_waveform('my_waveform2', waveform_data, False)
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform2')
        assert waveform_exists == True
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform3')
        assert waveform_exists == False

    def test_write_arb_waveform_numpy_interleaved_int16(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        test_rfsg_device_session.power_level_type = nirfsg.PowerLevelType.PEAK # Needed for writing unscaled int16 data
        waveform_data = np.array([1, 0] * 3000, dtype=np.int16)
        test_rfsg_device_session.write_arb_waveform('my_waveform3', waveform_data, False)
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform3')
        assert waveform_exists == True
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == False

    def test_write_arb_waveform_with_wrong_datatype(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data_wrong_numpy_type = np.array([1, 0] * 3000, dtype=np.int32)
        waveform_data_non_numpy_type = array.array('h', [1, 0] * 3000)
        try:
            test_rfsg_device_session.write_arb_waveform('my_waveform3', waveform_data_wrong_numpy_type, False)
            assert False
        except TypeError:
            pass
        try:
            test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data_non_numpy_type, False)
            assert False
        except TypeError:
            pass

    def test_clear_arb_waveform(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, False)
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == True
        test_rfsg_device_session.clear_arb_waveform('my_waveform1')
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == False

    def test_clear_all_arb_waveforms(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, False)
        test_rfsg_device_session.write_arb_waveform('my_waveform2', waveform_data, False)
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == True
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform2')
        assert waveform_exists == True
        test_rfsg_device_session.clear_all_arb_waveforms()
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform1')
        assert waveform_exists == False
        waveform_exists = test_rfsg_device_session.check_if_waveform_exists('my_waveform2')
        assert waveform_exists == False

    def test_allocate_arb_waveform(self, test_rfsg_device_session):
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.allocate_arb_waveform('foo', len(waveform_data))
        test_rfsg_device_session.write_arb_waveform('foo', waveform_data, False)

    def test_set_arb_waveform_next_write_position(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        test_rfsg_device_session.power_level_type = nirfsg.PowerLevelType.PEAK # To be able to call write multiple times on same waveform
        waveform_data = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, True)
        test_rfsg_device_session.set_arb_waveform_next_write_position('my_waveform1', nirfsg.RelativeTo.START_OF_WAVEFORM, 500)
        waveform_data_new_second_half = np.full(500, 0 + 1j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data_new_second_half, False)

    def test_write_script(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(1000, 0.707 + 0.707j, dtype=np.complex64)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, False)
        script = '''script myScript1
        repeat forever
        generate my_waveform1
        end repeat
        end script'''
        test_rfsg_device_session.write_script(script)

    @pytest.mark.skipif(use_simulated_session == True, reason="Cannot run on simulated device")
    def test_check_if_script_exists(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(1000, 0.707 + 0.707j, dtype=np.complex64)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data, False)
        script = '''script myScript1
        repeat forever
        generate my_waveform1
        end repeat
        end script'''
        test_rfsg_device_session.write_script(script)
        script_exists = test_rfsg_device_session.check_if_script_exists('myScript1')
        assert script_exists == True
        script_exists = test_rfsg_device_session.check_if_script_exists('myScript2')
        assert script_exists == False
    
    def test_write_script_with_bad_script(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        script = '''script myScript1
        repeat forever
        generate my_waveform1
        end repeat
        end script'''
        try:
            test_rfsg_device_session.write_script(script)
            assert False
        except nirfsg.Error as e:
            print(e)
            assert e.code == -1074135027  # Error : TODO
            assert e.description.find('Attribute is read-only.') != -1
    
    def test_configure_software_trigger(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_software_start_trigger()
        assert test_rfsg_device_session.start_trigger_type == nirfsg.StartTrigType.SOFTWARE
        test_rfsg_device_session.configure_software_script_trigger('scriptTrigger0')
        assert test_rfsg_device_session.script_triggers[0].script_trigger_type == nirfsg.ScriptTrigType.SOFTWARE

    def test_disable_trigger(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_software_start_trigger()
        assert test_rfsg_device_session.start_trigger_type == nirfsg.StartTrigType.SOFTWARE
        test_rfsg_device_session.disable_start_trigger()
        assert test_rfsg_device_session.start_trigger_type == nirfsg.StartTrigType.NONE
        test_rfsg_device_session.configure_software_script_trigger('scriptTrigger3')
        assert test_rfsg_device_session.script_triggers[3].script_trigger_type == nirfsg.ScriptTrigType.SOFTWARE
        test_rfsg_device_session.disable_script_trigger('scriptTrigger3')
        assert test_rfsg_device_session.script_triggers[3].script_trigger_type == nirfsg.ScriptTrigType.NONE

    def test_export_signal(self, test_rfsg_device_session):
        test_rfsg_device_session.export_signal(nirfsg.Signal.START_TRIGGER, '', 'PXI_Trig0')
        assert test_rfsg_device_session.exported_start_trigger_output_terminal == 'PXI_Trig0'
        test_rfsg_device_session.export_signal(nirfsg.Signal.SCRIPT_TRIGGER, 'scriptTrigger2', 'PXI_Trig1')
        assert test_rfsg_device_session.script_triggers[2].exported_script_trigger_output_terminal == 'PXI_Trig1'
        test_rfsg_device_session.export_signal(nirfsg.Signal.MARKER_EVENT, 'marker1', 'PXI_Trig2')
        assert test_rfsg_device_session.markers[1].exported_marker_event_output_terminal == 'PXI_Trig2'
        test_rfsg_device_session.export_signal(nirfsg.Signal.REF_CLOCK, '', '')
        assert test_rfsg_device_session.exported_ref_clock_output_terminal == ''
        test_rfsg_device_session.export_signal(nirfsg.Signal.STARTED_EVENT, '', 'PXI_Trig3')
        assert test_rfsg_device_session.exported_started_event_output_terminal == 'PXI_Trig3'
        test_rfsg_device_session.export_signal(nirfsg.Signal.DONE_EVENT, '', 'PFI0')
        assert test_rfsg_device_session.exported_done_event_output_terminal == 'PFI0'

    def test_export_signal_with_invalid_signal(self, test_rfsg_device_session):
        try:
            test_rfsg_device_session.export_signal(nirfsg.Signal.INVALID, '', 'PXI_Trig0')
            assert False
        except AttributeError as e:
            pass

    def test_export_signal_with_invalid_terminal(self, test_rfsg_device_session):
        try:
            test_rfsg_device_session.export_signal(nirfsg.Signal.START_TRIGGER, '', 'InvalidTerminal')
            test_rfsg_device_session.commit()
            assert False
        except nirfsg.Error as e:
            assert e.code == -1074135027  # Error : TODO
            assert 'Invalid terminal' in e.description
    
    def test_save_load_configuration(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_rf(2e9, -5.0)
        test_rfsg_device_session.iq_rate = 1e6
        test_rfsg_device_session.save_configurations_to_file(get_test_file_path('tempConfiguration.json'))
        assert os.path.exists(get_test_file_path('tempConfiguration.json'))
        test_rfsg_device_session.configure_rf(3e9, -15.0)
        test_rfsg_device_session.iq_rate = 2e6
        assert test_rfsg_device_session.frequency == 3e9
        assert test_rfsg_device_session.power_level == -15.0
        assert test_rfsg_device_session.iq_rate == 2e6
        test_rfsg_device_session.load_configurations_from_file(get_test_file_path('tempConfiguration.json'))
        assert test_rfsg_device_session.frequency == 2e9
        assert test_rfsg_device_session.power_level == -5.0
        assert test_rfsg_device_session.iq_rate == 1e6
        os.remove(get_test_file_path('tempConfiguration.json'))

# Basic tests for generation
    def test_cw_generation(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_rf(2e9, -5.0)
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False # IsDone will never be True in CW mode

    def test_abort(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_rf(2e9, -5.0)
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False # IsDone will never be True in CW mode
        IsDone = test_rfsg_device_session.check_generation_status()
        assert IsDone == True # IsDone should now be True after aborting

    def test_multiple_arb_waveform_generation(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.ARB_WAVEFORM)
        waveform_data1 = np.full(1000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform1', waveform_data1, False)
        waveform_data2 = np.full(8000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform2', waveform_data2, False)
        test_rfsg_device_session.arb_selected_waveform = 'my_waveform1'
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False # IsDone will never be True since we have not authored waveform_repeat_count
        test_rfsg_device_session.arb_selected_waveform = 'my_waveform2'
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False # IsDone will never be True since we have not authored waveform_repeat_count

    def test_multiple_script_generation(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(2000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform', waveform_data, False)
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
        test_rfsg_device_session.write_script(script0)
        test_rfsg_device_session.write_script(script1)
        test_rfsg_device_session.selected_script = 'myScript0'
        with test_rfsg_device_session.initiate():
            time.sleep(2)
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == True # IsDone will be True since we are repeating only once
        test_rfsg_device_session.selected_script = 'myScript1'
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False # IsDone will never be True since we are repeating forever

    def test_send_software_edge_trigger(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_generation_mode(nirfsg.GenerationMode.SCRIPT)
        waveform_data = np.full(2000, 1 + 0j, dtype=np.complex128)
        test_rfsg_device_session.write_arb_waveform('my_waveform', waveform_data, False)
        script0 = '''script myScript0
        wait until scriptrigger0
        generate my_waveform
        end script'''
        test_rfsg_device_session.write_script(script0)
        test_rfsg_device_session.configure_software_start_trigger()
        test_rfsg_device_session.configure_software_script_trigger('scriptTrigger0')
        with test_rfsg_device_session.initiate():
            test_rfsg_device_session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.START, '')
            test_rfsg_device_session.send_software_edge_trigger(nirfsg.SoftwareTriggerType.SCRIPT, 'scriptTrigger0')

    def test_deembedding_table_with_s2p_file(self, test_rfsg_device_session):
        test_rfsg_device_session.create_deembedding_sparameter_table_s2_p_file('', 'myTable1', get_test_file_path('samples2pfile.s2p'), nirfsg.SparameterOrientation.PORT2)
        test_rfsg_device_session.create_deembedding_sparameter_table_s2_p_file('', 'newTable2', get_test_file_path('samples2pfile.s2p'), nirfsg.SparameterOrientation.PORT1)
        test_rfsg_device_session.configure_deembedding_table_interpolation_linear('', 'myTable1', nirfsg.Format.MAGNITUDE_AND_PHASE)
        test_rfsg_device_session.deembedding_port[''].deembedding_selected_table = 'myTable1'
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False
        test_rfsg_device_session.delete_deembedding_table('', 'myTable1')
        test_rfsg_device_session.deembedding_port[''].deembedding_selected_table = 'myTable2'
        with test_rfsg_device_session.initiate():
            IsDone = test_rfsg_device_session.check_generation_status()
            assert IsDone == False
        test_rfsg_device_session.delete_all_deembedding_tables()
        with pytest.raises(nirfsg.Error) as exc_info:
            test_rfsg_device_session.commit()
        assert exc_info.value.code == -1074135027  # Error : TODO
        assert 'Invalid table' in exc_info.value.description
        test_rfsg_device_session.deembedding_port[''].deembedding_selected_table = ''
        try:
            test_rfsg_device_session.commit()
        except Exception as e:
            assert False, f"test_rfsg_device_session.commit() raised an exception: {e}"

    def test_wait_until_settled(self, test_rfsg_device_session):
        test_rfsg_device_session.configure_rf(2e9, -5.0)
        with test_rfsg_device_session.initiate():
            test_rfsg_device_session.wait_until_settled(15000)

class TestLibrary(SystemTests):
    @pytest.fixture(scope='class')
    def session_creation_kwargs(self):
        return {}
