from open_micro_stage_api import OpenMicroStageInterface
# source: https://github.com/HonakerM/MicroManipulatorStepper/tree/main

if __name__ == "__main__":
    print("Connecting to MicroManipuator Stage...")
    # create interface & connect
    oms = OpenMicroStageInterface(show_communication=True, show_log_messages=True)
    oms.connect('/dev/ttyACM0')

    for i in range(3): oms.calibrate_join(i, save_result=True)

    # homing
    oms.home()

    # to move => oms.move_to(x, y, z, f=10) -> in mm units
    # oms.wait_for_stop()

    """
    API functions (main ones)
        connect(port, baud_rate=921600)
        disconnect()
        set_workspace_transform(transform)
        get_workspace_transform()
        home(axis_list=None)
        calibrate_joint(joint_index, save_result)
        move_to(x, y, z, f, move_immediately, blocking, timeout)
        set_pose(x, y, z)
        dwell(time_s, blocking, timeout)
        enable_motors(enable)
        wait_for_stop(polling_interval_ms, disable_callbacks)
        set_max_acceleration(linear_accel, angular_accel)
        set_servo_parameter(pos_kp, pos_ki, vel_kp, vel_ki, vel_filter_tc)
    """
