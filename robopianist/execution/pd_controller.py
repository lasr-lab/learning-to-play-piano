import matplotlib.pyplot as plt
import numpy as np


class PDController:
    def __init__(self, kp: float, kd: float, time_step: float, mass: float = 0.6433644039547359):
        """
        Initialize the PD controller with proportional and derivative gains.
        :param kp: Proportional gain (Kp)
        :param kd: Derivative gain (Kd)
        """
        self.kp = kp
        self.kd = kd
        self.mass = mass
        self.target_position = 0
        self.target_velocity = 0
        self.actual_position = 0
        self.actual_velocity = 0
        self.time_step = time_step

    def _get_force(self) -> float:
        """
        Compute the control output (torque/force) based on desired and actual position/velocity.
        :return: The computed torque/force to apply to the system.
        """
        position_error = self.target_position - self.actual_position
        velocity_error = self.target_velocity - self.actual_velocity

        torque_p = self.kp * position_error
        torque_d = self.kd * velocity_error

        return torque_p + torque_d

    def step(self) -> float:
        """
        Make a physics step according to the initialized timestep.
        :return: The updated position.
        """
        # Compute the control force/torque
        force = self._get_force()

        # F = m * a => a = F / m
        acceleration = force / self.mass

        self.actual_velocity += acceleration * self.time_step
        self.actual_position += self.actual_velocity * self.time_step
        return self.actual_position

    def set_target_position(self, position: float) -> None:
        self.target_position = position


if __name__ == "__main__":
    time_step = 0.01
    controller = PDController(kp=2000, kd=71.742, time_step=time_step)
    controller.set_target_position(1.0)

    # Arrays to store the time, position, and velocity for plotting
    total_time = 1.0
    num_steps = int(total_time / time_step)
    time_array = np.linspace(0, total_time, num_steps)
    position_array = np.zeros(num_steps)
    velocity_array = np.zeros(num_steps)

    # Simulation loop
    for i in range(num_steps):
        position_array[i] = controller.actual_position
        velocity_array[i] = controller.actual_velocity
        controller.step()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, position_array, label='Position')
    plt.plot(time_array, velocity_array, label='Velocity', linestyle='--')
    plt.axhline(y=1, color='r', linestyle=':', label='Target Position')
    plt.title('PD Controller - Position and Velocity over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position / Velocity')
    plt.legend()
    plt.grid(True)
    plt.show()
