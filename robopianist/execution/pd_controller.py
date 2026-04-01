import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class PDController:
    def __init__(self, time_step: float,
                 kp: Union[float, list, np.ndarray],
                 kd: Union[float, list, np.ndarray],
                 sluggish: Union[float, list, np.ndarray] = 0.0,
                 max_u: Union[float, list, np.ndarray] = math.inf, ) -> None:
        """
        Initialize the PD controller.
        :param time_step: Time step in seconds
        :param kp: Proportional gain (Kp)
        :param kd: Derivative gain (Kd)
        :param sluggish: Real servos need some time until torque is applied. Should be 0 <= x < 1 with 0 -> no sluggishness
        :param max_u: Maximum control output
        """
        self.time_step = time_step
        self.target_position = None
        self.last_error = None
        self.last_u = 0.0

        self.kp = np.array(kp, dtype=np.float32, ndmin=1)
        self.kd = np.array(kd, dtype=np.float32, ndmin=1)
        self.sluggish = np.array(sluggish, dtype=np.float32, ndmin=1)
        self.max_u = np.array(max_u, dtype=np.float32, ndmin=1)

    @property
    def number_of_dofs(self) -> int:
        if self.kp is None:
            return 0
        return self.kp.shape[0]

    def step(self, current_position: Union[float, np.ndarray]) -> np.ndarray:
        if self.target_position is None:
            raise ValueError("Target position not set. Please set it using set_target_position().")

        error = self.target_position - current_position

        if self.last_error is None:
            self.last_error = error

        derivative_u = self.kd * (error - self.last_error) / self.time_step
        proportional_u = self.kp * error
        u = np.clip(derivative_u + proportional_u, -self.max_u, self.max_u)
        u = (1 - self.sluggish) * u + self.sluggish * self.last_u
        self.last_u = u
        self.last_error = error
        return u

    def set_target_position(self, position: Union[float, np.ndarray]) -> None:
        self.target_position = np.array(position, dtype=np.float32, ndmin=1)


if __name__ == "__main__":
    if __name__ == "__main__":
        # simple open-loop simulation to test PDController behavior
        controller = PDController(time_step=0.01, kp=8, kd=6.5, max_u=0.2)
        controller.set_target_position(0.33)

        dt = controller.time_step
        duration = 10.0
        t = np.arange(0.0, duration, dt)

        pos = 0.0
        vel = 0.0

        positions = []
        controls = []
        errors = []

        for _ in t:
            u = controller.step(pos).item()
            # simple dynamics: unit mass, no friction
            vel += u * dt
            pos += vel * dt

            positions.append(pos)
            controls.append(u)
            errors.append(controller.target_position - pos)

        print(f"Final position: {pos:.4f}, target: {controller.target_position}")

        plt.figure(figsize=(8, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, positions, label="position")
        plt.axhline(controller.target_position, color="k", linestyle=":", label="target")
        plt.ylabel("Position")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, errors, label="error", color="tab:orange")
        plt.ylabel("Error")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, controls, label="control (u)", color="tab:green")
        plt.xlabel("Time [s]")
        plt.ylabel("Control")
        plt.legend()

        plt.tight_layout()
        plt.show()
