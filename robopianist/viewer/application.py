# Copyright 2018 The dm_control Authors.
# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Viewer application module."""

import collections

from dm_control import _render

from robopianist import SF2_PATH
from robopianist.viewer import (
    figures,
    gui,
    renderer,
    runtime,
    user_input,
    util,
    viewer,
    views,
)

_PAUSE = user_input.KEY_SPACE
_RESTART = user_input.KEY_BACKSPACE
_ADVANCE_SIMULATION = user_input.KEY_RIGHT
_SPEED_UP_TIME = user_input.KEY_EQUAL
_SLOW_DOWN_TIME = user_input.KEY_MINUS
_HELP = user_input.KEY_F1
_STATUS = user_input.KEY_F2
_MIDI_STATUS = user_input.KEY_F3
_REWARD = user_input.KEY_F4
_MUTE_AUDIO = user_input.KEY_V

_MAX_FRONTBUFFER_SIZE = 2048
_MISSING_STATUS_ENTRY = "--"
_RUNTIME_STOPPED_LABEL = "EPISODE TERMINATED - hit backspace to restart"
_STATUS_LABEL = "Status"
_TIME_LABEL = "Time"
_CPU_LABEL = "CPU"
_FPS_LABEL = "FPS"
_CAMERA_LABEL = "Camera"
_PAUSED_LABEL = "Paused"
_MIDI_TITLE_LABEL = "MIDI"
_MIDI_N_NOTES_LABEL = "Notes"
_ERROR_LABEL = "Error"
_MUTED_LABEL = "Muted"
_DOFS_LABEL = "DoFs"
_CTRLS_LABEL = "Controls"
_SOUNDFONT_LABEL = "Soundfont"


class Help(views.ColumnTextModel):
    """Contains the description of input map employed in the application."""

    def __init__(self):
        """Instance initializer."""
        self._value = [
            ["Help", "F1"],
            ["Info", "F2"],
            ["Reward", "F3"],
            ["Stereo", "F5"],
            ["Frame", "F6"],
            ["Label", "F7"],
            ["--------------", ""],
            ["Pause", "Space"],
            ["Reset", "BackSpace"],
            ["Autoscale", "Ctrl A"],
            ["Geoms", "0 - 4"],
            ["Sites", "Shift 0 - 4"],
            ["Speed Up", "="],
            ["Slow Down", "-"],
            ["Switch Cam", "[ ]"],
            ["--------------", ""],
            ["Translate", "R drag"],
            ["Rotate", "L drag"],
            ["Zoom", "Scroll"],
            ["Select", "L dblclick"],
            ["Center", "R dblclick"],
            ["Track", "Ctrl R dblclick / Esc"],
            ["Perturb", "Ctrl [Shift] L/R drag"],
            ["--------------", ""],
            ["Mute", "V"],
        ]

    def get_columns(self):
        """Returns the text to display in two columns."""
        return self._value


class Status(views.ColumnTextModel):
    """Monitors and returns the status of the application."""

    def __init__(self, time_multiplier, pause, frame_timer):
        """Instance initializer.

        Args:
          time_multiplier: Instance of util.TimeMultiplier.
          pause: An observable pause subject, instance of util.ObservableFlag.
          frame_timer: A Timer instance counting duration of frames.
        """
        self._runtime = None
        self._time_multiplier = time_multiplier
        self._camera = None
        self._pause = pause
        self._frame_timer = frame_timer
        self._fps_counter = util.Integrator()
        self._cpu_counter = util.Integrator()

        self._value = collections.OrderedDict(
            [
                (_STATUS_LABEL, _MISSING_STATUS_ENTRY),
                (_TIME_LABEL, _MISSING_STATUS_ENTRY),
                (_CPU_LABEL, _MISSING_STATUS_ENTRY),
                (_FPS_LABEL, _MISSING_STATUS_ENTRY),
                (_CAMERA_LABEL, _MISSING_STATUS_ENTRY),
                (_PAUSED_LABEL, _MISSING_STATUS_ENTRY),
                (_ERROR_LABEL, _MISSING_STATUS_ENTRY),
                (_DOFS_LABEL, _MISSING_STATUS_ENTRY),
                (_CTRLS_LABEL, _MISSING_STATUS_ENTRY),
            ]
        )

    def set_camera(self, camera):
        """Updates the active camera instance.

        Args:
          camera: Instance of SceneCamera.
        """
        self._camera = camera

    def set_runtime(self, instance):
        """Updates the active runtime instance.

        Args:
          instance: Instance of runtime.Base.
        """
        if self._runtime:
            self._runtime.on_error -= self._on_error
            self._runtime.on_episode_begin -= self._clear_error
        self._runtime = instance
        if self._runtime:
            self._runtime.on_error += self._on_error
            self._runtime.on_episode_begin += self._clear_error

    def get_columns(self):
        """Returns the text to display in two columns."""
        if self._frame_timer.measured_time > 0:
            self._fps_counter.value = 1.0 / self._frame_timer.measured_time
        self._value[_FPS_LABEL] = "{0:.1f}".format(self._fps_counter.value)

        if self._runtime:
            if self._runtime.state == runtime.State.STOPPED:
                self._value[_STATUS_LABEL] = _RUNTIME_STOPPED_LABEL
            else:
                self._value[_STATUS_LABEL] = str(self._runtime.state)

            self._cpu_counter.value = self._runtime.simulation_time

            self._value[_TIME_LABEL] = "{0:.2f} ({1}x)".format(
                self._runtime.get_time(), str(self._time_multiplier)
            )
            self._value[_CPU_LABEL] = "{0:.2f} ms".format(
                self._cpu_counter.value * 1000.0
            )

            action_spec = self._runtime.environment.action_spec()
            self._value[_CTRLS_LABEL] = str(action_spec.shape[-1])
            dofs = self._runtime.environment.physics.model.nv
            self._value[_DOFS_LABEL] = str(dofs)
        else:
            self._value[_STATUS_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_TIME_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_CPU_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_CTRLS_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_DOFS_LABEL] = _MISSING_STATUS_ENTRY

        if self._camera:
            self._value[_CAMERA_LABEL] = self._camera.name
        else:
            self._value[_CAMERA_LABEL] = _MISSING_STATUS_ENTRY

        self._value[_PAUSED_LABEL] = str(self._pause.value)

        return list(self._value.items())  # For Python 2/3 compatibility.

    def _clear_error(self):
        self._value[_ERROR_LABEL] = _MISSING_STATUS_ENTRY

    def _on_error(self, error_msg):
        self._value[_ERROR_LABEL] = error_msg


class MidiStatus(views.ColumnTextModel):
    """MIDI-related status information."""

    def __init__(self, mute) -> None:
        """Instance initializer.

        Args:
          pause: An observable pause subject, instance of util.ObservableFlag.
        """
        self._runtime = None
        self._mute = mute

        self._value = collections.OrderedDict(
            [
                (_MIDI_TITLE_LABEL, _MISSING_STATUS_ENTRY),
                (_MIDI_N_NOTES_LABEL, _MISSING_STATUS_ENTRY),
                (_MUTED_LABEL, _MISSING_STATUS_ENTRY),
                (_SOUNDFONT_LABEL, _MISSING_STATUS_ENTRY),
            ]
        )

    def set_runtime(self, instance):
        """Updates the active runtime instance.

        Args:
          instance: Instance of runtime.Base.
        """
        self._runtime = instance

    def get_columns(self):
        """Returns the text to display in two columns."""
        if self._runtime:
            self._value[_MIDI_TITLE_LABEL] = (
                    self._runtime.environment.task.midi.title or _MISSING_STATUS_ENTRY
            )
            self._value[_MIDI_N_NOTES_LABEL] = str(
                self._runtime.environment.task.midi.n_notes
            )
            self._value[_SOUNDFONT_LABEL] = SF2_PATH.stem
            self._value[_MUTED_LABEL] = str(self._mute.value)
        return list(self._value.items())


class ReloadParams(collections.namedtuple("RefreshParams", ["zoom_to_scene"])):
    """Parameters of a reload request."""


class Application:
    """Viewer application."""

    def __init__(self, title="Explorer", width=1024, height=768):
        """Instance initializer."""
        self._render_surface = None
        self._renderer = renderer.NullRenderer()
        self._viewport = renderer.Viewport(width, height)
        self._window = gui.RenderWindow(width, height, title)

        self._pause_subject = util.ObservableFlag(True)
        self._mute_subject = util.ObservableFlag(False)
        self._time_multiplier = util.TimeMultiplier(1.0)
        self._frame_timer = util.Timer()
        self._viewer = viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard
        )
        self._viewer_layout = views.ViewportLayout()
        self._status = Status(
            self._time_multiplier,
            self._pause_subject,
            self._frame_timer,
        )
        self._midi_status = MidiStatus(self._mute_subject)

        self._runtime = None
        self._environment_loader = None
        self._environment = None
        self._policy = None
        self._deferred_reload_request = None

        status_view_toggle = self._build_view_toggle(
            views.ColumnTextView(self._status), views.PanelLocation.BOTTOM_LEFT
        )
        midi_status_view_toggle = self._build_view_toggle(
            views.ColumnTextView(self._midi_status), views.PanelLocation.TOP_LEFT
        )
        help_view_toggle = self._build_view_toggle(
            views.ColumnTextView(Help()), views.PanelLocation.TOP_RIGHT
        )
        status_view_toggle()
        midi_status_view_toggle()

        self._reward = figures.RewardTermsFigure(self._pause_subject)
        reward_view_toggle = self._build_view_toggle(
            views.MujocoFigureView(self._reward), views.PanelLocation.TOP_RIGHT
        )
        reward_view_toggle()

        self._input_map = user_input.InputMap(self._window.mouse, self._window.keyboard)
        self._input_map.bind(self._pause_subject.toggle, _PAUSE)
        self._input_map.bind(self._time_multiplier.increase, _SPEED_UP_TIME)
        self._input_map.bind(self._time_multiplier.decrease, _SLOW_DOWN_TIME)
        self._input_map.bind(self._advance_simulation, _ADVANCE_SIMULATION)
        self._input_map.bind(self._restart_runtime, _RESTART)
        self._input_map.bind(help_view_toggle, _HELP)
        self._input_map.bind(status_view_toggle, _STATUS)
        self._input_map.bind(midi_status_view_toggle, _MIDI_STATUS)
        self._input_map.bind(reward_view_toggle, _REWARD)
        self._input_map.bind(self._mute_subject.toggle, _MUTE_AUDIO)
        self._input_map.bind(lambda: self._move_cube((0, 0, 1)), user_input.KEY_KP_9)
        self._input_map.bind(lambda: self._move_cube((0, 0, -1)), user_input.KEY_KP_7)
        self._input_map.bind(lambda: self._move_cube((0, -1, 0)), user_input.KEY_KP_4)
        self._input_map.bind(lambda: self._move_cube((0, 1, 0)), user_input.KEY_KP_6)
        self._input_map.bind(lambda: self._move_cube((-1, 0, 0)), user_input.KEY_KP_8)
        self._input_map.bind(lambda: self._move_cube((1, 0, 0)), user_input.KEY_KP_2)

    def _move_cube(self, direction: tuple):
        self._environment.task.move_cube(direction)

    def _estimate_suitable_font_scale(self) -> int:
        """Estimates a suitable font scale for the current window properties.

        Returns:
          An integer font scale.
        """
        b2w = self._window.shape[0] / self._window.window_shape[0]
        ppi = b2w * self._window.pixels_per_inch
        if self._window.shape[0] > self._window.window_shape[0]:
            fs = int(round(b2w * 100))
        elif ppi > 50 and ppi < 350:
            fs = int(round(ppi))
        else:
            fs = 150
        fs = int(round(fs * 0.02) * 50)
        return min(max(fs, 100), 300)

    def _on_reload(self, zoom_to_scene=False):
        """Perform initialization related to Physics reload.

        Reset the components that depend on a specific Physics class instance.

        Args:
          zoom_to_scene: Should the camera zoom to show the entire scene after the
            reload is complete.
        """
        self._deferred_reload_request = ReloadParams(zoom_to_scene)
        self._viewer.deinitialize()
        self._status.set_camera(None)

    def _perform_deferred_reload(self, params):
        """Performs the deferred part of initialization related to Physics reload.

        Args:
          params: Deferred reload parameters, an instance of ReloadParams.
        """
        if self._render_surface:
            self._render_surface.free()
        if self._renderer:
            self._renderer.release()
        self._render_surface = _render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE
        )
        self._renderer = renderer.OffScreenRenderer(
            model=self._environment.physics.model,
            surface=self._render_surface,
            font_scale=self._estimate_suitable_font_scale(),
        )
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self._environment.physics, self._renderer, touchpad=False
        )
        self._status.set_camera(self._viewer.camera)
        if params.zoom_to_scene:
            self._viewer.zoom_to_scene()

    def _build_view_toggle(self, view, location):
        def toggle():
            if view in self._viewer_layout:
                self._viewer_layout.remove(view)
            else:
                self._viewer_layout.add(view, location)

        return toggle

    def _tick(self):
        """Handle GUI events until the main window is closed."""
        if self._deferred_reload_request:
            self._perform_deferred_reload(self._deferred_reload_request)
            self._deferred_reload_request = None
        time_elapsed = self._frame_timer.tick() * self._time_multiplier.get()
        if self._runtime:
            with self._viewer.perturbation.apply(self._pause_subject.value):
                self._runtime.tick(
                    time_elapsed, self._pause_subject.value, self._mute_subject.value
                )
        self._viewer.render()

    def _load_environment(self, zoom_to_scene):
        """Loads a new environment."""
        if self._runtime:
            del self._runtime
            self._runtime = None
        self._environment = None
        environment_instance = None
        if self._environment_loader:
            environment_instance = self._environment_loader()
        if environment_instance:
            self._environment = environment_instance
            self._runtime = runtime.Runtime(
                environment=self._environment, policy=self._policy
            )
            self._runtime.on_physics_changed += lambda: self._on_reload(False)
        self._status.set_runtime(self._runtime)
        self._midi_status.set_runtime(self._runtime)
        self._reward.set_runtime(self._runtime)
        self._on_reload(zoom_to_scene=zoom_to_scene)

    def _restart_runtime(self):
        """Restarts the episode, resetting environment, model, and data."""
        if self._runtime:
            self._runtime.stop()
        self._load_environment(zoom_to_scene=False)

        if self._policy:
            if hasattr(self._policy, "reset"):
                self._policy.reset()

    def _advance_simulation(self):
        if self._runtime:
            self._runtime.single_step()

    def launch(self, environment_loader, policy=None):
        """Starts the viewer with the specified policy and environment.

        Args:
          environment_loader: Either a callable that takes no arguments and returns
            an instance of dm_control.rl.control.Environment, or an instance of
            dm_control.rl.control.Environment.
          policy: An optional callable corresponding to a policy to execute
            within the environment. It should accept a `TimeStep` and return
            a numpy array of actions conforming to the output of
            `environment.action_spec()`. If the callable implements a method `reset`
            then this method is called when the viewer is reset.

        Raises:
          ValueError: If `environment_loader` is None.
        """
        if environment_loader is None:
            raise ValueError('"environment_loader" argument is required.')
        if callable(environment_loader):
            self._environment_loader = environment_loader
        else:
            self._environment_loader = lambda: environment_loader
        self._policy = policy
        self._load_environment(zoom_to_scene=True)

        def tick():
            self._viewport.set_size(*self._window.shape)
            self._tick()
            return self._renderer.pixels

        self._window.event_loop(tick_func=tick)
        self._window.close()
