#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError, TickRuntimeError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from agents.tools.misc import get_speed

class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None

        self._statistics_manager = statistics_manager

        self.tick_count = 0
        self.next_state = None
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)
        self.prev_frame_infos = []
        self.run_step = 0
        self.episode_return = 0
        self.episode_returns = []
        self.n_episode = 0
        self.best_train = -np.inf
        
    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(self, scenario, agent, route_index, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()

        while self._running:
            self._tick_scenario()
            # if len(self.prev_frame_infos) > 1:
            #     prev_frame_info = self.prev_frame_infos.pop(0)
            #     state = prev_frame_info["state"]
            #     action = prev_frame_info["action"]
            #     reward = prev_frame_info["reward"]
            #     done = prev_frame_info["done"]
            #     if done:
            #         next_state = state
            #     else:
            #         next_state = self.prev_frame_infos[0]["state"]
            #     self._agent_wrapper._agent.memory.append(state, action, reward, next_state, done)
                
        # self.n_episode += 1
        
        # if self.n_episode > 20 and np.mean(self.episode_returns) >= self.best_train:
        #     self.best_train = np.mean(self.episode_returns)
        #     self._agent_wrapper._agent.save_models(os.path.join(self._agent_wrapper._agent.model_dir, 'final'))
            
        # self.writer.add_scalar('reward/train', self.episode_return, self.run_step) 
        
        # print(f'Episode: {self.n_episode:<4}  '
        #       f'Episode steps: {self.tick_count:<4}  '
        #       f'Return: {self.episode_return:<5.1f}')
        
        # self.episode_returns.append(self.episode_return) 
        # if len(self.episode_returns) > 20:
        #     self.episode_returns.pop(0)
        # self.episode_return = 0
        
    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self.run_step += 1
            self._watchdog.pause()

            if self.tick_count > 4000:
                raise TickRuntimeError("RuntimeError, tick_count > 4000")

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action, buffer_data = self._agent_wrapper()
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()
                
            # reward = self.get_reward()
            # self.episode_return += reward
            
            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                          carla.Rotation(pitch=-90)))

            # self.prev_frame_info = {
            #     "state": {buffer_data["prev_frame_info"],buffer_data["input"]},
            #     "action": buffer_data["output"],
            #     "reward": reward,
            #     "done": self._running,
            # }
            # self.prev_frame_infos.append(self.prev_frame_info)
            
            # if self.run_step < 5000:
            #     continue
            # self._agent_wrapper._agent.learn()
            
            # if self.run_step % self._agent_wrapper._agent.target_update_interval == 0:
            #     self._agent_wrapper._agent.update_target()
                   
    def get_reward(self):
        reward = 0
        if len(self._agent_wrapper._agent.sensor_interface.collision_hist):
            self._running = False
            self._agent_wrapper._agent.sensor_interface.collision_hist = []
            reward -= 1
            
        if len(self._agent_wrapper._agent.sensor_interface.lane_invasion_hist):
            self._running = False
            self._agent_wrapper._agent.sensor_interface.lane_invasion_hist = []
            reward -= 1
        speed = get_speed(self.ego_vehicles[0])
        reward += speed * 0.001
        return reward

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        self._scenario_thread.join()
        self._scenario_thread = None

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
