import os
import numpy as np
import mujoco_py
import pickle
from promp import DeterministicProMP


class BallInACupCostFunction:
    count = 0
    rew = []
    suc = []
    def __init__(self, pid):
        self.pid = pid
        self.xml_raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml", "ball-in-a-cup-raw.xml")
        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml",
                                     "ball-in-a-cup" + str(pid) + ".xml")
        self.p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])
        self.max_ctrl = np.array([150., 125., 40., 60., 5., 5., 2.])
        self.min_ctrl = -self.max_ctrl
        self.sparse = False

        self.collision_objects = ["cup_geom1", "cup_geom2", "wrist_palm_link_convex_geom",
                                  "wrist_pitch_link_convex_decomposition_p1_geom",
                                  "wrist_pitch_link_convex_decomposition_p2_geom",
                                  "wrist_pitch_link_convex_decomposition_p3_geom",
                                  "wrist_yaw_link_convex_decomposition_p1_geom",
                                  "wrist_yaw_link_convex_decomposition_p2_geom",
                                  "forearm_link_convex_decomposition_p1_geom",
                                  "forearm_link_convex_decomposition_p2_geom"]

    def __call__(self, xs, render=False, save=False, returnsuc=False):
        xs = np.sqrt(0.02) * np.copy(xs)
        min_dist, action_costs, success, error = self._run_experiment(np.array([1.]), xs, render=render)

        if error:
            if save:
                os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "errors"),
                            exist_ok=True)
                log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "errors")
                exp_number = 0
                for path in os.listdir(log_dir):
                    if path.startswith("error-"):
                        num = int(path[6:].split(".")[0])
                        if num > exp_number:
                            exp_number = num

                new_path = os.path.join(log_dir, "error-" + str(exp_number + 1) + ".pkl")
                with open(new_path, "wb") as f:
                    pickle.dump(xs, f)

                print("Error in Experiment - wrote to errors-" + str(exp_number + 1) + ".pkl")

            # incrementing the global counter and storing the reward and success rate
            self.rew.append(-0.1)
            self.suc.append(success)
            self.count += 1

            # we return the success only if returnsuc is true
            if returnsuc:
                return -0.1, success
            else:
                return -0.1

        if self.sparse:
            reward = success - success * 7e-2 * np.sum(np.square(xs))
        else:
            reward = np.exp(-2 * min_dist) - 1e-4 * action_costs

        # incrementing the global counter and storing the reward and success rate
        self.rew.append(-reward)
        self.suc.append(success)
        self.count += 1

        # we return the success only if returnsuc is true
        # returning the negative of the reward as we want to maximize, and bicfun minimizes
        if returnsuc:
            return -reward, success
        else:
            return -reward

    def retcount(self):
        return self.count

    def retreward(self):
        return self.rew

    def newrew(self):
        self.rew.clear()

    def retsuc(self):
        return self.suc

    def newsuc(self):
        self.suc.clear()

    def set_seed(self, seed):
        print("Process " + str(self.pid) + " - Setting seed " + str(seed))
        np.random.seed(seed)
        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml",
                                     "ball-in-a-cup" + str(seed) + ".xml")

    @staticmethod
    def _check_collision(sim, ball_id, collision_ids):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 in collision_ids and con.geom2 == ball_id
            collision_trans = con.geom1 == ball_id and con.geom2 in collision_ids

            if collision or collision_trans:
                return True
        return False

    def _run_experiment(self, context, theta, render=False):
        # Create the simulator (we do that because it is easier to modify the XML and reloading takes only a fraction
        # of the simulation time)
        with open(self.xml_raw_path, "r") as f:
            raw_xml = f.read()
            scale = context[0]
            raw_xml = raw_xml.replace("[mesh_mag]", str(scale * 0.001))
            raw_xml = raw_xml.replace("[mesh_pos]", str(0.055 - (scale - 1.) * 0.023))
            raw_xml = raw_xml.replace("[goal_off]", str(0.1165 + (scale - 1.) * 0.0385))  # was 0.115
            raw_xml = raw_xml.replace("[base_scale]", str(scale * 0.038))
            with open(self.xml_path, "w") as f1:
                f1.write(raw_xml)

        sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(self.xml_path), nsubsteps=4)
        init_pos = sim.data.qpos.copy()
        init_vel = np.zeros_like(init_pos)

        ball_id = sim.model._body_name2id["ball"]
        ball_collision_id = sim.model._geom_name2id["ball_geom"]
        goal_id = sim.model._site_name2id["cup_goal"]
        goal_final_id = sim.model._site_name2id["cup_goal_final"]
        collision_ids = [sim.model._geom_name2id[name] for name in self.collision_objects]

        start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943,
                              -1.57])

        weights = np.reshape(theta, (-1, 3))
        n_steps = weights.shape[0]
        pmp = DeterministicProMP(n_basis=n_steps + 4, width=0.0035, off=0.01)
        weights = np.concatenate((np.zeros((n_steps, 1)), weights[:, 0][:, None], np.zeros((n_steps, 1)),
                                  weights[:, 1][:, None], np.zeros((n_steps, 1)), weights[:, 2][:, None],
                                  np.zeros((n_steps, 1))), axis=1)
        pmp.set_weights(3.5, np.concatenate((np.zeros((2, 7)), weights, np.zeros((2, 7))), axis=0))
        des_pos, des_vel = pmp.compute_trajectory(500, 1.)[1:3]
        des_pos += start_pos[None, :]

        # Reset the system
        sim.data.qpos[:] = init_pos
        sim.data.qvel[:] = init_vel
        sim.data.qpos[0:7] = start_pos

        sim.step()

        sim.data.qpos[:] = init_pos
        sim.data.qvel[:] = init_vel
        sim.data.qpos[0:7] = start_pos
        sim.data.body_xpos[ball_id, :] = np.copy(sim.data.site_xpos[goal_final_id, :]) - np.array([0., 0., 0.329])

        # Stabilize the system around the initial position
        for i in range(0, 500):
            sim.data.qpos[7:] = 0.
            sim.data.qvel[7:] = 0.
            sim.data.qpos[7] = -0.2
            cur_pos = sim.data.qpos[0:7].copy()
            cur_vel = sim.data.qvel[0:7].copy()
            trq = self.p_gains * (start_pos - cur_pos) + self.d_gains * (np.zeros_like(start_pos) - cur_vel)
            sim.data.qfrc_applied[0:7] = trq
            sim.step()

        for i in range(0, 500):
            cur_pos = sim.data.qpos[0:7].copy()
            cur_vel = sim.data.qvel[0:7].copy()
            trq = self.p_gains * (start_pos - cur_pos) + self.d_gains * (np.zeros_like(start_pos) - cur_vel)
            sim.data.qfrc_applied[0:7] = trq
            sim.step()

        if render:
            viewer = mujoco_py.MjViewer(sim)
        else:
            viewer = None

        dists = []
        dists_final = []
        k = 0
        torques = []
        error = False
        while k < des_pos.shape[0] + 350:
            # Compute the current distance from the ball to the inner part of the cup
            goal_pos = sim.data.site_xpos[goal_id]
            ball_pos = sim.data.body_xpos[ball_id]
            goal_final_pos = sim.data.site_xpos[goal_final_id]
            dists.append(np.linalg.norm(goal_pos - ball_pos))
            dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))

            # Compute the controls
            cur_pos = sim.data.qpos[0:7].copy()
            cur_vel = sim.data.qvel[0:7].copy()
            k_actual = np.minimum(des_pos.shape[0] - 1, k)
            trq = self.p_gains * (des_pos[k_actual, :] - cur_pos) + self.d_gains * (des_vel[k_actual, :] - cur_vel)
            torques.append(trq)

            # Advance the simulation
            sim.data.qfrc_applied[0:7] = trq
            try:
                sim.step()
            except mujoco_py.builder.MujocoException as e:
                print("Error in simulation: " + str(e))
                error = True
                # Copy the current torque as if it would have been applied until the end of the trajectory
                for i in range(k + 1, des_pos.shape[0] + 350):
                    torques.append(trq)
                break

            k += 1

            # Check for a collision - in which case we end the simulation
            if BallInACupCostFunction._check_collision(sim, ball_collision_id, collision_ids):
                # Copy the current torque as if it would have been applied until the end of the trajectory
                for i in range(k + 1, des_pos.shape[0]):
                    torques.append(trq)
                break

            if viewer is not None:
                viewer.render()

        # Remove the file after the simulation
        os.remove(self.xml_path)

        min_dist = np.min(dists)
        return 0.5 * min_dist + 0.5 * dists_final[-1], np.mean(np.sum(np.square(torques), axis=1), axis=0), \
               1. if dists_final[-1] < 0.05 * scale else 0., error