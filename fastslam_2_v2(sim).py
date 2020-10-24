# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 03:01:35 2020

@author: murta
"""
import matplotlib.pyplot as plt
import numpy as np
import math
dist_FOV_min = 0
dist_FOV_max =5
dist_FOV_angle = np.pi/2


Q = np.diag([3.0, np.deg2rad(10.0)])**2  # Qt variance due to measurement
R = np.diag([1.0, np.deg2rad(20.0)])**2


cam_displacement = 0

N_PARTICLE = 1  # number of particle
minimum_correspondence_likelihood = 0.01
NTH = N_PARTICLE / 1.5
DT = 0.1  # time tick [s]


class particle:

    def __init__(self):
        self.pose = np.zeros((3, 1))  # particle pose (x, y, yaw)
        self.w = 1.0 / N_PARTICLE   # particle weight
        self.lm = np.zeros((0, 2))  # landmark positions
        self.lm_cvar = np.zeros((0, 2, 2))  # landmark covarience
        self.lm_counter = np.zeros((0, 1))  # [] # landmark Confidence
        self.P = np.eye(3)

    def number_of_landmarks(self):
        """Utility: return current number of landmarks in this particle."""
        return len(self.lm)

    def h(self, landmark):
        """Measurement function. Takes a (x, y, theta) state and a (x, y)
           landmark, and returns the corresponding (range, bearing)."""
        dx = landmark[0,0] - (self.pose[0, 0] +
                            cam_displacement * math.cos(self.pose[2, 0]))
        dy = landmark[1,0] - (self.pose[1, 0] +
                            cam_displacement * math.sin(self.pose[2, 0]))
        r = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx) - self.pose[2, 0]
        alpha = pi_2_pi(angle)
        return np.array([r, alpha])

    def h_expected_measurement_for_landmark(self, landmark_number):
        """Returns the expected distance and bearing measurement for a given
           landmark number and the pose of this particle."""
        landmark = np.array(self.lm[landmark_number, :]).reshape(2, 1)
        expected = self.h(landmark)
        return expected  
    
    def compute_jacobians(self, xf, Pf, Q_cov):
        dx = xf[0, 0] - self.pose[0, 0]
        dy = xf[1, 0] - self.pose[1, 0]
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        zp = np.array(
            [d, pi_2_pi(math.atan2(dy, dx) - self.pose[2, 0])]).reshape(2, 1)

        Hv = np.array([[-dx / d, -dy / d, 0.0],
                       [dy / d2, -dx / d2, -1.0]])

        Hf = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])

        Sf = Hf @ Pf @ Hf.T + Q_cov

        return zp, Hv, Hf, Sf


    def compute_weight(self, z, landmark_number, Q_cov):
        lm_id = landmark_number
        xf = np.array(self.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(self.lm_cvar[lm_id, :])
        zp, Hv, Hf, Sf = self.compute_jacobians(xf, Pf, Q_cov)

        dx = z.reshape(2, 1) - zp
        dx[1, 0] = pi_2_pi(dx[1, 0])

        try:
            invS = np.linalg.inv(Sf)
        except np.linalg.linalg.LinAlgError:
            print("singular")
            return 1.0

        num = math.exp(-0.5 * dx.T @ invS @ dx)
        den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))

        w = num / den

        return w
    
    def proposal_sampling(self, z, landmark_number, Q_cov):
        lm_id = landmark_number
        xf = self.lm[lm_id, :].reshape(2, 1)
        Pf = self.lm_cvar[lm_id, :]
        # State
        x = self.pose
        P = self.P
        zp, Hv, Hf, Sf = self.compute_jacobians(xf, Pf, Q_cov)
    
        Sfi = np.linalg.inv(Sf)
        dz = z[0:2].reshape(2, 1) - zp
        dz[1] = pi_2_pi(dz[1])
    
        Pi = np.linalg.inv(P)
    
        particle.P = np.linalg.inv(Hv.T @ Sfi @ Hv + Pi)  # proposal covariance
        x += particle.P @ Hv.T @ Sfi @ dz  # proposal mean
    
        self.pose = x
        
        return self


    def compute_correspondence_likelihoods(self, measurement,
                                           number_of_landmarks,
                                           Qt_measurement_covariance):
        """For a given measurement, returns a list of all correspondence
           likelihoods (from index 0 to number_of_landmarks-1)."""
        likelihoods = []
        for i in range(number_of_landmarks):
            likelihoods.append(
                self.compute_weight(measurement, i, Qt_measurement_covariance))
            
        return likelihoods


    def initialize_new_landmark(self, measurement,
                                Qt_measurement_covariance):
        """Given a (x, y) measurement in the scanner's system, initializes a
           new landmark and its covariance."""
        s = np.sin(pi_2_pi(self.pose[2, 0] + measurement[1]))
        c = np.cos(pi_2_pi(self.pose[2, 0] + measurement[1]))
        temp = [self.pose[0, 0] + measurement[0] *
                c, self.pose[1, 0] + measurement[0] * s]

        self.lm = np.vstack((self.lm, temp))
        dx = measurement[0] * c
        dy = measurement[0] * s
        d2 = dx**2 + dy**2
        d = math.sqrt(d2)
        Gz = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])
        cvar = np.linalg.inv(
            Gz) @ Qt_measurement_covariance @ np.linalg.inv(Gz.T)  # Hinv @ Qt_measurement_covariance @ np.transpose(Hinv)

        # Replace this.
        self.lm_cvar = np.vstack((self.lm_cvar, cvar.reshape((1, 2, 2))))
        self.lm_counter = np.append(self.lm_counter, 1)
        
        return self
    
    def update_kf_with_cholesky(self, xf, Pf, v, Q_cov, Hf):
        PHt = Pf @ Hf.T
        S = Hf @ PHt + Q_cov
    
        S = (S + S.T) * 0.5
        s_chol = np.linalg.cholesky(S).T
        s_chol_inv = np.linalg.inv(s_chol)
        W1 = PHt @ s_chol_inv
        W = W1 @ s_chol_inv.T
    
        x = xf + W @ v
        P = Pf - W1 @ W1.T
    
        return x, P

    def update_landmark_slam(self, landmark_number, z, Q_cov):
        lm_id = landmark_number
        
        xf = np.array(self.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(self.lm_cvar[lm_id, :])
    
        zp, Hv, Hf, Sf = self.compute_jacobians(xf, Pf, Q_cov)
    
        dz = z.reshape(2, 1) - zp
        dz[1, 0] = pi_2_pi(dz[1, 0])
    
        xf, Pf = self.update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)
    
        self.lm_cvar[landmark_number, :] = Pf
        self.lm[landmark_number, :] = xf.T
        self.lm_counter[landmark_number] = self.lm_counter[landmark_number] + 2
    
        return self


    def update_particle(self, measurement):
        """Given a measurement, computes the likelihood that it belongs to any
           of the landmarks in the particle. If there are none, or if all
           likelihoods are below the minimum_correspondence_likelihood
           threshold, add a landmark to the particle. Otherwise, update the
           (existing) landmark with the largest likelihood."""
        likelihoods = self.compute_correspondence_likelihoods(measurement, self.number_of_landmarks(), Q)
        
        if (len(likelihoods) == 0) or (max(likelihoods) < minimum_correspondence_likelihood):
            self = self.initialize_new_landmark(measurement, Q)
            self.w *= minimum_correspondence_likelihood
        else:
            wj = np.max(likelihoods)
            index = np.argmax(likelihoods)
            self.w *= wj
            self = self.update_landmark_slam(index, measurement, Q)
            self = self.proposal_sampling(measurement, index, Q)

        self.decrement_visible_landmark_counters()
        self.remove_spurious_landmarks()
        return self
    

    def decrement_visible_landmark_counters(self):
        """Decrements the counter for every landmark which is potentially
           visible. This uses a simplified test: it is only checked if the
           bearing of the expected measurement is within the laser scanners
           range."""

        for i in range(self.number_of_landmarks()):
            r, alpha = self.h_expected_measurement_for_landmark(i)
            angle = (alpha < dist_FOV_angle/2)
            if(angle and r < dist_FOV_max):
                self.lm_counter[i] -= 1
        return  # Replace this.


    # Added: Removal of landmarks with negative counter.
    def remove_spurious_landmarks(self):
        """Remove all landmarks which have a counter less than zero."""
        
        remove = np.argwhere(self.lm_counter < 0)
        self.lm = np.delete(self.lm, remove, axis=0)
        self.lm_cvar = np.delete(self.lm_cvar, remove, axis=0)
        self.lm_counter = np.delete(self.lm_counter, remove)
        
        return  # Replace this.


"""
    FAST SLAM IMPLEMENTATION HERE
"""


def slam(particles, state, lm):
    
    particles = predict_particles(particles, state)
    particles = update_with_observation(particles, lm)
    particles = resampling(particles)
    
    return particles


def motion_model(x, u):
    """
    Compute predictions for a particle
    :param x: The state vector [x, y, theta]
    :param u: The input vector [linear vel Vt, angular vel Wt]
    :return: Returns predicted state vector x
    """
    # A 3x3 identity matrix
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # A 3x2 matrix to calculate new x, y, yaw given controls
    B = np.array([[DT * np.cos(x[2, 0]), 0],
                  [DT * np.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = F @ x + B @ u  # Formula: X = FX + BU

    x[2, 0] = pi_2_pi(x[2, 0])  # Ensure Theta is under pi radians

    return x


def predict_particles(particles, u):
    """
    Predict x, y, yaw values for new particles using motion model
    :param particles: An array of particles
    :param u: An input vector [linear vel, angular vel]
    :return: Returns predictions as particles
    """
    #print('PREDICTING PARTICLES')

    for i in range(N_PARTICLE):
        ud = u + (np.random.randn(1, 2) @ R).T  # Add noise
        particles[i].pose = motion_model(particles[i].pose, ud)

    return particles


def update_with_observation(particles, landmark_list):
    """Updates all particles and returns a list of their weights."""
    for p in particles:
        for i in range(np.shape(landmark_list)[1]):
            p.update_particle(landmark_list[:, i])

    return particles


# CODE SNIPPET #
def normalize_weight(particles):
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def resampling(particles):
    particles = normalize_weight(particles)
    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(n_eff)

    if n_eff < NTH:  # resampling
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)

        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].pose[0, 0] = tmp_particles[inds[i]].pose[0, 0]
            particles[i].pose[1, 0] = tmp_particles[inds[i]].pose[1, 0]
            particles[i].pose[2, 0] = tmp_particles[inds[i]].pose[2, 0]
            particles[i].lm = tmp_particles[inds[i]].lm[:, :]
            particles[i].lm_cvar = tmp_particles[inds[i]].lm_cvar[:, :]
            particles[i].lm_counter = tmp_particles[inds[i]].lm_counter[:]
            particles[i].w = 1.0 / N_PARTICLE

    return particles


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


"""For Simulation"""


#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
show_animation = True


def calc_input(time):
    if time <= 3.0:  # wait at first
        v = 0.0
        yaw_rate = 0.0
    else:
        v = 1.0  # [m/s]
        yaw_rate = 0.1  # [rad/s]

    u = np.array([v, yaw_rate]).reshape(2, 1)

    return u


def observation(xTrue, xd, u, rfid):
    xTrue = motion_model(xTrue, u)
    z = np.zeros((3, 0))
    for i in range(len(rfid[:, 0])):

        dx = rfid[i, 0] - xTrue[0, 0]
        dy = rfid[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_with_noize = angle + np.random.randn() * Q_sim[
                1, 1] ** 0.5  # add noise
            zi = np.array([dn, pi_2_pi(angle_with_noize), i]).reshape(3, 1)
            z = np.hstack((z, zi))

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[
        1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def calc_final_state(particles):
    xEst = np.zeros((3, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].pose[0]
        xEst[1, 0] += particles[i].w * particles[i].pose[1]
        xEst[2, 0] += particles[i].w * particles[i].pose[2]

    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    #  print(xEst)

    return xEst


def convert_z(z):
    num = np.shape(z)[1]
    zd = []
    for i in range(num):
        temp = z[:, i]
        # print(p)
        zd.append((temp[0]*math.cos(temp[1]), temp[0]*math.sin(temp[1])))

    return zd


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [15.0, 15.0],
                     [10.0, 20.0],
                     [3.0, 15.0],
                     [-5.0, 20.0],
                     [-5.0, 5.0],
                     [-10.0, 15.0]
                     ])
    xTrue = np.zeros((3, 1))  # True state
    xDR = np.zeros((3, 1))  # Dead reckoning

    particles = [particle() for _ in range(N_PARTICLE)]

    while SIM_TIME >= time:
        time += DT
        u = calc_input(time)

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)
        particles = slam(particles, ud, z[0:2, :])

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event', lambda event:
                [exit(0) if event.key == 'escape' else None])
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            
            for iz in range(np.shape(z)[1]):
                landmark_id = int(z[2, iz])
                plt.plot([particles[0].pose[0], RFID[landmark_id, 0]], [
                    particles[0].pose[1], RFID[landmark_id, 1]], "-k")

            for i in range(N_PARTICLE):
                plt.plot(particles[i].pose[0], particles[i].pose[1], ".r")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
    print("End")
    for i in range(N_PARTICLE):
        print("Particle[", i, "]x: ", (particles[i].lm), "\n")


if __name__ == '__main__':
    main()
