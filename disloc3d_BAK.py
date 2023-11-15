import numpy as np

m = np.array([0.976395985616554,1000000,1000000,90,51.9712056605602,171.608134885649,133.483131008597,5,0,0])
coordinates = np.array([[-394.371649557575,1778.22315975419,0],[-550.366250807550,1398.45333220161,0],[-394.371649557575,1778.22315975419,0],[-550.366250807550,1398.45333220161,0]])
shear_m = 30e9
poisson_ratio = 0.25

coordinates=coordinates.T

X = coordinates[0, :]
Y = coordinates[1, :]
Z = coordinates[2, :]

L = m[0]
W = m[1]
D = m[2]
angle = m[3]  # dip in degree


if D >= W * np.sin(np.radians(angle)) and np.array_equal(Z, -np.abs(Z)):
        strikeAngle = m[4]  # clockwise is positive
        Xc, Yc = m[5], m[6]
        slip_str, slip_dip, tensile = m[7], m[8], m[9]
        Gshear, nu = shear_m, poisson_ratio

        # unit transformation and constants
        delta = np.deg2rad(angle)
        angle_Str = -np.deg2rad(strikeAngle)

        if nu == 0.5:
                nu = 0.4999
                youngs = 2 * Gshear * (1 + nu)
                lambda_ = nu * youngs / ((1 + nu) * (1 - 2 * nu))
        else:
                youngs = 2 * Gshear * (1 + nu)
                lambda_ = nu * youngs / ((1 + nu) * (1 - 2 * nu))

        mu = Gshear
        c = D

        # coordinate transformation
        x = -np.sin(angle_Str) * (X - Xc) + np.cos(angle_Str) * (Y - Yc) + 1/2 * L
        y = -np.cos(angle_Str) * (X - Xc) - np.sin(angle_Str) * (Y - Yc)
        z = Z

        # integrating
        d = c - z
        p = y * np.cos(delta) + d * np.sin(delta)

        xi = np.array([x, x, x - L, x - L])
        eta = np.array([p, p - W, p, p - W])

        q = np.ones((4,x.size)) * y * np.sin(delta) - np.ones((4,x.size)) * d * np.cos(delta)

        alpha = (lambda_ + mu) / (lambda_ + 2 * mu)

        Rsquare = xi**2 + eta**2 + q**2
        R = np.sqrt(Rsquare)

        y_ = eta * np.cos(delta) + q * np.sin(delta)
        d_ = eta * np.sin(delta) - q * np.cos(delta)
        c_ = d_ + np.ones((4,x.size)) * z

        # For displacement
        X11 = 1 / (R * (R + xi))
        X32 = (2 * R + xi) / (R**3 * (R + xi)**2)
        X53 = (8 * R**2 + 9 * R * xi + 3 * xi**2) / (R**5 * (R + xi)**3)

        Y11 = 1 / (R * (R + eta))
        Y32 = (2 * R + eta) / (R**3 * (R + eta)**2)
        Y53 = (8 * R**2 + 9 * R * eta + 3 * eta**2) / (R**5 * (R + eta)**3)

        h = q * np.cos(delta) - np.ones((4,x.size)) * z
        Z32 = np.sin(delta) / R**3 - h * Y32
        Z53 = 3 * np.sin(delta) / R**5 - h * Y53

        Y0 = Y11 - xi**2 * Y32
        Z0 = Z32 - xi**2 * Z53

        # Selecting a right root for theta
        qsign = np.sign(q)
        theta = np.arctan2(xi * eta, np.abs(q) * R)
        theta = qsign * theta

        X = np.sqrt(xi**2 + q**2)

        if np.abs(np.cos(delta)) < 0.000001:
                I3 = 1/2 * (eta / (R + d_) + y_ * q / ((R + d_)**2) - np.log(R + eta))
                I4 = 1/2 * (xi * y_ / ((R + d_)**2))
        else:
                I3 = 1/np.cos(delta) * y_ / (R + d_) - 1 / np.cos(delta)**2 * (
                        np.log(R + eta) - np.sin(delta) * np.log(R + d_))
                I4 = np.sin(delta) / np.cos(delta) * xi / (R + d_) + 2 / (np.cos(delta)**2) * np.arctan2(
                        eta * (X + q * np.cos(delta)) + X * (R + X) * np.sin(delta),
                        xi * (R + X) * np.cos(delta))

        I1 = -(xi / (R + d_)) * np.cos(delta) - I4 * np.sin(delta)
        I2 = np.log(R + d_) + I3 * np.sin(delta)




        ############ Block 2!!!

        D11 = 1 / (R * (R + d_))

        if np.abs(np.cos(delta)) < 0.000001:
                K1 = (xi * q) / (R + d_) * D11
                K3 = np.sin(delta) / (R + d_) * (xi**2 * D11 - 1)
        else:
                K1 = xi / np.cos(delta) * (D11 - Y11 * np.sin(delta))
                K3 = 1 / np.cos(delta) * (q * Y11 - y_ * D11)

        K2 = 1 / R + K3 * np.sin(delta)
        K4 = xi * Y11 * np.cos(delta) - K1 * np.sin(delta)

        J5 = -(d_ + y_**2 / (R + d_)) * D11
        J2 = xi * y_ / (R + d_) * D11

        if np.abs(np.cos(delta)) < 0.000001:
                J6 = -y_ / (R + d_)**2 * (xi**2 * D11 - 1/2)
                J3 = -xi / (R + d_)**2 * (q**2 * D11 - 1/2)
        else:
                J6 = 1 / np.cos(delta) * (K3 - J5 * np.sin(delta))
                J3 = 1 / np.cos(delta) * (K1 - J2 * np.sin(delta))

        J1 = J5 * np.cos(delta) - J6 * np.sin(delta)
        J4 = -xi * Y11 - J2 * np.cos(delta) + J3 * np.sin(delta)

        # ki
        E = np.sin(delta) / R - y_ * q / R**3
        F = d_ / R**3 + xi**2 * Y32 * np.sin(delta)
        G = 2 * X11 * np.sin(delta) - y_ * q * X32
        H = d_ * q * X32 + xi * q * Y32 * np.sin(delta)
        P = np.cos(delta) / R**3 + q * Y32 * np.sin(delta)
        Q = 3 * c_ * d_ / R**5 - (np.ones((4, 1)) * z * Y32 + Z32 + Z0) * np.sin(delta)

        # li
        E_ = np.cos(delta) / R + d_ * q / R**3
        F_ = y_ / R**3 + xi**2 * Y32 * np.cos(delta)
        G_ = 2 * X11 * np.cos(delta) + d_ * q * X32
        H_ = y_ * q * X32 + xi * q * Y32 * np.cos(delta)
        P_ = np.sin(delta) / R**3 - q * Y32 * np.cos(delta)
        Q_ = (3 * c_ * y_) / R**5 + q * Y32 - (np.ones((4, 1)) * z * Y32 + Z32 + Z0) * np.cos(delta)


        if slip_str != 0:
                #%%%%% strike-slip
                #%% displacement
                #uA
                Su1A = theta/2 + alpha / 2 * xi * q * Y11
                Su2A = alpha / 2 * q / R
                Su3A = (1 - alpha) / 2 * np.log(R + eta) - alpha / 2 * q**2 * Y11

                #uB
                Su1B = -xi * q * Y11 - theta - (1 - alpha) / alpha * I1 * np.sin(delta)
                Su2B = -q / R + (1 - alpha) / alpha * y_ / (R + d_) * np.sin(delta)
                Su3B = q**2 * Y11 - (1 - alpha) / alpha * I2 * np.sin(delta)

                #uC
                Su1C = (1 - alpha) * xi * Y11 * np.cos(delta) - alpha * xi * q * Z32
                Su2C = (1 - alpha) * (np.cos(delta) / R + 2 * q * Y11 * np.sin(delta)) - alpha * c_ * q / R**3
                Su3C = (1 - alpha) * q * Y11 * np.cos(delta) - alpha * (c_ * eta / R**3 - np.ones((4, 1)) * z * Y11 + xi**2 * Z32)

                #%%%%% strike-slip
                #%% displacement gradient
                #jA
                Sj1A = -(1 - alpha) / 2 * q * Y11 - alpha / 2 * (xi**2) * q * Y32
                Sj2A = - alpha / 2 * xi * q / R**3
                Sj3A = (1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32

                #jB
                Sj1B = xi**2 * q * Y32 - (1 - alpha) / alpha * J1 * np.sin(delta)
                Sj2B = xi * q / R**3 - (1 - alpha) / alpha * J2 * np.sin(delta)
                Sj3B = -xi * q**2 * Y32 - (1 - alpha) / alpha * J3 * np.sin(delta)

                #jC
                Sj1C = (1 - alpha) * Y0 * np.cos(delta) - alpha * q * Z0
                Sj2C = -(1 - alpha) * xi * (np.cos(delta) / R**3 + 2 * q * Y32 * np.sin(delta)) + alpha * (
                                3 * c_ * xi * q) / R**5
                Sj3C = -(1 - alpha) * xi * q * Y32 * np.cos(delta) + alpha * xi * (
                                (3 * c_ * eta) / R**5 - np.ones((4, 1)) * z * Y32 - Z32 - Z0)

                #kA
                Sk1A = (1 - alpha) / 2 * xi * Y11 * np.sin(delta) + d_ / 2 * X11 + alpha / 2 * xi * F
                Sk2A = alpha / 2 * E
                Sk3A = (1 - alpha) / 2 * (
                                np.cos(delta) / R + q * Y11 * np.sin(delta)) - alpha / 2 * q * F

                #kB
                Sk1B = -xi * F - d_ * X11 + (1 - alpha) / alpha * (xi * Y11 + J4) * np.sin(delta)
                Sk2B = -E + (1 - alpha) / alpha * (1 / R + J5) * np.sin(delta)
                Sk3B = q * F - (1 - alpha) / alpha * (q * Y11 - J6) * np.sin(delta)

                #kC
                Sk1C = -(1 - alpha) * xi * P * np.cos(delta) - alpha * xi * Q
                Sk2C = 2 * (1 - alpha) * (d_ / R**3 - Y0 * np.sin(delta)) * np.sin(delta) - y_ / R**3 * np.cos(
                        delta) - alpha * ((c_ + d_) / R**3 * np.sin(delta) - eta / R**3 - 3 * c_ * y_ * q / R**5)
                Sk3C = -(1 - alpha) * q / R**3 + (y_ / R**3 - Y0 * np.cos(delta)) * np.sin(
                        delta) + alpha * ((c_ + d_) / R**3 * np.cos(delta) + 3 * c_ * d_ * q / R**5 - (
                                Y0 * np.cos(delta) + q * Z0) * np.sin(delta))

                #lA
                Sl1A = (1 - alpha) / 2 * xi * Y11 * np.cos(delta) + y_ / 2 * X11 + alpha / 2 * xi * F_
                Sl2A = alpha / 2 * E_
                Sl3A = -(1 - alpha) / 2 * (
                                np.sin(delta) / R - q * Y11 * np.cos(delta)) - alpha / 2 * q * F_

                #lB
                Sl1B = -xi * F_ - y_ * X11 + (1 - alpha) / alpha * K1 * np.sin(delta)
                Sl2B = -E_ + (1 - alpha) / alpha * y_ * D11 * np.sin(delta)
                Sl3B = q * F_ + (1 - alpha) / alpha * K2 * np.sin(delta)

                #lC
                Sl1C = (1 - alpha) * xi * P_ * np.cos(delta) - alpha * xi * Q_
                Sl2C = 2 * (1 - alpha) * (y_ / R**3 - Y0 * np.cos(delta)) * np.sin(delta) + d_ / R**3 * np.cos(
                        delta) - alpha * ((c_ + d_) / R**3 * np.cos(delta) + 3 * c_ * d_ * q / R**5)
                Sl3C = (y_ / R**3 - Y0 * np.cos(delta)) * np.cos(delta) - alpha * (
                                (c_ + d_) / R**3 * np.sin(delta) - 3 * c_ * y_ * q / R**5 - Y0 * np.sin(
                        delta)**2 + q * Z0 * np.cos(delta))

        if slip_dip != 0:
                # dip-slip
                # displacement
                ###uA
                Du1A = alpha / 2 * q / R
                Du2A = theta / 2 + alpha / 2 * eta * q * X11
                Du3A = (1 - alpha) / 2 * np.log(R + xi) - alpha / 2 * q**2 * X11

                ###
                #uB
                Du1B = -q / R + (1 - alpha) / alpha * I3 * np.sin(delta) * np.cos(delta)
                Du2B = -eta * q * X11 - theta - (1 - alpha) / alpha * xi / (R + d_) * np.sin(delta) * np.cos(delta)
                Du3B = q**2 * X11 + (1 - alpha) / alpha * I4 * np.sin(delta) * np.cos(delta)

                #uC
                Du1C = (1 - alpha) * np.cos(delta) / R - q * Y11 * np.sin(delta) - alpha * c_ * q / R**3
                Du2C = (1 - alpha) * y_ * X11 - alpha * c_ * eta * q * X32
                Du3C = -d_ * X11 - xi * Y11 * np.sin(delta) - alpha * c_ * (X11 - q**2 * X32)
                # displacement gradient
                # dip-slip
                #jA
                Dj1A = -alpha / 2 * xi * q / R**3
                Dj2A = -q / 2 * Y11 - alpha / 2 * eta * q / R**3
                Dj3A = (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3

                #jB
                Dj1B = xi * q / R**3 + (1 - alpha) / alpha * J4 * np.sin(delta) * np.cos(delta)
                Dj2B = eta * q / R**3 + q * Y11 + (1 - alpha) / alpha * J5 * np.sin(delta) * np.cos(delta)
                Dj3B = -q**2 / R**3 + (1 - alpha) / alpha * J6 * np.sin(delta) * np.cos(delta)

                #jC
                Dj1C = -(1 - alpha) * xi / R**3 * np.cos(delta) + xi * q * Y32 * np.sin(delta) + alpha * (
                                3 * c_ * xi * q / R**5)
                Dj2C = -(1 - alpha) * y_ / R**3 + alpha * 3 * c_ * eta * q / R**5
                Dj3C = d_ / R**3 - Y0 * np.sin(delta) + alpha * c_ / R**3 * (1 - 3 * q**2 / R**2)

                #kA
                Dk1A = alpha / 2 * E
                Dk2A = (1 - alpha) / 2 * d_ * X11 + xi / 2 * Y11 * np.sin(delta) + alpha / 2 * eta * G
                Dk3A = (1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G

                #kB
                Dk1B = -E + (1 - alpha) / alpha * J1 * np.sin(delta) * np.cos(delta)
                Dk2B = -eta * G - xi * Y11 * np.sin(delta) + (1 - alpha) / alpha * J2 * np.sin(delta) * np.cos(delta)
                Dk3B = q * G + (1 - alpha) / alpha * J3 * np.sin(delta) * np.cos(delta)

                #kC
                Dk1C = -(1 - alpha) * eta / R**3 + Y0 * np.sin(delta)**2 - alpha * (
                                (c_ + d_) / R**3 * np.sin(delta) - 3 * c_ * y_ * q / R**5)
                Dk2C = (1 - alpha) * (X11 - y_**2 * X32) - alpha * c_ * (
                                (d_ + 2 * q * np.cos(delta)) * X32 - y_ * eta * q * X53)
                Dk3C = xi * P * np.sin(delta) + y_ * d_ * X32 + alpha * c_ * (
                                (y_ + 2 * q * np.sin(delta)) * X32 - y_ * q**2 * X53)

                #lA
                Dl1A = alpha / 2 * E_
                Dl2A = (1 - alpha) / 2 * y_ * X11 + xi / 2 * Y11 * np.cos(delta) + alpha / 2 * eta * G_
                Dl3A = -(1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
                #lB
                Dl1B = -E_ - (1 - alpha) / alpha * K3 * np.sin(delta) * np.cos(delta)
                Dl2B = -eta * G_ - xi * Y11 * np.cos(delta) - (1 - alpha) / alpha * xi * D11 * np.sin(delta) * np.cos(delta)
                Dl3B = q * G_ - (1 - alpha) / alpha * K4 * np.sin(delta) * np.cos(delta)
                #lB
                Dl1C = -q / R**3 + Y0 * np.sin(delta) * np.cos(delta) - alpha * (
                                (c_ + d_) / R**3 * np.cos(delta) + 3 * c_ * d_ * q / R**5)
                Dl2C = (1 - alpha) * y_ * d_ * X32 - alpha * c_ * (
                                (y_ - 2 * q * np.sin(delta)) * X32 + d_ * eta * q * X53)
                Dl3C = -xi * P_ * np.sin(delta) + X11 - d_**2 * X32 - alpha * c_ * (
                                (d_ - 2 * q * np.cos(delta)) * X32 - d_ * q**2 * X53)

                # end B2
        if tensile != 0:
                # %% tensile
                # displacement
                # uA
                Tu1A = -(1 - alpha) / 2 * np.log(R + eta) - alpha / 2 * q**2 * Y11
                Tu2A = -(1 - alpha) / 2 * np.log(R + xi) - alpha / 2 * q**2 * X11
                Tu3A = theta / 2 - alpha / 2 * q * (eta * X11 + xi * Y11)
                # uB
                Tu1B = q**2 * Y11 - (1 - alpha) / alpha * I3 * np.sin(delta)**2
                Tu2B = q**2 * X11 + (1 - alpha) / alpha * xi / (R + d_) * np.sin(delta)**2
                Tu3B = q * (eta * X11 + xi * Y11) - theta - (1 - alpha) / alpha * I4 * np.sin(delta)**2
                # uC
                Tu1C = -(1 - alpha) * (np.sin(delta) / R + q * Y11 * np.cos(delta)) - alpha * (
                        np.ones((4, 1)) * z * Y11 - q**2 * Z32)
                Tu2C = (1 - alpha) * 2 * xi * Y11 * np.sin(delta) + d_ * X11 - alpha * c_ * (
                        X11 - q**2 * X32)
                Tu3C = (1 - alpha) * (y_ * X11 + xi * Y11 * np.cos(delta)) + alpha * q * (
                        c_ * eta * X32 + xi * Z32)
                # displacement gradient
                # tensile
                # jA
                Tj1A = - (1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32
                Tj2A = - (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
                Tj3A = - (1 - alpha) / 2 * q * Y11 - alpha / 2 * q**3 * Y32
                # jB
                Tj1B = -xi * q**2 * Y32 - (1 - alpha) / alpha * J4 * np.sin(delta)**2
                Tj2B = -q**2 / R**3 - (1 - alpha) / alpha * J5 * np.sin(delta)**2
                Tj3B = q**3 * Y32 - (1 - alpha) / alpha * J6 * np.sin(delta)**2
                # jC
                Tj1C = (1 - alpha) * xi / R**3 * np.sin(delta) + xi * q * Y32 * np.cos(delta) + alpha * xi * (
                        3 * c_ * eta / R**5 - 2 * Z32 - Z0)
                Tj2C = (1 - alpha) * 2 * Y0 * np.sin(delta) - d_ / R**3 + alpha * c_ / R**3 * (
                        1 - 3 * q**2 / R**2)
                Tj3C = -(1 - alpha) * (y_ / R**3 - Y0 * np.cos(delta)) - alpha * (
                        3 * c_ * eta * q / R**5 - q * Z0)
                # tensile
                # kA
                Tk1A = -(1 - alpha) / 2 * (np.cos(delta) / R + q * Y11 * np.sin(delta)) - alpha / 2 * q * F
                Tk2A = -(1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
                Tk3A = (1 - alpha) / 2 * (d_ * X11 + xi * Y11 * np.sin(delta)) + alpha / 2 * q * H
                # kB
                Tk1B = q * F - (1 - alpha) / alpha * J1 * np.sin(delta)**2
                Tk2B = q * G - (1 - alpha) / alpha * J2 * np.sin(delta)**2
                Tk3B = -q * H - (1 - alpha) / alpha * J3 * np.sin(delta)**2
                # kC
                Tk1C = (1 - alpha) * (q / R**3 + Y0 * np.sin(delta) * np.cos(delta)) + alpha * (
                        np.ones((4, 1)) * z / R**3 * np.cos(delta) + 3 * c_ * d_ * q / R**5 - q * Z0 * np.sin(delta))
                Tk2C = -(1 - alpha) * 2 * xi * P * np.sin(delta) - y_ * d_ * X32 + alpha * c_ * (
                        (y_ + 2 * q * np.sin(delta)) * X32 - y_ * q**2 * X53)
                Tk3C = -(1 - alpha) * (xi * P * np.cos(delta) - X11 + y_**2 * X32) + alpha * c_ * (
                        (d_ + 2 * q * np.cos(delta)) * X32 - y_ * eta * q * X53) + alpha * xi * Q
                # tensile
                Tl1A = (1 - alpha) / 2 * (np.sin(delta) / R - q * Y11 * np.cos(delta)) - alpha / 2 * q * F_
                Tl2A = (1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
                Tl3A = (1 - alpha) / 2 * (y_ * X11 + xi * Y11 * np.cos(delta)) + alpha / 2 * q * H_
                # lB
                Tl1B = q * F_ + (1 - alpha) / alpha * K3 * np.sin(delta)**2
                Tl2B = q * G_ + (1 - alpha) / alpha * xi * D11 * np.sin(delta)**2
                Tl3B = -q * H_ + (1 - alpha) / alpha * K4 * np.sin(delta)**2
                # lB
                Tl1C = -eta / R**3 + Y0 * np.cos(delta)**2 - alpha * (
                        np.ones((4, 1)) * z / R**3 * np.sin(delta) - 3 * c_ * y_ * q / R**5 - Y0 * np.sin(delta)**2 + q * Z0 * np.cos(delta))
                Tl2C = (1 - alpha) * 2 * xi * P_ * np.sin(delta) - X11 + d_**2 * X32 - alpha * c_ * (
                        (d_ - 2 * q * np.cos(delta)) * X32 - d_ * q**2 * X53)
                Tl3C = (1 - alpha) * (
                        xi * P_ * np.cos(delta) + y_ * d_ * X32) + alpha * c_ * (
                        (y_ - 2 * q * np.sin(delta)) * X32 + d_ * eta * q * X53) + alpha * xi * Q_

                # end B3
                # for slip_str
        if slip_str != 0:
                # strike-slip
                # displacement
                # u1A_
                Su1A_ = theta / 2 + alpha / 2 * xi * q * Y11
                Su2A_ = alpha / 2 * q / R
                Su3A_ = (1 - alpha) / 2 * np.log(R + eta) - alpha / 2 * q**2 * Y11
                # displacement gradient
                # jA_
                Sj1A_ = -(1 - alpha) / 2 * q * Y11 - alpha / 2 * xi**2 * q * Y32
                Sj2A_ = - alpha / 2 * xi * q / R**3
                Sj3A_ = (1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32
                # kA
                Sk1A_ = (1 - alpha) / 2 * xi * Y11 * np.sin(delta) + d_ / 2 * X11 + alpha / 2 * xi * F
                Sk2A_ = alpha / 2 * E
                Sk3A_ = (1 - alpha) / 2 * (np.cos(delta) / R + q * Y11 * np.sin(delta)) - alpha / 2 * q * F
                # lA
                Sl1A_ = (1 - alpha) / 2 * xi * Y11 * np.cos(delta) + y_ / 2 * X11 + alpha / 2 * xi * F_
                Sl2A_ = alpha / 2 * E_
                Sl3A_ = - (1 - alpha) / 2 * (np.sin(delta) / R - q * Y11 * np.cos(delta)) - alpha / 2 * q * F_

        # for slip_dip
        if slip_dip != 0:
                # dip-slip
                # displacement
                # u1A_
                Du1A_ = alpha / 2 * q / R
                Du2A_ = theta / 2 + alpha / 2 * eta * q * X11
                Du3A_ = (1 - alpha) / 2 * np.log(R + xi) - alpha / 2 * q**2 * X11
                # displacement gradient
                # jA_
                Dj1A_ = - alpha / 2 * xi * q / R**3
                Dj2A_ = - q / 2 * Y11 - alpha / 2 * eta * q / R**3
                Dj3A_ = (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
                # kA
                Dk1A_ = alpha / 2 * E
                Dk2A_ = (1 - alpha) / 2 * d_ * X11 + xi / 2 * Y11 * np.sin(delta) + alpha / 2 * eta * G
                Dk3A_ = (1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
                # lA
                Dl1A_ = alpha / 2 * E_
                Dl2A_ = (1 - alpha) / 2 * y_ * X11 + xi / 2 * Y11 * np.cos(delta) + alpha / 2 * eta * G_
                Dl3A_ = - (1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_

        # for tensile
        if tensile != 0:
                # tensile for u1A_
                # displacement
                Tu1A_ = - (1 - alpha) / 2 * np.log(R + eta) - alpha / 2 * q**2 * Y11
                Tu2A_ = - (1 - alpha) / 2 * np.log(R + xi) - alpha / 2 * q**2 * X11
                Tu3A_ = theta / 2 - alpha / 2 * q * (eta * X11 + xi * Y11)
                # displacement gradient
                # tensile
                # jA_
                Tj1A_ = - (1 - alpha) / 2 * xi * Y11 + alpha / 2 * xi * q**2 * Y32
                Tj2A_ = - (1 - alpha) / 2 * 1 / R + alpha / 2 * q**2 / R**3
                Tj3A_ = - (1 - alpha) / 2 * q * Y11 - alpha / 2 * q**3 * Y32
                # kA
                Tk1A_ = - (1 - alpha) / 2 * (np.cos(delta) / R + q * Y11 * np.sin(delta)) - alpha / 2 * q * F
                Tk2A_ = - (1 - alpha) / 2 * y_ * X11 - alpha / 2 * q * G
                Tk3A_ = (1 - alpha) / 2 * (d_ * X11 + xi * Y11 * np.sin(delta)) + alpha / 2 * q * H
                # lA
                Tl1A_ = (1 - alpha) / 2 * (np.sin(delta) / R - q * Y11 * np.cos(delta)) - alpha / 2 * q * F_
                Tl2A_ = (1 - alpha) / 2 * d_ * X11 - alpha / 2 * q * G_
                Tl3A_ = (1 - alpha) / 2 * (y_ * X11 + xi * Y11 * np.cos(delta)) + alpha / 2 * q * H_
        
        if slip_str != 0:
                # displacement
                Sux = 1 / (2 * np.pi) * slip_str * (Su1A - Su1A_ + Su1B + np.ones((4, 1)) * z * Su1C)
                Suy = 1 / (2 * np.pi) * slip_str * (
                        (Su2A - Su2A_ + Su2B + np.ones((4, 1)) * z * Su2C) * np.cos(delta) -
                        (Su3A - Su3A_ + Su3B + np.ones((4, 1)) * z * Su3C) * np.sin(delta)
                )
                Suz = 1 / (2 * np.pi) * slip_str * (
                        (Su2A - Su2A_ + Su2B - np.ones((4, 1)) * z * Su2C) * np.sin(delta) +
                        (Su3A - Su3A_ + Su3B - np.ones((4, 1)) * z * Su3C) * np.cos(delta)
                )

                # displacement gradients
                Sduxdx = 1 / (2 * np.pi) * slip_str * (Sj1A - Sj1A_ + Sj1B + np.ones((4, 1)) * z * Sj1C)
                Sduydx = 1 / (2 * np.pi) * slip_str * (
                        (Sj2A - Sj2A_ + Sj2B + np.ones((4, 1)) * z * Sj2C) * np.cos(delta) -
                        (Sj3A - Sj3A_ + Sj3B + np.ones((4, 1)) * z * Sj3C) * np.sin(delta)
                )
                Sduzdx = 1 / (2 * np.pi) * slip_str * (
                        (Sj2A - Sj2A_ + Sj2B - np.ones((4, 1)) * z * Sj2C) * np.sin(delta) +
                        (Sj3A - Sj3A_ + Sj3B - np.ones((4, 1)) * z * Sj3C) * np.cos(delta)
                )

                Sduxdy = 1 / (2 * np.pi) * slip_str * (Sk1A - Sk1A_ + Sk1B + np.ones((4, 1)) * z * Sk1C)
                Sduydy = 1 / (2 * np.pi) * slip_str * (
                        (Sk2A - Sk2A_ + Sk2B + np.ones((4, 1)) * z * Sk2C) * np.cos(delta) -
                        (Sk3A - Sk3A_ + Sk3B + np.ones((4, 1)) * z * Sk3C) * np.sin(delta)
                )
                Sduzdy = 1 / (2 * np.pi) * slip_str * (
                        (Sk2A - Sk2A_ + Sk2B - np.ones((4, 1)) * z * Sk2C) * np.sin(delta) +
                        (Sk3A - Sk3A_ + Sk3B - np.ones((4, 1)) * z * Sk3C) * np.cos(delta)
                )

                Sduxdz = 1 / (2 * np.pi) * slip_str * (Sl1A + Sl1A_ + Sl1B + Su1C + np.ones((4, 1)) * z * Sl1C)
                Sduydz = 1 / (2 * np.pi) * slip_str * (
                        (Sl2A + Sl2A_ + Sl2B + Su2C + np.ones((4, 1)) * z * Sl2C) * np.cos(delta) -
                        (Sl3A + Sl3A_ + Sl3B + Su3C + np.ones((4, 1)) * z * Sl3C) * np.sin(delta)
                )
                Sduzdz = 1 / (2 * np.pi) * slip_str * (
                        (Sl2A + Sl2A_ + Sl2B - Su2C - np.ones((4, 1)) * z * Sl2C) * np.sin(delta) +
                        (Sl3A + Sl3A_ + Sl3B - Su3C - np.ones((4, 1)) * z * Sl3C) * np.cos(delta)
                )
        else:
                Sux, Suy, Suz = 0, 0, 0
                Sduxdx, Sduydx, Sduzdx = 0, 0, 0
                Sduxdy, Sduydy, Sduzdy = 0, 0, 0
                Sduxdz, Sduydz, Sduzdz = 0, 0, 0

        # dip-slip
        if slip_dip != 0:
                # displacement
                Dux = 1 / (2 * np.pi) * slip_dip * (Du1A - Du1A_ + Du1B + np.ones((4, 1)) * z * Du1C)
                Duy = 1 / (2 * np.pi) * slip_dip * (
                        (Du2A - Du2A_ + Du2B + np.ones((4, 1)) * z * Du2C) * np.cos(delta) -
                        (Du3A - Du3A_ + Du3B + np.ones((4, 1)) * z * Du3C) * np.sin(delta)
                )
                Duz = 1 / (2 * np.pi) * slip_dip * (
                        (Du2A - Du2A_ + Du2B - np.ones((4, 1)) * z * Du2C) * np.sin(delta) +
                        (Du3A - Du3A_ + Du3B - np.ones((4, 1)) * z * Du3C) * np.cos(delta)
                )

                # displacement gradients
                Dduxdx = 1 / (2 * np.pi) * slip_dip * (Dj1A - Dj1A_ + Dj1B + np.ones((4, 1)) * z * Dj1C)
                Dduydx = 1 / (2 * np.pi) * slip_dip * (
                        (Dj2A - Dj2A_ + Dj2B + np.ones((4, 1)) * z * Dj2C) * np.cos(delta) -
                        (Dj3A - Dj3A_ + Dj3B + np.ones((4, 1)) * z * Dj3C) * np.sin(delta)
                )
                Dduzdx = 1 / (2 * np.pi) * slip_dip * (
                        (Dj2A - Dj2A_ + Dj2B - np.ones((4, 1)) * z * Dj2C) * np.sin(delta) +
                        (Dj3A - Dj3A_ + Dj3B - np.ones((4, 1)) * z * Dj3C) * np.cos(delta)
                )

                Dduxdy = 1 / (2 * np.pi) * slip_dip * (Dk1A - Dk1A_ + Dk1B + np.ones((4, 1)) * z * Dk1C)
                Dduydy = 1 / (2 * np.pi) * slip_dip * (
                        (Dk2A - Dk2A_ + Dk2B + np.ones((4, 1)) * z * Dk2C) * np.cos(delta) -
                        (Dk3A - Dk3A_ + Dk3B + np.ones((4, 1)) * z * Dk3C) * np.sin(delta)
                )
                Dduzdy = 1 / (2 * np.pi) * slip_dip * (
                        (Dk2A - Dk2A_ + Dk2B - np.ones((4, 1)) * z * Dk2C) * np.sin(delta) +
                        (Dk3A - Dk3A_ + Dk3B - np.ones((4, 1)) * z * Dk3C) * np.cos(delta)
                )

                Dduxdz = 1 / (2 * np.pi) * slip_dip * (Dl1A + Dl1A_ + Dl1B + Du1C + np.ones((4, 1)) * z * Dl1C)
                Dduydz = 1 / (2 * np.pi) * slip_dip * (
                        (Dl2A + Dl2A_ + Dl2B + Du2C + np.ones((4, 1)) * z * Dl2C) * np.cos(delta) -
                        (Dl3A + Dl3A_ + Dl3B + Du3C + np.ones((4, 1)) * z * Dl3C) * np.sin(delta)
                )
                Dduzdz = 1 / (2 * np.pi) * slip_dip * (
                        (Dl2A + Dl2A_ + Dl2B - Du2C - np.ones((4, 1)) * z * Dl2C) * np.sin(delta) +
                        (Dl3A + Dl3A_ + Dl3B - Du3C - np.ones((4, 1)) * z * Dl3C) * np.cos(delta)
                )
        else:
                Dux, Duy, Duz = 0, 0, 0
                Dduxdx, Dduydx, Dduzdx = 0, 0, 0
                Dduxdy, Dduydy, Dduzdy = 0, 0, 0
                Dduxdz, Dduydz, Dduzdz = 0, 0, 0

        if tensile != 0:
                # Tensile
                # Displacement
                Tux = (1 / (2 * np.pi)) * tensile * (Tu1A - Tu1A_ + Tu1B + np.ones((4, 1)) * z * Tu1C)
                Tuy = (1 / (2 * np.pi)) * tensile * ((Tu2A - Tu2A_ + Tu2B + np.ones((4, 1)) * z * Tu2C) * np.cos(delta) -
                                                        (Tu3A - Tu3A_ + Tu3B + np.ones((4, 1)) * z * Tu3C) * np.sin(delta))
                Tuz = (1 / (2 * np.pi)) * tensile * ((Tu2A - Tu2A_ + Tu2B - np.ones((4, 1)) * z * Tu2C) * np.sin(delta) +
                                                        (Tu3A - Tu3A_ + Tu3B - np.ones((4, 1)) * z * Tu3C) * np.cos(delta))

                # Displacement gradients
                Tduxdx = (1 / (2 * np.pi)) * tensile * (Tj1A - Tj1A_ + Tj1B + np.ones((4, 1)) * z * Tj1C)
                Tduydx = (1 / (2 * np.pi)) * tensile * ((Tj2A - Tj2A_ + Tj2B + np.ones((4, 1)) * z * Tj2C) * np.cos(delta) -
                                                        (Tj3A - Tj3A_ + Tj3B + np.ones((4, 1)) * z * Tj3C) * np.sin(delta))
                Tduzdx = (1 / (2 * np.pi)) * tensile * ((Tj2A - Tj2A_ + Tj2B - np.ones((4, 1)) * z * Tj2C) * np.sin(delta) +
                                                        (Tj3A - Tj3A_ + Tj3B - np.ones((4, 1)) * z * Tj3C) * np.cos(delta))

                Tduxdy = (1 / (2 * np.pi)) * tensile * (Tk1A - Tk1A_ + Tk1B + np.ones((4, 1)) * z * Tk1C)
                Tduydy = (1 / (2 * np.pi)) * tensile * ((Tk2A - Tk2A_ + Tk2B + np.ones((4, 1)) * z * Tk2C) * np.cos(delta) -
                                                        (Tk3A - Tk3A_ + Tk3B + np.ones((4, 1)) * z * Tk3C) * np.sin(delta))
                Tduzdy = (1 / (2 * np.pi)) * tensile * ((Tk2A - Tk2A_ + Tk2B - np.ones((4, 1)) * z * Tk2C) * np.sin(delta) +
                                                        (Tk3A - Tk3A_ + Tk3B - np.ones((4, 1)) * z * Tk3C) * np.cos(delta))

                Tduxdz = (1 / (2 * np.pi)) * tensile * (Tl1A + Tl1A_ + Tl1B + Tu1C + np.ones((4, 1)) * z * Tl1C)
                Tduydz = (1 / (2 * np.pi)) * tensile * ((Tl2A + Tl2A_ + Tl2B + Tu2C + np.ones((4, 1)) * z * Tl2C) * np.cos(delta) -
                                                        (Tl3A + Tl3A_ + Tl3B + Tu3C + np.ones((4, 1)) * z * Tl3C) * np.sin(delta))
                Tduzdz = (1 / (2 * np.pi)) * tensile * ((Tl2A + Tl2A_ + Tl2B - Tu2C - np.ones((4, 1)) * z * Tl2C) * np.sin(delta) +
                                                        (Tl3A + Tl3A_ + Tl3B - Tu3C - np.ones((4, 1)) * z * Tl3C) * np.cos(delta))
        else:
                # B8
                Tux, Tuy, Tuz = 0, 0, 0
                Tduxdx, Tduydx, Tduzdx = 0, 0, 0
                Tduxdy, Tduydy, Tduzdy = 0, 0, 0
                Tduxdz, Tduydz, Tduzdz = 0, 0, 0


        factor=np.ones((xi.shape))
        factor[1,:]=factor[1,:]*-1
        factor[2,:]=factor[2,:]*-1


        G1 = np.sum(factor * (Sux + Dux + Tux),axis=0)
        G2 = np.sum(factor * (Suy + Duy + Tuy),axis=0)
        G3 = np.sum(factor * (Suz + Duz + Tuz),axis=0)

        Dg11 = np.sum(factor * (Sduxdx + Dduxdx + Tduxdx),axis=0)
        Dg12 = np.sum(factor * (Sduxdy + Dduxdy + Tduxdy),axis=0)
        Dg13 = np.sum(factor * (Sduxdz + Dduxdz + Tduxdz),axis=0)

        Dg21 = np.sum(factor * (Sduydx + Dduydx + Tduydx),axis=0)
        Dg22 = np.sum(factor * (Sduydy + Dduydy + Tduydy),axis=0)
        Dg23 = np.sum(factor * (Sduydz + Dduydz + Tduydz),axis=0)

        Dg31 = np.sum(factor * (Sduzdx + Dduzdx + Tduzdx),axis=0)
        Dg32 = np.sum(factor * (Sduzdy + Dduzdy + Tduzdy),axis=0)
        Dg33 = np.sum(factor * (Sduzdz + Dduzdz + Tduzdz),axis=0)

        # Coordinate transformation
        Gx = np.cos(angle_Str) * (-G2) - np.sin(angle_Str) * G1
        Gy = np.sin(angle_Str) * (-G2) + np.cos(angle_Str) * G1
        Gz = G3

        displacement = np.array([Gx, Gy, Gz])

        # tempGrad1 = np.array([[Dg11, Dg12, Dg13], [Dg21, Dg22, Dg23], [Dg31, Dg32, Dg33]])
        Dg11_ = Dg22
        Dg12_ = -Dg21
        Dg13_ = -Dg23

        Dg21_ = -Dg12
        Dg22_ = Dg11
        Dg23_ = Dg13

        Dg31_ = -Dg32
        Dg32_ = Dg31
        Dg33_ = Dg33

        # Coordinate transformation
        Dgxx = (np.cos(angle_Str) * Dg11_ - np.sin(angle_Str) * Dg21_) * np.cos(angle_Str) + \
        (np.cos(angle_Str) * Dg12_ - np.sin(angle_Str) * Dg22_) * (-np.sin(angle_Str))
        Dgyx = (np.sin(angle_Str) * Dg11_ + np.cos(angle_Str) * Dg21_) * np.cos(angle_Str) - \
        (np.sin(angle_Str) * Dg12_ + np.cos(angle_Str) * Dg22_) * np.sin(angle_Str)
        Dgzx = Dg31_ * np.cos(angle_Str) - Dg32_ * np.sin(angle_Str)

        Dgxy = (np.cos(angle_Str) * Dg11_ - np.sin(angle_Str) * Dg21_) * np.sin(angle_Str) + \
        (np.cos(angle_Str) * Dg12_ - np.sin(angle_Str) * Dg22_) * np.cos(angle_Str)
        Dgyy = (np.sin(angle_Str) * Dg11_ + np.cos(angle_Str) * Dg21_) * np.sin(angle_Str) + \
        (np.sin(angle_Str) * Dg12_ + np.cos(angle_Str) * Dg22_) * np.cos(angle_Str)
        Dgzy = np.sin(angle_Str) * Dg31_ + np.cos(angle_Str) * Dg32_

        Dgxz = np.cos(angle_Str) * Dg13_ - np.sin(angle_Str) * Dg23_
        Dgyz = np.sin(angle_Str) * Dg13_ + np.cos(angle_Str) * Dg23_
        Dgzz = Dg33_

        gradient = np.array([[Dgxx, Dgxy, Dgxz],
                        [Dgyx, Dgyy, Dgyz],
                        [Dgzx, Dgzy, Dgzz]])

        # Strain components
        Ex = Dgxx
        Ey = Dgyy
        Ez = Dgzz
        Exy = 0.5 * (Dgyx + Dgxy)
        Eyz = 0.5 * (Dgyz + Dgzy)
        Ezx = 0.5 * (Dgzx + Dgxz)

        # Stress components
        Sx = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ex + nu * (Ey + Ez - Ex))
        Sy = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ey + nu * (Ex + Ez - Ey))
        Sz = youngs / ((1 + nu) * (1 - 2 * nu)) * (Ez + nu * (Ey + Ex - Ez))
        Sxy = 2 * Gshear * Exy
        Syz = 2 * Gshear * Eyz
        Szx = 2 * Gshear * Ezx

        # Stress vector
        Stress = np.array([Sx, Sxy, Szx, Sy, Syz, Sz])

    
else:
        if np.array_equal(Z, -np.abs(Z)):
                print('warning: physically impossible')
        else:
                print('warning: All z should be negative.')
 
        displacement = np.nan
        gradient = np.nan
        Stress = np.nan

for j in range(x.size):
    print('Displacement:',displacement[:,j])
    print('Stress:',Stress[:,j])