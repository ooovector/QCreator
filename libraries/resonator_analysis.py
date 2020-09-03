import sys
import os
import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.constants import epsilon_0, mu_0
epsilon=11.45
mu=1

# print( os.path.abspath('./'))
class ResonatorAnalyser():
    def __init__(self,Resonator):
        sys.path.insert(0, os.path.abspath('./'))
        self.frequency_value = Resonator.f
        self.widths=None
        self.cap=None
        self.ind=None
        self.coupler_l = Resonator.coupler_length
        self.short_end_l = Resonator.L-Resonator.coupler_length-Resonator.open_end_length
        self.claw_end_l = Resonator.open_end_length
        self.frequency=None
        self.nop=10000


    def get_conformal_tables(self,feedline, resonator):
        import conformal_mapping.сonformal_mapping as cm
        result=cm.ConformalMapping(self.widths)
        C, L = result.cl_and_Ll()
        reduced_table_C = np.asarray([[C[feedline, feedline], C[feedline, resonator]],
                                      [C[resonator, feedline], C[resonator, resonator]]])
        reduced_table_L  = np.asarray(np.linalg.inv(reduced_table_C)*(epsilon + 1)*epsilon_0/(1/(mu*mu_0)+1/mu_0))
        cap, ind = pd.DataFrame(reduced_table_C * 1e12), pd.DataFrame(reduced_table_L * 1e9)
        self.cap=reduced_table_C
        self.ind=reduced_table_L
        return cap,ind

    def simulate_S21(self,frequency=None,nop=None):
        def equation(om_re_val, om_im_val):
            determinant = circuit.boundary_condition_matrix_det(om_re_val+1j*om_im_val)
            return (determinant.real, determinant.imag)

        if nop:
            self.nop=nop
        if frequency is not None:
            self.frequency=frequency
        else:
            self.frequency = np.linspace(complex(self.frequency_value-0.02e9), complex(self.frequency_value+0.02e9), self.nop)
        # print(self.frequency)
        import Sparameters.transmission_line_simulator as tls
        claw = tls.capacitor()
        # qubit_cap = capacitor()
        # qubit_inductor = inductor()
        source = tls.port()
        analyzer = tls.port()

        GND = tls.short()
        resonator_short_end = tls.transmission_line_coupler(n=1)
        resonator_claw_end = tls.transmission_line_coupler(n=1)
        coupler = tls.transmission_line_coupler()

        circuit = tls.transmission_line_system()

        circuit.add_element(source, [1])
        circuit.add_element(coupler, [1, 2, 3, 4])
        circuit.add_element(analyzer, [3])
        circuit.add_element(resonator_short_end, [4, 0])
        circuit.add_element(resonator_claw_end, [2, 5])
        circuit.add_element(claw, [5, 0])
        # circuit.add_element(qubit_cap, [6, 0])
        # circuit.add_element(qubit_inductor, [6, 0])
        circuit.add_element(GND, [0])

        source.Z0 = 50
        analyzer.Z0 = 50

        coupler.l = self.coupler_l/1e6
        coupler.Cl = np.asarray(self.cap)
        coupler.Ll = np.asarray(self.ind)
        coupler.Rl = np.zeros(coupler.Ll.shape, dtype=np.int)
        coupler.Gl = np.zeros(coupler.Ll.shape, dtype=np.int)

        resonator_short_end.l = self.short_end_l/1e6
        resonator_short_end.Cl = self.cap[1,1]
        resonator_short_end.Ll = self.ind[1,1]
        resonator_short_end.Rl = 0
        resonator_short_end.Gl = 0

        resonator_claw_end.l = self.claw_end_l/1e6
        resonator_claw_end.Cl = self.cap[1,1]
        resonator_claw_end.Ll = self.ind[1,1]
        resonator_claw_end.Rl = 0
        resonator_claw_end.Gl = 0

        claw.C = 1e-15
        # qubit_cap.C=70e-15
        # qubit_inductor.L=19e-9

        # Simulate scattering parameter S21
        scale = np.asarray((2 * np.pi * self.frequency_value, 1e6))
        solution = optimize.fsolve(lambda x: equation(*(x * scale)), (1, 1)) * scale
        fr_numeric_num = solution[0] / np.pi / 2.
        Q_numeric_num = -solution[0] / (2 * solution[1])
        print('full numeric frequency, GHz: ', np.abs(solution[0] / np.pi / 2. / 1e9), ', Q: ',
              solution[0] / (2 * solution[1]))
        y = np.zeros(self.nop, dtype=complex)

        matrix_of_curcuit = circuit.create_boundary_problem_matrix(self.frequency[0] * 2 * np.pi)

        perturbation = np.zeros((matrix_of_curcuit.shape[0], 1))
        perturbation[0] = 1
        for i in range(self.nop):
            matrix_of_curcuit = circuit.create_boundary_problem_matrix(self.frequency[i] * 2 * np.pi)
            s21 = np.linalg.solve(matrix_of_curcuit, perturbation )
            y[i] = s21[2]
        abs_S21 = np.abs(y)
        angle_S21 = np.angle(y)
        self.S21=y
        return fr_numeric_num,Q_numeric_num

    def fit_S21(self):
        from resonator_tools.circuit import notch_port, reflection_port
        fitter = notch_port(f_data=self.frequency.real, z_data_raw=self.S21)
        fitter.autofit()
        return fitter
    def plot_S21(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(self.frequency.real, np.real(self.S21),label='$S_{21}$')
        plt.xlabel('Frequency, $f$ (GHz)')
        plt.ylabel('Power transmission, $S_{21}$ (dB)')
        plt.legend()
        plt.subplot(132)
        plt.plot(self.frequency.real, np.angle(self.S21), label='$\\angle S_{21}$')
        plt.xlabel('Frequency, $f$ (GHz)')
        plt.legend()
        plt.subplot(133)
        plt.plot(self.S21.real, self.S21.imag, label='$S_{21}$')
        plt.legend()
        plt.show()