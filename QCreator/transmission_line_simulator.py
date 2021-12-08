import numpy as np
from abc import *
from scipy.constants import e, hbar, h
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class TLSystemElement:
    @abstractmethod
    def num_terminals(self):
        pass

    @abstractmethod
    def num_degrees_of_freedom_dynamic(self):
        return self.num_degrees_of_freedom()

    @abstractmethod
    def num_degrees_of_freedom(self):
        pass

    @abstractmethod
    def boundary_condition(self, omega):
        pass

    @abstractmethod
    def energy(self, mode):
        try:
            return np.conj(mode).squeeze() @ self.energy_matrix() @ mode.squeeze()
        except:
            return 0

    @abstractmethod
    def energy_matrix(self):
        return 0

    def scdc(self, submode):
        return None

    def is_scdc(self):
        return False

    def get_capacitance_matrix(self):
        return 0

    def get_inv_inductance_matrix(self):
        return 0

    def nonlinearity_in_taylor_expansion(self):
        return 0

    def __init__(self, type_, name=''):
        self.name = name
        self.type_ = type_

    def __repr__(self):
        return "{} {}".format(self.type_, self.name)


class Resistor(TLSystemElement):
    def num_terminals(self):
        return 2

    def num_degrees_of_freedom(self):
        return 0

    def boundary_condition(self, omega):
        return np.asarray([[1, -1, self.R, 0], [0, 0, 1, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])  # derivatives
        a = np.asarray([[1, -1, self.R, 0], [0, 0, 1, 1]])  # current values
        return a, b

    def energy(self, mode):
        return 0

    def energy_matrix(self):
        return 0

    def __init__(self, r=None, name=''):
        super().__init__('R', name)
        self.R = r
        pass


class Capacitor(TLSystemElement):
    def num_terminals(self):
        return 2

    def num_degrees_of_freedom(self):
        return 0

    def boundary_condition(self, omega):
        return np.asarray([[1j * omega * self.C, -1j * omega * self.C, 1, 0], [0, 0, 1, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[self.C, -self.C, 0, 0], [0, 0, 0, 0]])  # derivatives
        a = np.asarray([[0, 0, -1, 0], [0, 0, 1, 1]])  # current values
        return a, b

    def energy(self, mode):
        # energy matrix
        emat = np.asarray([
            [self.C, -self.C, 0, 0],
            [-self.C, self.C, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]) / 2
        # return np.conj(mode)@emat@np.reshape(mode, (-1,1))
        return self.C / 2 * (mode[0] - mode[1]) ** 2

    def energy_matrix(self):
        # energy matrix
        emat = np.asarray([
            [self.C, -self.C, 0, 0],
            [-self.C, self.C, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]) / 2

        return emat

    def get_capacitance_matrix(self):
        return np.asarray([
            [self.C, -self.C],
            [-self.C, self.C]])

    def __init__(self, c=None, name=''):
        super().__init__('C', name)
        self.C = c
        pass


class Inductor(TLSystemElement):
    def num_terminals(self):
        return 2

    def num_degrees_of_freedom(self):
        return 0

    def boundary_condition(self, omega):
        return np.asarray([[1, -1, 1j * omega * self.L, 0], [0, 0, 1, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, -self.L, 0], [0, 0, 0, 0]])  # derivatives
        a = np.asarray([[1, -1, 0, 0], [0, 0, 1, 1]])  # current values
        return a, b

    def energy(self, mode):
        emat = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.L, 0],
            [0, 0, 0, 0]
        ]) / 2
        return np.conj(mode) @ emat @ mode

    # def scdc(self, submode):
    #     return [(submode[1] - submode[0] + submode[2] * 1e-6 * np.real(self.L) / (hbar / (2 * e))),
    #             (submode[2] + submode[3])]
    #
    # def scdc_energy(self, submode):
    #     return []
    #
    # def scdc_constrains(self):
    #     return [-1, 1, 1e-6 * np.real(self.L) / (hbar / (2 * e)),
    #              0, 0, 1, 1]
    #
    # def scdc_gradient(self, submode):
    #     return []

    def potential(self, submode):
        return (submode[1] - submode[0]) ** 2 / (2 * np.real(self.L)) * 1e-9

    def potential_gradient(self, submode):
        return [(submode[0] - submode[1]) / np.real(self.L) * 1e-9,
                (submode[1] - submode[0]) / np.real(self.L) * 1e-9]

    def potential_hessian(self, submode):
        return np.asarray([[1, -1], [-1, 1]]) / np.real(self.L) * 1e-9

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def energy_matrix(self):
        emat = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.L, 0],
            [0, 0, 0, 0]
        ]) / 2
        return emat

    def get_inv_inductance_matrix(self):
        return np.asarray([[1 / self.L, - 1 / self.L], [- 1 / self.L, 1 / self.L]])

    def __init__(self, l=None, name=''):
        super().__init__('L', name)
        self.L = l
        pass


class Short(TLSystemElement):
    def num_terminals(self):
        return 1

    def num_degrees_of_freedom(self):
        return 0

    def boundary_condition(self, omega):
        return np.asarray([[1, 0]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0]])  # derivatives
        a = np.asarray([[1, 0]])  # current values
        return a, b

    # def scdc(self, submode):
    #     #return [submode[0]]
    #     return ([], [1])
    #
    # def scdc_energy(self, submode):
    #     return []
    #
    # def scdc_constrains(self):
    #     return [1, 0]
    #
    # def scdc_gradient(self, submode):
    #     return []
    def potential(self, submode):
        return 0

    def potential_gradient(self, submode):
        return [0]

    def potential_hessian(self, submode):
        return [[0]]

    # def potential_constraints(self):
    #     return [[1]]

    def is_scdc(self):
        return True

    def energy(self, mode):
        return 0

    def energy_matrix(self):
        return 0

    def __init__(self):
        super().__init__('Short', '')
        pass


class Port(TLSystemElement):
    def num_terminals(self):
        return 1

    def num_degrees_of_freedom(self):
        return 1

    def boundary_condition(self, omega):
        return np.asarray([[1, self.Z0, 0], [1, -self.Z0, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, 0], [0, 0, 0]])  # derivatives
        a = np.asarray([[1, self.Z0, 0], [1, -self.Z0, 1]])  # current values
        return a, b

    # def scdc(self, submode):
    #     return [(submode[1] - self.idc)]
    #
    # def scdc_energy(self, submode):
    #     return [(submode[1] - self.idc)**2/2]
    #
    # def scdc_constrains(self):
    #     return []
    #
    # def scdc_gradient(self, submode):
    #     return [(submode[1] - self.idc)]

    def potential(self, submode):
        return -submode[0] * self.idc / (hbar / (2 * e)) * 1e-9

    def potential_gradient(self, submode):
        return [-self.idc / (hbar / (2 * e)) * 1e-9]

    def potential_hessian(self, submode):
        return [[0]]

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def energy(self, mode):
        return 0

    def energy_matrix(self):
        return 0

    def __init__(self, z0=None, name=''):
        super().__init__('Port', name)
        self.Z0 = z0
        self.idc = 0


class TLCoupler(TLSystemElement):
    '''
    Here n is a number of conductors in  TL CPW coupler
    '''

    def num_terminals(self):
        return self.n * 2

    def num_degrees_of_freedom(self):
        return self.n * 2

    def num_degrees_of_freedom_dynamic(self):
        return self.n * 2 * self.num_modes

    def propagating_modes(self):
        '''
        numpy.hstack puts together two matrix horizontally
        numpy.vstack puts together two matrix vertically
        '''
        M = np.hstack((np.vstack((self.Rl, self.Cl)), np.vstack((self.Ll, self.Gl))))

        cl, mode_amplitudes = np.linalg.eig(M)

        gammas = -cl
        modes = []

        for mode_id, gamma in enumerate(gammas):
            modes.append((gamma, mode_amplitudes[:, mode_id]))
        return modes

    def boundary_condition(self, omega):
        boundary_condition_matrix = np.zeros(
            (self.num_terminals() * 2, self.num_terminals() * 2 + self.num_degrees_of_freedom()), dtype=complex)
        boundary_condition_matrix[:, :self.num_terminals() * 2] = np.identity(self.num_terminals() * 2)

        for mode_pair_id, mode_pair in enumerate(self.propagating_modes()):
            boundary_condition_matrix[0:self.n, self.n * 4 + mode_pair_id] = -np.asarray(mode_pair[1][:self.n])
            boundary_condition_matrix[self.n:self.n * 2, self.n * 4 + mode_pair_id] = -np.asarray(
                mode_pair[1][:self.n]) * np.exp(1j * mode_pair[0] * self.l * omega)
            boundary_condition_matrix[self.n * 2:self.n * 3, self.n * 4 + mode_pair_id] = np.asarray(
                mode_pair[1][self.n:])
            boundary_condition_matrix[self.n * 3:, self.n * 4 + mode_pair_id] = -np.asarray(
                mode_pair[1][self.n:]) * np.exp(1j * mode_pair[0] * self.l * omega)
        # print(mode_pair)
        return boundary_condition_matrix

    def energy(self, state):
        energy_matrix = self.energy_matrix()

        # emat = np.vstack([np.hstack([ll, np.zeros_like(ll)]), np.hstack([np.zeros_like(cl), cl])])

        return np.conj(
            state.squeeze()) @ energy_matrix @ state.squeeze()  # TODO: energy stored in transmission line system

    def get_capacitance_matrix(self):
        """
        Capacitance matrix in dynamic basis
        """
        m = self.n * self.num_modes
        capacitance_matrix = np.zeros((self.num_terminals() + m, self.num_terminals() + m), dtype=complex)
        e_c = np.zeros((m, m))

        for i in range(self.num_modes):
            for j in range(self.num_modes):
                e_c_ = self.l * self.Cl / 2 * (
                        (1 / 2) ** (i + j + 1) - (- 1 / 2) ** (i + j + 1)) / (i + j + 1)
                space = np.zeros((self.num_modes, self.num_modes))
                space[i][j] = 1
                e_c += np.kron(e_c_, space)

        capacitance_matrix[self.num_terminals(): self.num_terminals() + m,
        self.num_terminals(): self.num_terminals() + m] = e_c * 2

        return capacitance_matrix

    def get_inv_inductance_matrix(self):
        """
        Inverse inductance matrix in dynamic basis
        """
        from numpy.linalg import inv
        m = self.n * self.num_modes
        inductance_matrix = np.zeros((self.num_terminals() + m, self.num_terminals() + m), dtype=complex)
        if self.n == 1:
            inv_Ll = 1 / self.Ll
        else:
            inv_Ll = inv(self.Ll)
        e_l = np.zeros((m, m))
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i + j == 1:
                    e_l_ = np.zeros((self.n, self.n))
                else:
                    e_l_ = 1 * inv_Ll / (2 * self.l) * i * j / (i + j - 1) * (
                            (1 / 2) ** (i + j - 1) - (- 1 / 2) ** (i + j - 1))
                # e_l_ = self.l * inv_Ll / 2 * (
                #         (1 / 2) ** (i + j + 1) - (- 1 / 2) ** (i + j + 1)) / (i + j + 1)
                space = np.zeros((self.num_modes, self.num_modes))
                space[i][j] = 1
                e_l += np.kron(e_l_, space)
                # e_l[i][j] = self.l * inv_Ll / 2 * (
                #         (1 / 2) ** (i + j + 1) - (- 1 / 2) ** (i + j + 1)) / (i + j + 1)
        inductance_matrix[self.num_terminals(): self.num_terminals() + m,
        self.num_terminals(): self.num_terminals() + m] = e_l * 2
        return inductance_matrix

    def energy_matrix(self):
        m = self.n * self.num_modes
        s = (-0.5) ** np.arange(self.num_modes)
        e = 0.5 * s * s.reshape((-1, 1)) * self.l
        e += np.abs(e)

        integral = 1 / (np.arange(self.num_modes).reshape(-1, 1) + np.arange(self.num_modes) + 1)
        e = e * integral

        ll = np.kron(self.Ll, e)
        cl = np.kron(self.Cl, e)

        energy_matrix = np.zeros((self.num_terminals() * 2 + m * 2, self.num_terminals() * 2 + m * 2))
        energy_matrix[-2 * m:-m, -2 * m:-m] = cl / 2
        energy_matrix[-m:, -m:] = ll / 2

        return energy_matrix

    def dynamic_equations(self):
        m = self.n * self.num_modes
        n_eq_internal = self.n * (self.num_modes - 1)

        b = np.zeros((self.num_terminals() * 2 + n_eq_internal * 2, self.num_terminals() * 2 + m * 2))
        a = np.zeros((self.num_terminals() * 2 + n_eq_internal * 2, self.num_terminals() * 2 + m * 2))

        # filling out telegrapher's equations
        E = np.zeros((self.num_modes - 1, self.num_modes))
        for i in range(self.num_modes - 1):
            # E[i, i] = 1#/cl_av
            E[i, i] = self.l

        Ll = np.kron(self.Ll, E)
        Cl = np.kron(self.Cl, E)
        Rl = np.kron(self.Rl, E)
        Gl = np.kron(self.Gl, E)

        b[:n_eq_internal, -m:] = Ll
        b[n_eq_internal:2 * n_eq_internal, -2 * m:-m] = Cl

        # Taylor-series expansion of I(x) = sum_i a_i x^i
        d = np.zeros((self.num_modes - 1, self.num_modes))
        for i in range(1, self.num_modes):
            d[i - 1, i] = i
        dmat = np.kron(np.eye(self.n), d)

        a[:n_eq_internal, -2 * m:-m] = -dmat
        a[n_eq_internal:2 * n_eq_internal, -m:] = -dmat
        a[:n_eq_internal, -m:] = -Rl
        a[n_eq_internal:2 * n_eq_internal, -2 * m:-m] = -Gl

        # filling out boundary conditions (voltage)
        a[-self.n * 4:, :self.n * 4] = np.eye(self.n * 4)

        mode_left = (-1 / 2) ** np.arange(self.num_modes)
        mode_right = (1 / 2) ** np.arange(self.num_modes)
        for k in range(self.n):
            c = np.zeros(self.n)
            c[k] = 1
            # Modal voltages
            a[2 * n_eq_internal + k, -2 * m:-m] = np.kron(c, mode_left)
            a[2 * n_eq_internal + self.n + k, -2 * m:-m] = np.kron(c, mode_right)
            # Modal currents
            a[2 * n_eq_internal + self.n * 2 + k, -m:] = -np.kron(c, mode_left)
            a[2 * n_eq_internal + self.n * 3 + k, -m:] = np.kron(c, mode_right)
        return a, b

    # def scdc(self, submode):
    #     dphi = submode[self.n:2*self.n] - submode[:self.n]
    #     phi = self.l * (self.Ll @ submode[2*self.n:3*self.n]) / (hbar/(2*e)) * 1e-6
    #     di = submode[2*self.n:3*self.n] + submode[3*self.n:4*self.n]
    #     return np.hstack([(phi + dphi), di]).tolist()
    #
    # def scdc_energy(self, submode):
    #     return []
    #
    # def scdc_constrains(self):
    #     equations_phase = np.zeros((self.n * 2, self.n * 4))
    #     equations_phase[:self.n, :self.n] = -np.identity(self.n)
    #     equations_phase[:self.n, self.n:self.n * 2] = np.identity(self.n)
    #     equations_phase[:self.n, self.n * 2:self.n * 3] = self.l * self.Ll / (hbar / (2 * e)) * 1e-6
    #
    #     equations_phase[self.n:, self.n * 3:self.n * 4] = np.identity(self.n)
    #     equations_phase[self.n:, self.n * 3:self.n * 4] = np.identity(self.n)
    #     return equations_phase.tolist()
    #
    # def scdc_gradient(self, submode):
    #     return []

    def potential(self, submode):
        dphi = -submode[self.n:] + submode[:self.n]
        potential = self.l * dphi @ np.linalg.inv(self.Ll) @ dphi / 2 * 1e-9
        return potential

    def potential_gradient(self, submode):
        dphi = -submode[self.n:] + submode[:self.n]
        gradient = self.l * np.linalg.inv(self.Ll) @ dphi * 1e-9
        return np.kron([1, -1], gradient)

    def potential_hessian(self, submode):
        hessian = self.l * np.linalg.inv(self.Ll) * 1e-9
        return np.kron([[1, -1], [-1, 1]], hessian)

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def __init__(self, n=2, l=None, ll=None, cl=None, rl=None, gl=None, name='', num_modes=10,
                 cutoff=None):
        super().__init__('TL', name)
        self.n = n
        self.l = l
        self.Ll = ll
        self.Cl = cl
        self.Rl = rl
        self.Gl = gl
        if cutoff is not None and np.isfinite(cutoff):
            cl_min = 1 / np.sqrt(np.linalg.norm(ll @ cl, ord=2))  # minimum speed of light in TL
            df = cl_min / (2 * l)
            num_modes = int(cutoff // df)
            if num_modes < 2:
                num_modes = 2
            self.num_modes = num_modes
        else:
            self.num_modes = num_modes

    def __repr__(self):
        return "TL {} (n={})".format(self.name, self.n)


class JosephsonJunctionChain(TLSystemElement):
    """
    JosephsonJunction is a nonlinear element with energy E = E_J(1 − cos φ).
    However, in approximation it can be represented as element with linear inductance L_J = Φ_0 / (2 pi I_c),
    where I_c is a critical current.
    """

    def num_terminals(self):
        return 2

    def num_degrees_of_freedom(self):
        return 0

    def boundary_condition(self, omega):
        return np.asarray([[1, -1, 1j * omega * self.L_lin(), 0], [0, 0, 1, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, -self.L_lin(), 0], [0, 0, 0, 0]])  # derivatives
        a = np.asarray([[1, -1, 0, 0], [0, 0, 1, 1]])  # current values
        return a, b

    def energy_matrix(self):
        energy = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.L_lin(), 0],
            [0, 0, 0, 0]
        ]) / 2
        return energy

    def get_inv_inductance_matrix(self):
        return np.asarray([[1 / self.L_lin(), - 1 / self.L_lin()], [- 1 / self.L_lin(), 1 / self.L_lin()]])

    def potential(self, submode):
        return self.E_J * (1 - np.cos((submode[0] - submode[1]) / self.n_junctions)) / (
                hbar / (2 * e)) ** 2 * 1e-9 * self.n_junctions

    def potential_gradient(self, submode):
        gradient = self.E_J * np.sin((submode[0] - submode[1]) / self.n_junctions) / (hbar / (2 * e)) ** 2 * 1e-9
        return gradient * np.asarray([1, -1])

    def potential_hessian(self, submode):
        hessian = self.E_J * np.cos((submode[0] - submode[1]) / self.n_junctions) / (
                hbar / (2 * e)) ** 2 * 1e-9 / self.n_junctions
        return hessian * np.asarray([[1, -1], [-1, 1]])

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def hob_phi_op(self, submode, num_levels):
        """
        Returns the matrix representation in harmonic oscillator basis of the phi operator across the junction for a given mode
        :param submode: mode voltages and currents
        :param num_levels: number of levels in harmonic oscillator basis
        :return:
        """
        prefactor = self.L_lin() / self.phi_0 * submode[2] / np.sqrt(2)
        sqrt_n = np.diag(np.sqrt(np.arange(num_levels)))
        operator = np.zeros(sqrt_n.shape, complex)
        operator[:-1, :] += prefactor * sqrt_n[1:, :]
        operator[:, :-1] += np.conj(prefactor) * sqrt_n[:, 1:]
        return operator

    def nonlinear_perturbation(self):
        p = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        Phi_0 = self.phi_0 * (2 * np.pi)
        v = - self.E_J / 4 * ((2 * np.pi / Phi_0) * self.L_lin()) ** 4 * np.kron(p, p) / (self.n_junctions ** 2)
        return v

    def nonlinearity_in_taylor_expansion(self):
        m = np.asarray([[1, -1],
                        [-1, 1]])
        p = np.kron(m, m)

        Phi_0 = self.phi_0 * (2 * np.pi)
        v = - self.E_J / 24 * (2 * np.pi / Phi_0) ** 4 * p / (self.n_junctions ** 2) * 6
        return v

    def nonlinear_perturbation4(self, mode1, mode2, mode3, mode4):
        v = - self.L_lin() ** 3 / self.phi_0 ** 2 * mode1[2] * mode2[2] * mode3[2] * mode4[2] / 4 / self.n_junctions
        return v

    def __init__(self, e_j=None, n_junctions=1, name=''):
        super().__init__('JJ', name)
        self.E_J = e_j
        self.n_junctions = n_junctions

        self.phi_0 = hbar / (2 * e)  # reduced flux quantum
        self.stationary_phase = 0

    def L_lin(self):
        return self.phi_0 ** 2 / (self.E_J * np.cos(self.stationary_phase)) * self.n_junctions  # linear part of JJ

    def set_stationary_phase(self, phase):
        self.stationary_phase = phase


JosephsonJunction = JosephsonJunctionChain


class TLSystem:
    def __init__(self):
        self.nodes = []  # list of all nodes [0,1,2...]
        self.elements = []  # list of all elements [<transmission_line_simulator_new.name_of_element>, ...]
        self.node_multiplicity = {}  # dictionary of node's multiplicity {node1: node1's multiplicity, node2: node2's multiplicity, ...}
        self.terminal_node_mapping = []  # list of terminals's nodes [terminal1's nodes=[], node2's multiplicity=[], ...]
        self.dof_mapping = []  # ???

        self.JJs = []  # all nonlinear elements

    def add_element(self, element, nodes):
        self.elements.append(element)

        if element.type_ == 'JJ':
            self.JJs.append(element)

        for node in nodes:
            if node not in self.nodes:
                self.node_multiplicity[node] = 0
                self.nodes.append(node)
            self.node_multiplicity[node] += 1
        self.terminal_node_mapping.append(nodes)
        return

    def grounded_node_ind(self):
        gnd_ind = []
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ in ['Port', 'Short']:
                voltages = self.get_element_dofs(elem)[0]
                for v in voltages:
                    if v not in gnd_ind:
                        gnd_ind.append(v)
        return gnd_ind

    def capacitance_matrix(self):
        """
        Create capacitance matrix in basis (node voltages: V, voltages degrees of freedom: v)
        """
        # number of nodes = number of voltages
        # dofs_nodes, grounded_nodes = self.dofs_of_hamiltonian()
        # number of nodes
        # node_no = len(dofs_nodes)
        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        # kinetic dofs = voltages + coefficients in taylor expansion for voltage
        kinetic_dofs = node_no + int(internal_dof_no / 2)
        capacitance_matrix = np.zeros((kinetic_dofs, kinetic_dofs), dtype=complex)

        internal_dof_offset = node_no
        for elem_id, elem in enumerate(self.elements):
            cap_mat = elem.get_capacitance_matrix()
            no_kinetic_internal_dofs = int(elem.num_degrees_of_freedom_dynamic() / 2)
            if type(cap_mat) is not int:
                elem_terminals = self.terminal_node_mapping[elem_id]
                # fill nodes voltages degrees of freedom
                for node1_id, node1 in enumerate(elem_terminals):
                    for node2_id, node2 in enumerate(elem_terminals):
                        ind1 = self.nodes.index(node1)
                        ind2 = self.nodes.index(node2)
                        capacitance_matrix[ind1][ind2] += cap_mat[node1_id][node2_id]
                if no_kinetic_internal_dofs > 0:
                    for dof1 in range(no_kinetic_internal_dofs):
                        for dof2 in range(no_kinetic_internal_dofs):
                            capacitance_matrix[internal_dof_offset + dof1][internal_dof_offset + dof2] += \
                                cap_mat[len(elem_terminals) + dof1][len(elem_terminals) + dof2]
                    internal_dof_offset += no_kinetic_internal_dofs

        return capacitance_matrix

    def inv_inductance_matrix(self, jj_lin=True):
        """
        Create capacitance matrix in basis (node currents: I, currents degrees of freedom: i)
        """
        # number of nodes = number of voltages
        # dofs_nodes, grounded_nodes = self.dofs_of_hamiltonian()
        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        # kinetic dofs = voltages + coefficients in taylor expansion for voltage
        potential_dofs = node_no + int(internal_dof_no / 2)
        inv_inductance_matrix = np.zeros((potential_dofs, potential_dofs), dtype=complex)

        internal_dof_offset = node_no
        for elem_id, elem in enumerate(self.elements):
            inv_ind_mat = elem.get_inv_inductance_matrix()
            no_potential_internal_dofs = int(elem.num_degrees_of_freedom_dynamic() / 2)
            if not jj_lin:
                if elem.type_ == 'JJ':
                    inv_ind_mat = 0
            if type(inv_ind_mat) is not int:
                elem_terminals = self.terminal_node_mapping[elem_id]
                # fill nodes currents degrees of freedom
                for node1_id, node1 in enumerate(elem_terminals):
                    for node2_id, node2 in enumerate(elem_terminals):
                        ind1 = self.nodes.index(node1)
                        ind2 = self.nodes.index(node2)
                        inv_inductance_matrix[ind1][ind2] += inv_ind_mat[node1_id][node2_id]

                if no_potential_internal_dofs > 0:
                    for dof1 in range(no_potential_internal_dofs):
                        for dof2 in range(no_potential_internal_dofs):
                            inv_inductance_matrix[internal_dof_offset + dof1][internal_dof_offset + dof2] += \
                                inv_ind_mat[len(elem_terminals) + dof1][len(elem_terminals) + dof2]
                    internal_dof_offset += no_potential_internal_dofs
        return inv_inductance_matrix

    def mode_vector_indices(self):
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        # kinetic dofs = voltages + coefficients in taylor expansion for voltage
        kinetic_dofs = node_no + int(internal_dof_no / 2)
        indexes = np.zeros(kinetic_dofs, dtype=int)

        # fill indexes corresponding to node voltages
        for i in range(node_no):
            indexes[i] = i

        # fill indexes corresponding to internal degrees of freedom
        internal_dof_offset = node_no
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ == 'TL':
                internal_dofs = self.get_element_dofs(elem)[2]
                for ind_id, ind in enumerate(internal_dofs[: int(len(internal_dofs) / 2)]):
                    indexes[internal_dof_offset + ind_id] = ind
                internal_dof_offset += int(len(internal_dofs) / 2)
        return indexes

    def phases_mode_vector(self, mode):
        """
        Returns vector voltages corresponded to nodes degrees of freedom and coefficients in Taylor series
        :param mode:
        :param dc_dofs: if dc_dofs=True dc degrees of freedom represented as dofs with constant value of Phi (withot vibrations)
        """
        indexes = list(self.mode_vector_indices())
        return mode[indexes].reshape(len(indexes), 1)

    def transition_matrix(self, modes):
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        potential_dofs = node_no + int(internal_dof_no / 2)
        u_matrix = np.zeros((potential_dofs, len(modes)), dtype=complex)
        for mode_id, mode in enumerate(modes):
            u_matrix[:, mode_id] = self.phases_mode_vector(mode).T
        return u_matrix

    def currents_mode_vector(self, omega, mode):
        voltage_vector = self.phases_mode_vector(mode)
        currents_vector = 1j * omega * self.capacitance_matrix() @ voltage_vector
        return currents_vector

    def harmonic_mode_constants(self, mode, i_dc, jj=False):
        p_ = self.phases_mode_vector(mode)

        # transition for quadratic form
        capacitance = np.real(np.conj(p_).T @ self.capacitance_matrix() @ p_)
        inv_inductance = np.real(np.conj(p_).T @ self.inv_inductance_matrix(jj_lin=jj) @ p_)

        # transition for liner form
        i = np.real(i_dc.T @ p_)
        # o = 1 / np.sqrt(capacitance / inv_inductance) / (2 * np.pi)
        # print('Omega', o / 1e9, 'GHz')
        phi_0 = hbar / (2 * e)
        e_c = e ** 2 / (2 * capacitance)
        e_l = phi_0 ** 2 * inv_inductance
        return e_c, e_l, i * phi_0

    def get_psi_dc(self, mode, i_dc):
        e_c, e_l, i = self.harmonic_mode_constants(mode, i_dc, jj=True)
        return i / e_l

    def get_phi_dc(self, modes, i_dc):
        phi_dc = []
        for mode_id, mode in enumerate(modes):
            phi_dc_i = self.get_psi_dc(mode, i_dc)
            phi_dc.append(phi_dc_i)
        dim = len(phi_dc)
        u_matrix = self.transition_matrix(modes)
        return u_matrix @ np.asarray(phi_dc).reshape(dim, 1)

    def map_dofs(self):
        # count nodes
        self.dof_mapping = [n for n in self.nodes]  # nodal voltages
        self.dof_mapping_dynamic = [n for n in self.nodes]  # nodal voltages
        # currents incident into each terminal
        self.dof_mapping.extend(
            [(e_id, p_id) for e_id, e in enumerate(self.elements) for p_id in range(e.num_terminals())])
        self.dof_mapping_dynamic.extend(
            [(e_id, p_id) for e_id, e in enumerate(self.elements) for p_id in range(e.num_terminals())])
        # number of element-internal degrees of freedom
        self.dof_mapping.extend([(e_id, int_dof_id) for e_id, e in enumerate(self.elements) for int_dof_id in
                                 range(e.num_degrees_of_freedom())])
        self.dof_mapping_dynamic.extend([(e_id, int_dof_id) for e_id, e in enumerate(self.elements) for int_dof_id in
                                         range(e.num_degrees_of_freedom_dynamic())])

    def get_modes(self):
        """
        Return the eigenenergies and eigenfunctions (field distributions) of the linear eigenmodes of the TL.
        Removes all infinite-frequency modes, zero-frequency modes and modes with low Q-factor.
        :return:
        """
        from scipy.linalg import eig
        a, b = self.create_dynamic_equation_matrices()

        w, v = eig(a, b)
        modes = []
        frequencies = []
        gammas = []
        mode_ind = []

        for state_id in range(len(w)):
            e = np.imag(w[state_id])
            gamma = -np.real(w[state_id])
            if e <= 0 or not np.isfinite(e):
                continue
            # modes.append((e, gamma, v[:, state_id]))
            frequencies.append(e)
            gammas.append(gamma)
            modes.append(v[:, state_id])
            mode_ind.append(state_id)

        order = np.argsort(frequencies)
        return np.asarray(frequencies)[order], np.asarray(gammas)[order], np.asarray(modes)[order]

    ###################################################################################################################
    # DC
    ###################################################################################################################
    def dc_dofs(self):
        """
        DC nodes for stationary phases dofs
        """
        dc_nodes = []
        dc_indices = []
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ == 'Port':
                if elem.dc_type:
                    dc_nodes.extend(self.terminal_node_mapping[elem_id])
                    dc_node_ind = self.nodes.index(self.terminal_node_mapping[elem_id][0])
                    dc_indices.append(dc_node_ind)
        return dc_nodes, dc_indices

    def get_scdc_subsystems(self):
        # scdc_nodes = self.get_scdc_nodes()
        unbound_scdc_nodes = self.get_scdc_nodes()
        bound_scdc_nodes = []
        subsystems = []
        shorts = []
        for element_id, element in enumerate(self.elements):
            if type(element) is Short:
                shorts.extend(self.terminal_node_mapping[element_id])
        unbound_scdc_nodes = list(set(unbound_scdc_nodes).difference(set(shorts)))

        while len(unbound_scdc_nodes):
            subsystem_nodes = [unbound_scdc_nodes[0]]
            subsystem_elements = []
            bound_scdc_nodes.append(unbound_scdc_nodes[0])
            del unbound_scdc_nodes[0]
            elements_found = True
            while elements_found:
                elements_found = False
                for element, connections in zip(self.elements, self.terminal_node_mapping):
                    if not element.is_scdc():
                        continue
                    for connection in connections:
                        if connection in subsystem_nodes:
                            if connection not in shorts:
                                subsystem_nodes = list(set(subsystem_nodes + connections))
                            if element not in subsystem_elements:
                                if connection not in shorts or type(element) is Short:
                                    subsystem_elements.append(element)
                                    elements_found = True
                                # print ('found ', element, connections)
                            bound_scdc_nodes = list(set(bound_scdc_nodes + subsystem_nodes))
                            unbound_scdc_nodes = list(set(unbound_scdc_nodes).difference(set(subsystem_nodes)))
            nodes_no_shorts = [node for node in subsystem_nodes if node not in shorts]
            nodes_shorts = [node for node in subsystem_nodes if node in shorts]
            subsystems.append((nodes_no_shorts, nodes_shorts, subsystem_elements))
        return subsystems

    def get_scdc_nodes(self):
        nodes = []
        for element, connections in zip(self.elements, self.terminal_node_mapping):
            if element.is_scdc():
                nodes.extend(connections)
        return list(set(nodes))

    def get_scdc_elements(self):
        return [e for e in self.elements if e.is_scdc()]

    def set_phases(self, state, subsystem_id):
        scdc_subnodes, scdc_shorts, scdc_subelements = self.get_scdc_subsystems()[subsystem_id]
        for e_id, e in enumerate(self.elements):
            if e not in scdc_subelements:
                continue
            if hasattr(e, 'set_stationary_phase'):
                if self.terminal_node_mapping[e_id][1] in scdc_shorts:
                    phase1 = 0
                else:
                    phase1 = state[scdc_subnodes.index(self.terminal_node_mapping[e_id][1])]

                if self.terminal_node_mapping[e_id][0] in scdc_shorts:
                    phase0 = 0
                else:
                    phase0 = state[scdc_subnodes.index(self.terminal_node_mapping[e_id][0])]
                phase = phase1 - phase0
                e.set_stationary_phase(phase)

    def scdc_energy(self, state, subsystem_id):
        # number of nodes
        scdc_subnodes, scdc_shorts, scdc_subelements = self.get_scdc_subsystems()[subsystem_id]
        energy = 0

        # work with decompressed state
        for e_id, e in enumerate(self.elements):
            if e not in scdc_subelements:
                continue
            element_state = np.zeros((e.num_terminals(),))
            for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                if terminal_node not in scdc_shorts:
                    node_id = scdc_subnodes.index(terminal_node)
                    element_state[terminal_id] = state[node_id]

            energy += e.potential(element_state)

        return energy

    def scdc_energy_gradient(self, state, subsystem_id):
        # number of nodes
        scdc_subnodes, scdc_shorts, scdc_subelements = self.get_scdc_subsystems()[subsystem_id]
        energy_gradient = np.zeros(len(scdc_subnodes))

        for e_id, e in enumerate(self.elements):
            if e not in scdc_subelements:
                continue
            element_state = np.zeros((e.num_terminals(),))
            for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                if terminal_node not in scdc_shorts:
                    node_id = scdc_subnodes.index(terminal_node)
                    element_state[terminal_id] = state[node_id]

            element_gradient = np.asarray(e.potential_gradient(element_state))
            matrix_indeces = [scdc_subnodes.index(node_id) for node_id in self.terminal_node_mapping[e_id] if
                              node_id not in scdc_shorts]
            submatrix_indeces = [i for i in range(len(self.terminal_node_mapping[e_id])) if
                                 self.terminal_node_mapping[e_id][i] not in scdc_shorts]
            energy_gradient[matrix_indeces] += element_gradient[submatrix_indeces]

        return energy_gradient

    def scdc_energy_hessian(self, state, subsystem_id):
        # number of nodes
        scdc_subnodes, scdc_shorts, scdc_subelements = self.get_scdc_subsystems()[subsystem_id]
        energy_hessian = np.zeros((len(scdc_subnodes), len(scdc_subnodes)))

        for e_id, e in enumerate(self.elements):
            if e not in scdc_subelements:
                continue
            element_state = np.zeros((e.num_terminals(),))
            for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                if terminal_node not in scdc_shorts:
                    node_id = scdc_subnodes.index(terminal_node)
                    element_state[terminal_id] = state[node_id]

            element_hessian = np.asarray(e.potential_hessian(element_state))
            matrix_indeces = [scdc_subnodes.index(node_id) for node_id in self.terminal_node_mapping[e_id] if
                              node_id not in scdc_shorts]
            submatrix_indeces = [i for i in range(len(self.terminal_node_mapping[e_id])) if
                                 self.terminal_node_mapping[e_id][i] not in scdc_shorts]
            if len(matrix_indeces):
                energy_hessian[np.meshgrid(matrix_indeces, matrix_indeces)] += element_hessian[
                    np.meshgrid(submatrix_indeces, submatrix_indeces)]

        return energy_hessian

    def set_scdc_subsystem_phases(self, subsystems):
        # subsystems = self.get_scdc_subsystems()
        for subsystem_id, subsystem in enumerate(subsystems):
            scdc_subnodes, scdc_shorts, scdc_subelements = subsystem
            initial = np.zeros(len(scdc_subnodes))
            solution = minimize(lambda x: self.scdc_energy(x, subsystem_id), initial,
                                jac=lambda x: self.scdc_energy_gradient(x, subsystem_id),
                                hess=lambda x: self.scdc_energy_hessian(x, subsystem_id))
            self.set_phases(solution.x, subsystem_id)

    def scdc_stationary_phases(self):
        """
        Returns vector of stationary phases phi_stationary
        """
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        potential_dofs_no = node_no + int(internal_dof_no / 2)

        stationary_phases = np.zeros((potential_dofs_no, 1))
        # calculate subsystems parameters
        subsystems = self.get_scdc_subsystems()

        for subsystem_id, subsystem in enumerate(subsystems):
            scdc_subnodes, scdc_shorts, scdc_subelements = subsystem
            initial = np.zeros(len(scdc_subnodes))
            solution = minimize(lambda x: self.scdc_energy(x, subsystem_id), initial,
                                jac=lambda x: self.scdc_energy_gradient(x, subsystem_id),
                                hess=lambda x: self.scdc_energy_hessian(x, subsystem_id))

            for node_id, node in enumerate(scdc_subnodes):
                node_ind = self.nodes.index(node)
                stationary_phases[node_ind] = solution.x[node_id]
        return stationary_phases

    def i_dc(self):
        """
        Returns I DC vector
        """
        node_no = len(self.nodes)
        # number of internal dofs from TL
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        potential_dofs_no = node_no + int(internal_dof_no / 2)
        i_dc = np.zeros((potential_dofs_no, 1))
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ == 'Port':
                port_nodes = self.terminal_node_mapping[elem_id]
                for node in port_nodes:
                    i_dc[self.nodes.index(node)][0] = elem.idc
        return i_dc

    # def scdc_diagonalization(self):
    #     """
    #     Returns Phi_DC
    #     """
    #     from numpy.linalg import solve
    #     matrix_a = 1 / 2 * (self.inv_inductance_matrix(jj_lin=False).T + self.inv_inductance_matrix(jj_lin=False))
    #     b_vector = self.i_dc()
    #     phi_dc = solve(a=matrix_a, b=b_vector)
    #     return phi_dc

    def get_element_dc_phase(self, elem: TLSystemElement, dc_phase):
        voltages_indices = self.get_element_dofs(elem)[0]
        dc_submode = []
        for ind in voltages_indices:
            dc_submode.append(dc_phase[ind])
        return np.asarray(dc_submode).reshape(len(voltages_indices), 1)

    # def scdc_equilibrium_stationary_phase(self):
    #     """
    #     This method defines phi_dc (a vector of equilibrium DC phases)
    #     """
    #     dc_nodes, dc_indices = self.dc_dofs()
    #
    #     node_no = len(self.nodes)
    #     # number of internal dofs from TL
    #     # internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
    #     internal_dof_no = 0
    #     # potential dofs = phases + coefficients in taylor expansion for phases
    #     potential_dofs_no = node_no + int(internal_dof_no / 2)
    #
    #     dc_phases = np.zeros((potential_dofs_no, 1))
    #
    #     # calculate subsystems parameters
    #     subsystems = self.get_scdc_subsystems()
    #     for subsystem_id, subsystem in enumerate(subsystems):
    #         scdc_subnodes, scdc_shorts, scdc_subelements = subsystem
    #         initial = np.zeros(len(scdc_subnodes))
    #         solution = minimize(lambda x: self.scdc_energy(x, subsystem_id), initial,
    #                             jac=lambda x: self.scdc_energy_gradient(x, subsystem_id),
    #                             hess=lambda x: self.scdc_energy_hessian(x, subsystem_id))
    #         for node_id, node in enumerate(scdc_subnodes):
    #             if node in dc_nodes:
    #                 node_ind = self.nodes.index(node)
    #                 dc_phases[node_ind][0] = solution.x[node_id]
    #     return dc_phases

    # def get_jj_elem_stationary_subphase(self, jj_elem, stationary_phase):
    #     voltages = self.get_element_dofs(jj_elem)[0]
    #     subphase = []
    #     vector_dim = len(voltages)
    #     for i in voltages:
    #         subphase.append(stationary_phase[i])
    #     return np.reshape(np.asarray(subphase), (vector_dim, 1))
    #
    # def get_jj_elem_submode(self, jj_elem, mode):
    #     dc_nodes, dc_indices = self.dc_dofs()
    #     voltages = self.get_element_dofs(jj_elem)[0]
    #     submode = []
    #     vector_dim = len(voltages)
    #     for i in voltages:
    #         if i in dc_indices:
    #             submode.append(0)
    #         else:
    #             submode.append(mode[i])
    #     return np.reshape(np.asarray(submode), (vector_dim, 1))

    ###################################################################################################################
    # Boundary matrix, dynamic matrix and energy
    ###################################################################################################################
    def get_element_energies_from_dynamic(self, state):
        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements)
        # number of terminals
        terminal_no = np.sum(e.num_terminals() for e in self.elements)

        dynamic_equation_no = terminal_no + internal_dof_no
        # kinetic equations are Kirchhof's law that the sum of nodal currents is zero
        kinetic_equation_no = node_no
        num_equations = dynamic_equation_no + kinetic_equation_no

        # todo: build energy and dissipation rate matrices
        e = np.zeros((num_equations, num_equations))
        p = np.zeros((num_equations, num_equations))

        current_offset = 0
        internal_dof_offset = 0
        energies = []
        for e_id, e in enumerate(self.elements):
            element_state_size = 2 * e.num_terminals() + e.num_degrees_of_freedom_dynamic()
            element_state = np.zeros((element_state_size,), dtype=complex)
            for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                node_id = self.nodes.index(terminal_node)
                element_state[terminal_id] = state[node_id]
                element_state[terminal_id + e.num_terminals()] = state[node_no + current_offset + terminal_id]
            for internal_dof_id in range(e.num_degrees_of_freedom_dynamic()):
                element_state[2 * e.num_terminals() + internal_dof_id] = state[
                    node_no + terminal_no + internal_dof_offset + internal_dof_id]
            internal_dof_offset += e.num_degrees_of_freedom_dynamic()
            current_offset += e.num_terminals()
            energies.append(e.energy(element_state))

        return energies

    def create_dynamic_equation_matrices(self):
        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements)
        # number of terminals
        terminal_no = np.sum(e.num_terminals() for e in self.elements)

        # dynamic equations reflect the element's IV characteristic
        dynamic_equation_no = terminal_no + internal_dof_no
        # kinetic equations are Kirchhof's law that the sum of nodal currents is zero
        kinetic_equation_no = node_no

        num_equations = dynamic_equation_no + kinetic_equation_no

        dynamic_equation_matrix_a = np.zeros((num_equations, num_equations), dtype=float)
        dynamic_equation_matrix_b = np.zeros((num_equations, num_equations), dtype=float)

        # filling dynamic equations
        equation_id = 0
        current_offset = 0
        internal_dof_offset = 0
        for e_id, e in enumerate(self.elements):
            equations_a, equations_b = e.dynamic_equations()
            for element_equation_id in range(equations_b.shape[0]):
                equation_b = equations_b[element_equation_id, :]
                equation_a = equations_a[element_equation_id, :]
                for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                    node_id = self.nodes.index(terminal_node)
                    dynamic_equation_matrix_a[equation_id, node_id] = equation_a[terminal_id]  # nodal voltages
                    dynamic_equation_matrix_a[equation_id, node_no + current_offset + terminal_id] = equation_a[
                        terminal_id + e.num_terminals()]  # nodal current
                    dynamic_equation_matrix_b[equation_id, node_id] = equation_b[terminal_id]  # nodal voltages
                    dynamic_equation_matrix_b[equation_id, node_no + current_offset + terminal_id] = equation_b[
                        terminal_id + e.num_terminals()]  # nodal current
                for internal_dof_id in range(e.num_degrees_of_freedom_dynamic()):
                    dynamic_equation_matrix_a[
                        equation_id, node_no + terminal_no + internal_dof_offset + internal_dof_id] = equation_a[
                        2 * e.num_terminals() + internal_dof_id]
                    dynamic_equation_matrix_b[
                        equation_id, node_no + terminal_no + internal_dof_offset + internal_dof_id] = equation_b[
                        2 * e.num_terminals() + internal_dof_id]
                equation_id += 1
            internal_dof_offset += e.num_degrees_of_freedom_dynamic()
            current_offset += e.num_terminals()

        full_terminal_id = 0
        # filling kinetic equations
        for e_id, e in enumerate(self.elements):
            for terminal_id, node in enumerate(self.terminal_node_mapping[e_id]):
                dynamic_equation_matrix_a[dynamic_equation_no + self.nodes.index(node), node_no + full_terminal_id] = 1
                full_terminal_id += 1
        return dynamic_equation_matrix_a, dynamic_equation_matrix_b

    def create_boundary_problem_matrix(self, omega):
        # full dof number
        self.map_dofs()
        num_dof = len(self.dof_mapping)

        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom() for e in self.elements)
        # number of terminals
        terminal_no = np.sum(e.num_terminals() for e in self.elements)

        # dynamic equations reflect the element's IV characteristic
        dynamic_equation_no = terminal_no + internal_dof_no
        # kinetic equations are Kirchhof's law that the sum of nodal currents is zero
        kinetic_equation_no = node_no

        num_equations = dynamic_equation_no + kinetic_equation_no

        boundary_condition_matrix = np.zeros((num_equations, num_dof), dtype=complex)

        # filling dynamic equations
        equation_id = 0
        current_offset = 0
        internal_dof_offset = 0
        for e_id, e in enumerate(self.elements):
            equations = e.boundary_condition(omega)
            for element_equation_id in range(equations.shape[0]):
                equation = equations[element_equation_id, :]
                for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                    node_id = self.nodes.index(terminal_node)
                    boundary_condition_matrix[equation_id, node_id] = equation[terminal_id]  # nodal voltages
                    boundary_condition_matrix[equation_id, node_no + current_offset + terminal_id] = equation[
                        terminal_id + e.num_terminals()]  # nodal current
                for internal_dof_id in range(e.num_degrees_of_freedom()):
                    boundary_condition_matrix[
                        equation_id, node_no + terminal_no + internal_dof_offset + internal_dof_id] = equation[
                        2 * e.num_terminals() + internal_dof_id]
                equation_id += 1
            internal_dof_offset += e.num_degrees_of_freedom()
            current_offset += e.num_terminals()

        full_terminal_id = 0
        # filling kinetic equations
        for e_id, e in enumerate(self.elements):
            for terminal_id, node in enumerate(self.terminal_node_mapping[e_id]):
                boundary_condition_matrix[dynamic_equation_no + self.nodes.index(node), node_no + full_terminal_id] = 1
                full_terminal_id += 1
        return boundary_condition_matrix

    def get_element_dofs(self, element: TLSystemElement, dynamic=True):
        self.map_dofs()
        e_id = self.elements.index(element)
        voltages = [self.dof_mapping[:len(self.nodes)].index(t) for t in self.terminal_node_mapping[e_id]]

        terminal_no = np.sum(e.num_terminals() for e in self.elements)

        current_variables = self.dof_mapping[len(self.nodes):len(self.nodes) + terminal_no]
        internal_dof_variables_dynamic = self.dof_mapping_dynamic[len(self.nodes) + terminal_no:]
        internal_dof_variables = self.dof_mapping[len(self.nodes) + terminal_no:]

        currents = [current_variables.index((e_id, p_id)) + len(self.nodes) for p_id in range(element.num_terminals())]
        if not dynamic:
            degrees_of_freedom = [internal_dof_variables.index((e_id, dof_id)) + len(self.nodes) + terminal_no \
                                  for dof_id in range(element.num_degrees_of_freedom())]
        else:
            degrees_of_freedom = [internal_dof_variables_dynamic.index((e_id, dof_id)) + len(self.nodes) + terminal_no \
                                  for dof_id in range(element.num_degrees_of_freedom_dynamic())]
        return voltages, currents, degrees_of_freedom

    def get_element_dynamic_equations(self, element: TLSystemElement):
        e_id = self.elements.index(element)
        offset = np.sum(e.num_terminals() + e.num_degrees_of_freedom() for e in self.elements[:e_id])

        return np.arange(offset, offset + element.num_terminals() + element.num_degrees_of_freedom())

    def get_element_submode(self, element: TLSystemElement, mode):
        voltages, currents, degrees_of_freedom = self.get_element_dofs(element)
        submode = []
        vector_dim = len(voltages + currents + degrees_of_freedom)
        for i in voltages + currents + degrees_of_freedom:
            submode.append(mode[i])

        return np.reshape(np.asarray(submode), (vector_dim, 1))

    def element_energy(self, element: TLSystemElement, mode, mode2=None):
        submode_element = self.get_element_submode(element, mode)
        if mode2 is None:
            submode_element2 = submode_element
        else:
            submode_element2 = self.get_element_submode(element, mode2)

        e = element.energy_matrix()
        if type(e) is not int:
            submode_energy = np.conj(submode_element.T).squeeze() @ element.energy_matrix() @ submode_element2.squeeze()
        else:
            submode_energy = 0

        return submode_energy

    def cross_energy(self, mode1, mode2):
        energy = 0
        for elem in self.elements:
            energy += self.element_energy(elem, mode1, mode2)
        return energy

    def get_total_linear_energy(self, list_of_modes_numbers: list):
        """
        :param list_of_modes_numbers: list of integer numbers corresponding to mode number
        """
        omega, kappa, modes = self.get_modes()

        modes_ = [modes[m] for m in list_of_modes_numbers]

        modes_energies = []  # list of energies for different modes

        for mode_ in modes_:
            total_circuit_energy = 0
            for elem in self.elements:
                total_circuit_energy += self.element_energy(elem, mode_)

            modes_energies.append(total_circuit_energy)

        return modes_energies

    def normalization_of_modes(self, list_of_modes_numbers: list):
        """
        Calculate normalized modes to satisfy normalization condition: total energy of a mode equals to energy quantum
        of this mode.
        """
        omega, kappa, modes = self.get_modes()

        modes_ = [modes[m] for m in list_of_modes_numbers]
        normalized_modes = np.zeros((len(modes_), modes.shape[1]), dtype=complex)

        for m in list_of_modes_numbers:
            energy_quantum = hbar * omega[m]

            total_circuit_energy = 0
            for elem in self.elements:
                total_circuit_energy += self.element_energy(elem, modes[m])

            mode_energy = total_circuit_energy
            normalization_coeff = mode_energy / energy_quantum
            normalized_mode = modes[m] / np.sqrt(normalization_coeff)
            normalized_modes[m:] = normalized_mode

        return normalized_modes

    def get_perturbation(self, list_of_modes_numbers: list):
        """
        Calculate Kerr matrix
        """
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized
        number_of_modes = len(modes_)
        JJ_kerr = np.zeros((number_of_modes, number_of_modes))
        if self.JJs:
            for JJ_ in self.JJs:
                perturbation_matrix = np.zeros((number_of_modes, number_of_modes))
                for i in range(number_of_modes):
                    for j in range(number_of_modes):
                        mode_i = self.get_element_submode(JJ_, modes_[i])
                        mode_j = self.get_element_submode(JJ_, modes_[j])
                        submode_ij = np.kron(mode_i, mode_j)
                        if j == i:

                            perturbation_matrix[i][j] = np.dot(np.conj(submode_ij.T),
                                                               np.dot(JJ_.nonlinear_perturbation(),
                                                                      submode_ij)).real / 2
                        else:
                            perturbation_matrix[i][j] = np.dot(np.conj(submode_ij.T),
                                                               np.dot(JJ_.nonlinear_perturbation(),
                                                                      submode_ij)).real

                kerr_coefficients_matrix = perturbation_matrix / (hbar * 2 * np.pi)  # in Hz
                JJ_kerr += kerr_coefficients_matrix
        return JJ_kerr

    ###################################################################################
    # Cross-kerr and self-kerr couplings
    ###################################################################################
    def check_self_non_linearity(self, omegas, kerr_matrix, epsilon=0.0001):
        """
        Self non linearity shows type of mode: harmonic(or quasi harmonic) or anharmonic
        :param omegas: Hz
        :param kerr_matrix: Hz
        :param epsilon:
        """
        num_modes = len(omegas)
        dict_ = {'Anharmonic modes': [], 'Quasi harmonic modes': [], 'Harmonic modes': []}
        for mode in range(num_modes):
            anharmonicity = kerr_matrix[mode][mode]
            if not omegas[mode]:
                ratio = 0
            else:
                ratio = np.abs(anharmonicity / omegas[mode])
            if ratio > epsilon:
                dict_['Anharmonic modes'].append(mode)
                # print('Mode {} is anharmonic!'.format(mode))
            elif ratio == 0:
                dict_['Harmonic modes'].append(mode)
                # print('Mode {} is harmonic!'.format(mode))
            else:
                dict_['Quasi harmonic modes'].append(mode)
                # print('Mode {} is quasi harmonic'.format(mode))
        return dict_

    def check_cross_non_linearity(self, omegas, kerr_matrix, epsilon=0.01):
        """
        Cross non linearity shows how mode1 and mode2 correlated
        :param omegas
        :param kerr_matrix:
        :param epsilon:
        """
        quasi_independent_subspaces = []
        non_independent_modes = []
        num_modes = len(omegas)
        dict_ = {'Coupled modes': [], 'Uncoupled modes': []}
        for mode_i in range(num_modes):
            for mode_j in range(num_modes):
                if mode_i == mode_j:
                    continue
                if mode_j > mode_i:
                    continue
                chi_ij = kerr_matrix[mode_i][mode_j]
                if not omegas[mode_i]:
                    ratio_i = 0
                else:
                    ratio_i = np.abs(chi_ij / omegas[mode_i])
                if not omegas[mode_j]:
                    ratio_j = 0
                else:
                    ratio_j = np.abs(chi_ij / omegas[mode_j])

                if (ratio_i > epsilon) or (ratio_j > epsilon):
                    dict_['Coupled modes'].append([mode_i, mode_j])
                    quasi_independent_subspaces.append([mode_i, mode_j])
                    non_independent_modes.extend([mode_i, mode_j])
        for mode in range(num_modes):
            if mode not in non_independent_modes:
                dict_['Uncoupled modes'].append(mode)
                quasi_independent_subspaces.append([mode])
        return quasi_independent_subspaces

    def define_modes_parameters(self, omegas, modes, kerr_matrix, dc_phase, epsilon_cross=0.01, epsilon_self=0.001):
        """
        This methods calculate effective hamiltonian for uncoupled or coupled subsystems for all modes presended
        in the circuit
        :param omegas:
        :param modes:
        :param kerr_matrix:
        :param dc_phase:
        :param epsilon_cross:
        :param epsilon_self:
        """
        independent_subspaces = self.check_cross_non_linearity(omegas, kerr_matrix, epsilon_cross)
        dict_self = self.check_self_non_linearity(omegas, kerr_matrix, epsilon_cross)
        hamiltonian_parameters = dict.fromkeys([str(i) for i in range(len(independent_subspaces))])
        num_junction = len(self.JJs)
        for subspace_id, subspace in enumerate(independent_subspaces):
            num = len(subspace)
            subspace_dict = {'subsystem_id': subspace, 'Ec': np.zeros((num, num)), 'El': np.zeros((num, num)), 'Ej': [],
                             'alpha': np.zeros((num_junction, num)), 'dc_phase': np.zeros((num_junction, num))}
            for mode_id, mode in enumerate(subspace):
                # define harmonic constants of a mode
                e_c, e_l, i = self.harmonic_mode_constants(modes[mode], self.i_dc())
                subspace_dict['Ec'][mode_id][mode_id] = e_c
                subspace_dict['El'][mode_id][mode_id] = e_l
                if num == 1:
                    mode = subspace[0]
                    # check harmonicity of a mode
                    if (mode in dict_self['Quasi harmonic modes']) or (mode in dict_self['Harmonic modes']):
                        for jj in range(num_junction):
                            subspace_dict['Ej'].append(0)
                    else:
                        for jj_id, jj in enumerate(self.JJs):
                            subspace_dict['Ej'].append(jj.E_J)
                            jj_submode_i, jj_submode_j = self.get_element_submode(jj, modes[mode])[:2]
                            subspace_dict['alpha'][jj_id][mode_id] = np.real(jj_submode_i - jj_submode_j)
                            jj_dc_phase_i, jj_dc_phase_j = self.get_element_dc_phase(jj, dc_phase)
                            subspace_dict['dc_phase'][jj_id][mode_id] = np.real(jj_dc_phase_i - jj_dc_phase_j)
                else:
                    for jj_id, jj in enumerate(self.JJs):
                        subspace_dict['Ej'].append(jj.E_J)
                        jj_submode_i, jj_submode_j = self.get_element_submode(jj, modes[mode])[:2]
                        subspace_dict['alpha'][jj_id][mode_id] = np.real(jj_submode_i - jj_submode_j)
                        jj_dc_phase_i, jj_dc_phase_j = self.get_element_dc_phase(jj, dc_phase)
                        subspace_dict['dc_phase'][jj_id][mode_id] = np.real(jj_dc_phase_i - jj_dc_phase_j)
            hamiltonian_parameters[str(subspace_id)] = subspace_dict

        return hamiltonian_parameters

    ###################################################################################
    # Plot
    ###################################################################################

    def plot_potential_1d(self, subsystem, phi):
        """
        Plot in phase basis, U GHz
        """
        phi_grid = phi
        u_1d = potential_1d(phi=phi_grid, e_l=subsystem['El'], e_j=subsystem['Ej'],
                            alpha=subsystem['alpha'], phi_dc=subsystem['dc_phase'])
        plt.plot(phi_grid, u_1d / h / 1e9)
        plt.ylabel('Energy, GHz')
        plt.xlabel('$\\phi$')
        plt.show()

    def plot_potential_2d(self, num_system: int, phi_start: list = None, phi_stop: list = None,
                          num_points: list = None):
        if not phi_start:
            phi_start = [-np.pi, -np.pi]
        if not phi_stop:
            phi_stop = [np.pi, np.pi]
        if not num_points:
            num_points = [201, 201]
        parameters = self.define_modes_parameters()['subsystem' + ' ' + str(num_system)]
        phi_x = np.linspace(phi_start[0], phi_stop[0], num_points[0])
        phi_y = np.linspace(phi_start[1], phi_stop[1], num_points[1])
        xx, yy, u_2d = potential_2d(phi_1=phi_x, phi_2=phi_y, e_l=parameters['El'], e_j=parameters['Ej'],
                                    alpha=parameters['alpha'])
        pot_plot = plt.contourf(xx, yy, u_2d / h / 1e9)
        plt.colorbar(pot_plot)
        plt.show()
        pass

    def solve_hamiltonian_eig_1d(self, subsystem, phi_grid, cutoff=4):
        from scipy.linalg import eig
        d = np.abs(phi_grid[0] - phi_grid[1])  # step of grid
        u = potential_1d(phi=phi_grid, e_l=subsystem['El'], e_j=subsystem['Ej'],
                         alpha=subsystem['alpha'], phi_dc=subsystem['dc_phase'])
        # u = np.asarray(
        #     [potential_1d(phi=phi_i, e_l=subsystem['El'], e_j=subsystem['Ej'], alpha=subsystem['alpha'],
        #                   phi_0=subsystem['dc_phase']) for phi_i in
        #      phi_grid])
        a_x = - 4 * subsystem['Ec'][0][0]
        # create L operator
        operator_l = np.diag(u - 2 * a_x / d ** 2)
        for i in range(len(phi_grid) - 1):
            operator_l[i][i + 1] = a_x / d ** 2
        for i in range(1, len(phi_grid)):
            operator_l[i][i - 1] = a_x / d ** 2
        # print('l_operator', operator_l)
        eigenvalues, eigenvectors = eig(operator_l)
        order = np.argsort(eigenvalues)
        energies = np.asarray(np.real(eigenvalues))[order][:cutoff]
        wavefunctions = []
        for state_id in order[:cutoff]:
            wavefunctions.append(eigenvectors[:, state_id])
        return energies, wavefunctions

    def plot_wavefunctions_1d(self, subsystem, phi, cutoff=5):
        """
        Returns plotted potential and wavefunctions
        :param subsystem:
        :param phi: phase grid
        :param cutoff:
        """
        phi_grid = phi
        u_1d = potential_1d(phi=phi_grid, e_l=subsystem['El'], e_j=subsystem['Ej'],
                            alpha=subsystem['alpha'], phi_dc=subsystem['dc_phase'])
        energies, wavefunctions = self.solve_hamiltonian_eig_1d(subsystem, phi_grid, cutoff)

        plt.ylabel('Energy, GHz')
        plt.xlabel('$\\phi$')
        for energy_id, energy in enumerate(energies):
            plt.plot(phi_grid, (energy + 2 * energy * wavefunctions[energy_id]) / h / 1e9)
            plt.fill_between(phi_grid, energy / h / 1e9, (energy + 2 * energy * wavefunctions[energy_id]) / h / 1e9,
                             alpha=0.25)

        plt.plot(phi_grid, u_1d / h / 1e9)
        plt.show()

    def dc_energy_spectrum_1d(self, num_subsystem, dc_line: TLSystemElement, phi, currents, omegas, modes, kerr_matrix,
                              cutoff=4):
        """
        :param num_subsystem:
        :param dc_line:
        :param phi:
        :param currents:
        :param cutoff:
        """
        energies = np.zeros((len(currents), cutoff))
        for current_id, current in enumerate(currents):
            dc_line.idc = current
            dc_phase = self.get_phi_dc(modes, self.i_dc())
            modes_subsystems = self.define_modes_parameters(omegas, modes, kerr_matrix, dc_phase)
            E, wavefunctions = self.solve_hamiltonian_eig_1d(modes_subsystems[str(num_subsystem)], phi, cutoff)
            energies[current_id, :] = np.asarray(E)
        return currents, energies

    def dc_plot_energy_spectrum_1d(self, num_subsystem, dc_line: TLSystemElement, phi, currents, omegas, modes,
                                   kerr_matrix, cutoff=4):
        currents, energies = self.dc_energy_spectrum_1d(num_subsystem, dc_line, phi, currents, omegas, modes,
                                                        kerr_matrix, cutoff)
        for energy_level in range(cutoff):
            plt.plot(currents, (energies[:, energy_level]) / h / 1e9)
            plt.ylabel('Energy, GHz')
            plt.xlabel('$I_{DC}$')

    ###################################################################################
    # Second order perturbation analysis
    ###################################################################################
    def get_perturbation_nondiagonal(self, list_of_modes_numbers: list):
        """
        Calculate matrix of perturbation in basis of |g>, |e>, |f>, |h> states, truncate to n states
        """
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized

        number_of_modes = len(modes_)

        JJ_kerr = np.zeros((2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes))
        for JJ_ in self.JJs:
            for i in range(number_of_modes * 2):
                mode_i = self.get_element_submode(JJ_, modes_[i % number_of_modes])
                if i > number_of_modes:
                    mode_i = np.conj(mode_i)
                for j in range(number_of_modes * 2):
                    mode_j = self.get_element_submode(JJ_, modes_[j % number_of_modes])
                    if j > number_of_modes:
                        mode_j = np.conj(mode_j)
                    submode_ij = np.kron(mode_i, mode_j)
                    for k in range(number_of_modes * 2):
                        mode_k = self.get_element_submode(JJ_, modes_[k % number_of_modes])
                        if k > number_of_modes:
                            mode_k = np.conj(mode_k)
                        for l in range(number_of_modes * 2):
                            mode_l = self.get_element_submode(JJ_, modes_[l % number_of_modes])
                            if l > number_of_modes:
                                mode_l = np.conj(mode_l)
                            submode_kl = np.kron(mode_k, mode_l)
                            JJ_kerr[i, j, k, l] += np.dot(submode_ij.T,
                                                          np.dot(JJ_.nonlinear_perturbation(),
                                                                 submode_kl)).ravel()[0] / 6

        return JJ_kerr

    def get_second_order_perturbation(self, initial_state: list, list_of_modes_numbers: list):
        """
        Calculate second order correction to energy with perturbation operator, return this correction in Hz
        :param initial_state: state for which second order correction should be calculated
        :param list_of_modes_numbers:
        """

        from collections import defaultdict
        from itertools import product

        number_of_modes = len(list_of_modes_numbers)  # number of modes in the system

        omega, kappa, modes = self.get_modes()
        omegas = np.asarray([omega[m] for m in list_of_modes_numbers])

        # first order corrections for mode energies
        kerr_matrix = self.get_perturbation(list_of_modes_numbers)

        # create initial state of the system corresponding to M modes
        s = [i for i in range(1, number_of_modes + 1)]
        state = defaultdict(int)
        for k in s:
            state[k] = initial_state[k - 1]

        # energy of initial state (зочем)
        self_kerr_matrix = np.diag(kerr_matrix)
        cross_kerr_matrix = kerr_matrix - np.diag(self_kerr_matrix)

        kerr_correction1 = np.asarray(initial_state) @ self_kerr_matrix.T * 2
        kerr_correction2 = np.asarray(initial_state) @ sum(
            [cross_kerr_matrix[:][i] for i in range(number_of_modes)]).T / 2

        energy_initial_state = hbar * np.asarray(initial_state) @ omegas + hbar * 2 * np.pi * (
                kerr_correction1 + kerr_correction2)

        JJ_kerr = self.get_perturbation_nondiagonal2(list_of_modes_numbers) / 4

        operators = []
        for i in range(1, number_of_modes + 1):
            operators.extend([i, -i])

        # create a list with perturbation terms: 'n' -- creation operator of mode n,
        # '-n' -- annihilation operator of mode n

        perturbation_terms = []

        list_of_all_perturbation_terms = []  # all matrix elements

        for i in product(operators, repeat=4):
            perturbation_terms.append(i)

        for t in perturbation_terms:
            coefficient = 1

            for operator in t[::-1]:  # operators act in reverse order
                if np.sign(operator) == 1:

                    coefficient *= np.sqrt(state[np.abs(operator)] + 1)
                    state[np.abs(operator)] += 1

                elif np.sign(operator) == -1:

                    if state[np.abs(operator)] > 0:

                        coefficient *= np.sqrt(state[np.abs(operator)])
                        state[np.abs(operator)] -= 1

                    elif state[np.abs(operator)] == 0:

                        coefficient *= 0
                        state[np.abs(operator)] = 0
                        break

            indexes = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))
            J_kerr_ijkl = hbar * 2 * np.pi * JJ_kerr[indexes]

            final_state = list(state.values())
            list_of_all_perturbation_terms.append(np.asarray([coefficient * J_kerr_ijkl, final_state]))

            # clear state
            for k in s:
                state[k] = initial_state[k - 1]

        second_order_energy_correction = 0
        states_f = [s[1] for s in list_of_all_perturbation_terms]  # states for summation

        # remove initial state in states_i, because we can not sum up by f = |initial state>
        states_f = [s for s in states_f if s != initial_state]

        for f in states_f:
            kerr_correction1 = np.asarray(f) @ self_kerr_matrix.T * 2
            kerr_correction2 = np.asarray(f) @ sum(
                [cross_kerr_matrix[:][i] for i in range(number_of_modes)]).T / 2

            energy_final_state = hbar * np.asarray(f) @ omegas + hbar * 2 * np.pi * (
                    kerr_correction1 + kerr_correction2)

            all_matrix_elements = 0
            for term in list_of_all_perturbation_terms:
                matrix_elem = term[0]
                final_state = term[1]

                if f == final_state:
                    all_matrix_elements += matrix_elem
                else:
                    all_matrix_elements += 0

            numerator = np.abs(all_matrix_elements) ** 2
            energy_diff = energy_final_state - energy_initial_state
            second_order_energy_correction += numerator / energy_diff

        return second_order_energy_correction / (hbar * 2 * np.pi)

    def get_perturbation_nondiagonal2(self, list_of_modes_numbers: list):
        """
        Calculate matrix of perturbation in basis of |g>, |e>, |f>, |h> states, truncate to n states
        """
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized

        number_of_modes = len(modes_)

        JJ_kerr = np.zeros((2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes),
                           dtype=complex)
        for JJ_ in self.JJs:
            for i in range(number_of_modes * 2):
                mode_i = self.get_element_submode(JJ_, modes_[i % number_of_modes])
                if i >= number_of_modes:
                    mode_i = np.conj(mode_i)
                for j in range(number_of_modes * 2):
                    if j < i:
                        continue
                    mode_j = self.get_element_submode(JJ_, modes_[j % number_of_modes])
                    if j >= number_of_modes:
                        mode_j = np.conj(mode_j)
                    for k in range(number_of_modes * 2):
                        if k < j:
                            continue
                        mode_k = self.get_element_submode(JJ_, modes_[k % number_of_modes])
                        if k >= number_of_modes:
                            mode_k = np.conj(mode_k)
                        for l in range(number_of_modes * 2):
                            if l < k:
                                continue
                            mode_l = self.get_element_submode(JJ_, modes_[l % number_of_modes])
                            if l >= number_of_modes:
                                mode_l = np.conj(mode_l)

                            JJ_kerr[i, j, k, l] += JJ_.nonlinear_perturbation4(mode_i, mode_j, mode_k, mode_l) / 6

        return JJ_kerr / (hbar * 2 * np.pi)

    def get_second_order_perturbation_kerr2(self, list_of_modes_numbers: list, raw_frequencies=0):
        omega, kappa, modes = self.get_modes()
        omega_ = np.asarray([omega[m] for m in list_of_modes_numbers]) / (2 * np.pi)

        JJ_kerr = self.get_perturbation_nondiagonal2(list_of_modes_numbers) / 4

        first_order_kerr = self.get_perturbation(list_of_modes_numbers)
        kerr = np.zeros(first_order_kerr.shape)

        mode = [0 for i in range(len(list_of_modes_numbers))]
        ground_state_energy = self.get_second_order_perturbation3(mode, omega_, JJ_kerr, first_order_kerr,
                                                                  raw_frequencies)
        first_state_corrections = [0 for i in range(len(list_of_modes_numbers))]

        for mode_id in range(len(list_of_modes_numbers)):
            mode = [0 if i != mode_id else 1 for i in range(len(list_of_modes_numbers))]
            # corrected_energy = self.get_second_order_perturbation(mode, list_of_modes_numbers)
            # omega_corrected[mode_id] += first_order_kerr[mode_id, mode_id] + corrected_energy - ground_state_energy
            first_state_corrections[mode_id] = self.get_second_order_perturbation3(mode, omega_, JJ_kerr,
                                                                                   first_order_kerr, raw_frequencies)

        for mode1_id in range(len(list_of_modes_numbers)):
            for mode2_id in range(len(list_of_modes_numbers)):
                mode = [0 for i in range(len(list_of_modes_numbers))]
                mode[mode1_id] += 1
                mode[mode2_id] += 1
                corrected_energy = self.get_second_order_perturbation3(mode, omega_, JJ_kerr, first_order_kerr,
                                                                       raw_frequencies)
                kerr[mode1_id, mode2_id] = first_order_kerr[
                                               mode1_id, mode2_id] + corrected_energy + ground_state_energy - \
                                           first_state_corrections[mode1_id] - first_state_corrections[mode2_id]

        return kerr

    def get_second_order_perturbation2(self, initial_state: list, list_of_modes_numbers: list):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        omega, kappa, modes = self.get_modes()
        omega_ = np.asarray([omega[m] for m in list_of_modes_numbers]) / (2 * np.pi)
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized
        # first_order_kerr = self.get_perturbation(list_of_modes_numbers)
        # first_order_kerr = (first_order_kerr + np.diag(np.diag(first_order_kerr))) / 2
        JJ_kerr = self.get_perturbation_nondiagonal2(list_of_modes_numbers) / 4

        initial_state_ = np.asarray(initial_state)
        initial_state = tuple(initial_state)
        initial_state_energy = initial_state_ @ omega_  # + initial_state@first_order_kerr@initial_state

        number_of_modes = JJ_kerr.shape[0] // 2  # number of modes in the system

        operators = []
        for i in range(1, number_of_modes + 1):
            operators.extend([i, -i])

        final_states = defaultdict(lambda: 0.0)

        # create a list with perturbation terms: 'n' -- creation operator of mode n,
        # '-n' -- annihilation operator of mode n

        for t in product(operators, repeat=4):
            # print ('Operator: ', t)
            state = {k + 1: v for k, v in enumerate(initial_state)}

            matrix_element_factor = 1
            indeces = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))
            # print ('JJ kerr indeces: ', indeces)
            matrix_element_number = JJ_kerr[indeces]

            for operator in t[::-1]:
                # print('operator: ', operator)
                if np.sign(operator) == 1:
                    state[np.abs(operator)] += 1
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                elif np.sign(operator) == -1:
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                    state[np.abs(operator)] -= 1
                    if state[np.abs(operator)] < 0:
                        # print (matrix_element_factor)
                        break
                # print ('matrix element factor: ', matrix_element_factor)

            final_state_tuple = tuple([state[s] for s in range(1, len(initial_state) + 1)])
            final_states[final_state_tuple] += matrix_element_factor * matrix_element_number
            # print (state[1], matrix_element_factor)
            # print('Arrived in state: ', state,
            #      'got pre-factor: ', matrix_element_factor,
            #      'modal matrix element: ', matrix_element_number,
            #      'final state: ', final_states)
            # print (final_states)
        correction = 0
        for final_state, matrix_element in final_states.items():
            final_state_ = np.asarray(final_state)
            final_state_energy = final_state_ @ omega_  # + final_state_ @ first_order_kerr @ final_state_
            denominator = initial_state_energy - final_state_energy
            numerator = np.abs(matrix_element) ** 2
            correction_new = numerator / denominator
            # print('i: ', initial_state, 'f: ', final_state, 'E_i', initial_state_energy/1e9, 'E_f:', final_state_energy/1e9,
            #      'Vif:', np.abs(matrix_element)/1e9, 'correction:', correction_new/1e9)
            if final_state != initial_state:
                correction += correction_new

        return correction

    def get_second_order_perturbation3(self, initial_state: list, omega_, JJ_kerr, first_order_kerr=None,
                                       raw_frequencies=0):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        initial_state_ = np.asarray(initial_state)
        initial_state = tuple(initial_state)

        if raw_frequencies == 0:
            initial_state_energy = initial_state_ @ omega_  # + initial_state@first_order_kerr@initial_state
        elif raw_frequencies == 1:
            initial_state_energy = initial_state_ @ omega_ + 0.5 * initial_state_ @ first_order_kerr @ initial_state_

        number_of_modes = len(initial_state)  # number of modes in the system

        operators = []
        for i in range(1, number_of_modes + 1):
            operators.extend([i, -i])

        final_states = defaultdict(lambda: 0.0)

        # create a list with perturbation terms: 'n' -- creation operator of mode n,
        # '-n' -- annihilation operator of mode n

        for t in product(operators, repeat=4):
            # print ('Operator: ', t)
            state = {k + 1: v for k, v in enumerate(initial_state)}

            matrix_element_factor = 1
            indeces = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))
            # print ('JJ kerr indeces: ', indeces, JJ_kerr[indeces])
            matrix_element_number = JJ_kerr[indeces]

            for operator in t[::-1]:
                # print('operator: ', operator)
                if np.sign(operator) == 1:
                    state[np.abs(operator)] += 1
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                elif np.sign(operator) == -1:
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                    state[np.abs(operator)] -= 1
                    if state[np.abs(operator)] < 0:
                        # print (matrix_element_factor)
                        break
                # print ('matrix element factor: ', matrix_element_factor)

            final_state_tuple = tuple([state[s] for s in range(1, len(initial_state) + 1)])
            final_states[final_state_tuple] += matrix_element_factor * matrix_element_number
            # print (state[1], matrix_element_factor)
            # print('Arrived in state: ', state,
            #       'got pre-factor: ', matrix_element_factor,
            #       'modal matrix element: ', matrix_element_number,
            #       'final state: ', final_states)
            # print (final_states)
        correction = 0
        for final_state, matrix_element in final_states.items():
            final_state_ = np.asarray(final_state)
            if raw_frequencies == 0:
                final_state_energy = final_state_ @ omega_
            elif raw_frequencies == 1:
                final_state_energy = final_state_ @ omega_ + 0.5 * final_state_ @ first_order_kerr @ final_state_
            denominator = initial_state_energy - final_state_energy
            numerator = np.abs(matrix_element) ** 2
            correction_new = numerator / denominator
            # print('i: ', initial_state, 'f: ', final_state, 'E_i', initial_state_energy/1e9, 'E_f:', final_state_energy/1e9,
            #      'Vif:', np.abs(matrix_element)/1e9, 'correction:', correction_new/1e9)
            if final_state != initial_state:
                correction += correction_new

        return correction

    def get_perturbation_hamiltonian(self, modes: list, num_levels: list):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        omega, kappa, modes_ = self.get_modes()
        omega_ = np.asarray([omega[m] for m in modes]) / (2 * np.pi)

        jj_kerr = self.get_perturbation_nondiagonal2(modes) / 4

        number_of_modes = len(modes)  # number of modes in the system

        basis = [b for b in product(*[tuple([i for i in range(dof_levels)]) for dof_levels in num_levels])]
        # print ([tuple([i for i in range(dof_levels)]) for dof_levels in num_levels])
        # print (basis)
        ham = np.zeros((len(basis), len(basis)), complex)

        for initial_state_id, initial_state in enumerate(basis):
            # print ('initial state: ', initial_state)
            initial_state_ = np.asarray(initial_state)
            initial_state = tuple(initial_state)
            initial_state_energy = initial_state_ @ omega_

            ham[initial_state_id, initial_state_id] = initial_state_energy

            operators = []
            for i in range(1, number_of_modes + 1):
                operators.extend([i, -i])

            # create a list with perturbation terms: 'n' -- creation operator of mode n,
            # '-n' -- annihilation operator of mode n

            for t in product(operators, repeat=4):
                state = {k + 1: v for k, v in enumerate(initial_state)}
                # print (t)

                matrix_element_factor = 1
                indeces = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))

                for operator in t[::-1]:
                    if np.sign(operator) == 1:
                        state[np.abs(operator)] += 1
                        matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                    elif np.sign(operator) == -1:
                        matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                        state[np.abs(operator)] -= 1
                        if state[np.abs(operator)] < 0:
                            break

                final_state_tuple = tuple([state[s] for s in range(1, len(initial_state) + 1)])
                try:
                    final_state_id = basis.index(final_state_tuple)
                    matrix_element_number = jj_kerr[indeces]
                    ham[final_state_id, initial_state_id] += matrix_element_factor * matrix_element_number
                except ValueError:
                    # print ('not found: ', final_state_tuple)
                    pass

        return ham, basis

    def get_perturbation_hamiltonian2(self, modes: list, num_levels: list):
        """
        Calculate second order correction to energy with perturbation operator
        """
        from collections import defaultdict
        from itertools import product

        omega, kappa, modes_ = self.get_modes()
        omega_ = np.asarray([omega[m] for m in modes]) / (2 * np.pi)
        normalized_modes = self.normalization_of_modes(modes)

        # build junction operators
        ham = np.zeros((np.prod(num_levels), np.prod(num_levels)), complex)
        basis = [b for b in product(*[tuple([i for i in range(dof_levels)]) for dof_levels in num_levels])]
        basis_ = np.asarray(basis)

        ham += np.diag(basis_ @ omega_)  # linear part of hamiltonian
        for jj in self.JJs:
            jj_phi_operator = np.zeros_like(ham)
            for mode_id in range(len(modes)):
                jj_phi_mode_operator = np.identity(1)
                for mode2_id in range(len(modes)):
                    if mode_id == mode2_id:
                        submode = self.get_element_submode(jj, normalized_modes[mode_id, :])
                        op = jj.hob_phi_op(submode, num_levels[mode_id])
                    else:
                        op = np.identity(num_levels[mode2_id])
                    jj_phi_mode_operator = np.kron(jj_phi_mode_operator, op)
                jj_phi_operator += jj_phi_mode_operator
            # jj_phi_operators.append(jj_phi_operator)
            phi2 = jj_phi_operator @ jj_phi_operator
            phi4 = phi2 @ phi2
            phi6 = phi4 @ phi2

            ham -= phi4 * jj.E_J / 24 / (hbar * 2 * np.pi) / jj.n_junctions
            ham += phi6 * jj.E_J / 720 / (hbar * 2 * np.pi) / (jj.n_junctions ** 2)

        return ham, basis

    def get_perturbation_hamiltonian_kerr(self, modes: list, num_levels: list, truncation='initial'):
        if truncation == 'initial':
            ham, basis = self.get_perturbation_hamiltonian(modes, num_levels)
        elif truncation == 'intermediate':
            ham, basis = self.get_perturbation_hamiltonian2(modes, num_levels)

        vals, vecs = np.linalg.eigh(ham)

        ground_state_id = np.argmax(np.abs(vecs[:, 0]))
        single_excitation_state_ids = np.zeros(len(modes), int)
        double_excitation_state_ids = np.zeros((len(modes), len(modes)), int)

        all_ids = [ground_state_id]

        for mode_id in range(len(modes)):
            reference_vec = np.zeros(num_levels, complex).ravel()
            index = np.zeros(len(num_levels), int)
            index[mode_id] += 1
            reference_vec[np.ravel_multi_index(index, num_levels)] = 1
            # print (reference_vec)
            mode_state_id = np.argmax(np.abs(reference_vec @ vecs))
            single_excitation_state_ids[mode_id] = mode_state_id
            all_ids.append(mode_state_id)

        for mode1_id in range(len(modes)):
            for mode2_id in range(len(modes)):
                if mode2_id <= mode1_id:
                    reference_vec = np.zeros(num_levels, complex).ravel()
                    index = np.zeros(len(num_levels), int)
                    index[mode1_id] += 1
                    index[mode2_id] += 1
                    reference_vec[np.ravel_multi_index(index, num_levels)] = 1
                    # print(reference_vec)
                    mode_state_id = np.argmax(
                        np.abs(reference_vec @ vecs))  # TODO: probably this scalar product is wrong
                    double_excitation_state_ids[mode1_id, mode2_id] = mode_state_id
                    all_ids.append(mode_state_id)

        for mode1_id in range(len(modes)):
            for mode2_id in range(len(modes)):
                if mode2_id > mode1_id:
                    double_excitation_state_ids[mode1_id, mode2_id] = double_excitation_state_ids[mode2_id, mode1_id]

        if len(set(all_ids)) < len(all_ids):
            raise ValueError('Shifts are non-dispersive, cannot attribute states to modes')

        omegas = vals[single_excitation_state_ids] - vals[ground_state_id]
        kerrs = vals[double_excitation_state_ids] + vals[ground_state_id] - vals[single_excitation_state_ids] - vals[
            single_excitation_state_ids.reshape(-1, 1)]

        return omegas, kerrs

    ###################################################################################
    # Energy distribution
    ###################################################################################
    def energies_participations(self, modes):
        """
        Returns sorted list of energies in modes
        :param modes:
        """
        number_of_modes = modes.shape[0]
        dict_of_modes = dict.fromkeys([str(i) for i in range(number_of_modes)])
        for i in range(number_of_modes):
            total_mode_energy = 0
            energies_list = []
            for elem in self.elements:
                elem_energy = self.element_energy(element=elem, mode=modes[i]).real
                total_mode_energy += elem_energy
                energies_list.append((elem, elem_energy))
            # sort elements by energies
            for k in range(len(energies_list) - 1):
                for p in range(len(energies_list) - k - 1):
                    if energies_list[p][1] > energies_list[p + 1][1]:
                        energies_list[p], energies_list[p + 1] = energies_list[p + 1], energies_list[p]
            energies_list.reverse()
            dict_of_modes[str(i)] = energies_list
        return dict_of_modes

    def energy_distribution_cell(self, modes, element_cell):
        """
        Returns sorted modes for the element cell where element cell is a list of elements
        :param modes:
        :param element_cell:
        """
        number_of_modes = modes.shape[0]
        energies_list = []
        for i in range(number_of_modes):
            if type(element_cell) == list:
                elem_energy = 0
                for element in element_cell:
                    elem_energy += self.element_energy(element=element, mode=modes[i]).real
            else:
                elem_energy = self.element_energy(element=element_cell, mode=modes[i]).real
            energies_list.append((str(i), elem_energy))

        # sort modes by energies values
        for k in range(len(energies_list) - 1):
            for p in range(len(energies_list) - k - 1):
                if energies_list[p][1] > energies_list[p + 1][1]:
                    energies_list[p], energies_list[p + 1] = energies_list[p + 1], energies_list[p]

        energies_list.reverse()
        return energies_list


def potential_1d(phi, e_l, e_j, alpha, phi_dc):
    potential = 1 / 2 * e_l[0][0] * phi ** 2
    for jj_id, e_j in enumerate(e_j):
        phi_dc_jj = phi_dc[jj_id][0]
        potential += e_j * (1 - np.cos(alpha[jj_id][0] * phi + phi_dc_jj))
    return potential


def potential_2d(phi_1, phi_2, e_l, e_j, alpha):
    xx, yy = np.meshgrid(phi_1, phi_2)
    potential = 1 / 2 * e_l[0][0] * xx ** 2 + 1 / 2 * e_l[1][1] * yy ** 2
    for jj_id, e_j in enumerate(e_j):
        potential += e_j * (1 - np.cos(alpha[jj_id][0] * xx + alpha[jj_id][1] * yy))
    return xx, yy, potential
