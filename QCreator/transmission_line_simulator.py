import numpy as np
from abc import *
from scipy.constants import e, hbar


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
            return np.conj(mode).squeeze()@self.energy_matrix()@mode.squeeze()
        except:
            return 0

    @abstractmethod
    def energy_matrix(self):
        return 0

    def scdc(self, submode):
        return None

    def is_scdc(self):
        return False

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
        return (submode[1] - submode[0])**2  / (2 * np.real(self.L)) * 1e-9

    def potential_gradient(self, submode):
        return [(submode[0] - submode[1]) /  np.real(self.L) * 1e-9,
                (submode[1] - submode[0]) /  np.real(self.L) * 1e-9]

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
        return -submode[0]*self.idc/ (hbar / (2 * e)) * 1e-9

    def potential_gradient(self, submode):
        return [-self.idc/ (hbar / (2 * e)) * 1e-9]

    def potential_hessian(self, submode):
        return [[0]]

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def energy(self, mode):
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

        return np.conj(state.squeeze()) @ energy_matrix @ state.squeeze()  # TODO: energy stored in transmission line system

    def energy_matrix(self):
        m = self.n * self.num_modes

        s = (-0.5) ** np.arange(self.num_modes)
        e = 0.5 * s * s.reshape((-1, 1)) * self.l
        e += np.abs(e)

        integral = 1/(np.arange(self.num_modes).reshape(-1,1) + np.arange(self.num_modes) + 1)
        e = e * integral

        ll = np.kron(self.Ll, e)
        cl = np.kron(self.Cl, e)

        energy_matrix = np.zeros((self.num_terminals() * 2 + m * 2, self.num_terminals() * 2 + m * 2))
        energy_matrix[-2*m:-m, -2*m:-m] = cl/2
        energy_matrix[-m:, -m:] = ll/2

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

    def __init__(self, n=2, l=None, ll=None, cl=None, rl=None, gl=None, name='', num_modes=10, cutoff=None):
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


class JosephsonJunction(TLSystemElement):
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

    # def scdc(self, submode):
    #     return [self.E_J / self.phi_0 * np.sin((submode[1] - submode[0])) * 1e6 + submode[2], ## TODO: fix JJ energy
    #             (submode[2] + submode[3])]
    #
    # def scdc_energy(self, submode):
    #     return [self.E_J / self.phi_0 * ]
    #
    # def scdc_constrains(self):
    #     return [[0, 0, 1, 1]]
    #
    # def scdc_gradient(self, submode):
    #     return []

    def potential(self, submode):
        return self.E_J * (1 - np.cos(submode[0] - submode[1])) / (hbar / (2 * e)) ** 2 * 1e-9

    def potential_gradient(self, submode):
        gradient = self.E_J * np.sin(submode[0] - submode[1]) / (hbar / (2 * e)) ** 2 * 1e-9
        return gradient * np.asarray([1, -1])

    def potential_hessian(self, submode):
        hessian = self.E_J * np.cos(submode[0] - submode[1]) / (hbar / (2 * e)) ** 2 * 1e-9
        return hessian * np.asarray([[1, -1], [-1, 1]])

    # def potential_constraints(self):
    #     return []

    def is_scdc(self):
        return True

    def nonlinear_perturbation(self):
        p = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        Phi_0 = self.phi_0 * (2 * np.pi)
        v = - self.E_J / 4 * ((2 * np.pi / Phi_0) * self.L_lin()) ** 4 * np.kron(p, p)

        # np.conj(np.kron(mode1, mode2)) @ v @ (np.kron(mode1, mode2))

        return v

    def nonlinear_perturbation4(self, mode1, mode2, mode3, mode4):

        v = - self.L_lin() ** 3 / self.phi_0**2 *mode1[2]*mode2[2]*mode3[2]*mode4[2]/ 4

        # np.conj(np.kron(mode1, mode2)) @ v @ (np.kron(mode1, mode2))

        return v

    def __init__(self, e_j=None, name=''):
        super().__init__('JJ', name)
        self.E_J = e_j

        self.phi_0 = hbar / (2 * e)  # reduced flux quantum
        self.stationary_phase = 0

    def L_lin(self):
        return self.phi_0 ** 2 / (self.E_J * np.cos(self.stationary_phase))  # linear part of JJ

    def set_stationary_phase(self, phase):
        self.stationary_phase = phase


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

        for state_id in range(len(w)):
            e = np.imag(w[state_id])
            gamma = -np.real(w[state_id])
            if e <= 0 or not np.isfinite(e):
                continue
            # modes.append((e, gamma, v[:, state_id]))
            frequencies.append(e)
            gammas.append(gamma)
            modes.append(v[:, state_id])

        order = np.argsort(frequencies)
        return np.asarray(frequencies)[order], np.asarray(gammas)[order], np.asarray(modes)[order]

    def get_scdc_subsystems(self):
        #scdc_nodes = self.get_scdc_nodes()
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
                                #print ('found ', element, connections)
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
            matrix_indeces = [scdc_subnodes.index(node_id) for node_id in self.terminal_node_mapping[e_id] if node_id not in scdc_shorts]
            submatrix_indeces = [i for i in range(len(self.terminal_node_mapping[e_id])) if self.terminal_node_mapping[e_id][i] not in scdc_shorts]
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
            matrix_indeces = [scdc_subnodes.index(node_id) for node_id in self.terminal_node_mapping[e_id] if node_id not in scdc_shorts]
            submatrix_indeces = [i for i in range(len(self.terminal_node_mapping[e_id])) if self.terminal_node_mapping[e_id][i] not in scdc_shorts]
            if len(matrix_indeces):
                energy_hessian[np.meshgrid(matrix_indeces, matrix_indeces)] += element_hessian[np.meshgrid(submatrix_indeces, submatrix_indeces)]

        return energy_hessian

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
        internal_dof_variables = self.dof_mapping_dynamic[len(self.nodes) + terminal_no:]

        currents = [current_variables.index((e_id, p_id)) + len(self.nodes) for p_id in range(element.num_terminals())]
        if not dynamic:
            degrees_of_freedom = [internal_dof_variables.index((e_id, dof_id)) + len(self.nodes) + terminal_no \
                                  for dof_id in range(element.num_degrees_of_freedom())]
        else:
            degrees_of_freedom = [internal_dof_variables.index((e_id, dof_id)) + len(self.nodes) + terminal_no \
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
            submode_energy = np.conj(submode_element.T).squeeze()@element.energy_matrix()@submode_element2.squeeze()
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

        modes_energies = []

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

        if self.JJs:
            JJ_kerr = np.zeros((number_of_modes, number_of_modes))
            for JJ_ in self.JJs:
                perturbation_matrix = np.zeros((number_of_modes, number_of_modes))
                for i in range(number_of_modes):
                    for j in range(number_of_modes):
                        mode_i = self.get_element_submode(JJ_, modes_[i])
                        mode_j = self.get_element_submode(JJ_, modes_[j])

                        submode_ij = np.kron(mode_i, mode_j)

                        if j == i:

                            perturbation_matrix[i][j] = np.dot(np.conj(submode_ij.T),
                                                           np.dot(JJ_.nonlinear_perturbation(), submode_ij)).real / 2
                        else:
                            perturbation_matrix[i][j] = np.dot(np.conj(submode_ij.T),
                                                               np.dot(JJ_.nonlinear_perturbation(),
                                                                      submode_ij)).real

                kerr_coefficients_matrix = perturbation_matrix / (hbar*2*np.pi)
                JJ_kerr += kerr_coefficients_matrix

        return JJ_kerr

    def get_perturbation_nondiagonal(self, list_of_modes_numbers: list):
        """
        Calculate matrix of perturbation in basis of |g>, |e>, |f>, |h> states, truncate to n states
        """
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized

        number_of_modes = len(modes_)

        JJ_kerr = np.zeros((2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes))
        for JJ_ in self.JJs:
            for i in range(number_of_modes*2):
                mode_i = self.get_element_submode(JJ_, modes_[i%number_of_modes])
                if i > number_of_modes:
                    mode_i = np.conj(mode_i)
                for j in range(number_of_modes*2):
                    mode_j = self.get_element_submode(JJ_, modes_[j%number_of_modes])
                    if j > number_of_modes:
                        mode_j = np.conj(mode_j)
                    submode_ij = np.kron(mode_i, mode_j)
                    for k in range(number_of_modes*2):
                        mode_k = self.get_element_submode(JJ_, modes_[k%number_of_modes])
                        if k > number_of_modes:
                            mode_k = np.conj(mode_k)
                        for l in range(number_of_modes*2):
                            mode_l = self.get_element_submode(JJ_, modes_[l%number_of_modes])
                            if l > number_of_modes:
                                mode_l = np.conj(mode_l)
                            submode_kl = np.kron(mode_k, mode_l)
                            JJ_kerr[i, j, k, l] += np.dot(submode_ij.T,
                                                                   np.dot(JJ_.nonlinear_perturbation(),
                                                                          submode_kl)).ravel()[0] / 6

        return JJ_kerr

    def get_second_order_perturbation(self, list_of_modes_numbers: list):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        number_of_modes = len(list_of_modes_numbers)  # number of modes in the system

        omega, kappa, modes = self.get_modes()
        omegas = np.asarray([omega[m] for m in list_of_modes_numbers])

        kerr_matrix = self.get_perturbation(list_of_modes_numbers)

        # first order corrections for mode energies
        energies_in_modes_with_first_correction = np.zeros(number_of_modes)

        for m in range(number_of_modes):
            energy_corrected = hbar * omegas[m]
            for n in range(number_of_modes):
                energy_corrected += hbar * 2 * np.pi * kerr_matrix[m][n]
            energies_in_modes_with_first_correction[m] = energy_corrected

        s = [i for i in range(1, number_of_modes + 1)]
        state = defaultdict(int)  # ground state of the system corresponding to M modes
        for k in s:
            state[k] = 0

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

            list_of_modes_numbers_ijkl = []  # [i,j,k,l] for calculation J_kerr[i,j,k,l]

            for operator in t[::-1]:  # operators act in reverse order

                if np.sign(operator) == 1:

                    coefficient *= np.sqrt(state[np.abs(operator)] + 1)
                    list_of_modes_numbers_ijkl.append(np.abs(operator))
                    state[np.abs(operator)] += 1

                elif np.sign(operator) == -1:

                    if state[np.abs(operator)] > 0:

                        coefficient *= np.sqrt(state[np.abs(operator)])
                        list_of_modes_numbers_ijkl.append(np.abs(operator))
                        state[np.abs(operator)] -= 1

                    elif state[np.abs(operator)] == 0:

                        coefficient = 0
                        list_of_modes_numbers_ijkl.append(np.abs(operator))
                        state[np.abs(operator)] = 0

                        break

                    else:
                        raise ValueError('Sign of the mode is not correct')

            JJ_kerr = self.get_perturbation_nondiagonal(list_of_modes_numbers)
            indexes = tuple([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t])
            J_kerr_ijkl = hbar * 2 * np.pi * JJ_kerr[indexes]

            final_state = list(state.values())

            list_of_all_perturbation_terms.append(np.asarray([coefficient * J_kerr_ijkl, final_state]))

            # clear state
            for k in s:
                state[k] = 0

        second_order_energy_correction = 0

        states_i = [s[1] for s in list_of_all_perturbation_terms]  # states for summation

        ground_state = [0 * i for i in range(number_of_modes)]

        energy_ground = 0

        # remove ground state in states_i, because we can not sum up by f = |ground state>
        states_i = [s for s in states_i if s != ground_state]

        for i in states_i:

            all_matrix_elements = 0

            for term in list_of_all_perturbation_terms:

                matrix_elem = term[0]

                final_state = term[1]

                # condition of orthogonal states: < i | final state > = delta(i, final state)

                if i == final_state:
                    all_matrix_elements += matrix_elem
                else:
                    all_matrix_elements += 0

            term_for_correction = np.abs(all_matrix_elements) ** 2
            print(term_for_correction)
            energy_i = 0

            for mode in range(number_of_modes):
                number_of_excitations = i[mode]
                energy_i += number_of_excitations * energies_in_modes_with_first_correction[mode]

            energy_diff = energy_i - energy_ground

            second_order_energy_correction += term_for_correction / energy_diff

        return second_order_energy_correction

    def get_perturbation_nondiagonal2(self, list_of_modes_numbers: list):
        """
        Calculate matrix of perturbation in basis of |g>, |e>, |f>, |h> states, truncate to n states
        """
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized

        number_of_modes = len(modes_)

        JJ_kerr = np.zeros((2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes, 2 * number_of_modes), dtype=complex)
        for JJ_ in self.JJs:
            for i in range(number_of_modes*2):
                mode_i = self.get_element_submode(JJ_, modes_[i % number_of_modes])
                if i > number_of_modes:
                    mode_i = np.conj(mode_i)
                for j in range(number_of_modes*2):
                    if j < i:
                        continue
                    mode_j = self.get_element_submode(JJ_, modes_[j % number_of_modes])
                    if j > number_of_modes:
                        mode_j = np.conj(mode_j)
                    #submode_ij = np.kron(mode_i, mode_j)
                    for k in range(number_of_modes*2):
                        if k < j:
                            continue
                        mode_k = self.get_element_submode(JJ_, modes_[k % number_of_modes])
                        if k > number_of_modes:
                            mode_k = np.conj(mode_k)
                        for l in range(number_of_modes*2):
                            if l < k:
                               continue
                            mode_l = self.get_element_submode(JJ_, modes_[l % number_of_modes])
                            if l > number_of_modes:
                                mode_l = np.conj(mode_l)
                            #submode_kl = np.kron(mode_k, mode_l)
                            #JJ_kerr[i, j, k, l] += (submode_ij.T@JJ_.nonlinear_perturbation()@submode_kl).ravel()[0] / 6
                            JJ_kerr[i, j, k, l] += JJ_.nonlinear_perturbation4(mode_i, mode_j, mode_k, mode_l) / 6
                            # print(i, j, k, l)

        return JJ_kerr / (hbar * 2 * np.pi)

    def get_second_order_perturbation_kerr2(self, list_of_modes_numbers: list):
        omega, kappa, modes = self.get_modes()
        omega_ = np.asarray([omega[m] for m in list_of_modes_numbers])/(2*np.pi)
        omega_corrected = np.asarray([omega[m] for m in list_of_modes_numbers])/(2*np.pi)

        JJ_kerr = self.get_perturbation_nondiagonal2(list_of_modes_numbers) / 4

        first_order_kerr = self.get_perturbation(list_of_modes_numbers)
        kerr = np.zeros(first_order_kerr.shape)

        mode = [0 for i in range(len(list_of_modes_numbers))]
        ground_state_energy = self.get_second_order_perturbation3(mode, omega_, JJ_kerr)
        first_state_corrections = [0 for i in range(len(list_of_modes_numbers))]

        for mode_id in range(len(list_of_modes_numbers)):
            mode = [0 if i != mode_id else 1 for i in range(len(list_of_modes_numbers))]
            #corrected_energy = self.get_second_order_perturbation(mode, list_of_modes_numbers)
            #omega_corrected[mode_id] += first_order_kerr[mode_id, mode_id] + corrected_energy - ground_state_energy
            first_state_corrections[mode_id] = self.get_second_order_perturbation3(mode, omega_, JJ_kerr)

        for mode1_id in range(len(list_of_modes_numbers)):
            for mode2_id in range(len(list_of_modes_numbers)):
                mode = [0 for i in range(len(list_of_modes_numbers))]
                mode[mode1_id] += 1
                mode[mode2_id] += 1
                corrected_energy = self.get_second_order_perturbation3(mode, omega_, JJ_kerr)
                kerr[mode1_id, mode2_id] = first_order_kerr[mode1_id, mode2_id] + corrected_energy + ground_state_energy - \
                                                first_state_corrections[mode1_id] - first_state_corrections[mode2_id]

        return kerr

    def get_second_order_perturbation2(self, initial_state: list, list_of_modes_numbers: list):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        omega, kappa, modes = self.get_modes()
        omega_ = np.asarray([omega[m] for m in list_of_modes_numbers])/(2*np.pi)
        modes_ = self.normalization_of_modes(list_of_modes_numbers)  # here modes are normalized
        #first_order_kerr = self.get_perturbation(list_of_modes_numbers)
        #first_order_kerr = (first_order_kerr + np.diag(np.diag(first_order_kerr))) / 2
        JJ_kerr = self.get_perturbation_nondiagonal2(list_of_modes_numbers) / 4

        initial_state_ = np.asarray(initial_state)
        initial_state = tuple(initial_state)
        initial_state_energy = initial_state_@omega_# + initial_state@first_order_kerr@initial_state

        number_of_modes = JJ_kerr.shape[0] // 2  # number of modes in the system

        operators = []
        for i in range(1, number_of_modes + 1):
            operators.extend([i, -i])

        final_states = defaultdict(lambda: 0.0)

        # create a list with perturbation terms: 'n' -- creation operator of mode n,
        # '-n' -- annihilation operator of mode n

        for t in product(operators, repeat=4):
            #print ('Operator: ', t)
            state = {k+1:v for k,v in enumerate(initial_state)}

            matrix_element_factor = 1
            indeces = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))
            #print ('JJ kerr indeces: ', indeces)
            matrix_element_number = JJ_kerr[indeces]

            for operator in t[::-1]:
                #print('operator: ', operator)
                if np.sign(operator) == 1:
                    state[np.abs(operator)] += 1
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                elif np.sign(operator) == -1:
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                    state[np.abs(operator)] -= 1
                    if state[np.abs(operator)] < 0:
                        #print (matrix_element_factor)
                        break
                #print ('matrix element factor: ', matrix_element_factor)

            final_state_tuple = tuple([state[s] for s in range(1, len(initial_state)+1)])
            final_states[final_state_tuple] += matrix_element_factor * matrix_element_number
            #print (state[1], matrix_element_factor)
            #print('Arrived in state: ', state,
            #      'got pre-factor: ', matrix_element_factor,
            #      'modal matrix element: ', matrix_element_number,
            #      'final state: ', final_states)
            # print (final_states)
        correction = 0
        for final_state, matrix_element in final_states.items():
            final_state_ = np.asarray(final_state)
            final_state_energy = final_state_@omega_ # + final_state_ @ first_order_kerr @ final_state_
            denominator = initial_state_energy - final_state_energy
            numerator = np.abs(matrix_element)**2
            correction_new = numerator / denominator
            #print('i: ', initial_state, 'f: ', final_state, 'E_i', initial_state_energy/1e9, 'E_f:', final_state_energy/1e9,
            #      'Vif:', np.abs(matrix_element)/1e9, 'correction:', correction_new/1e9)
            if final_state != initial_state:
                correction += correction_new

        return correction

    def get_second_order_perturbation3(self, initial_state: list, omega_, JJ_kerr):
        """
        Calculate second order correction to energy with perturbation operator
        """

        from collections import defaultdict
        from itertools import product

        initial_state_ = np.asarray(initial_state)
        initial_state = tuple(initial_state)
        initial_state_energy = initial_state_@omega_# + initial_state@first_order_kerr@initial_state

        number_of_modes = len(initial_state)  # number of modes in the system

        operators = []
        for i in range(1, number_of_modes + 1):
            operators.extend([i, -i])

        final_states = defaultdict(lambda: 0.0)

        # create a list with perturbation terms: 'n' -- creation operator of mode n,
        # '-n' -- annihilation operator of mode n

        for t in product(operators, repeat=4):
            #print ('Operator: ', t)
            state = {k+1: v for k, v in enumerate(initial_state)}

            matrix_element_factor = 1
            indeces = tuple(sorted([np.abs(operator) - 1 + number_of_modes * (operator < 0) for operator in t]))
            # print ('JJ kerr indeces: ', indeces, JJ_kerr[indeces])
            matrix_element_number = JJ_kerr[indeces]

            for operator in t[::-1]:
                #print('operator: ', operator)
                if np.sign(operator) == 1:
                    state[np.abs(operator)] += 1
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                elif np.sign(operator) == -1:
                    matrix_element_factor = matrix_element_factor * np.sqrt(state[np.abs(operator)])
                    state[np.abs(operator)] -= 1
                    if state[np.abs(operator)] < 0:
                        #print (matrix_element_factor)
                        break
                #print ('matrix element factor: ', matrix_element_factor)

            final_state_tuple = tuple([state[s] for s in range(1, len(initial_state)+1)])
            final_states[final_state_tuple] += matrix_element_factor * matrix_element_number
            #print (state[1], matrix_element_factor)
            # print('Arrived in state: ', state,
            #       'got pre-factor: ', matrix_element_factor,
            #       'modal matrix element: ', matrix_element_number,
            #       'final state: ', final_states)
            # print (final_states)
        correction = 0
        for final_state, matrix_element in final_states.items():
            final_state_ = np.asarray(final_state)
            final_state_energy = final_state_@omega_ # + final_state_ @ first_order_kerr @ final_state_
            denominator = initial_state_energy - final_state_energy
            numerator = np.abs(matrix_element)**2
            correction_new = numerator / denominator
            # print('i: ', initial_state, 'f: ', final_state, 'E_i', initial_state_energy/1e9, 'E_f:', final_state_energy/1e9,
            #      'Vif:', np.abs(matrix_element)/1e9, 'correction:', correction_new/1e9)
            if final_state != initial_state:
                correction += correction_new

        return correction
