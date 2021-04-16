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
        b = np.asarray([[0,  0, 0, 0], [0, 0, 0, 0]]) # derivatives
        a = np.asarray([[1, -1, self.R, 0], [0, 0, 1, 1]]) # current values
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
        return np.asarray([[1j*omega*self.C, -1j*omega*self.C, 1, 0], [0,0,1,1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[self.C, -self.C, 0, 0], [0, 0, 0, 0]]) # derivatives
        a = np.asarray([[0, 0, -1, 0], [0, 0, 1, 1]]) # current values
        return a, b

    def energy(self, mode):
        # energy matrix
        emat = np.asarray([
            [self.C, -self.C, 0, 0],
            [-self.C, self.C, 0, 0],
            [0,       0,      0, 0],
            [0,       0,      0, 0]
        ])/2
        return np.conj(mode)@emat@np.reshape(mode, (-1,1))
        #return self.C/2 * (mode[0]-mode[1])**2

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
        return np.asarray([[1, -1, 1j*omega*self.L, 0], [0,0,1,1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, -self.L, 0], [0, 0, 0, 0]]) # derivatives
        a = np.asarray([[1,-1, 0, 0], [0, 0, 1, 1]]) # current values
        return a, b

    def energy(self, mode):
        emat = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.L, 0],
            [0, 0, 0, 0]
        ])/2
        return np.conj(mode)@emat@mode

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
        b = np.asarray([[0, 0]]) # derivatives
        a = np.asarray([[1, 0]]) # current values
        return a, b

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
        b = np.asarray([[0, 0, 0], [0, 0, 0]]) # derivatives
        a = np.asarray([[1, self.Z0, 0], [1, -self.Z0, 1]]) # current values
        return a, b

    def energy(self, mode):
        return 0

    def __init__(self, z0=None, name=''):
        super().__init__('Port', name)
        self.Z0 = z0


class TLCoupler(TLSystemElement):
    '''
    Here n is a number of conductors in  TL CPW coupler
    '''
    def num_terminals(self):
        return self.n*2

    def num_degrees_of_freedom(self):
        return self.n*2

    def num_degrees_of_freedom_dynamic(self):
        return self.n*2*self.num_modes

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
        boundary_condition_matrix = np.zeros((self.num_terminals()*2, self.num_terminals()*2+self.num_degrees_of_freedom()), dtype=complex)
        boundary_condition_matrix[:, :self.num_terminals()*2] = np.identity(self.num_terminals()*2)

        for mode_pair_id, mode_pair in enumerate(self.propagating_modes()):
            boundary_condition_matrix[       0:self.n,  self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][:self.n])
            boundary_condition_matrix[  self.n:self.n*2,self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][:self.n])*np.exp(1j*mode_pair[0]*self.l*omega)
            boundary_condition_matrix[self.n*2:self.n*3,self.n*4+mode_pair_id] = np.asarray(mode_pair[1][self.n:])
            boundary_condition_matrix[self.n*3:        ,self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][self.n:])*np.exp(1j*mode_pair[0]*self.l*omega)
        #print(mode_pair)
        return boundary_condition_matrix

    def energy(self, state):
        m = self.n * self.num_modes
        v = state[-2*m:-m]
        i = state[-m:]
        s = (-0.5)**np.arange(self.num_modes)
        e = 0.5*s*s.reshape((-1, 1))*self.l
        e += np.abs(e)

        ll = np.kron(self.Ll, e)
        cl = np.kron(self.Cl, e)

        #emat = np.vstack([np.hstack([ll, np.zeros_like(ll)]), np.hstack([np.zeros_like(cl), cl])])

        return 0.5*np.conj(v)@cl@v+0.5*np.conj(i)@ll@i # TODO: energy stored in transmission line system

    def dynamic_equations(self):
        m = self.n * self.num_modes
        n_eq_internal = self.n * (self.num_modes - 1)

        b = np.zeros((self.num_terminals()*2+n_eq_internal*2, self.num_terminals()*2+m*2))
        a = np.zeros((self.num_terminals()*2+n_eq_internal*2, self.num_terminals()*2+m*2))

        # filling out telegrapher's equations
        E = np.zeros((self.num_modes - 1, self.num_modes))
        for i in range(self.num_modes-1):
            #E[i, i] = 1#/cl_av
            E[i, i] = self.l

        Ll = np.kron(self.Ll, E)
        Cl = np.kron(self.Cl, E)
        Rl = np.kron(self.Rl, E)
        Gl = np.kron(self.Gl, E)

        b[ :n_eq_internal, -m:] = Ll
        b[n_eq_internal:2 * n_eq_internal, -2 * m:-m] = Cl

        # Taylor-series expansion of I(x) = sum_i a_i x^i
        d = np.zeros((self.num_modes-1, self.num_modes))
        for i in range(1, self.num_modes):
            d[i-1, i] = i
        dmat = np.kron(np.eye(self.n), d)

        a[:n_eq_internal,    -2*m:-m] = -dmat
        a[n_eq_internal:2*n_eq_internal, -m:] = -dmat
        a[:n_eq_internal,    -m:] = -Rl
        a[n_eq_internal:2*n_eq_internal, -2*m:-m] = -Gl

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

    def __init__(self, n=2, l=None, ll=None, cl=None, rl=None, gl=None, name='', num_modes=10):
        super().__init__('TL', name)
        self.n = n
        self.l = l
        self.Ll = ll
        self.Cl = cl
        self.Rl = rl
        self.Gl = gl
        self.num_modes = num_modes
        pass

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
        return np.asarray([[1, -1, 1j * omega * self.L_lin, 0], [0, 0, 1, 1]], dtype=complex)

    def dynamic_equations(self):
        b = np.asarray([[0, 0, self.L_lin, 0], [0, 0, 0, 0]])  # derivatives
        a = np.asarray([[1, -1, 0, 0], [0, 0, 1, 1]])  # current values
        return a, b

    def energy_matrix(self):
        energy = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.L_lin, 0],
            [0, 0, 0, 0]
        ]) / 2
        return energy

    def nonlinear_perturbation(self):
        p = np.asarray([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        v = - (2 * e / hbar) ** 4 * self.E_J / 24 * self.L_lin ** 4 * np.kron(p, p)

        # np.conj(np.kron(mode1, mode2)) @ v @ (np.kron(mode1, mode2))

        return v

    def __init__(self, e_j=None, name=''):
        super().__init__('JJ', name)
        self.E_J = e_j
        phi_0 = hbar / (2 * e)  # reduced flux quantum
        self.L_lin = phi_0 ** 2 / self.E_J  # linear part of JJ

class TLSystem:
    def __init__(self):
        self.nodes = []   #list of all nodes [0,1,2...]
        self.elements = []   #list of all elements [<transmission_line_simulator_new.name_of_element>, ...]
        self.node_multiplicity = {}   #dictionary of node's multiplicity {node1: node1's multiplicity, node2: node2's multiplicity, ...}
        self.terminal_node_mapping = []   #list of terminals's nodes [terminal1's nodes=[], node2's multiplicity=[], ...]
        self.dof_mapping = []   #???

    def add_element(self, element, nodes):
        self.elements.append(element)
        for node in nodes:
            if node not in self.nodes:
                self.node_multiplicity[node] = 0
                self.nodes.append(node)
            self.node_multiplicity[node] += 1
        self.terminal_node_mapping.append(nodes)
        return

    def map_dofs(self):
        # count nodes
        self.dof_mapping = [n for n in self.nodes] # nodal voltages
        self.dof_mapping.extend([(e_id, p_id) for e_id, e in enumerate(self.elements) for p_id in range(e.num_terminals())])
                                                     # currents incident into each terminal
        self.dof_mapping.extend([(e_id, int_dof_id) for e_id, e in enumerate(self.elements) for int_dof_id in range(e.num_degrees_of_freedom())])
                                                     # number of element-internal degrees of freedom

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
            #modes.append((e, gamma, v[:, state_id]))
            frequencies.append(e)
            gammas.append(gamma)
            modes.append(v[:, state_id])

        order = np.argsort(frequencies)
        return np.asarray(frequencies)[order], np.asarray(gammas)[order], np.asarray(modes)[order]

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
                element_state[terminal_id+e.num_terminals()] = state[node_no+current_offset+terminal_id]
            for internal_dof_id in range(e.num_degrees_of_freedom_dynamic()):
                element_state[2*e.num_terminals() + internal_dof_id] = state[node_no+terminal_no+internal_dof_offset+internal_dof_id]
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
                    dynamic_equation_matrix_a[equation_id, node_id] = equation_a[terminal_id] #nodal voltages
                    dynamic_equation_matrix_a[equation_id, node_no+current_offset+terminal_id] = equation_a[terminal_id+e.num_terminals()] #nodal current
                    dynamic_equation_matrix_b[equation_id, node_id] = equation_b[terminal_id] #nodal voltages
                    dynamic_equation_matrix_b[equation_id, node_no+current_offset+terminal_id] = equation_b[terminal_id+e.num_terminals()] #nodal current
                for internal_dof_id in range(e.num_degrees_of_freedom_dynamic()):
                    dynamic_equation_matrix_a[equation_id, node_no+terminal_no+internal_dof_offset+internal_dof_id] = equation_a[2*e.num_terminals() + internal_dof_id]
                    dynamic_equation_matrix_b[equation_id, node_no + terminal_no + internal_dof_offset + internal_dof_id] = equation_b[2 * e.num_terminals() + internal_dof_id]
                equation_id += 1
            internal_dof_offset += e.num_degrees_of_freedom_dynamic()
            current_offset += e.num_terminals()

        full_terminal_id = 0
        # filling kinetic equations
        for e_id, e in enumerate(self.elements):
            for terminal_id, node in enumerate(self.terminal_node_mapping[e_id]):
                dynamic_equation_matrix_a[dynamic_equation_no+self.nodes.index(node), node_no+full_terminal_id] = 1
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
                    boundary_condition_matrix[equation_id, node_id] = equation[terminal_id] #nodal voltages
                    boundary_condition_matrix[equation_id, node_no+current_offset+terminal_id] = equation[terminal_id+e.num_terminals()] #nodal current
                for internal_dof_id in range(e.num_degrees_of_freedom()):
                    boundary_condition_matrix[equation_id, node_no+terminal_no+internal_dof_offset+internal_dof_id] = equation[2*e.num_terminals() + internal_dof_id]
                equation_id += 1
            internal_dof_offset += e.num_degrees_of_freedom()
            current_offset += e.num_terminals()

        full_terminal_id = 0
        # filling kinetic equations
        for e_id, e in enumerate(self.elements):
            for terminal_id, node in enumerate(self.terminal_node_mapping[e_id]):
                boundary_condition_matrix[dynamic_equation_no+self.nodes.index(node), node_no+full_terminal_id] = 1
                full_terminal_id += 1
        return boundary_condition_matrix

    def get_element_dofs(self, element: TLSystemElement):
        self.map_dofs()
        e_id = self.elements.index(element)
        voltages = [self.dof_mapping[:len(self.nodes)].index(t) for t in self.terminal_node_mapping[e_id]]

        terminal_no = np.sum(e.num_terminals() for e in self.elements)

        current_variables = self.dof_mapping[len(self.nodes):len(self.nodes) + terminal_no]
        internal_dof_variables = self.dof_mapping[len(self.nodes) + terminal_no:]

        currents = [current_variables.index((e_id, p_id)) + len(self.nodes) for p_id in range(element.num_terminals())]
        degrees_of_freedom = [internal_dof_variables.index((e_id, dof_id)) + len(self.nodes) + terminal_no \
                              for dof_id in range(element.num_degrees_of_freedom())]

        return voltages, currents, degrees_of_freedom

    def get_element_dynamic_equations(self, element: TLSystemElement):
        e_id = self.elements.index(element)
        offset = np.sum(e.num_terminals()+e.num_degrees_of_freedom() for e in self.elements[:e_id])

        return np.arange(offset, offset+element.num_terminals()+element.num_degrees_of_freedom())

"""
    def boundary_condition_matrix_det(self, omega):
        matrix = self.create_boundary_problem_matrix(omega)
        return np.linalg.det(matrix)
    def boundary_condition_matrix_abs_det(self, omega):
        matrix = self.create_boundary_problem_matrix(omega)
        det = np.linalg.det(matrix)
        return np.log10((det.real)**2 + (det.imag)**2)
    def solve_problem(self, frequency_approximation, epsilon, step):
        '''
        This is a stupid method for solving the problem. It looks like gradient descent method
        '''
        x = frequency_approximation
        func = self.create_boundary_problem_matrix(frequency_approximation)
        number_of_iterations = 0
        while (self.create_boundary_problem_matrix(x) - 0) > epsilon:
            number_of_iterations = number_of_iterations + 1
            print('not', number_of_iterations)
            print(x)
            grad = (self.create_boundary_problem_matrix(x+step) - self.create_boundary_problem_matrix(x))/step
            if grad < 0:
                x = x + step#*grad
            elif grad > 0:
                x = x - step#*grad
            else:
                print('Problem')
                break
        result = x
        return result
    def res(self):
        print('self.nodes', self.nodes)
        print('self.elements', self.elements)
        print('self.node_multiplicity', self.node_multiplicity)
        print('self.terminal_node_mapping', self.terminal_node_mapping)
        print('self.dof_mapping', self.dof_mapping)
        print('self.nodes', type(self.nodes))
        print('self.elements', type(self.elements))
        print('self.node_multiplicity', type(self.node_multiplicity))
        print('self.terminal_node_mapping', type(self.terminal_node_mapping))
        print('self.dof_mapping', type(self.dof_mapping))
        return
"""