import numpy as np
from abc import *


class TLSystemElement:
    @abstractmethod
    def num_terminals(self):
        pass

    @abstractmethod
    def num_degrees_of_freedom(self):
        pass

    @abstractmethod
    def boundary_condition(self, omega):
        pass

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
            modes.append((gamma, mode_amplitudes[:,mode_id]))
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

    def __init__(self, n=2, l=None, ll=None, cl=None, rl=None, gl=None, name=''):
        super().__init__('TL', name)
        self.n = n
        self.l = l
        self.Ll = ll
        self.Cl = cl
        self.Rl = rl
        self.Gl = gl
        pass

    def __repr__(self):
        return "TL {} (n={})".format(self.name, self.n)

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

