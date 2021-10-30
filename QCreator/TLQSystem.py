import matplotlib.pyplot as plt
import numpy as np
from transmission_line_simulator import TLSystemElement, Resistor, Capacitor, Inductor, Short, Port, TLCoupler, \
    JosephsonJunctionChain, TLSystem


class TLQSystem(TLSystem):
    def change_tl_basis(self, elem):
        """
        Change basis for TL element from v to (V_nodes, v)
        """
        if elem.type_ != 'TL':
            raise ValueError('Element is not TL!')

        # create transition matrix
        no_int_dofs = elem.num_modes
        if no_int_dofs < 2:
            raise ValueError('Cannot change basis!')

        m = elem.n * no_int_dofs
        s_matrix = np.zeros((m, m))

        alpha = (-1 / 2) ** np.arange(no_int_dofs)
        beta = (1 / 2) ** np.arange(no_int_dofs)
        space = np.eye(elem.n)

        a = np.kron(space, alpha)
        b = np.kron(space, beta)
        t = np.vstack((a, b))
        e = np.eye(elem.n * (no_int_dofs - 2))
        e_ = np.hstack((e, np.zeros((elem.n * (no_int_dofs - 2), elem.num_terminals()))))

        s_matrix = np.vstack((t, e_))

        return s_matrix

    def cap_mat_in_changed_basis(self, elem):
        if elem.type_ != 'TL':
            raise ValueError('Element is not TL!')
        s_matrix = self.change_tl_basis(elem)

        cap_mat = elem.get_capacitance_matrix()

        cap_mat_ = s_matrix.T @ cap_mat @ s_matrix

        return cap_mat_

    def dofs_of_hamiltonian(self):
        """
        This method removes all short and port nodes from degrees of freedom and makes them grounded
        """
        from copy import deepcopy
        dofs_nodes = deepcopy(self.nodes)

        grounded_nodes = []
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ in ['Port', 'Short']:
                nodes = self.terminal_node_mapping[elem_id]
                for node in nodes:
                    dofs_nodes.remove(node)
                    grounded_nodes.append(node)
        return dofs_nodes, grounded_nodes

    def create_capacitance_system_matrix(self, changed_basis=True):
        """
        This method returns capacitance matrix of the system in the basis (V, v), where V are nodes voltages
        and v are internal degrees of freedom corresponded coefficients in Taylor series for TL
        """
        dofs_nodes, grounded_nodes = self.dofs_of_hamiltonian()
        # number of kinetic dofs
        no_dof = len(dofs_nodes)

        # number of internal kinetic dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')
        # for TL type elements we use changed basis
        no_tl_terminals = np.sum(e.num_terminals() for e in self.elements if e.type_ == 'TL')

        no_kinetic_dofs = no_dof + int(internal_dof_no / 2) - no_tl_terminals

        capacitance_matrix = np.zeros((no_kinetic_dofs, no_kinetic_dofs))

        internal_dof_offset = 0
        for elem_id, elem in enumerate(self.elements):
            if elem.type_ == 'TL':
                cap_mat = self.cap_mat_in_changed_basis(elem)
            else:
                cap_mat = elem.get_capacitance_matrix()
            if type(cap_mat) is not int:
                elem_nodes = self.terminal_node_mapping[elem_id]
                for node1_id, node1 in enumerate(elem_nodes):
                    if node1 not in grounded_nodes:
                        for node2_id, node2 in enumerate(elem_nodes):
                            if node2 not in grounded_nodes:
                                ind1 = dofs_nodes.index(node1)
                                ind2 = dofs_nodes.index(node2)
                                capacitance_matrix[ind1][ind2] += cap_mat[node1_id][node2_id]

                no_kinetic_internal_dofs = int(elem.num_degrees_of_freedom_dynamic() / 2) - elem.num_terminals()
                if no_kinetic_internal_dofs > 0:
                    for dof1 in range(no_kinetic_internal_dofs):
                        for dof2 in range(no_kinetic_internal_dofs):
                            capacitance_matrix[no_dof + internal_dof_offset + dof1][
                                no_dof + internal_dof_offset + dof2] += \
                                cap_mat[len(elem_nodes) + dof1][len(elem_nodes) + dof2]
                    internal_dof_offset += no_kinetic_internal_dofs

                # if no_kinetic_internal_dofs > 0:
                #     for dof1 in range(no_kinetic_internal_dofs):
                #         for dof2 in range(no_kinetic_internal_dofs):
                #             capacitance_matrix[no_dof + internal_dof_offset + dof1][
                #                 no_dof + internal_dof_offset + dof2] += \
                #                 cap_mat[len(elem_nodes) + dof1][len(elem_nodes) + dof2]
                #     internal_dof_offset += no_kinetic_internal_dofs

        return capacitance_matrix

    def create_inductance_system_matrix(self, linear_jj=True):
        dofs_nodes, grounded_nodes = self.dofs_of_hamiltonian()
        # number of kinetic dofs
        no_dof = len(dofs_nodes)

        # number of internal kinetic dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom_dynamic() for e in self.elements if e.type_ == 'TL')

        no_potential_dofs = no_dof + int(internal_dof_no / 2)

        inductance_matrix = np.zeros((no_potential_dofs, no_potential_dofs))

        internal_dof_offset = 0
        for elem_id, elem in enumerate(self.elements):
            ind_mat = elem.get_inductance_matrix()
            if type(ind_mat) is not int:
                elem_nodes = self.terminal_node_mapping[elem_id]

                for node1_id, node1 in enumerate(elem_nodes):
                    if node1 not in grounded_nodes:
                        for node2_id, node2 in enumerate(elem_nodes):
                            if node2 not in grounded_nodes:
                                ind1 = dofs_nodes.index(node1)
                                ind2 = dofs_nodes.index(node2)
                                inductance_matrix[ind1][ind2] += ind_mat[node1_id][node2_id]

                no_kinetic_internal_dofs = int(elem.num_degrees_of_freedom_dynamic() / 2)
                if no_kinetic_internal_dofs > 0:
                    for dof1 in range(no_kinetic_internal_dofs):
                        for dof2 in range(no_kinetic_internal_dofs):
                            capacitance_matrix[no_dof + internal_dof_offset + dof1][
                                no_dof + internal_dof_offset + dof2] += \
                                cap_mat[len(elem_nodes) + dof1][len(elem_nodes) + dof2]
                    internal_dof_offset += no_kinetic_internal_dofs

        return capacitance_matrix


    def inverse_capacitance_matrix(self):
        from numpy.linalg import inv, det
        cap_mat = self.create_capacitance_system_matrix()
        if det(cap_mat) == 0:
            raise ValueError('Capacitance matrix is singular!')

        return inv(cap_mat)
