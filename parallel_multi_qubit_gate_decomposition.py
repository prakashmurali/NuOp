import math
import time
import numpy as np
import multiprocessing as mp
from os import getpid

from scipy.optimize import minimize
from qiskit.quantum_info import Operator
from qiskit.converters import dag_to_circuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library.standard_gates import U1Gate, U2Gate, U3Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.unitary import UnitaryGate
from gates_numpy import get_gate_unitary_qiskit
from cirq.circuits.qasm_output import QasmUGate, QasmTwoQubitGate

from qiskit.transpiler.passes.optimization.optimize_1q_gates import Optimize1qGates
optimise1qgates = Optimize1qGates()

class GateTemplate:
    def __init__(self, two_qubit_gate, two_qubit_gate_params, n_qubit=2):
        self.two_qubit_gate = two_qubit_gate
        self.two_qubit_gate_params = two_qubit_gate_params
        self.n_qubit = n_qubit
        self.identity = np.eye(2)
        
    def u3_gate(self, theta, phi, lam):
        return np.matrix([
            [
                np.cos(theta / 2),
                -np.exp(1j * lam) * np.sin(theta / 2)
            ],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2)
            ]])
    
    def multiply_all(self, matrices):
        product = np.eye(2**self.n_qubit)
        for i in range(len(matrices)):
            product = np.matmul(matrices[i], product)
        return product
    
    def u3_layer(self, x, init=False):
        t = np.kron(self.u3_gate(*x[0:3]), self.u3_gate(*x[3:6]))
        if init:
            if self.n_qubit > 2:
                for i in range(2, self.n_qubit):
                    t = np.kron(t, self.u3_gate(*x[i*3:i*3+3]))
        return t   

    def kronk_identity(self, gate, positions=[0,1]):
        gate_list = []
        # assume only use two-qubit gates on nearest neighbours of a linear array
        # TODO for any pairs
        if positions[0]>self.n_qubit-2:
            raise ValueError('pj={pj} is too big ')
        for i in range(self.n_qubit):
            if i not in positions:
                gate_list.append(self.identity)
            elif i == positions[0]:
                gate_list.append(gate)
            else:
                pass 
        if len(gate_list) > 1:
            t = np.kron(gate_list[0], gate_list[1])
            if len(gate_list) > 2:
                for i in range(2, len(gate_list)):
                    t = np.kron(t, gate_list[i])
        else:
            t = gate_list[0]
        return t
    
    def n_layer_unitary(self, nth_layer, params, previous_layers, positions):
        # it might be better to use a q simulator instead of numpy matrix multiplication
        
        if len(self.two_qubit_gate_params):
            two_qubit_gate = self.two_qubit_gate(*self.two_qubit_gate_params)
        else:
            two_qubit_gate = self.two_qubit_gate()
        gate_list = []
        idx = 3*self.n_qubit
        gate_list.append(self.u3_layer(params[0:idx], init=True))
        if nth_layer:
            for i in range(nth_layer):
                pj = previous_layers[i]
                if pj> self.n_qubit -2:
                    raise ValueError('pj={pj} is too big ')
                gate_list.append(self.kronk_identity(two_qubit_gate, positions=[pj, pj+1]))
                gate_list.append(self.kronk_identity(self.u3_layer(params[idx:idx+6]), positions=[pj, pj+1]))
                idx += 6

        gate_list.append(self.kronk_identity(two_qubit_gate, positions=positions))
        gate_list.append(self.kronk_identity(self.u3_layer(params[idx:idx+6]), positions=positions))
        
        return self.multiply_all(gate_list)

    def get_num_params(self, n_layers):
        return 3*self.n_qubit+6*(n_layers+1)

class MultiQubitGateSynthesizer:
    def __init__(self, target_unitary, gate_template_obj):
        self.target_unitary = target_unitary
        self.gate_template_obj = gate_template_obj
        self.n_qubit = int(np.log2(np.shape(target_unitary)[0]))
        
    def unitary_distance_function(self, A, B):
        """
        TODO: Explore other distance functions
        Can we output a "fidelity" value from here to specify the quality of approximation?
        """
        #return (1 - np.abs(np.sum(np.multiply(B,np.conj(np.transpose(A))))) / np.shape(A)[0])
        # return (1 - (np.abs(np.sum(np.multiply(B,np.conj(A)))))**2+np.shape(A)[0] / np.shape(A)[0]**2) # quantum volume paper
        return (1 - np.abs(np.sum(np.multiply(B,np.conj(A)))) / np.shape(A)[0])

    def make_cost_function(self, n_layers, previous_layers, positions):
        target_unitary = self.target_unitary
        def cost_function(x):
            A = self.gate_template_obj.n_layer_unitary(n_layers, x, previous_layers, positions)
            B = target_unitary
            return self.unitary_distance_function(A, B)
        return cost_function
        
    def get_num_params(self, n_layers):
        return self.gate_template_obj.get_num_params(n_layers)

    def rand_initialize(self, n_layers):
        params = self.get_num_params(n_layers)
        return [np.pi*2*np.random.random() for i in range(params)]

    def gen_constraints(self, n_layers):
        params = self.get_num_params(n_layers)
        cons = []
        for i in range(params):
            cons.append({
                'type': 'ineq', 
                'fun': lambda x: -x[i] + 2*np.pi 
            })
        return cons
    
    def solve_instance(self, n_layers, trials, previous_layers):
        current_layers = []
        constraints = self.gen_constraints(n_layers)
        results = []
        best_idx = 0
        best_val = float('inf')
        best_loc = 0
        # TODO j needs to be variational
        # I tried to choose j that has minimises the cost function, but somehow not working well
        # ks = [(int(n_layers/2))%(self.n_qubit-1), int(n_layers)%(self.n_qubit-1)]
        if self.gate_template_obj.two_qubit_gate_params == [0, np.pi]:
            ks = [int(n_layers)%(self.n_qubit-1)]
        else:
            ks = [int(n_layers/2)%(self.n_qubit-1)]
        for j in range(1):
            # if using syc gates
            # k = (int(n_layers/2))%(self.n_qubit-1)
            # if using cphase gates
            # k = int(n_layers)%(self.n_qubit-1)
            k = ks[j]
            self.cost_function = self.make_cost_function(n_layers, previous_layers, positions=[k, k+1])
            for i in range(trials):
                init = self.rand_initialize(n_layers)
                res = minimize(self.cost_function, init, method='BFGS',
                               #constraints=constraints,
                               options={'maxiter':1000*30})
                results.append(res)
                current_layers.append(k)
                if best_val > res.fun:
                    best_val = res.fun
                    best_idx = i+j*trials  
                    best_loc = i+j*trials         
        return results[best_idx], current_layers[best_loc]

    def optimal_decomposition(self, tol=1e-5, fidelity_2q_gate=[1.0], fidelity_1q_gate=[1.0, 1.0], trials=1, eval_fidelty=True):
        max_num_layers = 8
        cutoff_with_tol=True
        results = []
        fidelity = []
        best_idx = 0
        best_val = float('inf')
        best_fidelity = 0
        previous_layers = []
        for layer in range(max_num_layers):
            if cutoff_with_tol and best_fidelity > 1.0-tol:
                break 
            res = self.solve_instance(n_layers=layer, trials=trials, previous_layers=previous_layers)
            results.append(res[0])
            previous_layers.append(res[1])
            print('current two-qubit positions', res[1], 'current cost function', res[0].fun)

            # evaluate the fidelity after adding one layer
            hw_fidelity = 1
            if eval_fidelty:
                for fi in fidelity_1q_gate:
                    hw_fidelity *= fi 
                if layer:
                    for j in previous_layers[1:]:
                        hw_fidelity *= fidelity_2q_gate[j]
                        for pi in range(self.n_qubit):
                            if pi not in [j, j+1]:
                                hw_fidelity *= fidelity_1q_gate[pi]

            unitary_fidelity = 1.0 - res[0].fun
            current_fidelity = hw_fidelity * unitary_fidelity
            fidelity.append(current_fidelity)
            
            # Update if the best_fidelity so far has been 0 (initial case)            
            if best_fidelity == 0:
                best_idx = layer
                best_fidelity = current_fidelity
                
            # Update if the current value is much smaller than the previous minimum 
            if current_fidelity - best_fidelity > tol*0.1:
                best_idx = layer
                best_fidelity = current_fidelity

        return best_idx+1, results[best_idx], previous_layers, fidelity[best_idx]
        
           
def _driver_func(target_unitary, gate_def, gate_param, fidelity_2q_gate, fidelity_1q_gate, tol, trials=1, eval_fidelty=True):
    attempts = 1
    for i in range(attempts): #Max 3 attempts. Typically 1st attempt always succeeds
        gt = GateTemplate(gate_def, gate_param, int(np.log2(np.shape(target_unitary)[0])))
        gs = MultiQubitGateSynthesizer(target_unitary, gt)
        layer_count, result_obj, pre_layers, fidelity = gs.optimal_decomposition(tol,
                                                                     fidelity_2q_gate,
                                                                     fidelity_1q_gate, 
                                                                     trials=trials, eval_fidelty=eval_fidelty)
        if result_obj.success == True:
            return [layer_count, result_obj, pre_layers, fidelity]
        else:
            if i==attempts-1:
                return [layer_count, result_obj, [],0.0]

class MultiQubitGateReplacementPass:
    """
    Takes a hardware mapped/routed Qiskit circuit and performs gate replacement
    Parallelizes over two-qubit gates and different hardware gates (jobs = n_algorithm_gates * n_hardware_gates)
    """
    def __init__(self, 
                 gate_defs, 
                 gate_params, 
                 gate_labels, 
                 fidelity_dict_2q_gate, 
                 fidelity_list_1q_gate, 
                 tol=1e-3):
        self.gate_defs = gate_defs
        self.gate_params = gate_params
        self.gate_labels = gate_labels
        self.fidelity_dict_2q_gate = fidelity_dict_2q_gate
        self.fidelity_list_1q_gate = fidelity_list_1q_gate
        self.tol = tol
        self.num_target_gates = len(self.gate_defs)
        
        assert self.num_target_gates == len(self.gate_params)
        assert self.num_target_gates == len(self.gate_labels)
        
    def run(self, circ, num_threads=1, trials=5, eval_fidelty=False):
        job_list = []
        node_list = {}
        dag = circuit_to_dag(circ)     
        job_id = 0
        for gate in dag.topological_op_nodes():
            if len(gate.qargs) == 1:
                continue                              
            target_unitary = get_gate_unitary_qiskit(gate.op)           
            
            gate_tup = [gate.qargs[i].index for i in range(len(gate.qargs))]
                 
            for i in range(self.num_target_gates):
                fidelity_2q_gate = []
                for j in range(len(gate_tup)-1):
                    idx1 = min(gate_tup[j], gate_tup[j+1])
                    idx2 = max(gate_tup[j], gate_tup[j+1])
                    fidelity_2q_gate.append(self.fidelity_dict_2q_gate[(idx1, idx2)][i])
                fidelity_1q_gate = tuple([self.fidelity_list_1q_gate[i] for i in gate_tup])
                
                job_list.append((
                    target_unitary,
                    self.gate_defs[i],
                    self.gate_params[i],
                    fidelity_2q_gate,
                    fidelity_1q_gate,
                    self.tol,
                    trials,
                    eval_fidelty
                ))
                node_list[(gate, i)] = job_id
                job_id += 1
        #print(job_list)
        
        if len(job_list) == 1:
            results = [_driver_func(*job_list[0])]
        else:         
            start = time.time()
            #print("Jobs:", len(job_list))
            #print("Threads:", num_threads)
            pool = mp.Pool(num_threads)
            #starmap guarentees ordering of results
            results = pool.starmap(_driver_func, job_list)
            pool.close()    
            end = time.time()
            #print("Compile time:", end - start)
        
        #Stitch outputs
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        # qr = new_dag.qregs['q']
        new_circ = dag_to_circuit(new_dag)
        
        for gate in dag.topological_op_nodes():
            if len(gate.qargs) == 1:
                new_circ.compose(gate.op, gate.qargs, gate.cargs, inplace=True)
                continue
            #Pick out the best implementation for this gate
            best_fidelity = 0
            best_res_obj = None
            best_layer_count = 0
            best_layers = None
            best_idx = 0 
            for i in range(self.num_target_gates):
                tmp = results[node_list[(gate, i)]]
                if tmp == []:
                    continue
                else:
                    my_layer_count, my_result_obj, my_layers, my_fidelity = results[node_list[(gate, i)]]
                if my_fidelity >= best_fidelity:
                    best_fidelity = my_fidelity
                    best_res_obj = my_result_obj
                    best_layer_count = my_layer_count
                    best_layers = my_layers
                    best_idx = i
                    
            print(best_fidelity)
            print(best_res_obj.fun)
            idx = best_idx
            gate_func = self.gate_defs[idx]
            param = self.gate_params[idx]
            n_layers = best_layer_count
            all_layers = best_layers
            angles = best_res_obj.x

            param_idx = 0
           
            for q in range(len(gate.qargs)):
                q = len(gate.qargs) - 1 - q
                new_circ.u3(angles[param_idx], angles[param_idx+1], angles[param_idx+2], [gate.qargs[q]])
                param_idx += 3

            for i in range(n_layers):
                qbt = all_layers[i]
                qbt = len(gate.qargs) - 1 - qbt
                new_circ.unitary(Operator(gate_func(*param)), 
                                [gate.qargs[qbt].index, gate.qargs[qbt-1].index], 
                                label=self.gate_labels[idx])
                new_circ.u3(angles[param_idx], angles[param_idx+1], angles[param_idx+2],
                            [gate.qargs[qbt].index])
                new_circ.u3(angles[param_idx+3], angles[param_idx+4], angles[param_idx+5],
                            [gate.qargs[qbt-1].index])
                param_idx += 6

        optimized_dag = optimise1qgates.run(circuit_to_dag(new_circ))
        new_circ = dag_to_circuit(optimized_dag)
        return new_circ             