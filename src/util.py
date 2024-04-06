import numpy as np
from qiskit_experiments.library import InterleavedRB
import qiskit.circuit.library as circuits

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

from qiskit_aer import AerSimulator

from scipy.stats import variation

def interleaved_rb_QEC(backend, lengths, num_samples, seed):
    
    num_qubits = backend.num_qubits
    
    
    # get couple map for cnot
    coordinate_pairs = backend.coupling_map.get_edges()
    unique_pairs = []
    for pair in coordinate_pairs:
        # 检查反转的边是否已经存在
        if (pair[1], pair[0]) not in unique_pairs:
            # 如果不存在，将边添加到集合中
            unique_pairs.append(pair)
    
    # for H gate
    h_gate_error = {}
    id_gate_error = {}
    cx_gate_error = {}
    
    qec_basis_gate = ['h', 'id', 'cx']
    
    for gate in qec_basis_gate:
        print(f"{gate}")
        if gate == 'h':
            for qubit_id in range(num_qubits):
                qubits = (qubit_id,)
                int_exp2 = InterleavedRB(
                    circuits.HGate(), qubits, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                h_gate_error[qubit_id] = int_results2[2].value
                print(f"{qubit_id}_")
        elif gate == 'id':
            for qubit_id in range(num_qubits):
                qubits = (qubit_id,)
                int_exp2 = InterleavedRB(circuits.IGate(), qubits, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                id_gate_error[qubit_id] = int_results2[2].value
                print(f"{qubit_id}_")
        elif gate == 'cx':
            for qubit_pairs in unique_pairs:
                int_exp2 = InterleavedRB(circuits.CXGate(), qubit_pairs, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                cx_gate_error[qubit_pairs] = int_results2[2].value
                print(f"{qubit_pairs}_")
                
    return h_gate_error, id_gate_error, cx_gate_error, unique_pairs

def get_noise_model(h_gate_error, id_gate_error, cx_gate_error, backend, backend_noise_model):

    # Create an empty noise model
    noise_model = NoiseModel()

    # consider down bound - (int_results2[2].value.std_dev)

    # Add H and id depolarizing error to qubit
    for qubit_id, depolarizing_param in h_gate_error.items():
        dep_error = depolarizing_error(1 - depolarizing_param.nominal_value, 1)
        noise_model.add_quantum_error(dep_error, ['h'], [qubit_id,])
    for qubit_id, depolarizing_param in id_gate_error.items():
        dep_error = depolarizing_error(1 - depolarizing_param.nominal_value, 1)
        noise_model.add_quantum_error(dep_error, ['id'], [qubit_id,])
    for qubit_pairs, depolarizing_param in cx_gate_error.items():
        dep_error = depolarizing_error(1 - depolarizing_param.nominal_value, 2)
        # Because the error about IRB, so we consider the cx error about (0,1) and (1,0) is same.
        noise_model.add_quantum_error(dep_error, ['cx'], qubit_pairs)
        noise_model.add_quantum_error(dep_error, ['cx'], (qubit_pairs[1], qubit_pairs[0]))
        
    # add readout error and reset error
    # NoiseModel.from_backend(backend)._local_readout_errors[(0,)]
    for i in range(backend.num_qubits):
        noise_model.add_readout_error(backend_noise_model._local_readout_errors[(i,)], [i,])
        # noise_model.add_quantum_error(backend_noise_model._local_quantum_errors['reset'][(i,)], ['reset'], [i,])

    sim_noise = AerSimulator(noise_model = noise_model)
    return sim_noise

def analytical_simulation(sim_noise, backend, circ, repeat_times=100, shot_times=100, print_times=10):
    counts1_sum = {}
    counts2_sum = {}
    for i in range(repeat_times):
        if i%print_times == 0:
            print(i)
        counts1 = sim_noise.run(circ,shots = shot_times).result().get_counts()
        counts2 = backend.run(circ,shots = shot_times).result().get_counts()
        for key, value in counts1.items():
            counts1_sum[key] = counts1_sum.get(key, 0) + value
        for key, value in counts2.items():
            counts2_sum[key] = counts2_sum.get(key, 0) + value

    from qiskit.quantum_info import hellinger_distance, hellinger_fidelity
    # print(counts1_sum, counts2_sum)
    hellinger_dis = hellinger_distance(counts1_sum, counts2_sum)

    hellinger_fid = hellinger_fidelity(counts1_sum, counts2_sum)
    print(f"hellinger_dis is: {hellinger_dis}, hellinger_fid is: {hellinger_fid}")
    return counts1_sum, counts2_sum

def total_variation_distance(counts1_sum, counts2_sum):
    distribution_A = counts1_sum
    distribution_B = counts2_sum

    # 提取分布的概率向量
    probabilities_A = {key: value / sum(distribution_A.values()) for key, value in distribution_A.items()}
    probabilities_B = {key: value / sum(distribution_B.values()) for key, value in distribution_B.items()}

    # 计算总变差距离
    total_variation_distance = sum(abs(probabilities_A.get(key, 0) - probabilities_B.get(key, 0)) for key in set(probabilities_A) | set(probabilities_B))

    print("总变差距离：", total_variation_distance)
    return probabilities_A, probabilities_B