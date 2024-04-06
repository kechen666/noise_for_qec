# For simulation
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeSingaporeV2

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel

from util import  interleaved_rb_QEC, get_noise_model, analytical_simulation, total_variation_distance

from qtcodes import XXZZQubit, RepetitionQubit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer

if __name__ == "__main__":
    backend = AerSimulator.from_backend(FakeSingaporeV2())
    backend_noise_model = NoiseModel.from_backend(backend)
    
    lengths = np.arange(1, 121+1, 30)
    num_samples = 10
    seed = 1010
    h_gate_error, id_gate_error, cx_gate_error, unique_pairs = interleaved_rb_QEC(backend, lengths, num_samples, seed)
    
    sim_noise = get_noise_model(h_gate_error, id_gate_error, cx_gate_error, backend, backend_noise_model)
    
    # RepetitionQubit
    print("RepetitionQubit")
    qubit_rep = RepetitionQubit({"d":3},"t")
    qubit_rep.stabilize()
    qubit_rep.stabilize()
    # qubit.x()
    qubit_rep.lattice_readout_z()
    
    circ_rep = qubit_rep.circ
    
    counts1_sum_rep, counts2_sum_rep = analytical_simulation(sim_noise, backend, circ_rep, repeat_times=100, shot_times=1024, print_times=10)
    
    probabilities_A, probabilities_B = total_variation_distance(counts1_sum_rep, counts2_sum_rep)
    
    # XXZZ
    print("XXZZQubit")
    qubit_xxzz = XXZZQubit() #d=3 default
    # print(qubit)
    qubit_xxzz.stabilize()
    qubit_xxzz.stabilize()
    qubit_xxzz.lattice_readout_z()
    
    circ_xxzz = qubit_xxzz.circ
    
    counts1_sum_xxzz, counts2_sum_xxzz = analytical_simulation(sim_noise, backend, circ_xxzz, repeat_times=100, shot_times=1024, print_times=10)
    
    probabilities_A, probabilities_B = total_variation_distance(counts1_sum_xxzz, counts2_sum_xxzz)