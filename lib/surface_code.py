from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

class SurfaceCode():


    def __init__(self, d: int = 3):
        """Generate lattice structure of surface code"""
        self.d = d
        self.q_data = QuantumRegister(d, name="data")
        self.q_ancilla = QuantumRegister(d-1, name="ancilla")
        self.circuit = QuantumCircuit(self.q_data, self.q_ancilla) # Data is copied

        return

    def measure_syndrome(n: int = 1):
        """This function goes through a single round of syndrome measurements

        If n is provided, it undergoes n rounds of syndrome measurement"""



        return

    def generate_bell_state(self):

        self.circuit.h(0)
        self.circuit.cx(0, 1)


    def simulate(self, shots: int = 10):

        sim = AerSimulator

        job = sim.run(circuits=self.circuit, shots=shots)
        result = job.result()

        return result
