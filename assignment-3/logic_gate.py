import numpy as np

class LogicGate:
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = self.x2 = None

    def print_output(self, gate):
        if gate == 'AND':
            print(f'{self.x1} AND {self.x2} is {self.out}')
        elif gate == 'OR':    
            print(f'{self.x1} OR {self.x2} is {self.out}')
        elif gate == 'XOR':
            print(f'{self.x1} XOR {self.x2} is {self.out}')
        else:
            print(f'{self.x1} NAND {self.x2} is {self.out}')

    def nand_gate(self, x1, x2):
        self.w1 = -0.2
        self.w2 = -0.4
        self.th = -0.5

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.sum(x * w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0   

    def xor_gate(self, x1, x2):
        self.w1 = -0.5
        self.w2 = -0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.sum(x * w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0   
    
    def or_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.sum(x * w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0        

    def and_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        
        if np.sum(x * w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0

if __name__ == "__main__":
    print("Usage: Create an instance of LogicGate and use its methods to perform logic operations.")
    print("Example:")
    
    logic_gate = LogicGate()
    
    x1 = 1
    x2 = 0
    
    result = logic_gate.and_gate(x1, x2)
    logic_gate.print_output('AND')
    
    result = logic_gate.or_gate(x1, x2)
    logic_gate.print_output('OR')
    
    result = logic_gate.xor_gate(x1, x2)
    logic_gate.print_output('XOR')
    
    result = logic_gate.nand_gate(x1, x2)
    logic_gate.print_output('NAND')