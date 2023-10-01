import logic_gate as lg

logic_gate=lg.LogicGate()

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