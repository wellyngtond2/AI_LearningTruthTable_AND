import numpy as np


_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
_output = np.array([0,0,0,1])
_weights = np.array([0.0,0.0])

learningRate = 0.5

def Sum (e,p):
    return e.dot(p)

s = Sum(_inputs,_weights)

def StepFunctions(sum):
    if(sum >= 1):
        return 1
    return 0

def CalculationOutput(inpt):
    s = inpt.dot(_weights) # (1 * 0.5) + (0 * 0.5)
    return StepFunctions(s)


def LearnUpdate():
    totalError = 1
    while (totalError != 0):
        totalError = 0
        for op in range(len(_output)):
            calcOutput = CalculationOutput(np.array(_inputs[op]))
            erro = abs(_output[op] - calcOutput)
            totalError += erro
            for wt in range(len(_weights)):
                _weights[wt] = _weights[wt] + (learningRate * _inputs[op][wt] * erro)
                print('Updated weights: ' + str(_weights[wt]))
        print('total errors: ' + str(totalError))
        
 

LearnUpdate()