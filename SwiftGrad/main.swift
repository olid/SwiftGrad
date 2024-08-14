//
//  main.swift
//  SwiftGrad
//

import Foundation

typealias Backward = () -> ()

class Value: CustomStringConvertible, Identifiable, Hashable {
    let id = UUID()
    var value: Double
    let label: String
    let children: [Value]
    var grad: Double
    private var backwardFn: () -> ()
    
    init(value: Double, label: String = "") {
        self.value = value
        self.children = []
        self.label = label
        self.grad = 0
        self.backwardFn = { }
    }
    
    private init(value: Double, children: [Value], label: String = "") {
        self.value = value
        self.children = children
        self.grad = 0
        self.label = label
        self.backwardFn = { }
    }
    
    func multiply(by other: Value, label: String = "") -> Value {
        let result = Value(value: value * other.value, children: [self, other], label: label)
        
        result.backwardFn = {
            self.grad += other.value * result.grad
            other.grad += self.value * result.grad
        }
        
        return result
    }
    
    func add(to other: Value, label: String = "") -> Value {
        let result = Value(value: value + other.value, children: [self, other], label: label)
        
        result.backwardFn = {
            self.grad += result.grad
            other.grad += result.grad
        }
        
        return result
    }
    
    func subtract(from other: Value, label: String = "") -> Value {
        let result = Value(value: other.value - value, children: [self, other], label: label)
        
        result.backwardFn = {
            self.grad += result.grad
            other.grad += result.grad
        }
        
        return result
    }
    
    func pow(other: Double, label: String = "") -> Value {
        let result = Value(value: powl(value, other), children: [self], label: label)
        
        result.backwardFn = {
            self.grad += other * (powl(self.value, other - 1)) * result.grad
        }
        
        return result
    }
    
    func divide(by other: Value, label: String = "") -> Value {
        self * other.pow(other: -1, label: "div")
    }
    
    func tanh(label: String = "") -> Value {
        let result = Value(value: tanhl(value), children: [self], label: label)
        
        result.backwardFn = {
            self.grad += (1 - (result.value * result.value)) * result.grad
        }
        
        return result
    }
    
    func exp(label: String = "") -> Value {
        let result = Value(value: powl(M_E, value), children: [self], label: label)
        
        result.backwardFn = {
            self.grad += result.value * result.grad
        }
        
        return result
    }
    
    func backward() {
        grad = 1.0
        
        var sorted = [Value]()
        var visited = Set<Value>()
        
        sort(node: self)
        
        for node in sorted.reversed() {
            node.backwardFn()
        }
        
        func sort(node: Value) {
            if !visited.contains(node) {
                visited.insert(node)
                
                for child in node.children {
                    sort(node: child)
                }
                
                sorted.append(node)
            }
        }
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    var description: String {
        "(\(label) value: \(value), grad: \(grad))[\(children.map { c in c.description }.joined(separator: ", "))]"
    }
    
    static func +(lhs: Value, rhs: Value) -> Value { lhs.add(to: rhs) }
    static func +(lhs: Value, rhs: Double) -> Value { lhs.add(to: Value(value: rhs)) }
    static func +(lhs: Double, rhs: Value) -> Value { Value(value: lhs).add(to: rhs) }
    
    static func -(lhs: Value, rhs: Value) -> Value { lhs.subtract(from: rhs) }
    static func -(lhs: Value, rhs: Double) -> Value { lhs.subtract(from: Value(value: rhs)) }
    static func -(lhs: Double, rhs: Value) -> Value { Value(value: lhs).subtract(from: rhs) }
    
    static func *(lhs: Value, rhs: Value) -> Value { lhs.multiply(by: rhs) }
    static func *(lhs: Value, rhs: Double) -> Value { lhs.multiply(by: Value(value: rhs)) }
    static func *(lhs: Double, rhs: Value) -> Value { Value(value: lhs).multiply(by: rhs) }
    
    static func /(lhs: Value, rhs: Value) -> Value { lhs.divide(by: rhs) }
    static func /(lhs: Value, rhs: Double) -> Value { lhs.divide(by: Value(value: rhs)) }
    static func /(lhs: Double, rhs: Value) -> Value { Value(value: lhs).divide(by: rhs) }
    
    static func ^(lhs: Value, rhs: Value) -> Value { lhs.pow(other: rhs.value) }
    static func ^(lhs: Value, rhs: Double) -> Value { lhs.pow(other: rhs) }
    static func ^(lhs: Double, rhs: Value) -> Value { Value(value: lhs).pow(other: rhs.value) }
    
    static func == (lhs: Value, rhs: Value) -> Bool {
        lhs.id == rhs.id
    }
}

struct Random {
    static func uniformBipolar() -> Double {
        Double(arc4random_uniform(0xffff)) / 0x7fff - 1
    }
}

struct Neuron {
    let weights: [Value]
    let bias: Value
    
    init(inputCount: Int) {
        weights = (0..<inputCount).map { _ in Value(value: Random.uniformBipolar(), label: "") }
        bias = Value(value: Random.uniformBipolar(), label: "")
    }
    
    var parameters: [Value] { weights + [bias] }
    
    func go(inputs: [Value]) -> Value {
        let activation = zip(weights, inputs).reduce(bias) { t, p in t + p.0 * p.1 }
        
        return activation.tanh(label: "")
    }
}

struct Layer {
    let neurons: [Neuron]
    
    init(inputCountPerNeuron: Int, outputs: Int) {
        neurons = (0..<outputs).map { _ in Neuron(inputCount: inputCountPerNeuron) }
    }
    
    var parameters: [Value] { neurons.flatMap { n in n.parameters } }
    
    func go(inputs: [Value]) -> [Value] {
        neurons.map { n in n.go(inputs: inputs) }
    }
}

struct MultiLayerPerceptron {
    let layers: [Layer]
    
    init(inputCount: Int, outputCounts: [Int]) {
        let counts = [inputCount] + outputCounts
        
        layers = (0..<outputCounts.count).map { ix in Layer(inputCountPerNeuron: counts[ix], outputs: counts[ix + 1]) }
    }
    
    var parameters: [Value] { layers.flatMap { l in l.parameters } }
    
    func go(inputs: [Double]) -> [Value] {
        var outs = inputs.map { i in Value(value: i, label: "") }
        
        for layer in layers {
            outs = layer.go(inputs: outs)
        }
        
        return outs
    }
    
    func nudge(amount: Double) {
        for parameter in parameters {
            parameter.value += -amount * parameter.grad
            parameter.grad = 0.0
        }
    }
}

let net = MultiLayerPerceptron(inputCount: 3, outputCounts: [4, 4, 1])

let examples: [[Double]] = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

let desiredOutput = [1.0, -1.0, -1.0, 1.0]

for _ in 1...50 {
    let predictions = examples.map { e in net.go(inputs: e) }
    
    let loss = zip(desiredOutput, predictions)
        .map { (d, p) in (d - p.first!).pow(other: 2.0) }
        .reduce(Value(value: 0.0)) { t, e in t + e }
    
    print(loss.value, predictions.flatMap { $0 }.map { $0.value })
    
    loss.backward()
    
    net.nudge(amount: 0.05)
}





