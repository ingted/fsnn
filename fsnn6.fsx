open System
open System.Collections.Concurrent

type axonId = int
type dendriteId = int
type tick = int64
type synapseId = int
type synapsePositionId = int //一個軸突可以連接到一個突觸的多個位置
type electronicSignal = float
type chemicalSignal = float
type activated = bool
type windowSize = int

//neuronInput: 一個樹突結構會接到多個突觸位置，有不同電信號
type neuronInput = ConcurrentDictionary<tick, ConcurrentDictionary<dendriteId, ConcurrentDictionary<synapseId, (synapsePositionId*electronicSignal)[]>>>

//neuronOutput: 一個軸突也會接到多個突觸位置，但是輸出信號相同(激發態 or 非激發態) 
type neuronOutput = ConcurrentDictionary<tick, ConcurrentDictionary<synapseId, (synapsePositionId*activated)[]>> 
type activationCurve = windowSize -> neuronInput -> neuronOutput -> activated 

//突觸範圍假設足夠小，化學信號能有效影響整個範圍，輸出受自我歷史化學信號與軸突輸入電信號影響
type aggregationCurve = windowSize -> activated[] -> chemicalSignal[] -> electronicSignal * chemicalSignal

type Neuron = { 
    id: axonId
    output: neuronOutput
    input: neuronInput
    activationCurve: activationCurve
}
type SynapseRegion = { 
    id: synapseId; 
    connection: ConcurrentDictionary<tick, (synapsePositionId*axonId)[] * (synapsePositionId*dendriteId)[]>
    aggregationCurve: aggregationCurve
}