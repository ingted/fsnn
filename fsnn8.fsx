open System

/// A richer implementation that extends the fsnn6 MVP type design so it can execute
/// the original four-pre/two-post proof-of-concept while remaining flexible enough
/// for future FSharp.FGL and GPU work.
module Fsnn =

    type CellId = Guid
    type AxonId = Guid
    type DendriteId = Guid
    type SynapseRegionId = Guid
    type SynapsePositionId = int

    type tick = int64
    type windowSize = int
    type electronicSignal = float
    type chemicalSignal = float
    type activated = bool

    type ActivationState =
        { electrical: electronicSignal
          chemical: chemicalSignal
          activated: activated }

    type ActivationContext =
        { tick: tick
          inputs: electronicSignal list
          previousChemical: chemicalSignal }

    type ActivationCurve = ActivationContext -> ActivationState

    type AggregationContext =
        { tick: tick
          inputs: (electronicSignal * activated) list
          previousChemical: chemicalSignal }

    type AggregationCurve = AggregationContext -> electronicSignal * chemicalSignal

    type Axon =
        { id: AxonId
          neuronId: CellId
          name: string }

    type Dendrite =
        { id: DendriteId
          neuronId: CellId
          name: string }

    type Neuron =
        { id: CellId
          name: string
          activation: ActivationCurve
          mutable state: ActivationState
          axons: Axon list
          dendrites: Dendrite list }

    type SynapseRegion =
        { id: SynapseRegionId
          name: string
          aggregation: AggregationCurve
          mutable electrical: electronicSignal
          mutable chemical: chemicalSignal }

    type AxonProjection =
        { id: Guid
          axonId: AxonId
          regionId: SynapseRegionId
          weight: float }

    type RegionProjection =
        { id: Guid
          regionId: SynapseRegionId
          dendriteId: DendriteId
          weight: float }

    type NeuralGraph =
        { neurons: Neuron list
          axons: Axon list
          dendrites: Dendrite list
          regions: SynapseRegion list
          axonToRegion: AxonProjection list
          regionToDendrite: RegionProjection list }

    type NeuronBlueprint =
        { name: string
          activation: ActivationCurve
          initialElectrical: electronicSignal
          initialChemical: chemicalSignal
          axonCount: int
          dendriteCount: int }

    type SynapseRegionBlueprint =
        { name: string
          aggregation: AggregationCurve }

    type AxonStimulus =
        { axonId: AxonId
          value: electronicSignal }

    type ConnectionCardinality =
        | OneToOne
        | UpTo of int

    type ConnectionPolicy =
        { cardinality: ConnectionCardinality
          allowDuplicates: bool }

    type WiringStrategy =
        { rng: Random
          axonPolicy: ConnectionPolicy
          dendritePolicy: ConnectionPolicy }

    type ProcessingUnit =
        | CPU
        | GPU of string

    /// Tick snapshot keeps enough state for validation and future actor forwarding.
    type TickSnapshot =
        { tick: tick
          device: ProcessingUnit
          regionSignals: (string * (electronicSignal * chemicalSignal)) list
          neuronStates: (string * ActivationState) list }

    // -------- Activation & aggregation curves --------
    module ActivationCurves =

        let constant value : ActivationCurve =
            fun ctx ->
                let chemical = (ctx.previousChemical * 0.95) + value
                { electrical = value
                  chemical = chemical
                  activated = value > 0.0 }

        let threshold thresholdValue leak gain : ActivationCurve =
            fun ctx ->
                let total = ctx.inputs |> List.sum
                let isActive = total >= thresholdValue
                let electrical = if isActive then 1.0 else 0.0
                let chemical =
                    (ctx.previousChemical * leak) + (if isActive then gain else 0.0)
                { electrical = electrical
                  chemical = chemical
                  activated = isActive }

        let passThrough leak : ActivationCurve =
            fun ctx ->
                let total = ctx.inputs |> List.sum
                let chemical = (ctx.previousChemical * leak) + total
                { electrical = total
                  chemical = chemical
                  activated = total > 0.0 }

    module AggregationCurves =

        let weightedSum gain leak : AggregationCurve =
            fun ctx ->
                let raw = ctx.inputs |> List.sumBy fst
                let electrical = raw * gain
                let chemical = (ctx.previousChemical * leak) + electrical
                electrical, chemical

        let spikeOrHold leak : AggregationCurve =
            fun ctx ->
                let electrical =
                    ctx.inputs
                    |> List.tryFind (snd >> id)
                    |> Option.map fst
                    |> Option.defaultValue 0.0
                let chemical = (ctx.previousChemical * leak) + electrical
                electrical, chemical

    // -------- Builders --------
    module Builder =

        let private createCells (blueprints: NeuronBlueprint list) =
            let neurons = ResizeArray<Neuron>()
            let axons = ResizeArray<Axon>()
            let dendrites = ResizeArray<Dendrite>()

            for bp in blueprints do
                let cellId = Guid.NewGuid()
                let createdAxons : Axon list =
                    [ for idx in 1 .. bp.axonCount ->
                          { id = Guid.NewGuid()
                            neuronId = cellId
                            name = $"{bp.name}.axon{idx}" } ]

                let createdDendrites : Dendrite list =
                    [ for idx in 1 .. bp.dendriteCount ->
                          { id = Guid.NewGuid()
                            neuronId = cellId
                            name = $"{bp.name}.den{idx}" } ]

                let neuron =
                    { id = cellId
                      name = bp.name
                      activation = bp.activation
                      state =
                        { electrical = bp.initialElectrical
                          chemical = bp.initialChemical
                          activated = bp.initialElectrical > 0.0 }
                      axons = createdAxons
                      dendrites = createdDendrites }

                neurons.Add(neuron)
                createdAxons |> List.iter axons.Add
                createdDendrites |> List.iter dendrites.Add

            (List.ofSeq neurons, List.ofSeq axons, List.ofSeq dendrites)

        let private createRegions (blueprints: SynapseRegionBlueprint list) =
            blueprints
            |> List.map (fun bp ->
                { id = Guid.NewGuid()
                  name = bp.name
                  aggregation = bp.aggregation
                  electrical = 0.0
                  chemical = 0.0 })

        module RandomConnect =

            let private pickTargets (policy: ConnectionPolicy) (rng: Random) (items: 'a array) =
                if items.Length = 0 then
                    Array.empty
                else
                    let desire =
                        match policy.cardinality with
                        | OneToOne -> 1
                        | UpTo n -> max 1 (min n items.Length)
                    if policy.allowDuplicates then
                        Array.init desire (fun _ -> items[rng.Next(items.Length)])
                    else
                        let bag = ResizeArray(items)
                        let acc = ResizeArray()
                        while acc.Count < desire && bag.Count > 0 do
                            let idx = rng.Next(bag.Count)
                            acc.Add(bag[idx])
                            bag.RemoveAt(idx)
                        acc.ToArray()

            let connectAxons policy rng (axons: Axon list) (regions: SynapseRegion list) =
                let regionArr = regions |> List.toArray
                [
                    for axon in axons do
                        let targets = pickTargets policy rng regionArr
                        for region in targets do
                            yield
                                { id = Guid.NewGuid()
                                  axonId = axon.id
                                  regionId = region.id
                                  weight = 1.0 }
                ]

            let connectRegions policy rng (regions: SynapseRegion list) (dendrites: Dendrite list) =
                let dendArr = dendrites |> List.toArray
                [
                    for region in regions do
                        let targets = pickTargets policy rng dendArr
                        for dendrite in targets do
                            yield
                                { id = Guid.NewGuid()
                                  regionId = region.id
                                  dendriteId = dendrite.id
                                  weight = 1.0 }
                ]

        let initNetwork cellBlueprints regionBlueprints wiringStrategy =
            let neurons, axons, dendrites = createCells cellBlueprints
            let regions = createRegions regionBlueprints

            let axonToRegion, regionToDendrite =
                match wiringStrategy with
                | None -> ([], [])
                | Some strategy ->
                    let a2r =
                        RandomConnect.connectAxons
                            strategy.axonPolicy
                            strategy.rng
                            axons
                            regions
                    let r2d =
                        RandomConnect.connectRegions
                            strategy.dendritePolicy
                            strategy.rng
                            regions
                            dendrites
                    (a2r, r2d)

            { neurons = neurons
              axons = axons
              dendrites = dendrites
              regions = regions
              axonToRegion = axonToRegion
              regionToDendrite = regionToDendrite }

    // -------- Simulation --------
    module Simulation =

        let private runCpu (graph: NeuralGraph) (stimuli: AxonStimulus list) tick =
            let stimMap =
                stimuli
                |> List.fold (fun acc s -> Map.add s.axonId s.value acc) Map.empty

            let axonMap =
                graph.axons
                |> List.map (fun a -> a.id, a)
                |> Map.ofList

            let neuronMap =
                graph.neurons
                |> List.map (fun n -> n.id, n)
                |> Map.ofList

            let dendriteMap =
                graph.dendrites
                |> List.map (fun d -> d.id, d)
                |> Map.ofList

            let regionMap =
                graph.regions
                |> List.map (fun r -> r.id, r)
                |> Map.ofList

            let getAxonSignal axonId =
                match Map.tryFind axonId stimMap with
                | Some value -> value, value > 0.0
                | None ->
                    let axon = Map.find axonId axonMap
                    let neuron = Map.find axon.neuronId neuronMap
                    neuron.state.electrical, neuron.state.activated

            let regionInputs : Map<SynapseRegionId, (electronicSignal * activated) list> =
                graph.axonToRegion
                |> List.fold
                    (fun acc proj ->
                        let value, active = getAxonSignal proj.axonId
                        let weighted = value * proj.weight
                        let entry = weighted, active
                        let bucket = Map.tryFind proj.regionId acc |> Option.defaultValue []
                        Map.add proj.regionId (entry :: bucket) acc)
                    Map.empty

            let regionOutputs : Map<SynapseRegionId, (electronicSignal * chemicalSignal)> =
                graph.regions
                |> List.map (fun region ->
                    let inputs = Map.tryFind region.id regionInputs |> Option.defaultValue []
                    let ctx =
                        { tick = tick
                          inputs = inputs
                          previousChemical = region.chemical }
                    let electrical, chemical = region.aggregation ctx
                    region.electrical <- electrical
                    region.chemical <- chemical
                    region.id, (electrical, chemical))
                |> Map.ofList

            let addNeuronInput neuronId value bucket =
                let current = Map.tryFind neuronId bucket |> Option.defaultValue []
                Map.add neuronId (value :: current) bucket

            let neuronInputs : Map<CellId, electronicSignal list> =
                graph.regionToDendrite
                |> List.fold
                    (fun acc proj ->
                        let dendrite = Map.find proj.dendriteId dendriteMap
                        let signal =
                            regionOutputs
                            |> Map.tryFind proj.regionId
                            |> Option.defaultValue (0.0, 0.0)
                            |> fst
                            |> fun v -> v * proj.weight
                        addNeuronInput dendrite.neuronId signal acc)
                    Map.empty

            let neuronStates : (string * ActivationState) list =
                graph.neurons
                |> List.map (fun neuron ->
                    let inputs = Map.tryFind neuron.id neuronInputs |> Option.defaultValue []
                    let ctx : ActivationContext =
                        { tick = tick
                          inputs = inputs |> List.rev
                          previousChemical = neuron.state.chemical }
                    let updated = neuron.activation ctx
                    neuron.state <- updated
                    neuron.name, updated)

            { tick = tick
              device = CPU
              regionSignals =
                graph.regions
                |> List.map (fun region -> region.name, (region.electrical, region.chemical))
              neuronStates = neuronStates }

        let run device graph stimuli tick =
            match device with
            | CPU -> runCpu graph stimuli tick
            | GPU backend ->
                let message =
                    $"GPU backend '{backend}' is not implemented yet. The CPU path is used today but the signature keeps the hook for future TorchSharp integration."
                raise (NotSupportedException(message))

    // -------- POC Example --------
    module Example =

        let private rng = Random(13)

        let wiringStrategy =
            { rng = rng
              axonPolicy = { cardinality = UpTo 2; allowDuplicates = false }
              dendritePolicy = { cardinality = UpTo 2; allowDuplicates = true } }

        let preInputs =
            [ "pre1", 1.0
              "pre2", 1.0
              "pre3", 0.0
              "pre4", 1.0 ]

        let cellBlueprints =
            let pres =
                preInputs
                |> List.map (fun (name, value) ->
                    { name = name
                      activation = ActivationCurves.constant value
                      initialElectrical = value
                      initialChemical = value
                      axonCount = 1
                      dendriteCount = 0 })

            let posts =
                [ { name = "post1"
                    activation = ActivationCurves.threshold 0.75 0.4 0.8
                    initialElectrical = 0.0
                    initialChemical = 0.0
                    axonCount = 1
                    dendriteCount = 2 }
                  { name = "post2"
                    activation = ActivationCurves.threshold 0.25 0.4 0.8
                    initialElectrical = 0.0
                    initialChemical = 0.0
                    axonCount = 1
                    dendriteCount = 2 } ]

            pres @ posts

        let regionBlueprints =
            [ { name = "r1"; aggregation = AggregationCurves.weightedSum 1.0 0.25 }
              { name = "r2"; aggregation = AggregationCurves.weightedSum 1.0 0.25 }
              { name = "r3"; aggregation = AggregationCurves.weightedSum 1.0 0.25 } ]

        let previewRandomWiring () =
            let preview = Builder.initNetwork cellBlueprints regionBlueprints (Some wiringStrategy)
            printfn "Preview wiring: axon->region=%d region->dendrite=%d" preview.axonToRegion.Length preview.regionToDendrite.Length

        let private baseGraph = Builder.initNetwork cellBlueprints regionBlueprints None

        let private findAxon name =
            baseGraph.axons |> List.find (fun axon -> axon.name = name)

        let private findRegion name =
            baseGraph.regions |> List.find (fun region -> region.name = name)

        let private findDendrite name =
            baseGraph.dendrites |> List.find (fun den -> den.name = name)

        let graph =
            let axonToRegion =
                [ { id = Guid.NewGuid()
                    axonId = (findAxon "pre1.axon1").id
                    regionId = (findRegion "r1").id
                    weight = 1.0 }
                  { id = Guid.NewGuid()
                    axonId = (findAxon "pre2.axon1").id
                    regionId = (findRegion "r2").id
                    weight = 0.5 }
                  { id = Guid.NewGuid()
                    axonId = (findAxon "pre3.axon1").id
                    regionId = (findRegion "r2").id
                    weight = 0.5 }
                  { id = Guid.NewGuid()
                    axonId = (findAxon "pre4.axon1").id
                    regionId = (findRegion "r3").id
                    weight = 1.0 } ]

            let regionToDendrite =
                [ { id = Guid.NewGuid()
                    regionId = (findRegion "r1").id
                    dendriteId = (findDendrite "post1.den1").id
                    weight = 1.0 }
                  { id = Guid.NewGuid()
                    regionId = (findRegion "r2").id
                    dendriteId = (findDendrite "post1.den2").id
                    weight = 1.0 }
                  { id = Guid.NewGuid()
                    regionId = (findRegion "r2").id
                    dendriteId = (findDendrite "post2.den1").id
                    weight = 1.0 }
                  { id = Guid.NewGuid()
                    regionId = (findRegion "r3").id
                    dendriteId = (findDendrite "post2.den2").id
                    weight = 1.0 } ]

            { baseGraph with
                axonToRegion = axonToRegion
                regionToDendrite = regionToDendrite }

        let private stimuliFrom inputs =
            inputs
            |> List.map (fun (name, value) ->
                { axonId = (findAxon $"{name}.axon1").id
                  value = value })

        let runDeterministic () =
            let stimuli = stimuliFrom preInputs
            Simulation.run CPU graph stimuli 0L

#if INTERACTIVE
    Example.previewRandomWiring ()
    let snapshot = Example.runDeterministic ()
    printfn "post1 electrical=%f" (snapshot.neuronStates |> List.find (fst >> (=) "post1") |> snd |> fun s -> s.electrical)
    printfn "post2 electrical=%f" (snapshot.neuronStates |> List.find (fst >> (=) "post2") |> snd |> fun s -> s.electrical)
#endif
