open System
open System.Collections.Generic

// ---------- Core domain ----------
type CellId = Guid
type AxonId = Guid
type DendriteId = Guid
type SynapseRegionId = Guid

type ActivationState =
    { electrical: float
      chemical: float }

type ActivationContext =
    { tick: int
      inputs: float list }

type ActivationCurve = ActivationContext -> ActivationState -> ActivationState

type Cell =
    { id: CellId
      name: string
      activation: ActivationCurve
      mutable state: ActivationState }

type Axon =
    { id: AxonId
      name: string
      cellId: CellId }

type Dendrite =
    { id: DendriteId
      name: string
      cellId: CellId }

type SynapseRegion =
    { id: SynapseRegionId
      name: string
      gain: float
      mutable signal: float }

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

/// All graph information is collected here so it can later be pushed into FSharp.FGL.
type NeuralGraph =
    { cells: Cell list
      axons: Axon list
      dendrites: Dendrite list
      regions: SynapseRegion list
      axonToRegion: AxonProjection list
      regionToDendrite: RegionProjection list }

type CellBlueprint =
    { name: string
      activation: ActivationCurve
      initialElectrical: float
      initialChemical: float
      axonCount: int
      dendriteCount: int }

type SynapseRegionBlueprint =
    { name: string
      gain: float }

type AxonStimulus =
    { axonId: AxonId
      value: float }

// ---------- Activation curves ----------
module ActivationCurves =

    let holdValue : ActivationCurve =
        fun _ state ->
            // Keeps pre-synaptic cells stable, but lets chemistry accumulate slowly.
            let chemical = (state.chemical * 0.95) + state.electrical
            { state with chemical = chemical }

    let passThrough : ActivationCurve =
        fun ctx state ->
            let value =
                match ctx.inputs with
                | [] -> state.electrical
                | xs -> List.sum xs
            let chemical = (state.chemical * 0.9) + value
            { electrical = value; chemical = chemical }

    let thresholdWithChemical threshold leak gain : ActivationCurve =
        fun ctx state ->
            let total = ctx.inputs |> List.sum
            let fired = if total >= threshold then 1.0 else 0.0
            let chemical = (state.chemical * leak) + (fired * gain)
            { electrical = fired; chemical = chemical }

// ---------- Builders ----------
module private Builders =

    let createCells (blueprints: CellBlueprint list) =
        let cells = ResizeArray<Cell>()
        let axons = ResizeArray<Axon>()
        let dendrites = ResizeArray<Dendrite>()

        for bp in blueprints do
            let cellId = Guid.NewGuid()
            let cell =
                { id = cellId
                  name = bp.name
                  activation = bp.activation
                  state =
                    { electrical = bp.initialElectrical
                      chemical = bp.initialChemical } }
            cells.Add(cell)

            for idx in 1 .. bp.axonCount do
                axons.Add(
                    { id = Guid.NewGuid()
                      name = $"{bp.name}.axon{idx}"
                      cellId = cellId }
                )

            for idx in 1 .. bp.dendriteCount do
                dendrites.Add(
                    { id = Guid.NewGuid()
                      name = $"{bp.name}.den{idx}"
                      cellId = cellId }
                )

        (List.ofSeq cells, List.ofSeq axons, List.ofSeq dendrites)

    let createRegions (blueprints: SynapseRegionBlueprint list) =
        blueprints
        |> List.map (fun bp ->
            { id = Guid.NewGuid()
              name = bp.name
              gain = bp.gain
              signal = 0.0 })

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

module RandomConnect =

    let private pickTargets (policy: ConnectionPolicy) (rng: Random) (items: 'a array) =
        if items.Length = 0 then
            Array.empty
        else
            let desired =
                match policy.cardinality with
                | OneToOne -> 1
                | UpTo n -> max 1 (min n items.Length)

            if policy.allowDuplicates then
                Array.init desired (fun _ -> items[rng.Next(items.Length)])
            else
                let bag = ResizeArray(items)
                let results = ResizeArray()
                while results.Count < desired && bag.Count > 0 do
                    let idx = rng.Next(bag.Count)
                    results.Add(bag[idx])
                    bag.RemoveAt(idx)
                results.ToArray()

    let connectAxons policy rng (axons: Axon list) (regions: SynapseRegion list) =
        let regionArray = regions |> List.toArray
        [
            for axon in axons do
                let targets = pickTargets policy rng regionArray
                for region in targets do
                    yield
                        { id = Guid.NewGuid()
                          axonId = axon.id
                          regionId = region.id
                          weight = 1.0 }
        ]

    let connectRegions policy rng (regions: SynapseRegion list) (dendrites: Dendrite list) =
        let dendriteArray = dendrites |> List.toArray
        [
            for region in regions do
                let targets = pickTargets policy rng dendriteArray
                for dendrite in targets do
                    yield
                        { id = Guid.NewGuid()
                          regionId = region.id
                          dendriteId = dendrite.id
                          weight = 1.0 }
        ]

let initNetwork cellBlueprints regionBlueprints wiringStrategy =
    let cells, axons, dendrites = Builders.createCells cellBlueprints
    let regions = Builders.createRegions regionBlueprints

    let (axonToRegion, regionToDendrite) =
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

    { cells = cells
      axons = axons
      dendrites = dendrites
      regions = regions
      axonToRegion = axonToRegion
      regionToDendrite = regionToDendrite }

// ---------- Simulation ----------
type TickSnapshot =
    { tick: int
      regionSignals: (string * float) list
      cellStates: (string * ActivationState) list }

module Simulation =
    /// Takes external axonId/value pairs and treats them as proxy synapse inputs for this tick.
    let runTick (graph: NeuralGraph) (stimuli: AxonStimulus list) tick =
        let stimMap =
            stimuli
            |> List.fold (fun acc s -> Map.add s.axonId s.value acc) Map.empty

        let axonMap = graph.axons |> List.map (fun a -> a.id, a) |> Map.ofList
        let cellMap = graph.cells |> List.map (fun c -> c.id, c) |> Map.ofList
        let dendriteMap = graph.dendrites |> List.map (fun d -> d.id, d) |> Map.ofList
        let regionMap = graph.regions |> List.map (fun r -> r.id, r) |> Map.ofList

        let getAxonSignal axonId =
            match Map.tryFind axonId stimMap with
            | Some value -> value
            | None ->
                let axon = Map.find axonId axonMap
                let cell = Map.find axon.cellId cellMap
                cell.state.electrical

        let regionInputs =
            graph.axonToRegion
            |> List.fold
                (fun acc proj ->
                    let contribution = (getAxonSignal proj.axonId) * proj.weight
                    match Map.tryFind proj.regionId acc with
                    | None -> Map.add proj.regionId contribution acc
                    | Some value -> Map.add proj.regionId (value + contribution) acc)
                Map.empty

        let regionSignals =
            graph.regions
            |> List.map (fun region ->
                let weighted = Map.tryFind region.id regionInputs |> Option.defaultValue 0.0
                let newSignal = region.gain * weighted
                region.signal <- newSignal
                region.id, newSignal)
            |> Map.ofList

        let addInput cellId value bucket =
            match Map.tryFind cellId bucket with
            | None -> Map.add cellId [ value ] bucket
            | Some xs -> Map.add cellId (value :: xs) bucket

        let cellInputs =
            graph.regionToDendrite
            |> List.fold
                (fun acc proj ->
                    let dendrite = Map.find proj.dendriteId dendriteMap
                    let cellId = dendrite.cellId
                    let signal =
                        (Map.tryFind proj.regionId regionSignals |> Option.defaultValue 0.0)
                        * proj.weight
                    addInput cellId signal acc)
                Map.empty

        let cellStates =
            graph.cells
            |> List.map (fun cell ->
                match Map.tryFind cell.id cellInputs with
                | None -> cell.name, cell.state
                | Some inputs ->
                    let ctx =
                        { tick = tick
                          inputs = List.rev inputs }
                    let updated = cell.activation ctx cell.state
                    cell.state <- updated
                    cell.name, updated)

        { tick = tick
          regionSignals =
            graph.regions
            |> List.map (fun region -> region.name, region.signal)
          cellStates = cellStates }

// ---------- Example based on the original POC ----------
module Example =

    let rng = Random(7)

    let wiringStrategy =
        { rng = rng
          axonPolicy =
            { cardinality = UpTo 2
              allowDuplicates = false }
          dendritePolicy =
            { cardinality = UpTo 2
              allowDuplicates = true } }

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
                  activation = ActivationCurves.holdValue
                  initialElectrical = value
                  initialChemical = value
                  axonCount = 1
                  dendriteCount = 0 })

        let posts =
            [ { name = "post1"
                activation = ActivationCurves.thresholdWithChemical 0.75 0.4 0.8
                initialElectrical = 0.0
                initialChemical = 0.0
                axonCount = 1
                dendriteCount = 2 }
              { name = "post2"
                activation = ActivationCurves.thresholdWithChemical 0.25 0.4 0.8
                initialElectrical = 0.0
                initialChemical = 0.0
                axonCount = 1
                dendriteCount = 2 } ]

        pres @ posts

    let regionBlueprints =
        [ { name = "r1"; gain = 1.0 }
          { name = "r2"; gain = 1.0 }
          { name = "r3"; gain = 1.0 } ]

    let previewRandomWiring () =
        let preview = initNetwork cellBlueprints regionBlueprints (Some wiringStrategy)
        printfn "Random wiring preview: A2R=%d, R2D=%d" preview.axonToRegion.Length preview.regionToDendrite.Length

    previewRandomWiring ()

    let baseGraph = initNetwork cellBlueprints regionBlueprints None

    let findAxon name =
        baseGraph.axons |> List.find (fun axon -> axon.name = name)

    let findRegion name =
        baseGraph.regions |> List.find (fun region -> region.name = name)

    let findDendrite name =
        baseGraph.dendrites |> List.find (fun den -> den.name = name)

    let deterministicGraph =
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

    let stimuli =
        preInputs
        |> List.map (fun (name, value) ->
            { axonId = (findAxon $"{name}.axon1").id
              value = value })

    let snapshot = Simulation.runTick deterministicGraph stimuli 0

    let printSnapshot (snap: TickSnapshot) =
        printfn "--- Tick %d ---" snap.tick
        printfn "Regions"
        snap.regionSignals
        |> List.iter (fun (name, value) -> printfn "  %s -> %.2f" name value)

        printfn "Posts"
        snap.cellStates
        |> List.filter (fun (name, _) -> name.StartsWith("post"))
        |> List.iter (fun (name, state) ->
            printfn "  %s : electrical=%.2f chemical=%.2f" name state.electrical state.chemical)

    printSnapshot snapshot
