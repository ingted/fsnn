open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Threading

type axonId = int
type dendriteId = int
type tick = int64
type synapseId = int
type synapsePositionId = int
type electronicSignal = float
type chemicalSignal = float
type activated = bool
type windowSize = int

type neuronInput =
    ConcurrentDictionary<
        tick,
        ConcurrentDictionary<
            dendriteId,
            ConcurrentDictionary<synapseId, (synapsePositionId * electronicSignal)[]>>>

type neuronOutput =
    ConcurrentDictionary<
        tick,
        ConcurrentDictionary<synapseId, (synapsePositionId * activated)[]>>

type activationCurve = windowSize -> neuronInput -> neuronOutput -> activated

type aggregationCurve = windowSize -> activated[] -> chemicalSignal[] -> electronicSignal * chemicalSignal

type Neuron =
    { id: axonId
      output: neuronOutput
      input: neuronInput
      mutable activationCurve: activationCurve }

type SynapseRegion =
    { id: synapseId
      connection: ConcurrentDictionary<tick, (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[]>
      mutable aggregationCurve: aggregationCurve }

type RunContext =
    { stateTick: tick
      mutable curTick: tick
      curConnection: (synapseId * (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[])[]
      curNeurons: ConcurrentDictionary<axonId, Neuron>
      curSynapseRegions: ConcurrentDictionary<synapseId, SynapseRegion> }

type RunState = RunContext []

module Fsnn =

    open System
    open System.Collections.Concurrent
    open System.Collections.Generic
    open System.Threading

    type ProcessingUnit =
        | CPU
        | GPU of string

    type NeuronBlueprint =
        { name: string
          dendriteCount: int
          activationFactory: axonId -> activationCurve }

    type SynapseRegionBlueprint =
        { name: string
          aggregationFactory: synapseId -> aggregationCurve }

    type NamedAxonLink =
        { region: string
          axon: string
          weight: float
          position: synapsePositionId option }

    type NamedDendriteLink =
        { region: string
          neuron: string
          dendriteIndex: int
          weight: float
          position: synapsePositionId option }

    type NamedWiringPlan =
        { axonLinks: NamedAxonLink list
          dendriteLinks: NamedDendriteLink list }

    type AxonRegionLink =
        { regionId: synapseId
          axonId: axonId
          positionId: synapsePositionId
          weight: float }

    type RegionDendriteLink =
        { regionId: synapseId
          dendriteId: dendriteId
          positionId: synapsePositionId
          weight: float }

    type WiringPlan =
        { axonLinks: AxonRegionLink list
          dendriteLinks: RegionDendriteLink list }

    type ConnectionCardinality =
        | OneToOne
        | UpTo of int

    type ConnectionPolicy =
        { cardinality: ConnectionCardinality
          allowDuplicates: bool }

    type WiringStrategy =
        { rng: Random
          axonPolicy: ConnectionPolicy
          dendritePolicy: ConnectionPolicy
          weightRange: float * float }

    type NeuronRuntime =
        { neuron: Neuron
          name: string
          dendrites: dendriteId[] }

    type RegionRuntime =
        { region: SynapseRegion
          name: string
          mutable inbound: AxonRegionLink[]
          mutable outbound: RegionDendriteLink[]
          mutable lastSignal: electronicSignal * chemicalSignal }

    type NetworkInstance =
        { mutable context: RunContext
          mutable state: RunState
          neuronRuntimes: ConcurrentDictionary<axonId, NeuronRuntime>
          regionRuntimes: ConcurrentDictionary<synapseId, RegionRuntime>
          neuronNames: ConcurrentDictionary<axonId, string>
          neuronIds: ConcurrentDictionary<string, axonId>
          regionNames: ConcurrentDictionary<synapseId, string>
          regionIds: ConcurrentDictionary<string, synapseId>
          dendriteOwners: ConcurrentDictionary<dendriteId, axonId>
          dendriteNames: ConcurrentDictionary<dendriteId, string>
          dendriteIds: ConcurrentDictionary<string * int, dendriteId>
          axonOutbound: ConcurrentDictionary<axonId, ConcurrentDictionary<synapseId, AxonRegionLink[]>>
          wiringPlan: WiringPlan
          connectionShape: (synapseId * (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[])[]
          mutable proxyRegions: (axonId * SynapseRegion) list }

    type Stimulus =
        { axonId: axonId
          value: activated }

    type RunSnapshot =
        { tick: tick
          device: ProcessingUnit
          neuronSpikes: (string * activated) list
          regionSignals: (string * (electronicSignal * chemicalSignal)) list
          inputProxies: (string * synapseId) list }

    module Id =

        let mutable axonCounter = 0
        let mutable dendriteCounter = 0
        let mutable synapseCounter = 0
        let mutable positionCounter = 0

        let nextAxon () = Interlocked.Increment(&axonCounter)
        let nextDendrite () = Interlocked.Increment(&dendriteCounter)
        let nextSynapse () = Interlocked.Increment(&synapseCounter)
        let nextPosition () = Interlocked.Increment(&positionCounter)

    module Storage =

        let recordNeuronInput (neuron: Neuron) (tick: tick) (dendriteId: dendriteId) (regionId: synapseId) (position: synapsePositionId) (signal: electronicSignal) =
            let tickBucket =
                neuron.input.GetOrAdd(
                    tick,
                    fun _ -> ConcurrentDictionary<dendriteId, ConcurrentDictionary<synapseId, (synapsePositionId * electronicSignal)[]>>()
                )
            let dendriteBucket =
                tickBucket.GetOrAdd(
                    dendriteId,
                    fun _ -> ConcurrentDictionary<synapseId, (synapsePositionId * electronicSignal)[]>()
                )
            dendriteBucket.AddOrUpdate(
                regionId,
                (fun _ -> [| position, signal |]),
                (fun _ existing ->
                    existing
                    |> Array.filter (fun (pos, _) -> pos <> position)
                    |> Array.append [| position, signal |])
            )
            |> ignore

        let recordNeuronOutput (neuron: Neuron) (tick: tick) (regionId: synapseId) (payload: (synapsePositionId * activated)[]) =
            let tickBucket =
                neuron.output.GetOrAdd(
                    tick,
                    fun _ -> ConcurrentDictionary<synapseId, (synapsePositionId * activated)[]>()
                )
            tickBucket.AddOrUpdate(
                regionId,
                payload,
                (fun _ _ -> payload)
            )
            |> ignore

        let sumRecentInputs (window: windowSize) (input: neuronInput) =
            input
            |> Seq.sortByDescending (fun (KeyValue(tick, _)) -> tick)
            |> Seq.truncate window
            |> Seq.collect (fun (KeyValue(_, dendrites)) ->
                dendrites
                |> Seq.collect (fun (KeyValue(_, synapses)) ->
                    synapses.Values
                    |> Seq.collect (Seq.map snd)))
            |> Seq.sum

    module ActivationCurves =

        let constant (value: activated) : axonId -> activationCurve =
            fun _ ->
                fun _ _ _ -> value

        let threshold (thresholdValue: electronicSignal) (leak: float) (gain: float) : axonId -> activationCurve =
            fun _ ->
                let mutable chemicalState = 0.0
                fun window input _ ->
                    let total = Storage.sumRecentInputs window input
                    let fired = total >= thresholdValue
                    if fired then
                        chemicalState <- (chemicalState * leak) + gain
                    else
                        chemicalState <- chemicalState * leak
                    fired

    module AggregationCurves =

        let weightedSum (leak: float) : synapseId -> aggregationCurve =
            fun _ ->
                let mutable chemical = 0.0
                fun _ _ chemicalInputs ->
                    let electrical = chemicalInputs |> Array.sum
                    chemical <- (chemical * leak) + electrical
                    electrical, chemical

        let proxy (value: float) : synapseId -> aggregationCurve =
            fun _ ->
                fun _ _ _ -> value, value

    module Wiring =

        let private assignPositions (links: NamedAxonLink list) =
            let tracker = Dictionary<string, int>()
            links
            |> List.map (fun link ->
                let position =
                    match link.position with
                    | Some p -> p
                    | None ->
                        let next =
                            match tracker.TryGetValue link.region with
                            | true, existing ->
                                let value = existing + 1
                                tracker[link.region] <- value
                                value
                            | _ ->
                                tracker[link.region] <- 1
                                1
                        next
                { link with position = Some position })

        let private assignPositionsForDendrites (links: NamedDendriteLink list) =
            let tracker = Dictionary<string, int>()
            links
            |> List.map (fun link ->
                let position =
                    match link.position with
                    | Some p -> p
                    | None ->
                        let next =
                            match tracker.TryGetValue link.region with
                            | true, existing ->
                                let value = existing + 1
                                tracker[link.region] <- value
                                value
                            | _ ->
                                tracker[link.region] <- 1
                                1
                        next
                { link with position = Some position })

        let fromNamedPlan
            (plan: NamedWiringPlan)
            (regionIds: ConcurrentDictionary<string, synapseId>)
            (neuronIds: ConcurrentDictionary<string, axonId>)
            (dendriteIds: ConcurrentDictionary<string * int, dendriteId>) =

            let axonLinks =
                plan.axonLinks
                |> assignPositions
                |> List.map (fun link ->
                    let regionId =
                        match regionIds.TryGetValue link.region with
                        | true, v -> v
                        | _ -> failwithf "Region '%s' not found while wiring axons." link.region
                    let axonId =
                        match neuronIds.TryGetValue link.axon with
                        | true, v -> v
                        | _ -> failwithf "Neuron '%s' not found while wiring axons." link.axon
                    { regionId = regionId
                      axonId = axonId
                      positionId = link.position.Value
                      weight = link.weight })

            let dendriteLinks =
                plan.dendriteLinks
                |> assignPositionsForDendrites
                |> List.map (fun link ->
                    let regionId =
                        match regionIds.TryGetValue link.region with
                        | true, v -> v
                        | _ -> failwithf "Region '%s' not found while wiring dendrites." link.region
                    let dendriteKey = link.neuron, link.dendriteIndex
                    let dendriteId =
                        match dendriteIds.TryGetValue dendriteKey with
                        | true, v -> v
                        | _ -> failwithf "Dendrite %s[%d] not found." link.neuron link.dendriteIndex
                    { regionId = regionId
                      dendriteId = dendriteId
                      positionId = link.position.Value
                      weight = link.weight })

            { axonLinks = axonLinks
              dendriteLinks = dendriteLinks }

        let private determineCount policy available =
            match policy.cardinality with
            | OneToOne -> min 1 available
            | UpTo limit -> min limit (if policy.allowDuplicates then limit else available)

        let private shuffle (rng: Random) (items: 'a[]) =
            let clone = Array.copy items
            for i = clone.Length - 1 downto 1 do
                let j = rng.Next(i + 1)
                let tmp = clone[i]
                clone[i] <- clone[j]
                clone[j] <- tmp
            clone

        let private pickWithoutReplacement rng count (items: 'a[]) =
            if count <= 0 then
                [||]
            else
                items
                |> shuffle rng
                |> Array.truncate count

        let private pickWithReplacement (rng: Random) count (items: 'a[]) =
            if count <= 0 then
                [||]
            else
                Array.init count (fun _ -> items[rng.Next(items.Length)])

        let randomPlan
            (strategy: WiringStrategy)
            (regionIds: ConcurrentDictionary<string, synapseId>)
            (neurons: ConcurrentDictionary<axonId, NeuronRuntime>) =

            let rng = strategy.rng
            let axonPool = neurons.Keys |> Seq.toArray
            let dendritePool =
                neurons.Values
                |> Seq.collect (fun runtime -> runtime.dendrites :> seq<_>)
                |> Seq.toArray

            let inbound = ResizeArray<AxonRegionLink>()
            let outbound = ResizeArray<RegionDendriteLink>()

            for KeyValue(_, regionId) in regionIds do
                let inboundCount = determineCount strategy.axonPolicy axonPool.Length
                let outboundCount = determineCount strategy.dendritePolicy dendritePool.Length

                let pickAxons =
                    if strategy.axonPolicy.allowDuplicates then
                        pickWithReplacement rng inboundCount axonPool
                    else
                        pickWithoutReplacement rng inboundCount axonPool

                let pickDendrites =
                    if strategy.dendritePolicy.allowDuplicates then
                        pickWithReplacement rng outboundCount dendritePool
                    else
                        pickWithoutReplacement rng outboundCount dendritePool

                pickAxons
                |> Array.iteri (fun idx axonId ->
                    let weight =
                        let minWeight, maxWeight = strategy.weightRange
                        minWeight + (rng.NextDouble() * (maxWeight - minWeight))
                    inbound.Add(
                        { regionId = regionId
                          axonId = axonId
                          positionId = idx + 1
                          weight = weight }
                    ))

                pickDendrites
                |> Array.iteri (fun idx dendriteId ->
                    let weight =
                        let minWeight, maxWeight = strategy.weightRange
                        minWeight + (rng.NextDouble() * (maxWeight - minWeight))
                    outbound.Add(
                        { regionId = regionId
                          dendriteId = dendriteId
                          positionId = idx + 1
                          weight = weight }
                    ))

            { axonLinks = inbound |> List.ofSeq
              dendriteLinks = outbound |> List.ofSeq }

    module Builder =

        let private instantiateNeurons (blueprints: NeuronBlueprint list) =
            let neurons = ConcurrentDictionary<axonId, NeuronRuntime>()
            let neuronNames = ConcurrentDictionary<axonId, string>()
            let neuronIds = ConcurrentDictionary<string, axonId>()
            let dendriteOwners = ConcurrentDictionary<dendriteId, axonId>()
            let dendriteNames = ConcurrentDictionary<dendriteId, string>()
            let dendriteIds = ConcurrentDictionary<string * int, dendriteId>()

            for bp in blueprints do
                let neuronId = Id.nextAxon ()
                let neuron =
                    { id = neuronId
                      output = ConcurrentDictionary()
                      input = ConcurrentDictionary()
                      activationCurve = fun _ _ _ -> false }
                neuron.activationCurve <- bp.activationFactory neuronId

                let dendrites =
                    [| for idx in 1 .. bp.dendriteCount ->
                        let dendriteId = Id.nextDendrite ()
                        dendriteOwners.TryAdd(dendriteId, neuronId) |> ignore
                        let dendriteName = $"{bp.name}.den{idx}"
                        dendriteNames.TryAdd(dendriteId, dendriteName) |> ignore
                        dendriteIds.TryAdd((bp.name, idx), dendriteId) |> ignore
                        dendriteId |]

                let runtime =
                    { neuron = neuron
                      name = bp.name
                      dendrites = dendrites }

                neurons.TryAdd(neuronId, runtime) |> ignore
                neuronNames.TryAdd(neuronId, bp.name) |> ignore
                neuronIds.TryAdd(bp.name, neuronId) |> ignore

            neurons, neuronNames, neuronIds, dendriteOwners, dendriteNames, dendriteIds

        let private instantiateRegions (blueprints: SynapseRegionBlueprint list) =
            let regions = ConcurrentDictionary<synapseId, RegionRuntime>()
            let regionNames = ConcurrentDictionary<synapseId, string>()
            let regionIds = ConcurrentDictionary<string, synapseId>()

            for bp in blueprints do
                let regionId = Id.nextSynapse ()
                let region =
                    { id = regionId
                      connection = ConcurrentDictionary()
                      aggregationCurve = fun _ _ _ -> 0.0, 0.0 }
                region.aggregationCurve <- bp.aggregationFactory regionId
                let runtime =
                    { region = region
                      name = bp.name
                      inbound = [||]
                      outbound = [||]
                      lastSignal = 0.0, 0.0 }
                regions.TryAdd(regionId, runtime) |> ignore
                regionNames.TryAdd(regionId, bp.name) |> ignore
                regionIds.TryAdd(bp.name, regionId) |> ignore

            regions, regionNames, regionIds

        let private buildConnectionShape (regions: ConcurrentDictionary<synapseId, RegionRuntime>) =
            regions
            |> Seq.map (fun kv ->
                let inbound = kv.Value.inbound |> Array.map (fun link -> link.positionId, link.axonId)
                let outbound = kv.Value.outbound |> Array.map (fun link -> link.positionId, link.dendriteId)
                kv.Key, inbound, outbound)
            |> Seq.toArray

        let private groupConnections (plan: WiringPlan) =
            let inbound = Dictionary<synapseId, ResizeArray<AxonRegionLink>>()
            let outbound = Dictionary<synapseId, ResizeArray<RegionDendriteLink>>()

            for link in plan.axonLinks do
                let bucket =
                    match inbound.TryGetValue link.regionId with
                    | true, existing -> existing
                    | _ ->
                        let created = ResizeArray()
                        inbound[link.regionId] <- created
                        created
                bucket.Add(link)

            for link in plan.dendriteLinks do
                let bucket =
                    match outbound.TryGetValue link.regionId with
                    | true, existing -> existing
                    | _ ->
                        let created = ResizeArray()
                        outbound[link.regionId] <- created
                        created
                bucket.Add(link)

            inbound, outbound

        let private populateRegionConnections
            (regions: ConcurrentDictionary<synapseId, RegionRuntime>)
            (plan: WiringPlan) =

            let inbound, outbound = groupConnections plan

            for KeyValue(regionId, runtime) in regions do
                let inboundArr =
                    match inbound.TryGetValue regionId with
                    | true, bucket -> bucket |> Seq.toArray
                    | _ -> [||]
                let outboundArr =
                    match outbound.TryGetValue regionId with
                    | true, bucket -> bucket |> Seq.toArray
                    | _ -> [||]

                runtime.inbound <- inboundArr
                runtime.outbound <- outboundArr

                runtime.region.connection.Clear()
                runtime.region.connection.TryAdd(
                    0L,
                    (inboundArr |> Array.map (fun link -> link.positionId, link.axonId),
                     outboundArr |> Array.map (fun link -> link.positionId, link.dendriteId))
                )
                |> ignore

        let private buildAxonOutboundIndex (plan: WiringPlan) =
            let index = ConcurrentDictionary<axonId, ConcurrentDictionary<synapseId, AxonRegionLink[]>>()
            for link in plan.axonLinks do
                let regionMap =
                    index.GetOrAdd(
                        link.axonId,
                        fun _ -> ConcurrentDictionary<synapseId, AxonRegionLink[]>()
                    )
                regionMap.AddOrUpdate(
                    link.regionId,
                    (fun _ -> [| link |]),
                    (fun _ existing -> existing |> Array.append [| link |])
                )
                |> ignore
            index

        let initNetwork
            (neurons: NeuronBlueprint list)
            (regions: SynapseRegionBlueprint list)
            (namedPlan: NamedWiringPlan option)
            (strategy: WiringStrategy option) =

            let neuronRuntimes, neuronNames, neuronIds, dendriteOwners, dendriteNames, dendriteIds =
                instantiateNeurons neurons
            let regionRuntimes, regionNames, regionIds = instantiateRegions regions

            let wiringPlan =
                match namedPlan, strategy with
                | Some plan, _ -> Wiring.fromNamedPlan plan regionIds neuronIds dendriteIds
                | None, Some strat -> Wiring.randomPlan strat regionIds neuronRuntimes
                | _ -> failwith "A deterministic plan or wiring strategy is required."

            populateRegionConnections regionRuntimes wiringPlan
            let connectionShape = buildConnectionShape regionRuntimes
            let axonOutbound = buildAxonOutboundIndex wiringPlan

            let neuronTable = ConcurrentDictionary<axonId, Neuron>()
            for KeyValue(id, runtime) in neuronRuntimes do
                neuronTable.TryAdd(id, runtime.neuron) |> ignore

            let regionTable = ConcurrentDictionary<synapseId, SynapseRegion>()
            for KeyValue(id, runtime) in regionRuntimes do
                regionTable.TryAdd(id, runtime.region) |> ignore

            let context =
                { stateTick = 0L
                  curTick = 0L
                  curConnection = connectionShape
                  curNeurons = neuronTable
                  curSynapseRegions = regionTable }

            { context = context
              state = [| context |]
              neuronRuntimes = neuronRuntimes
              regionRuntimes = regionRuntimes
              neuronNames = neuronNames
              neuronIds = neuronIds
              regionNames = regionNames
              regionIds = regionIds
              dendriteOwners = dendriteOwners
              dendriteNames = dendriteNames
              dendriteIds = dendriteIds
              axonOutbound = axonOutbound
              wiringPlan = wiringPlan
              connectionShape = connectionShape
              proxyRegions = [] }

        let neuronId (network: NetworkInstance) name =
            match network.neuronIds.TryGetValue name with
            | true, id -> id
            | _ -> failwithf "Neuron '%s' not found." name

        let dendriteId (network: NetworkInstance) neuronName dendriteIndex =
            match network.dendriteIds.TryGetValue((neuronName, dendriteIndex)) with
            | true, id -> id
            | _ -> failwithf "Dendrite %s[%d] not found." neuronName dendriteIndex

        let regionId (network: NetworkInstance) name =
            match network.regionIds.TryGetValue name with
            | true, id -> id
            | _ -> failwithf "Region '%s' not found." name

    module Runner =

        let private floatOfBool value = if value then 1.0 else 0.0

        let private sameConnectionShape
            (current: (synapseId * (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[])[])
            (expected: (synapseId * (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[])[]) =
            if current.Length <> expected.Length then
                false
            else
                current
                |> Array.forall (fun (regionId, axons, dendrites) ->
                    match expected |> Array.tryFind (fun (rid, _, _) -> rid = regionId) with
                    | None -> false
                    | Some (_, targetAxons, targetDendrites) ->
                        axons = targetAxons && dendrites = targetDendrites)

        let private createStimulusMap (stimuli: Stimulus list) =
            let map = Dictionary<axonId, activated>()
            for stimulus in stimuli do
                map[stimulus.axonId] <- stimulus.value
            map

        let run (device: ProcessingUnit) (window: windowSize) (network: NetworkInstance) (stimuli: Stimulus list) =
            match device with
            | GPU backend ->
                raise (NotSupportedException($"GPU backend '{backend}' is planned but not implemented in fsnn6 yet."))
            | CPU ->
                let nextTick = network.context.curTick + 1L
                let stimulusMap = createStimulusMap stimuli
                let activationCache = Dictionary<axonId, activated>()

                let proxies =
                    stimuli
                    |> List.map (fun stim ->
                        let proxyId = Id.nextSynapse ()
                        let proxy =
                            { id = proxyId
                              connection = ConcurrentDictionary()
                              aggregationCurve = AggregationCurves.proxy (floatOfBool stim.value) proxyId }
                        stim.axonId, proxy)

                let evaluateNeuron axonId =
                    match activationCache.TryGetValue axonId with
                    | true, value -> value
                    | _ ->
                        let value =
                            match stimulusMap.TryGetValue axonId with
                            | true, v -> v
                            | _ ->
                                match network.context.curNeurons.TryGetValue axonId with
                                | true, neuron -> neuron.activationCurve window neuron.input neuron.output
                                | _ -> false
                        activationCache[axonId] <- value
                        value

                for KeyValue(_, regionRuntime) in network.regionRuntimes do
                    if regionRuntime.inbound.Length = 0 then
                        regionRuntime.lastSignal <- 0.0, 0.0
                    else
                        let activations = Array.zeroCreate regionRuntime.inbound.Length
                        let weightedSignals = Array.zeroCreate regionRuntime.inbound.Length
                        for idx in 0 .. regionRuntime.inbound.Length - 1 do
                            let link = regionRuntime.inbound[idx]
                            let fired = evaluateNeuron link.axonId
                            activations[idx] <- fired
                            weightedSignals[idx] <- if fired then link.weight else 0.0
                        let electrical, chemical =
                            regionRuntime.region.aggregationCurve window activations weightedSignals
                        regionRuntime.lastSignal <- electrical, chemical
                        for outLink in regionRuntime.outbound do
                            match network.dendriteOwners.TryGetValue outLink.dendriteId with
                            | true, owner ->
                                let signal = electrical * outLink.weight
                                let targetNeuron = network.neuronRuntimes[owner].neuron
                                Storage.recordNeuronInput targetNeuron nextTick outLink.dendriteId regionRuntime.region.id outLink.positionId signal
                            | _ -> ()

                for KeyValue(axonId, runtime) in network.neuronRuntimes do
                    if activationCache.ContainsKey axonId |> not then
                        let value =
                            match stimulusMap.TryGetValue axonId with
                            | true, v -> v
                            | _ -> runtime.neuron.activationCurve window runtime.neuron.input runtime.neuron.output
                        activationCache[axonId] <- value

                for KeyValue(axonId, runtime) in network.neuronRuntimes do
                    match activationCache.TryGetValue axonId with
                    | true, fired ->
                        match network.axonOutbound.TryGetValue axonId with
                        | true, perRegion ->
                            for KeyValue(regionId, links) in perRegion do
                                let payload = links |> Array.map (fun link -> link.positionId, fired)
                                Storage.recordNeuronOutput runtime.neuron nextTick regionId payload
                        | _ -> ()
                    | _ -> ()

                let newContext =
                    let adjusted =
                        if sameConnectionShape network.context.curConnection network.connectionShape then
                            network.context
                        else
                            { network.context with curConnection = network.connectionShape }
                    { adjusted with
                        curTick = nextTick
                        stateTick = nextTick }

                network.context <- newContext
                network.state <- [| newContext |]
                network.proxyRegions <- proxies |> List.map (fun (axonId, region) -> axonId, region)

                let neuronSpikes =
                    network.neuronNames
                    |> Seq.map (fun kv ->
                        let fired =
                            match activationCache.TryGetValue kv.Key with
                            | true, value -> value
                            | _ -> false
                        kv.Value, fired)
                    |> Seq.sortBy fst
                    |> List.ofSeq

                let regionSignals =
                    network.regionNames
                    |> Seq.map (fun kv ->
                        let runtime = network.regionRuntimes[kv.Key]
                        kv.Value, runtime.lastSignal)
                    |> Seq.sortBy fst
                    |> List.ofSeq

                let proxySummary =
                    proxies
                    |> List.map (fun (axonId, region) ->
                        match network.neuronNames.TryGetValue axonId with
                        | true, name -> name, region.id
                        | _ -> $"axon-{axonId}", region.id)

                { tick = nextTick
                  device = device
                  neuronSpikes = neuronSpikes
                  regionSignals = regionSignals
                  inputProxies = proxySummary },
                network.state

    module Example =

        let private preBlueprints =
            [ "pre1", true
              "pre2", true
              "pre3", false
              "pre4", true ]
            |> List.map (fun (name, value) ->
                { name = name
                  dendriteCount = 0
                  activationFactory = ActivationCurves.constant value })

        let private postBlueprints =
            [ { name = "post1"
                dendriteCount = 2
                activationFactory = ActivationCurves.threshold 0.75 0.4 0.6 }
              { name = "post2"
                dendriteCount = 2
                activationFactory = ActivationCurves.threshold 0.25 0.4 0.6 } ]

        let private regionBlueprints =
            [ "r1"
              "r2"
              "r3" ]
            |> List.map (fun name ->
                { name = name
                  aggregationFactory = AggregationCurves.weightedSum 0.65 })

        let private wiringPlan : NamedWiringPlan =
            { axonLinks =
                [ { region = "r1"; axon = "pre1"; weight = 1.0; position = Some 1 }
                  { region = "r2"; axon = "pre2"; weight = 0.5; position = Some 1 }
                  { region = "r2"; axon = "pre3"; weight = 0.5; position = Some 2 }
                  { region = "r3"; axon = "pre4"; weight = 1.0; position = Some 1 } ]
              dendriteLinks =
                [ { region = "r1"; neuron = "post1"; dendriteIndex = 1; weight = 1.0; position = Some 1 }
                  { region = "r2"; neuron = "post1"; dendriteIndex = 2; weight = 1.0; position = Some 1 }
                  { region = "r2"; neuron = "post2"; dendriteIndex = 1; weight = 1.0; position = Some 2 }
                  { region = "r3"; neuron = "post2"; dendriteIndex = 2; weight = 1.0; position = Some 1 } ] }

        let network =
            Builder.initNetwork
                (preBlueprints @ postBlueprints)
                regionBlueprints
                (Some wiringPlan)
                None

        let private stimuli =
            [ "pre1", true
              "pre2", true
              "pre3", false
              "pre4", true ]
            |> List.map (fun (name, value) ->
                { axonId = Builder.neuronId network name
                  value = value })

        let run () =
            Runner.run CPU 1 network stimuli

#if INTERACTIVE
        let snapshot, _ = run ()
        printfn "=== Example Tick %d ===" snapshot.tick
        snapshot.neuronSpikes
        |> List.iter (fun (name, fired) -> printfn "Neuron %s fired=%b" name fired)
        snapshot.regionSignals
        |> List.iter (fun (name, (e, c)) ->
            printfn "Region %s: electrical=%.3f chemical=%.3f" name e c)
#endif
