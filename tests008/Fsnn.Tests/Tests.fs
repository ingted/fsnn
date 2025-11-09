module Fsnn.Tests

open Xunit
open FsUnit.Xunit
open Fsnn6.Fsnn

let private stateMap (snapshot: TickSnapshot) =
    snapshot.neuronStates |> Map.ofList

let private regionMap (snapshot: TickSnapshot) =
    snapshot.regionSignals |> Map.ofList

[<Fact>]
let ``POC graph fires both post cells`` () =
    let snapshot = Example.runDeterministic ()
    let neurons = stateMap snapshot
    let regions = regionMap snapshot

    neurons["post1"].electrical |> should equal 1.0
    neurons["post2"].electrical |> should equal 1.0
    regions["r1"] |> fst |> should equal 1.0
    regions["r2"] |> fst |> should equal 0.5
    regions["r3"] |> fst |> should equal 1.0

[<Fact>]
let ``Random wiring policy honors fan-out limits`` () =
    let neuronBlueprints =
        [ { name = "n1"
            activation = ActivationCurves.passThrough 0.5
            initialElectrical = 0.0
            initialChemical = 0.0
            axonCount = 1
            dendriteCount = 1 }
          { name = "n2"
            activation = ActivationCurves.passThrough 0.5
            initialElectrical = 0.0
            initialChemical = 0.0
            axonCount = 1
            dendriteCount = 1 }
          { name = "n3"
            activation = ActivationCurves.passThrough 0.5
            initialElectrical = 0.0
            initialChemical = 0.0
            axonCount = 1
            dendriteCount = 1 } ]

    let regionBlueprints =
        [ { name = "r1"; aggregation = AggregationCurves.weightedSum 1.0 0.1 }
          { name = "r2"; aggregation = AggregationCurves.weightedSum 1.0 0.1 }
          { name = "r3"; aggregation = AggregationCurves.weightedSum 1.0 0.1 } ]

    let strategy =
        { rng = System.Random(99)
          axonPolicy = { cardinality = UpTo 2; allowDuplicates = false }
          dendritePolicy = { cardinality = OneToOne; allowDuplicates = false } }

    let graph = Builder.initNetwork neuronBlueprints regionBlueprints (Some strategy)

    let connectionsByAxon =
        graph.axonToRegion
        |> List.groupBy (fun proj -> proj.axonId)
        |> List.map snd

    connectionsByAxon
    |> List.iter (fun edges ->
        edges.Length |> should (lessThanOrEqualTo) 2
        edges.Length |> should be (greaterThan 0))

    let regionTargets =
        graph.regionToDendrite
        |> List.groupBy (fun proj -> proj.regionId)

    regionTargets
    |> List.iter (fun (_, edges) ->
        edges.Length |> should equal 1)
