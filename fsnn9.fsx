#r "nuget: FSharp.FGL"
#r "nuget: Expecto"

open System
open System.Collections.Concurrent
open FSharp.FGL
open Expecto

// ---- 基本型別與資料結構 ----
type axonId = int
type dendriteId = int
type tick = int64
type tickId = int64
type synapseId = int
type synapsePositionId = int
type electronicSignal = float
type chemicalSignal = float
type activated = bool
type windowSize = int

type BioSignal =
    { electronic: (synapsePositionId * electronicSignal)[]
      chemical: Map<string, chemicalSignal> }

type neuronInput =
    ConcurrentDictionary<
        tick,
        ConcurrentDictionary<
            dendriteId,
            ConcurrentDictionary<synapseId, BioSignal>>>

type neuronOutput =
    ConcurrentDictionary<
        tick,
        ConcurrentDictionary<synapseId, activated * BioSignal>>

// windowSiz 多少個 tick 內的輸入會被考慮
// 根據本次的輸入與之前的輸入計算是否發射
type activationCurve = windowSize * neuronInput -> windowSize * neuronOutput -> activated

// 神經元觸發後怎麼更新網路
type aggregationCurve = windowSize * activated[] -> windowSize * BioSignal[] -> BioSignal

type Neuron =
    { id: axonId
      output: neuronOutput
      input: neuronInput
      mutable activationCurve: activationCurve }

type SynapseRegion =
    { id: synapseId
      connection: ConcurrentDictionary<tick, (synapsePositionId * axonId)[] * (synapsePositionId * dendriteId)[]>
      mutable aggregationCurve: aggregationCurve }

// 執行就類似建立一個樹狀結構，有向無環，curTick 紀錄現在 HEAD 在時間線的哪裡
type RunContext =
    { stateTick: tickId // 表示整個 graph 的 tick 位置
      mutable curTick: tick
      mutable 結構圖: Graph<int, string, string>
      curConnection: ConcurrentDictionary<synapseId, SynapseRegion>
      curNeurons: ConcurrentDictionary<axonId, Neuron> }

type RunState = RunContext []

// ---- Graph 專用標籤 ----
type 節點標籤 =
    | 軸突節點 of axonId
    | 樹突節點 of dendriteId
    | 突觸節點 of synapseId * synapsePositionId
    with
        override this.ToString() =
            match this with
            | 軸突節點 i -> $"軸突:{i}"
            | 樹突節點 i -> $"樹突:{i}"
            | 突觸節點 (s, p) -> $"突觸:{s}@{p}"

type 邊種類 =
    | 軸突指向突觸
    | 突觸指向樹突
    | 突觸串接

[<CLIMutable>]
type 邊標籤 =
    { 種類: 邊種類
      突觸編號: synapseId
      位置: synapsePositionId option
      權重: float }

// ---- 建立與修改網路結構 ----
module 生物網路 =
    let 建立空執行環境 () : RunContext =
        { stateTick = 0L
          curTick = 0L
          結構圖 = Graph.empty
          curConnection = ConcurrentDictionary()
          curNeurons = ConcurrentDictionary() }

    /// 預設激發曲線：簡單求和，超過 threshold 即發射
    let 預設激發曲線 (threshold: float) : activationCurve =
        fun (窗, 輸入) (_窗, _輸出) ->
            let mutable 線性和 = 0.0
            let 最新Tick = if 輸入.IsEmpty then 0L else 輸入.Keys |> Seq.max
            for t in int64(Math.Max(0, int 最新Tick - (窗 |> int))) .. 最新Tick do
                match 輸入.TryGetValue t with
                | true, dendriteMap ->
                    for kv in dendriteMap do
                        let synapses = kv.Value
                        for syn in synapses do
                            let bio = syn.Value
                            線性和 <- 線性和 + (bio.electronic |> Array.sumBy snd)
                | _ -> ()
            線性和 >= threshold

    /// 預設聚合曲線：把 window 內的電子訊號平均，化學訊號累加
    let 預設聚合曲線 : aggregationCurve =
        fun (窗, 激發狀態) (窗2, 訊號們) ->
            let recentBio =
                訊號們
                |> Seq.rev
                |> Seq.truncate 窗
                |> Seq.toArray
            let 電訊號平均 =
                recentBio
                |> Array.collect (fun b -> b.electronic)
                |> Array.map snd
                |> fun xs -> if xs.Length = 0 then 0.0 else xs |> Array.average
            let 化學訊號 =
                recentBio
                |> Array.fold
                    (fun acc b ->
                        b.chemical
                        |> Map.fold (fun m k v -> m |> Map.change k (function None -> Some v | Some old -> Some(old + v))) acc)
                    Map.empty
            { electronic = [||]; chemical = 化學訊號 }

    /// 加入神經元，並在 Graph 中建立軸突與樹突節點
    let 新增神經元 (ctx: RunContext) (神經元編號: axonId) (樹突清單: dendriteId list) (激發曲線: activationCurve option) =
        let nodeId 軸突 = 神經元編號 * 10 + 軸突
        let addNode (id, label: 節點標籤) = ctx.結構圖 <- ctx.結構圖 |> Graph.addNode (id, label.ToString())
        addNode (nodeId 1, 軸突節點 神經元編號)
        樹突清單 |> List.iteri (fun i d -> addNode (nodeId (i + 2), 樹突節點 d))

        let 神經元 =
            { id = 神經元編號
              output = ConcurrentDictionary()
              input = ConcurrentDictionary()
              activationCurve = defaultArg 激發曲線 (預設激發曲線 0.5) }
        ctx.curNeurons.TryAdd(神經元編號, 神經元) |> ignore
        ctx

    /// 新增突觸區域，於 Graph 建立突觸位置節點集合
    let 新增突觸區 (ctx: RunContext) (突觸編號: synapseId) (位置數量: int) (聚合曲線: aggregationCurve option) =
        let baseId = 突觸編號 * 1000
        for p in 0 .. 位置數量 - 1 do
            ctx.結構圖 <- ctx.結構圖 |> Graph.addNode (baseId + p, 突觸節點 (突觸編號, p) |> string)
        let region =
            { id = 突觸編號
              connection = ConcurrentDictionary()
              aggregationCurve = defaultArg 聚合曲線 預設聚合曲線 }
        ctx.curConnection.TryAdd(突觸編號, region) |> ignore
        ctx

    /// 驗證並連結軸突 -> 突觸位置，需考慮 asMax 與 axonAsMax
    let 連接軸突到突觸 (ctx: RunContext) (asMax: int) (axonAsMax: int) (軸突: axonId) (突觸: synapseId) (位置: synapsePositionId) (權重: float) =
        // 計算當前圖中該突觸的入邊數
        let 突觸入邊數 =
            ctx.結構圖
            |> Graph.edges
            |> List.filter (fun (s, t, l) -> l.Contains($"syn:{突觸}") && s.ToString().StartsWith("軸突:"))
            |> List.length
        if 突觸入邊數 >= asMax then invalidOp "超過 synapse 全域 asMax"

        let 同軸突入邊 =
            ctx.結構圖
            |> Graph.edges
            |> List.filter (fun (s, t, l) -> l.Contains($"syn:{突觸}") && l.Contains($"ax:{軸突}"))
            |> List.length
        if 同軸突入邊 >= axonAsMax then invalidOp "超過同軸突 axonAsMax"

        let 邊標籤字串 = $"ax:{軸突}|syn:{突觸}|pos:{位置}|w:{權重}" |> string
        ctx.結構圖 <- ctx.結構圖 |> Graph.addEdge (軸突 * 10 + 1, 突觸 * 1000 + 位置, 邊標籤字串)
        ctx

    /// 驗證並連結突觸位置 -> 樹突，需考慮 dsMax 與 dendriteDsMax
    let 連接突觸到樹突 (ctx: RunContext) (dsMax: int) (dendriteDsMax: int) (突觸: synapseId) (位置: synapsePositionId) (樹突: dendriteId) (權重: float) =
        let 突觸出邊數 =
            ctx.結構圖
            |> Graph.edges
            |> List.filter (fun (s, t, l) -> l.Contains($"syn:{突觸}") && l.Contains("樹突:"))
            |> List.length
        if 突觸出邊數 >= dsMax then invalidOp "超過 synapse 全域 dsMax"

        let 同樹突出邊 =
            ctx.結構圖
            |> Graph.edges
            |> List.filter (fun (s, t, l) -> l.Contains($"syn:{突觸}") && l.Contains($"den:{樹突}"))
            |> List.length
        if 同樹突出邊 >= dendriteDsMax then invalidOp "超過 dendrite dsMax"

        let 邊標籤字串 = $"syn:{突觸}|pos:{位置}|den:{樹突}|w:{權重}" |> string
        ctx.結構圖 <- ctx.結構圖 |> Graph.addEdge (突觸 * 1000 + 位置, 突觸 + 樹突, 邊標籤字串)
        ctx

    /// 記錄單一 tick 的輸入 (範例: 外部感測訊號)
    let 紀錄輸入 (ctx: RunContext) (目標樹突: dendriteId) (突觸: synapseId) (tick值: tick) (訊號: BioSignal) =
        let dendriteMap = ctx.curNeurons |> Seq.collect (fun kv -> kv.Value.input) |> ignore
        let tickMap = ctx.curNeurons.Values |> Seq.tryHead |> Option.map (fun n -> n.input)
        let 真實輸入 = defaultArg tickMap (ConcurrentDictionary())
        let dendrites =
            真實輸入.GetOrAdd(
                tick值,
                fun _ -> ConcurrentDictionary())
        let synMap = dendrites.GetOrAdd(目標樹突, fun _ -> ConcurrentDictionary())
        synMap.AddOrUpdate(突觸,訊號,(fun _ _ ->訊號)) |> ignore
        // 同步到每個 neuron 的 input，方便 activationCurve 直接使用
        for n in ctx.curNeurons.Values do
            n.input[tick值] <- dendrites
        ctx

    /// 執行一次 tick (會寫入 neuronOutput 並更新 curTick)
    let 執行一步 (窗: windowSize) (ctx: RunContext) =
        let 下一Tick = ctx.curTick + 1L
        ctx.curTick <- 下一Tick
        for n in ctx.curNeurons.Values do
            let 是否激發 = n.activationCurve (窗, n.input) (窗, n.output)
            let bio =
                { electronic =
                    [| if 是否激發 then (0, 1.0) else (0, 0.0) |]
                  chemical = Map.empty }
            let synOut = ConcurrentDictionary()
            synOut.TryAdd(0, (是否激發, bio)) |> ignore
            n.output[下一Tick] <- synOut
        ctx

    /// 以目前 ctx 為基礎建立新的 run 分支 (類似 git checkout)
    let 分支狀態 (新的TickId: tickId) (ctx: RunContext) =
        { stateTick = 新的TickId
          curTick = ctx.curTick
          結構圖 = ctx.結構圖
          curConnection = ctx.curConnection
          curNeurons = ctx.curNeurons }

// ---- 單元測試與範例 ----
module 測試 =
    open 生物網路

    let 範例激發窗 = 5

    let 建立簡單網路 () =
        let ctx = 建立空執行環境 ()
        ctx |> 新增神經元 1 [ 101 ] None |> ignore
        ctx |> 新增突觸區 10 2 None |> ignore
        ctx |> 連接軸突到突觸 4 2 1 10 0 1.0 |> ignore
        ctx |> 連接突觸到樹突 4 2 10 0 101 1.0 |> ignore
        ctx

    let 測試集合 =
        testList "fsnn9" [
            test "Graph 連線數符合限制" {
                let ctx = 建立簡單網路 ()
                let 邊數 = ctx.結構圖 |> Graph.edges |> List.length
                Expect.equal 邊數 2 "應該有兩條有向邊"
            }

            test "激發曲線會在輸入累積後發射" {
                let ctx = 建立簡單網路 ()
                let 訊號 = { electronic = [| (0, 1.0) |]; chemical = Map.empty }
                ctx |> 紀錄輸入 101 10 0L 訊號 |> ignore
                ctx |> 執行一步 範例激發窗 |> ignore
                let 神經元 = ctx.curNeurons[1]
                let 有輸出 = 神經元.output[ctx.curTick].Values |> Seq.head |> fst
                Expect.isTrue 有輸出 "因為輸入足夠，應該觸發"
            }

            test "分支狀態複製 tick 位置" {
                let ctx = 建立簡單網路 ()
                ctx |> 執行一步 範例激發窗 |> ignore
                let 分支 = ctx |> 分支狀態 99L
                Expect.equal 分支.curTick ctx.curTick "分支時應複製 tick" |> ignore
            }
        ]

    [<EntryPoint>]
    let main _ =
        runTestsWithCLIArgs [] [||] 測試集合 |> ignore
        0

