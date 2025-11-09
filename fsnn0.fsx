open System

// ===== 定義結構 =====
type Cell = { name: string; mutable output: float }
type SynapseRegion = { name: string; weight: float; mutable signal: float }

// 輸入層（前級）
let pre1 = { name = "pre1"; output = 1.0 }
let pre2 = { name = "pre2"; output = 1.0 }
let pre3 = { name = "pre3"; output = 0.0 }
let pre4 = { name = "pre4"; output = 1.0 }

// 突觸區域
let r1 = { name = "r1"; weight = 1.0; signal = 0.0 }
let r2 = { name = "r2"; weight = 0.5; signal = 0.0 }
let r3 = { name = "r3"; weight = 1.0; signal = 0.0 }

// 輸出層（後級）
let post1 = { name = "post1"; output = 0.0 }
let post2 = { name = "post2"; output = 0.0 }

// ===== 定義傳遞與激活規則 =====
let firePreToRegion pre region =
    region.signal <- pre.output * region.weight

let activatePost post inputs threshold =
    let sumInput = inputs |> List.sum
    post.output <- if sumInput >= threshold then 1.0 else 0.0
    printfn "%s receives %.2f → output %.1f" post.name sumInput post.output

// ===== 模擬前向傳遞 =====
// 前級到突觸
firePreToRegion pre1 r1
firePreToRegion pre2 r2
firePreToRegion pre3 r2
firePreToRegion pre4 r3

// 匯聚突觸信號
let post1Inputs = [ r1.signal; r2.signal ]   // pre1+r1, pre2+r2
let post2Inputs = [ r2.signal; r3.signal ]   // pre3+r2, pre4+r3

// 激活後級
activatePost post1 post1Inputs 0.75
activatePost post2 post2Inputs 0.25

// ===== 輸出結果 =====
printfn "\n=== 最終狀態 ==="
for r in [r1;r2;r3] do
    printfn "Region %s: signal=%.2f weight=%.2f" r.name r.signal r.weight
for p in [post1;post2] do
    printfn "Post %s: output=%.1f" p.name p.output