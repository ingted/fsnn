#r "nuget: TorchSharp"
#r "nuget: TorchSharp-cuda-windows, 0.105.1"

open System
open TorchSharp

let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
printfn "Running on %A" device

// ===============================
// 簡化：4 前級 / 3 區域 / 2 後級（1 tick）
// ===============================
//
// pre 輸入 = [1; 1; 0; 1]
// 權重：
//   pre1 -> r1 (w1=1.0)
//   pre2 -> r2 (w2=0.5)
//   pre3 -> r2 (w3=0.5)
//   pre4 -> r3 (w4=1.0)
// post1 讀取 r1 與 r2，門檻 0.75
// post2 讀取 r2 與 r3，門檻 0.25
//
// 「共用區域」語義：r2 是單一區域，其輸出同時被 post1/2 讀取
// → r2 = sum(pre2*w2 + pre3*w3)

let pre =
    torch.tensor([| 1.0; 1.0; 0.0; 1.0 |], device=device, dtype=torch.ScalarType.Float32) // [4]

// W_pr: [region(3) x pre(4)]，描述 pre→region 的權重
// r1: 只接 pre1
// r2: 接 pre2 與 pre3（共享）
// r3: 只接 pre4

open FSharp.Collections

let W_pr =
    [|
        [| 1.0; 0.0; 0.0; 0.0 |]
        [| 0.0; 0.5; 0.5; 0.0 |]
        [| 0.0; 0.0; 0.0; 1.0 |]
    |]
    |> array2D
    |> fun a -> torch.tensor(a, dtype=torch.ScalarType.Float32, device=device)

// P: [post(2) x region(3)]，描述 post 從哪些區域讀值
// post1: r1 + r2
// post2: r2 + r3
let P =
    torch.tensor(
        array2D [|
            [| 1.0; 1.0; 0.0 |] // post1
            [| 0.0; 1.0; 1.0 |] // post2
        |],
        device = device,
        dtype=torch.ScalarType.Float32
    ) // [2x3]

// 門檻（對應 post1, post2）
let thresh =
    torch.tensor([| 0.75; 0.25 |], device=device, dtype=torch.ScalarType.Float32) // [2]

// ---- 1 tick forward step ----
let pre_col = pre.view([| 4L; 1L |])          // [4,1]
let region   = torch.mm(W_pr, pre_col).squeeze()  // [3]
let region_col = region.view([| 3L; 1L |])     // [3,1]
let post_sum = torch.mm(P, region_col).squeeze()  // [2]

let dtype = torch.ScalarType.Float32
// 二值化輸出（>= 門檻 → 1，否則 0）
let post_out = post_sum.ge(thresh).to_type(dtype)      // [2],   Float32

torch.set_printoptions(style = torch.numpy)
// ---- 顯示結果 ----
printfn "region   = \n%s\n" <| region.cpu().str()
printfn "post_sum = \n%s\n" <| post_sum.cpu().str()
printfn "post_out = \n%s\n" <| post_out.cpu().str()
// 預期：
// region = [1.0; 0.5; 1.0]
// post_sum = [1.5; 1.5]
// post_out = [1.0; 1.0]