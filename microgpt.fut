-- Port of Karpathy's microgpt to Futhark
-- Part I: forward pass
-- Blog post: https://www.kmjn.org/notes/microgpt_futhark.html
-- Mark Nelson, 2026

def n_layer : i64 = 1
def n_embd : i64 = 16
def block_size : i64 = 16
def n_head : i64 = 4
def head_dim : i64 = n_embd / n_head

type params [v] = {
  wte:      [v][n_embd]f32,                   -- token embeddings
  wpe:      [block_size][n_embd]f32,          -- position embeddings
  lm_head:  [v][n_embd]f32,                   -- output projection
  attn_wq:  [n_layer][n_embd][n_embd]f32,     -- query weights
  attn_wk:  [n_layer][n_embd][n_embd]f32,     -- key weights
  attn_wv:  [n_layer][n_embd][n_embd]f32,     -- value weights
  attn_wo:  [n_layer][n_embd][n_embd]f32,     -- output weights
  mlp_fc1:  [n_layer][4 * n_embd][n_embd]f32, -- MLP up-projection
  mlp_fc2:  [n_layer][n_embd][4 * n_embd]f32  -- MLP down-projection
}

-- Model components

def linear [n][m] (x: [n]f32) (w: [m][n]f32) : [m]f32 =
  map (\w_row -> f32.sum (map2 (*) w_row x)) w

def softmax [n] (logits: [n]f32) : [n]f32 =
  let max_val = f32.maximum logits
  let exps = map (\v -> f32.exp (v - max_val)) logits
  let total = f32.sum exps
  in map (/ total) exps

def rmsnorm [n] (x: [n]f32) : [n]f32 =
  let ms = f32.sum (map2 (*) x x) / f32.i64 n
  let scale = 1f32 / f32.sqrt (ms + 1e-5)
  in map (* scale) x


-- GPT forward pass

def gpt [v]
  (p: params [v])
  (token_id: i64) (pos_id: i64)
  (keys:   *[n_layer][block_size][n_embd]f32)
  (values: *[n_layer][block_size][n_embd]f32)
  : ([v]f32,
     *[n_layer][block_size][n_embd]f32,
     *[n_layer][block_size][n_embd]f32) =

  let tok_emb = p.wte[token_id]
  let pos_emb = p.wpe[pos_id]
  let x = map2 (+) tok_emb pos_emb
  let x = rmsnorm x

  let (x, keys, values) =
    loop (x, keys, values) for li < n_layer do
      -- 1) Multi-head Attention block
      let x_residual = x
      let x_norm = rmsnorm x
      let q     = linear x_norm p.attn_wq[li]
      let k     = linear x_norm p.attn_wk[li]
      let v_vec = linear x_norm p.attn_wv[li]
      let keys   = keys   with [li, pos_id] = k
      let values = values with [li, pos_id] = v_vec
      let x_attn = flatten (
        tabulate n_head (\h ->
          let hs = h * head_dim
          let q_h = q[hs : hs + head_dim]
          let scale = 1f32 / f32.sqrt (f32.i64 head_dim)
          let attn_logits = tabulate block_size (\t ->
            let dot = f32.sum (map2 (*) q_h keys[li, t, hs : hs + head_dim])
            in if t <= pos_id then dot * scale else -1e30f32
          )
          let attn_weights = softmax attn_logits
          in tabulate head_dim (\j ->
            f32.sum (map2 (*) attn_weights values[li, :block_size, hs + j])
          )
        )
      ) :> [n_embd]f32
      let x_out = linear x_attn p.attn_wo[li]
      let x = map2 (+) x_out x_residual

      -- 2) MLP block
      let x_residual = x
      let x_norm = rmsnorm x
      let x_mlp = linear x_norm p.mlp_fc1[li]
      let x_mlp = map (f32.max 0) x_mlp
      let x_mlp = linear x_mlp p.mlp_fc2[li]
      let x = map2 (+) x_mlp x_residual
      in (x, keys, values)
  let logits = linear x p.lm_head
  in (logits, keys, values)
