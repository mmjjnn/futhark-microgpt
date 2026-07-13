-- Port of Karpathy's microgpt to Futhark
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


import "lib/github.com/diku-dk/cpprandom/random"

module pcg32_engine = pcg32
module dist = uniform_real_distribution f32 u32 pcg32_engine
module norm_dist = normal_distribution f32 u32 pcg32_engine

-- Random number generator

type rng = pcg32_engine.rng

def rng_f32 (s: rng) : (rng, f32) =
  dist.rand (0f32, 1f32) s

def rng_gauss (s: rng) (std: f32) : (rng, f32) =
  norm_dist.rand {mean = 0f32, stddev = std} s

-- Deterministic subseed for parallel initialization
def rng_for_index (seed: i32) (idx: i64) : rng =
  pcg32_engine.rng_from_seed [seed, i32.i64 (idx >> 32), i32.i64 idx]

-- Weighted random sampling (like random.choices)
def weighted_sample [n] (probs: [n]f32) (rng_state: rng) : (rng, i64) =
  let (rng_state, u) = rng_f32 rng_state
  let cdf = scan (+) 0f32 probs
  -- The number of elements <= u is exactly the index of the first element > u
  let idx = i64.sum (map (\c -> if c <= u then 1i64 else 0i64) cdf)
  in (rng_state, idx)


-- Elementwise operations on parameter records

def params_map2 [v] (f: f32 -> f32 -> f32)
              (a: params [v]) (b: params [v]) : params [v] = {
  wte     = map2 (map2 f) a.wte b.wte,
  wpe     = map2 (map2 f) a.wpe b.wpe,
  lm_head = map2 (map2 f) a.lm_head b.lm_head,
  attn_wq = map2 (map2 (map2 f)) a.attn_wq b.attn_wq,
  attn_wk = map2 (map2 (map2 f)) a.attn_wk b.attn_wk,
  attn_wv = map2 (map2 (map2 f)) a.attn_wv b.attn_wv,
  attn_wo = map2 (map2 (map2 f)) a.attn_wo b.attn_wo,
  mlp_fc1 = map2 (map2 (map2 f)) a.mlp_fc1 b.mlp_fc1,
  mlp_fc2 = map2 (map2 (map2 f)) a.mlp_fc2 b.mlp_fc2
}

def params_map3 [v] (f: f32 -> f32 -> f32 -> f32)
              (a: params [v]) (b: params [v]) (c: params [v])
    : params [v] = {
  wte     = map3 (map3 f) a.wte b.wte c.wte,
  wpe     = map3 (map3 f) a.wpe b.wpe c.wpe,
  lm_head = map3 (map3 f) a.lm_head b.lm_head c.lm_head,
  attn_wq = map3 (map3 (map3 f)) a.attn_wq b.attn_wq c.attn_wq,
  attn_wk = map3 (map3 (map3 f)) a.attn_wk b.attn_wk c.attn_wk,
  attn_wv = map3 (map3 (map3 f)) a.attn_wv b.attn_wv c.attn_wv,
  attn_wo = map3 (map3 (map3 f)) a.attn_wo b.attn_wo c.attn_wo,
  mlp_fc1 = map3 (map3 (map3 f)) a.mlp_fc1 b.mlp_fc1 c.mlp_fc1,
  mlp_fc2 = map3 (map3 (map3 f)) a.mlp_fc2 b.mlp_fc2 c.mlp_fc2
}


-- Parameter initialization

def mk_params (v: i64) (seed: i32) : params [v] =
  let std = 0.08f32
  -- Compute byte offsets so each weight gets a unique subseed
  let o0 = 0i64
  let o1 = o0 + v * n_embd
  let o2 = o1 + block_size * n_embd
  let o3 = o2 + v * n_embd
  let o4 = o3 + n_layer * n_embd * n_embd
  let o5 = o4 + n_layer * n_embd * n_embd
  let o6 = o5 + n_layer * n_embd * n_embd
  let o7 = o6 + n_layer * n_embd * n_embd
  let o8 = o7 + n_layer * (4 * n_embd) * n_embd
  let g s o i j d = (rng_gauss (rng_for_index s (o + i*d + j)) std).1
  in {
    wte     = tabulate_2d v n_embd
                (\i j -> g seed o0 i j n_embd),
    wpe     = tabulate_2d block_size n_embd
                (\i j -> g seed o1 i j n_embd),
    lm_head = tabulate_2d v n_embd
                (\i j -> g seed o2 i j n_embd),
    attn_wq = tabulate_3d n_layer n_embd n_embd
                (\l i j -> g seed (o3 + l*n_embd*n_embd) i j n_embd),
    attn_wk = tabulate_3d n_layer n_embd n_embd
                (\l i j -> g seed (o4 + l*n_embd*n_embd) i j n_embd),
    attn_wv = tabulate_3d n_layer n_embd n_embd
                (\l i j -> g seed (o5 + l*n_embd*n_embd) i j n_embd),
    attn_wo = tabulate_3d n_layer n_embd n_embd
                (\l i j -> g seed (o6 + l*n_embd*n_embd) i j n_embd),
    mlp_fc1 = tabulate_3d n_layer (4 * n_embd) n_embd
                (\l i j -> g seed (o7 + l*(4*n_embd)*n_embd) i j n_embd),
    mlp_fc2 = tabulate_3d n_layer n_embd (4 * n_embd)
                (\l i j -> g seed (o8 + l*n_embd*(4*n_embd)) i j (4*n_embd))
  }


-- Average cross-entropy loss over a token sequence
-- (This is what we differentiate with vjp2 to get gradients.)

def compute_loss [v]
  (p: params [v])
  (tokens: [block_size + 1]i64)  -- padded token sequence
  (n_pos: i64)                   -- actual positions to process
  : f32 =
  let keys_init   = replicate n_layer
                      (replicate block_size (replicate n_embd 0f32))
  let values_init = replicate n_layer
                      (replicate block_size (replicate n_embd 0f32))
  let (total_loss, _, _) =
    loop (loss_acc, keys, values) =
         (0f32, keys_init, values_init)
    for pos < n_pos do
      let token_id  = tokens[pos]
      let target_id = tokens[pos + 1]
      let (logits, keys', values') = gpt p token_id pos keys values
      let probs = softmax logits
      -- Cross-entropy: -log P(target) (line 167)
      let loss_t = -(f32.log (f32.max probs[target_id] 1e-10))
      in (loss_acc + loss_t, keys', values')
  in total_loss / f32.i64 n_pos


-- Training loop and entry point

def learning_rate : f32 = 0.01
def beta1 : f32 = 0.85
def beta2 : f32 = 0.99
def eps_adam : f32 = 1e-8

entry train [n_docs][max_doc_len]
  (vocab_size: i64)
  (seed:       i32)
  (docs:      [n_docs][max_doc_len]i64) -- tokenized docs, padded with -1
  (doc_lens:  [n_docs]i64)              -- actual length of each doc
  (num_steps: i64)
  : (f32,                                              -- final loss
     [vocab_size][n_embd]f32, [block_size][n_embd]f32, -- wte, wpe
     [vocab_size][n_embd]f32,                          -- lm_head
     [n_layer][n_embd][n_embd]f32,                     -- wq
     [n_layer][n_embd][n_embd]f32,                     -- wk
     [n_layer][n_embd][n_embd]f32,                     -- wv
     [n_layer][n_embd][n_embd]f32,                     -- wo
     [n_layer][4 * n_embd][n_embd]f32,                 -- fc1
     [n_layer][n_embd][4 * n_embd]f32                  -- fc2
    ) =

  let bos = vocab_size - 1
  let p = mk_params vocab_size seed

  -- Initialize Adam moment buffers to zero (original lines 148-149)
  -- tabulate instead of replicate: avoids self-aliased arrays, which
  -- Futhark uniqueness types require to be consumable in loops.
  let mom : params [vocab_size] = {
    wte     = tabulate_2d vocab_size n_embd (\_ _ -> 0f32),
    wpe     = tabulate_2d block_size n_embd (\_ _ -> 0f32),
    lm_head = tabulate_2d vocab_size n_embd (\_ _ -> 0f32),
    attn_wq = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wk = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wv = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wo = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    mlp_fc1 = tabulate_3d n_layer (4 * n_embd) n_embd (\_ _ _ -> 0f32),
    mlp_fc2 = tabulate_3d n_layer n_embd (4 * n_embd) (\_ _ _ -> 0f32)
  }
  let vel : params [vocab_size] = {
    wte     = tabulate_2d vocab_size n_embd (\_ _ -> 0f32),
    wpe     = tabulate_2d block_size n_embd (\_ _ -> 0f32),
    lm_head = tabulate_2d vocab_size n_embd (\_ _ -> 0f32),
    attn_wq = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wk = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wv = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    attn_wo = tabulate_3d n_layer n_embd n_embd       (\_ _ _ -> 0f32),
    mlp_fc1 = tabulate_3d n_layer (4 * n_embd) n_embd (\_ _ _ -> 0f32),
    mlp_fc2 = tabulate_3d n_layer n_embd (4 * n_embd) (\_ _ _ -> 0f32)
  }

  let (p, _mom, _vel, final_loss) =
    loop (p, mom, vel, _loss) = (copy p, copy mom, copy vel, 0f32)
    for step < num_steps do

      let doc_idx = step % n_docs
      let doc = docs[doc_idx]
      let doc_len = doc_lens[doc_idx]

      let padded : [block_size + 1]i64 =
        tabulate (block_size + 1) (\i ->
          if i == 0 then bos
          else if i - 1 < doc_len then doc[i - 1]
          else bos
        )
      let n_pos = i64.min block_size (doc_len + 1)

      let (loss, grads) =
        vjp2 (\p_ -> compute_loss p_ padded n_pos) p 1f32

      -- Adam update
      let lr_t = learning_rate * (1f32 - f32.i64 step / f32.i64 num_steps)
      let bc1  = 1f32 - beta1 ** (f32.i64 step + 1)
      let bc2  = 1f32 - beta2 ** (f32.i64 step + 1)

      let mom' = params_map2 (\m g -> beta1 * m + (1 - beta1) * g) mom grads
      let vel' = params_map2 (\v_ g -> beta2 * v_ + (1 - beta2) * g * g) vel grads
      let p' = params_map3
        (\p_val m_val v_val ->
          let m_hat = m_val / bc1
          let v_hat = v_val / bc2
          in p_val - lr_t * m_hat / (f32.sqrt v_hat + eps_adam)
        ) p mom' vel'
      in (p', mom', vel', loss)
  in (final_loss,
      p.wte, p.wpe, p.lm_head,
      p.attn_wq, p.attn_wk, p.attn_wv, p.attn_wo,
      p.mlp_fc1, p.mlp_fc2)


-- Inference entry point
--
-- Note: Returns token IDs as a 2D array. Each sample is a [block_size]i64
-- array; -1 marks unused positions.

-- Helper needed b/c Futhark entry points can't directly accept records
def pack_params [v]
  (wte: [v][n_embd]f32) (wpe: [block_size][n_embd]f32)
  (lm_head: [v][n_embd]f32)
  (wq: [n_layer][n_embd][n_embd]f32)
  (wk: [n_layer][n_embd][n_embd]f32)
  (wv: [n_layer][n_embd][n_embd]f32)
  (wo: [n_layer][n_embd][n_embd]f32)
  (fc1: [n_layer][4 * n_embd][n_embd]f32)
  (fc2: [n_layer][n_embd][4 * n_embd]f32)
  : params [v] =
  { wte, wpe, lm_head,
    attn_wq = wq, attn_wk = wk, attn_wv = wv, attn_wo = wo,
    mlp_fc1 = fc1, mlp_fc2 = fc2 }

entry generate [v]
  (wte:   [v][n_embd]f32)
  (wpe:   [block_size][n_embd]f32)
  (lmh:   [v][n_embd]f32)
  (wq:    [n_layer][n_embd][n_embd]f32)
  (wk:    [n_layer][n_embd][n_embd]f32)
  (wv:    [n_layer][n_embd][n_embd]f32)
  (wo:    [n_layer][n_embd][n_embd]f32)
  (fc1:   [n_layer][4 * n_embd][n_embd]f32)
  (fc2:   [n_layer][n_embd][4 * n_embd]f32)
  (temperature:  f32)
  (num_samples:  i64)
  (infer_seed:   i32)
  : [num_samples][block_size]i64 =

  let bos = v - 1
  let p = pack_params wte wpe lmh wq wk wv wo fc1 fc2

  in tabulate num_samples (\sample_idx ->
    let rng_state = rng_for_index infer_seed sample_idx

    -- Fresh KV caches per sample
    let keys_init   = replicate n_layer
                        (replicate block_size (replicate n_embd 0f32))
    let values_init = replicate n_layer
                        (replicate block_size (replicate n_embd 0f32))

    let (_rng, _tok, _keys, _vals, result) =
      loop (rng_state, token_id, keys, vals, tokens) =
           (rng_state, bos, keys_init, values_init,
            replicate block_size (-1i64))
      for pos_id < block_size do
        if token_id == bos && pos_id > 0
        then (rng_state, token_id, keys, vals, tokens)
        else
          let (logits, keys', vals') =
            gpt p token_id pos_id keys vals
          let probs = softmax (map (/ temperature) logits)
          let (rng_state', next_token) =
            weighted_sample probs rng_state
          let tokens =
            if next_token == bos then tokens
            else tokens with [pos_id] = next_token
          in (rng_state', next_token, keys', vals', tokens)
    in result
  )

-- Convenience entry point: train then generate
entry demo [n_docs][max_doc_len]
  (vocab_size:  i64)
  (seed:        i32)
  (docs:       [n_docs][max_doc_len]i64)
  (doc_lens:   [n_docs]i64)
  (num_steps:   i64)
  (temperature: f32)
  (num_samples: i64)
  (infer_seed:  i32)
  : (f32, [num_samples][block_size]i64) =
  let (loss, wte, wpe, lmh, wq, wk, wv, wo, fc1, fc2) =
    train vocab_size seed docs doc_lens num_steps
  in (loss,
      generate wte wpe lmh wq wk wv wo fc1 fc2 temperature num_samples infer_seed)
