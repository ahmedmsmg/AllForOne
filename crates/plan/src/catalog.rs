//! Models the planner can choose from, with the numbers needed to size them.
//!
//! Only single-file GGUFs of the `llama` architecture with the Llama 3 chat template are listed,
//! since that is what the engine and client run. File sizes are exact (from the download host).

pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub n_layer: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// The output head reuses the token embedding matrix.
    pub tied: bool,
    pub repo: &'static str,
    pub file_prefix: &'static str,
    pub tokenizer_repo: &'static str,
    pub quants: &'static [(&'static str, u64)],
}

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "llama-3.3-70b",
        name: "Llama 3.3 70B Instruct",
        n_layer: 80,
        d_model: 8192,
        n_head: 64,
        n_kv_head: 8,
        ffn: 28672,
        vocab: 128256,
        tied: false,
        repo: "bartowski/Llama-3.3-70B-Instruct-GGUF",
        file_prefix: "Llama-3.3-70B-Instruct",
        tokenizer_repo: "unsloth/Llama-3.3-70B-Instruct",
        // Larger quants of this model are published split into parts, which the engine can't load.
        quants: &[("Q4_K_M", 42_520_398_816), ("Q4_K_S", 40_347_225_056), ("Q3_K_M", 34_267_499_488)],
    },
    ModelSpec {
        id: "llama-3.1-8b",
        name: "Llama 3.1 8B Instruct",
        n_layer: 32,
        d_model: 4096,
        n_head: 32,
        n_kv_head: 8,
        ffn: 14336,
        vocab: 128256,
        tied: false,
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file_prefix: "Meta-Llama-3.1-8B-Instruct",
        tokenizer_repo: "unsloth/Meta-Llama-3.1-8B-Instruct",
        quants: &[
            ("Q8_0", 8_540_775_840),
            ("Q6_K", 6_596_011_424),
            ("Q5_K_M", 5_732_992_416),
            ("Q4_K_M", 4_920_739_232),
        ],
    },
    ModelSpec {
        id: "llama-3.2-3b",
        name: "Llama 3.2 3B Instruct",
        n_layer: 28,
        d_model: 3072,
        n_head: 24,
        n_kv_head: 8,
        ffn: 8192,
        vocab: 128256,
        tied: true,
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        file_prefix: "Llama-3.2-3B-Instruct",
        tokenizer_repo: "unsloth/Llama-3.2-3B-Instruct",
        quants: &[
            ("Q8_0", 3_421_899_296),
            ("Q6_K", 2_643_853_856),
            ("Q5_K_M", 2_322_154_016),
            ("Q4_K_M", 2_019_377_696),
        ],
    },
    ModelSpec {
        id: "llama-3.2-1b",
        name: "Llama 3.2 1B Instruct",
        n_layer: 16,
        d_model: 2048,
        n_head: 32,
        n_kv_head: 8,
        ffn: 8192,
        vocab: 128256,
        tied: true,
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
        file_prefix: "Llama-3.2-1B-Instruct",
        tokenizer_repo: "unsloth/Llama-3.2-1B-Instruct",
        quants: &[
            ("Q8_0", 1_321_083_008),
            ("Q6_K", 1_021_800_576),
            ("Q5_K_M", 911_503_488),
            ("Q4_K_M", 807_694_464),
        ],
    },
];

/// One downloadable file: a model at a given quantization.
#[derive(Clone, Copy)]
pub struct Candidate {
    pub model: &'static ModelSpec,
    pub quant: &'static str,
    pub file_bytes: u64,
}

/// Every model/quant pair, best first: more parameters wins, then more bits per weight. A big
/// model at 4 bits generally beats a smaller one at 8.
pub fn candidates() -> Vec<Candidate> {
    MODELS
        .iter()
        .flat_map(|m| m.quants.iter().map(move |&(quant, file_bytes)| Candidate { model: m, quant, file_bytes }))
        .collect()
}

pub fn find(id: &str) -> Option<Candidate> {
    candidates().into_iter().find(|c| c.id().eq_ignore_ascii_case(id))
}

impl Candidate {
    pub fn id(&self) -> String {
        format!("{}:{}", self.model.id, self.quant)
    }

    pub fn file_name(&self) -> String {
        format!("{}-{}.gguf", self.model.file_prefix, self.quant)
    }

    pub fn tokenizer_name(&self) -> String {
        format!("{}.tokenizer.json", self.model.file_prefix)
    }

    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.model.repo, self.file_name())
    }

    pub fn tokenizer_url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/tokenizer.json", self.model.tokenizer_repo)
    }

    fn head_dim(&self) -> usize {
        self.model.d_model / self.model.n_head
    }

    fn layer_params(&self) -> u64 {
        let m = self.model;
        let kv = m.n_kv_head * self.head_dim();
        (2 * m.d_model * m.d_model + 2 * m.d_model * kv + 3 * m.d_model * m.ffn) as u64
    }

    fn vocab_params(&self) -> u64 {
        (self.model.vocab * self.model.d_model) as u64
    }

    /// Average stored bytes per parameter in this file.
    fn bytes_per_param(&self) -> f64 {
        self.file_bytes as f64 / self.params() as f64
    }

    /// Quantized weights of one transformer block.
    pub fn layer_bytes(&self) -> u64 {
        (self.layer_params() as f64 * self.bytes_per_param()) as u64
    }

    /// What the client holds: the embedding (dequantized to f32 on load) and the quantized head.
    pub fn client_bytes(&self) -> u64 {
        self.vocab_params() * 4 + self.head_bytes()
    }

    pub fn head_bytes(&self) -> u64 {
        (self.vocab_params() as f64 * self.bytes_per_param()) as u64
    }

    /// f32 keys and values for `ctx` tokens in one block.
    pub fn kv_bytes_per_layer(&self, ctx: usize) -> u64 {
        (ctx * 2 * self.model.n_kv_head * self.head_dim() * 4) as u64
    }

    /// Peak scratch while a node prefills a prompt of `tokens` in one pass: the attention scores
    /// (plus the masked and softmaxed copies) and the MLP activations.
    pub fn prefill_bytes(&self, tokens: usize) -> u64 {
        let m = self.model;
        (3 * m.n_head * tokens * tokens * 4 + 3 * tokens * m.ffn * 4) as u64
    }

    pub fn params(&self) -> u64 {
        self.model.n_layer as u64 * self.layer_params() + self.vocab_params() * if self.model.tied { 1 } else { 2 }
    }
}
