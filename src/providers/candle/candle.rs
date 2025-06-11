use rig::OneOrMany;
use rig::client::{AsEmbeddings, AsTranscription, CompletionClient, ProviderClient};
use rig::message::{AssistantContent, Message, Text, UserContent};
use serde::Deserialize;
use serde::de::Deserializer;
use std::collections::HashSet;
use std::marker::PhantomData;

use anyhow::Result;

use candle_transformers::models::mistral::{Config, Model as Mistral};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use tokenizers::Tokenizer;

pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle_core::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> candle_core::Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> candle_core::Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

impl<T> From<(T, Device, Tokenizer)> for CompletionModel<T>
where
    T: CandleModel + Clone + std::fmt::Debug + Sync + Send + 'static,
{
    fn from((model, device, tokenizer): (T, Device, Tokenizer)) -> Self {
        Self {
            model,
            device,
            tokenizer,
        }
    }
}

impl<T> From<CompletionModel<T>> for TextGeneration<T>
where
    T: CandleModel + Clone + std::fmt::Debug + Sync + Send + 'static,
{
    fn from(e: CompletionModel<T>) -> Self {
        Self::new(
            e.model,
            e.tokenizer,
            299792458, // seed RNG
            Some(0.),  // temperature
            None,      // top_p - Nucleus sampling probability stuff
            1.1,       // repeat penalty
            64,        // context size to consider for the repeat penalty
            &e.device,
        )
    }
}

impl<T> From<&CompletionModel<T>> for TextGeneration<T>
where
    T: CandleModel + Clone + std::fmt::Debug + Sync + Send + 'static,
{
    fn from(e: &CompletionModel<T>) -> Self {
        Self::new(
            e.model.clone(),
            e.tokenizer.clone(),
            299792458, // seed RNG
            Some(0.),  // temperature
            None,      // top_p - Nucleus sampling probability stuff
            1.1,       // repeat penalty
            64,        // context size to consider for the repeat penalty
            &e.device,
        )
    }
}

struct TextGeneration<T> {
    model: T,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<T> TextGeneration<T>
where
    T: CandleModel + Clone + std::fmt::Debug + Sync + Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: T,
        tokenizer: Tokenizer,
        seed: u64,
        _temp: Option<f64>,
        _top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, Some(0.0), None);

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(mut self, prompt: String, sample_len: usize) -> CompletionResponse {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => panic!("cannot find the </s> token"),
        };

        let mut string = String::new();

        let mut token_usage = 0;

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.forward(&input, start_pos).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if next_token == eos_token {
                token_usage = index + 1;
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                string.push_str(&t);
            }
        }

        CompletionResponse {
            response: string,
            token_usage,
        }
    }
}

pub struct CompletionResponse {
    response: String,
    pub token_usage: usize,
}

impl TryFrom<CompletionResponse> for rig::completion::CompletionResponse<CompletionResponse> {
    type Error = rig::completion::CompletionError;

    fn try_from(raw_response: CompletionResponse) -> std::result::Result<Self, Self::Error> {
        let text = raw_response.response.clone();
        Ok(Self {
            choice: OneOrMany::one(AssistantContent::Text(Text { text })),
            raw_response,
        })
    }
}

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> anyhow::Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}

#[derive(Debug, Clone)]
pub struct Client<T> {
    api_key: String,
    model_ty: PhantomData<T>,
}

impl<T> Client<T> {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model_ty: PhantomData,
        }
    }
}

impl<T> ProviderClient for Client<T>
where
    T: CandleModel + Clone + std::fmt::Debug + Send + Sync + 'static,
{
    fn from_env() -> Self {
        let api_key = std::env::var("HUGGINGFACE_API_KEY")
            .expect("HUGGINGFACE_API_KEY (HuggingFace API key) to exist as env var");

        Self {
            api_key,
            model_ty: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel<T> {
    model: T,
    device: Device,
    tokenizer: Tokenizer,
}

impl<T> rig::completion::CompletionModel for CompletionModel<T>
where
    T: CandleModel + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = String;

    async fn completion(
        &self,
        request: rig::completion::CompletionRequest,
    ) -> std::result::Result<
        rig::completion::CompletionResponse<Self::Response>,
        rig::completion::CompletionError,
    > {
        let max_tokens = if let Some(max_tokens) = request.max_tokens {
            max_tokens as usize
        } else {
            1024
        };
        let text_generation = TextGeneration::from(self);
        let prompt = request.chat_history.into_iter().last().unwrap();
        let Message::User { content } = prompt else {
            panic!("couldn't get user message");
        };
        let UserContent::Text(Text { text }) = content.into_iter().last().unwrap() else {
            panic!("User message was not text");
        };

        let response = text_generation.run(text, max_tokens);

        response.try_into()
    }

    async fn stream(
        &self,
        _request: rig::completion::CompletionRequest,
    ) -> std::result::Result<
        rig::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        rig::completion::CompletionError,
    > {
        todo!()
    }
}

// impl_conversion_traits!(
//     AsEmbeddings,
//     AsTranscription,
//     AsAudioGeneration,
//     AsImageGeneration for Client
// );
//
impl<T> AsEmbeddings for Client<T>
where
    T: CandleModel + std::fmt::Debug + Clone + Send + Sync,
{
    fn as_embeddings(&self) -> Option<Box<dyn rig::client::embeddings::EmbeddingsClientDyn>> {
        None
    }
}

impl<T> AsTranscription for Client<T>
where
    T: CandleModel + std::fmt::Debug + Clone + Send + Sync,
{
    fn as_transcription(
        &self,
    ) -> Option<Box<dyn rig::client::transcription::TranscriptionClientDyn>> {
        None
    }
}

impl<T> CompletionClient for Client<T>
where
    T: CandleModel + std::fmt::Debug + Clone + Send + Sync + 'static,
    T::Config: Clone + std::fmt::Debug,
{
    type CompletionModel = CompletionModel<T>;
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        let api = ApiBuilder::new()
            .with_token(Some(self.api_key.clone()))
            .build()
            .expect("to successfully build the HuggingFace API client");

        let repo = api.repo(Repo::with_revision(
            model.to_string(),
            RepoType::Model,
            "latest".to_string(),
        ));

        let tokenizer = {
            let tokenizer_filename = repo.get("tokenizer.json").unwrap();
            Tokenizer::from_file(tokenizer_filename).unwrap()
        };

        let device = Device::Cpu;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap();

        let model = {
            let dtype = DType::F32;
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
            T::new(vb)
        };

        CompletionModel::from((model, device, tokenizer))
    }
}

trait CandleModel {
    type Config: Clone + std::fmt::Debug;
    fn new(vb: VarBuilder<'_>) -> Self;

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle_core::Result<Tensor>;
}

impl CandleModel for Mistral {
    type Config = Config;

    fn new(vb: VarBuilder<'_>) -> Self {
        let config = Config::config_7b_v0_1(false);
        Self::new(&config, vb).unwrap()
    }

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle_core::Result<Tensor> {
        self.forward(input_ids, seqlen_offset)
    }
}
