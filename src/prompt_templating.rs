use rig::{
    agent::Agent,
    completion::{Chat, CompletionModel, Prompt, PromptError},
    message::Message,
};
use serde::Serialize;
use std::path::Path;
use tera::Context;

/// Prompting templates.
/// Create your own template using Jinja formatting, then use the fluent builder to set variables (or add them in from a type that implements Serialize).
///
/// Usage:
/// ```rust
/// let str = "Hello {{ user }}!";
///
/// let template = PromptTemplate::new(str)
///     .with_variable("user", "Rig");
///
/// let res = template.render_to_string();
/// assert_eq!(res, "Hello Rig!".to_string());
/// ```
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: Context,
}

impl PromptTemplate {
    /// Create a new PromptTemplate instance from a string.
    pub fn new(str: &str) -> Self {
        Self {
            template: str.to_string(),
            variables: Context::new(),
        }
    }

    /// Create a new PromptTemplate instance from the text contents of a file.
    pub fn from_file<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let str = std::fs::read_to_string(path).unwrap();

        Self {
            template: str.to_string(),
            variables: Context::new(),
        }
    }

    /// Set a variable for use in the prompt template.
    pub fn with_variable<V>(mut self, k: &str, v: V) -> Self
    where
        V: Serialize,
    {
        self.variables.insert(k, &v);
        self
    }

    /// Set a list of variables to be used in a PromptTemplate from a type that implements Serialize (ie, a hashmap, a btree, etc...).
    pub fn with_variables_from_serialize<V>(mut self, v: V) -> Result<Self, tera::Error>
    where
        V: Serialize,
    {
        self.variables = Context::from_serialize(v)?;
        Ok(self)
    }

    /// Sets a variable using &mut.
    pub fn set_variable(&mut self, k: &str, v: &str) {
        self.variables.insert(k, v);
    }

    /// Renders the template as a string.
    pub fn render_to_string(&self) -> String {
        tera::Tera::one_off(&self.template, &self.variables, false).unwrap()
    }
}

/// A helper trait to make it easier to idiomatically convert types into custom types that can easily use prompt templating.
pub trait PromptTemplating<T> {
    fn into_prompt_template(self, template: &str) -> PromptTemplatingWrapper<T>;
}

/// A prompt templating wrapper (that wraps over a type).
/// Not intended to be instantiated outside of the crate as this is primarily to be used with [`PromptTemplating<T>`].
#[derive(Debug)]
pub struct PromptTemplatingWrapper<T> {
    template: PromptTemplate,
    inner: T,
}

impl<T> PromptTemplatingWrapper<T>
where
    T: Sized,
{
    /// Set a variable for usage with your prompt template.
    pub fn with_variable<V>(mut self, k: &str, v: V) -> Self
    where
        V: Serialize,
    {
        self.template = self.template.with_variable(k, v);
        self
    }

    /// Set a list of variables to be used in a PromptTemplate from a type that implements Serialize (ie, a hashmap, a btree, etc...).
    pub fn with_variables_from_serialize<V>(mut self, v: V) -> Result<Self, tera::Error>
    where
        V: Serialize,
    {
        self.template = self.template.with_variables_from_serialize(v)?;
        Ok(self)
    }
}

impl<M> PromptTemplating<Agent<M>> for Agent<M>
where
    M: CompletionModel + 'static,
{
    fn into_prompt_template(self, template: &str) -> PromptTemplatingWrapper<Agent<M>> {
        PromptTemplatingWrapper {
            template: PromptTemplate::new(template),
            inner: self,
        }
    }
}

impl<M> PromptTemplatingWrapper<Agent<M>>
where
    M: CompletionModel,
{
    /// Prompt your agent using your prompt template and the variables you've set.
    pub async fn prompt(self) -> Result<String, PromptError> {
        let res = self.template.render_to_string();

        self.inner.prompt(res).await
    }

    /// Prompt your agent using your prompt template and the variables you've set, as well as enabling automatic multi-turn.
    pub async fn prompt_multi_turn(self, turns: usize) -> Result<String, PromptError> {
        let res = self.template.render_to_string();

        self.inner.prompt(res).multi_turn(turns).await
    }

    /// Chat with your agent using your prompt template and the variables you've set, as well as a message history.
    pub async fn chat(self, message_history: Vec<Message>) -> Result<String, PromptError> {
        let res = self.template.render_to_string();

        self.inner.chat(res, message_history).await
    }
}
