//! An example of how you can use the Realtime OpenAI API to send and receive audio bytes.
//! This example uses OpenAI's Speech Activity Detection and as such does not need to manually commit audio buffers (by sending the relevant input event).
//!
//! In production, you would typically use something like CPAL and stream the audio bytes into the sender
//! (or potentially something else, depending on what you're trying to do).
//! You would then open the stream and convert the received audio deltas into bytes then use something like rodio to play the soundbytes back.
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use futures::StreamExt;
use hound::{SampleFormat, WavSpec, WavWriter};
use rig_experimental::providers::openai_realtime::realtime::{
    AudioFormat, GPT_4O_REALTIME_PREVIEW_20250603, InputEvent, Modality, ReceivedEventKind,
    ReceivedItemEventKind, Session, SessionEvent,
};
use rig_experimental::providers::openai_realtime::{
    Client,
    realtime::{RealtimeClient, RealtimeVoice, RealtimeVoiceRequest},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let audio_bytes = std::fs::read("examples/voice_clips/hello_world.wav").unwrap();
    let audio_bytes_base64 = BASE64_STANDARD.encode(audio_bytes);

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var should exist");
    let openai_client = Client::new(&api_key).realtime_client(GPT_4O_REALTIME_PREVIEW_20250603);

    let session = Session::new()
        .voice("sage")
        .input_audio_format(AudioFormat::Pcm16)
        .output_audio_format(AudioFormat::Pcm16)
        .modalities(vec![Modality::Text, Modality::Audio]);

    let req = RealtimeVoiceRequest::with_session(session);

    let (sender, mut stream) = openai_client.realtime_voice(req).await?;

    sender
        .send(InputEvent::append_audio(&audio_bytes_base64))
        .await
        .unwrap();

    let mut output_bytes: Vec<u8> = Vec::new();

    while let Some(evt) = stream.next().await {
        match evt.data {
            ReceivedEventKind::Item {
                data: ReceivedItemEventKind::AudioDelta { delta },
                ..
            } => {
                let bytes = BASE64_STANDARD.decode(delta).unwrap();
                output_bytes.extend(bytes);
            }
            ReceivedEventKind::Session(SessionEvent::SessionUpdated { session }) => {
                println!("Updated session: {session:?}");
            }
            ReceivedEventKind::Session(SessionEvent::SessionCreated { session }) => {
                println!("Created session: {session:?}");
            }
            ReceivedEventKind::Item {
                data: ReceivedItemEventKind::AudioDone,
                ..
            } => break,
        }
    }

    println!("{len} bytes received from OpenAI", len = output_bytes.len());

    let mut rdr = Cursor::new(&output_bytes);
    let mut samples = Vec::new();

    // We need to convert Vec<u8> to Vec<i16> as PCM bytes are i16 by default
    while let Ok(sample) = rdr.read_i16::<LittleEndian>() {
        samples.push(sample);
    }

    // It should be noted that OpenAI returns pcm16 data at a sample rate of 24kHz.
    // We should therefore reflect this in our sample rate/etc to avoid distorted audio... although if you wanted to change the pitch a bit
    // you can do so by playing around with the sampling rate
    let spec = WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int, // this gives you signed 16-bit PCM
    };

    let mut writer = WavWriter::create("output.wav", spec).unwrap();

    // Assuming you have i16 samples:
    for sample in samples {
        writer.write_sample(sample).unwrap();
    }

    writer.finalize().unwrap();

    Ok(())
}
