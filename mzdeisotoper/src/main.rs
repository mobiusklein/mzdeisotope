use std::fs;
use std::io;
use std::marker::PhantomData;
use std::path::PathBuf;

use clap::Parser;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};

use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::{
    field::MakeVisitor,
    fmt::{
        self,
        format::{DefaultVisitor, Writer},
    },
    prelude::*,
    EnvFilter,
};

use mzdeisotoper::{MZDeiosotoper, MZDeisotoperError};

#[cfg(windows)]
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// See https://github.com/tokio-rs/tracing/issues/3065#issuecomment-2318179647
struct CustomFormatter<T>{
    _t: PhantomData<T>
}

impl<T> CustomFormatter<T> {
    fn new() -> Self {
        Self { _t: PhantomData }
    }
}

impl<'a, T> MakeVisitor<Writer<'a>> for CustomFormatter<T> {
    type Visitor = DefaultVisitor<'a>;

    #[inline]
    fn make_visitor(&self, target: Writer<'a>) -> Self::Visitor {
        DefaultVisitor::new(target, true)
    }
}

pub fn main() -> Result<(), MZDeisotoperError> {
    let subscriber = tracing_subscriber::registry().with(
        fmt::layer()
            .with_timer(fmt::time::ChronoLocal::rfc_3339())
            .with_writer(io::stderr)
            .with_filter(
                EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .from_env_lossy(),
            ),
    );

    let mut config = Figment::new();
    let args = MZDeiosotoper::parse();

    if let Some(cpath) = args.config_file.clone() {
        config = config.merge(Serialized::defaults(args));
        config = config.merge(Toml::file_exact(cpath));
    } else {
        config = config.merge(Serialized::defaults(args));
    }

    if PathBuf::from("mzdeisotoper.toml").exists() {
        config = config.merge(Toml::file_exact("mzdeisotoper.toml"));
    }

    let args: MZDeiosotoper = config
        .merge(Env::prefixed("MZDEISOTOPER_"))
        .extract()
        .unwrap();

    if let Some(log_path) = args.log_file.as_ref() {
        let log_file = fs::File::create(log_path)?;
        let (log_file, _guard) = tracing_appender::non_blocking(log_file);
        let subscriber = subscriber.with(
            fmt::layer()
                .fmt_fields(CustomFormatter::<fs::File>::new())
                .with_span_events(FmtSpan::ACTIVE)
                .with_timer(fmt::time::ChronoLocal::rfc_3339())
                .with_ansi(false)
                .with_writer(log_file)
                .with_filter(
                    EnvFilter::builder()
                        .with_default_directive(tracing::Level::INFO.into())
                        .from_env_lossy(),
                ),
        );
        subscriber.init();
        args.main()?;
    } else {
        subscriber.init();
        args.main()?;
    }
    Ok(())
}
