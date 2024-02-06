use std::fs;
use std::io;

use clap::Parser;

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use mzdeisotoper::{MZDeiosotoper, MZDeisotoperError};


pub fn main() -> Result<(), MZDeisotoperError> {
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::TRACE.into()))
        .with(
            fmt::layer().compact().with_writer(io::stderr).with_filter(
                EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .from_env_lossy(),
            ),
        );

    let args = MZDeiosotoper::parse();

    if let Some(log_path) = args.log_file.as_ref() {
        let log_file = fs::File::create(log_path)?;
        let (log_file, _guard) = tracing_appender::non_blocking(log_file);
        let subscriber = subscriber.with(
            fmt::layer()
                .compact()
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
