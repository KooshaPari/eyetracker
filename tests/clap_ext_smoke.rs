//! Smoke tests verifying that clap-ext's Verbosity and ConfigArg work in this CLI.

use clap::Parser;
use clap_ext::prelude::*;

#[test]
fn clap_ext_verbosity_parses_quiet_flag() {
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        verbosity: Verbosity,
    }
    let p = Probe::try_parse_from(["probe", "--quiet"]).expect("parse");
    let filter = p.verbosity.to_filter();
    assert_eq!(format!("{:?}", filter), "LevelFilter::ERROR");
}

#[test]
fn clap_ext_verbosity_parses_double_v() {
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        verbosity: Verbosity,
    }
    let p = Probe::try_parse_from(["probe", "-vv"]).expect("parse");
    let filter = p.verbosity.to_filter();
    assert_eq!(format!("{:?}", filter), "LevelFilter::TRACE");
}

#[test]
fn clap_ext_config_arg_default_is_none() {
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        config: ConfigArg,
    }
    let p = Probe::try_parse_from(["probe"]).expect("parse");
    assert!(p.config.config.is_none());
}

#[test]
fn clap_ext_config_arg_parses_short_flag() {
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        config: ConfigArg,
    }
    let p = Probe::try_parse_from(["probe", "-c", "/tmp/cfg.toml"]).expect("parse");
    assert_eq!(p.config.config.unwrap().to_str().unwrap(), "/tmp/cfg.toml");
}
