use figment::{
    providers::{Format, Toml},
    Figment,
};

#[test_log::test]
#[test_log(default_log_filter = "debug")]
fn test_waters() {
    let mut config = Figment::new();
    config = config.merge(Toml::file_exact("../test/data/waters_test.toml"));
    let driver: mzdeisotoper::MZDeiosotoper = config.extract().unwrap();
    driver.main().unwrap();
}
