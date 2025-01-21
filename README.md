# `mzdeisotope` - Tools for deisotoping and charge deconvolution for mass spectra

## Components

- `mzdeisotope` - A Rust library with tools for deconvolution of peak lists
- `mzdeisotope-map` - A Rust library that expands `mzdeisotope` for processing ion mobility frames and LC-MS feature maps
- `mzdeisotoper` - An executable tool for applying the deconvolution process whole mass spectrometry data files

## Library Usage

**TODO**

## Tool Usage

```
$ mzdeisotoper --help
Deisotoping and charge state deconvolution of mass spectrometry files.

Read a file or stream, transform the spectra, and write out a processed mzML file or stream.

Usage: mzdeisotoper [OPTIONS] <INPUT_FILE>

Arguments:
  <INPUT_FILE>
          The path to read the input spectra from, or if '-' is passed, read from STDIN

Options:
  -o, --output-file <OUTPUT_FILE>
          The path to write the output file to, or if '-' is passed, write to STDOUT

          [default: -]

  -t, --threads <THREADS>
          The number of threads to use, passing a value < 1 to use all available threads

          [default: -1]

  -r, --time-range <BEGIN-END>
          The time range to process, denoted [start?]-[stop?]

          If a start is not specified, processing begins from the start of the run.
          If a stop is not specified, processing stops at the end of the run.


  -g, --ms1-averaging-range <MS1_AVERAGING_RANGE>
          The number of MS1 spectra before and after to average with prior to peak picking

          [default: 0]

  -b, --ms1-background-reduction <MS1_DENOISING>
          The magnitude of background noise reduction to use on MS1 spectra prior to peak picking

          [default: 0]

  -a, --ms1-isotopic-model <MS1_ISOTOPIC_MODEL>
          The isotopic model to use for MS1 spectra

          [default: peptide]
          [possible values: peptide, glycan, glycopeptide, permethylated-glycan, heparin, heparan-sulfate]

  -s, --ms1-score-threshold <MS1_SCORE_THRESHOLD>
          The minimum isotopic pattern fit score for MS1 spectra

          [default: 20]

  -A, --msn-isotopic-model <MSN_ISOTOPIC_MODEL>
          The isotopic model to use for MSn spectra

          [default: peptide]
          [possible values: peptide, glycan, glycopeptide, permethylated-glycan, heparin, heparan-sulfate]

  -S, --msn-score-threshold <MSN_SCORE_THRESHOLD>
          The minimum isotopic pattern fit score for MSn spectra

          [default: 10]

  -v, --precursor-processing <PRECURSOR_PROCESSING>
          How to treat precursor ranges

          [default: selected-precursors]

          Possible values:
          - full:                Process the entire MS1 mass range and all MSn spectra
          - selected-precursors: Process only the MS1 regions that are selected for MSn and all MSn spectra
          - tandem-only:         Process only MSn spectra without examining MS1 spectra
          - ms1-only:            Process only the MS1 spectra without examining MSn spectra

  -z, --charge-range <CHARGE_RANGE>
          The range of charge states to consider for each peak denoted [low]-[high] or [high]

          [default: 1-8]

  -m, --max-missed-peaks <MS1_MISSED_PEAKS>
          The maximum number of missed peaks for MS1 spectra

          [default: 1]

  -M, --msn-max-missed-peaks <MSN_MISSED_PEAKS>
          The maximum number of missed peaks for MSn spectra

          [default: 1]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Input Files

`mzdeisotoper` can read mzML, MGF, and if built with the `mzmlb` feature, mzMLb from the file
system. The program can also receive mzML or MGF over STDIN, letting you pipe the output of a
tool like `msconvert` or `curl` into it.

| Format | Read Files | Read Pipe | Feature<br>Requirements |
|:------ | ---------- | --------- | :--------------------: |
| mzML   | :white_check_mark: | :white_check_mark: | Always |
| MGF    | :white_check_mark: | :white_check_mark: | Always |
| mzMLb  | :white_check_mark: | :x: | `mzmlb` |
| Thermo | :white_check_mark: | :x: | `thermo` |
