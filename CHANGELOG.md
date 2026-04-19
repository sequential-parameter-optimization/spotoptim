## [0.12.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.12.0...v0.12.1) (2026-04-19)

### Bug Fixes

* **deps:** bump scikit-learn floor in requirements-docs.txt to >=1.5.0 ([d9e776e](https://github.com/sequential-parameter-optimization/spotoptim/commit/d9e776efcafd884d1d74c5b7b894e9c1fd686cd2))

## [0.12.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.5...v0.12.0) (2026-04-18)

### Features

* **spotoptim:** add max_restarts patience-based early stopping ([7d6fe70](https://github.com/sequential-parameter-optimization/spotoptim/commit/7d6fe70ee03bb1549b2fc098c995714aaf9219c2))

### Documentation

* **early-stopping:** add dedicated chapter for max_restarts ([73557cc](https://github.com/sequential-parameter-optimization/spotoptim/commit/73557ccd06b83b22bc6a5d7c87aed1ddedb5a5ba))
* **slurm:** add GWDG NHR cluster recipe + reference scripts ([eca2cbc](https://github.com/sequential-parameter-optimization/spotoptim/commit/eca2cbcb338a41106bfec9db9e0e0c8d96f3a5cd))

## [0.11.5](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.4...v0.11.5) (2026-04-12)

### Bug Fixes

* show figures in plot user guide instead of print statements ([57ec92b](https://github.com/sequential-parameter-optimization/spotoptim/commit/57ec92bc58925bb99008d7866811941ea36acd8b))

## [0.11.4](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.3...v0.11.4) (2026-04-12)

### Bug Fixes

* commit _freeze/ outputs and fix user guide code examples ([a992fa3](https://github.com/sequential-parameter-optimization/spotoptim/commit/a992fa32ea033e3734d01b33d649752f3b16bd70))

### Documentation

* add comprehensive User Guide with 18 module pages ([5559729](https://github.com/sequential-parameter-optimization/spotoptim/commit/55597298a416ae78244ef2fce41524e2a71dd28e))
* update _is_gil -> is_gil ([50a08d9](https://github.com/sequential-parameter-optimization/spotoptim/commit/50a08d9c5072865730249cef19d7bdae608aa504))

## [0.11.3](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.2...v0.11.3) (2026-04-05)

### Bug Fixes

* ty ([738ba02](https://github.com/sequential-parameter-optimization/spotoptim/commit/738ba02f34a5b80ae650e1c731adb292c6c7f997))
* warnings_type ([18151b3](https://github.com/sequential-parameter-optimization/spotoptim/commit/18151b3f497c6830c716cf1c98e8ba5bf2940c15))

### Code Refactoring

* add SpotOptimProtocol for type-safe extracted modules ([5a38c02](https://github.com/sequential-parameter-optimization/spotoptim/commit/5a38c02e9fa8c954b3fa8c8f423b4bd8b532b77b))

## [0.11.2](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.1...v0.11.2) (2026-04-05)

### Bug Fixes

* update is_gil_disabled import path in optimize_parallel.qmd ([f17edeb](https://github.com/sequential-parameter-optimization/spotoptim/commit/f17edeb579c2e2e1027c7d56a75d8f20c2f74f95))

## [0.11.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.11.0...v0.11.1) (2026-04-05)

### Code Refactoring

* extract acquisition optimization methods from SpotOptim class ([4ec71db](https://github.com/sequential-parameter-optimization/spotoptim/commit/4ec71db004bb7343bcf0b1b251a05807decf6f60))
* extract data management methods from SpotOptim class ([e6d6558](https://github.com/sequential-parameter-optimization/spotoptim/commit/e6d65580ccd9888749a456d37bfeb870f324c917))
* extract dimension reduction methods from SpotOptim class ([e80757c](https://github.com/sequential-parameter-optimization/spotoptim/commit/e80757c842d37308757c8968dff7b24bbc2ca802))
* extract OCBA methods from SpotOptim class ([d1db547](https://github.com/sequential-parameter-optimization/spotoptim/commit/d1db54714050377dc521c5655056c1befe0520cb))
* extract reporting and analysis methods from SpotOptim class ([0fd5750](https://github.com/sequential-parameter-optimization/spotoptim/commit/0fd5750c2372efb4ffb1d33875c4a9f1276d8c8d))
* extract serialization methods from SpotOptim class ([63fa4e4](https://github.com/sequential-parameter-optimization/spotoptim/commit/63fa4e49eb448e5c2c1da72988d7ad75c76051b9))
* extract steady-state parallel optimization from SpotOptim class ([916b9ed](https://github.com/sequential-parameter-optimization/spotoptim/commit/916b9edfd6e800107871464e861af2274a634f40))
* extract variable and transformation methods from SpotOptim class ([9d7da12](https://github.com/sequential-parameter-optimization/spotoptim/commit/9d7da12c9b6103067249f036d74b70fa11dd6226))

## [0.11.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.10.3...v0.11.0) (2026-04-05)

### Features

* parallel and wrapper ([847bec0](https://github.com/sequential-parameter-optimization/spotoptim/commit/847bec025186f90558704243f61b33cbd76158eb))

### Bug Fixes

* add --extra docs to uv run commands in release workflow ([f568290](https://github.com/sequential-parameter-optimization/spotoptim/commit/f5682904640802a844e288fb9afd008e0a91548b))
* ci ([497ae0d](https://github.com/sequential-parameter-optimization/spotoptim/commit/497ae0da0a4032ea20bc5a6932dd52aec185be3f))
* global structure ([b8f89ac](https://github.com/sequential-parameter-optimization/spotoptim/commit/b8f89acd0c1bdba4fd734f045f17e035b4bb70fa))
* ip plots dataframe colnames ([cbee260](https://github.com/sequential-parameter-optimization/spotoptim/commit/cbee2605e10055bbcb5d1e9ee1f040e64e83e70d))
* labels ([ff400e0](https://github.com/sequential-parameter-optimization/spotoptim/commit/ff400e0942f39edad5539a0a15e2030956768a9c))
* preflight and release ([13eaa27](https://github.com/sequential-parameter-optimization/spotoptim/commit/13eaa27010174ddb1652fd8f8ce4a91d580ea5f8))
* quartodoc 2nd ([4c4837b](https://github.com/sequential-parameter-optimization/spotoptim/commit/4c4837b5f673842ab1cea8e9dc46293d265cbc13))
* quartodoc ru ([e976ced](https://github.com/sequential-parameter-optimization/spotoptim/commit/e976cedcd3a573f883e6314e9d13f0eea306957c))
* release ([09c8d3f](https://github.com/sequential-parameter-optimization/spotoptim/commit/09c8d3fcc70eef23beea2ae4065807c93cb095c9))
* release-preflight ([eed5cc0](https://github.com/sequential-parameter-optimization/spotoptim/commit/eed5cc094e7de81c0b6b9deedde5d1a35a599e91))
* renaming fit() ([39ebe63](https://github.com/sequential-parameter-optimization/spotoptim/commit/39ebe63d30e3ab8358ee9b4316a11aba9d71c555))
* TASKS ([4a4d94d](https://github.com/sequential-parameter-optimization/spotoptim/commit/4a4d94d268edaf2d337541d278760689a821889c))
* tolerance ([b0d6e25](https://github.com/sequential-parameter-optimization/spotoptim/commit/b0d6e258dc71dc60ffbd82eb03dab5bf9f68d1e8))
* update preflight ([a3a838a](https://github.com/sequential-parameter-optimization/spotoptim/commit/a3a838a4d7a5465ba27d0d9e70537192d5522bf3))
* via claude ([8fa065a](https://github.com/sequential-parameter-optimization/spotoptim/commit/8fa065a79f48c54d0cd9a228023569ac2810c545))

### Documentation

* fit ([8d6e61f](https://github.com/sequential-parameter-optimization/spotoptim/commit/8d6e61f8e2f05bef7277ee09a8847087d3a1aefc))

### Code Refactoring

* extract TensorBoard and safe_float from SpotOptim class ([5417505](https://github.com/sequential-parameter-optimization/spotoptim/commit/5417505b6685cacfce1cc180f79c136b435c2131))

## [0.10.3](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.10.2...v0.10.3) (2026-03-15)


### Bug Fixes

* y0eval can be passed to parallel ([d1be9dd](https://github.com/sequential-parameter-optimization/spotoptim/commit/d1be9dd8ed58e876555adefdf42c4386e868cebb))


### Documentation

* seq/parallel ([fc20db5](https://github.com/sequential-parameter-optimization/spotoptim/commit/fc20db55f3b6defa6b19034121e9e02f7f39dd5a))

## [0.10.2](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.10.1...v0.10.2) (2026-03-15)


### Bug Fixes

* **security:** address Scorecard supply-chain findings ([4f76eae](https://github.com/sequential-parameter-optimization/spotoptim/commit/4f76eaed7308a76efaf62c06b4d4d76c2bb7dad0))

## [0.10.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.10.0...v0.10.1) (2026-03-15)


### Bug Fixes

* **ci:** add custom CodeQL workflow for Python-only analysis ([25de631](https://github.com/sequential-parameter-optimization/spotoptim/commit/25de631908ab3f8a283d6fb732ae65066900e898))

## [0.10.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.9.0...v0.10.0) (2026-03-15)


### Features

* GIL awareness ([82d4bc0](https://github.com/sequential-parameter-optimization/spotoptim/commit/82d4bc0f54e7879377c770d8efd32b922217497b))


### Bug Fixes

* 3.14 added ([57e4548](https://github.com/sequential-parameter-optimization/spotoptim/commit/57e4548699d51102a4fadb31a779ef527888572a))
* cleanup ([d03fde8](https://github.com/sequential-parameter-optimization/spotoptim/commit/d03fde817c425b525b7b683f5620d71f5c37f391))
* overshooting managed ([c0c6df1](https://github.com/sequential-parameter-optimization/spotoptim/commit/c0c6df1528b097ebbbf2bec56d26ab2a362c3650))

## [0.9.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.8.0...v0.9.0) (2026-03-15)


### Features

* 0.10.0 parall finisihed ([5ddd33b](https://github.com/sequential-parameter-optimization/spotoptim/commit/5ddd33b9aaae625d1c64e7a6b3e81d286fe8b67f))

## [0.8.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.7.0...v0.8.0) (2026-03-15)


### Features

* parallel 0.9.0 ([6bfb9ef](https://github.com/sequential-parameter-optimization/spotoptim/commit/6bfb9efe67d90979ce4906cc9f06e10a6bb5086a))

## [0.7.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.6.0...v0.7.0) (2026-03-15)


### Features

* parallel step 1 ([3632f58](https://github.com/sequential-parameter-optimization/spotoptim/commit/3632f58338fb84f959d27f560908d9d848336de2))

## [0.6.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.5.1...v0.6.0) (2026-03-15)


### Features

* plot_design_points ([c65c0c7](https://github.com/sequential-parameter-optimization/spotoptim/commit/c65c0c7e29d38d5374052081486e6e3a52421368))

## [0.5.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.5.0...v0.5.1) (2026-03-14)


### Bug Fixes

* **ci:** resolve all pipeline failures ([eb94af5](https://github.com/sequential-parameter-optimization/spotoptim/commit/eb94af57e19e72a09b86c2f8c530800d8c5da590))
* correctness of mlp_surrogate ([af1f59e](https://github.com/sequential-parameter-optimization/spotoptim/commit/af1f59ef6bf752b35c3b3c65edc183eb7cebc250))
* problem with scipy's DE bound handling ([bf5f2dd](https://github.com/sequential-parameter-optimization/spotoptim/commit/bf5f2dda39911535e6c9b6dd07ac16842523f156))
* **surrogate+test:** vectorise MC dropout predict and fix timing assertions ([18d0749](https://github.com/sequential-parameter-optimization/spotoptim/commit/18d07492e0453d102f0d3744002754bd29948d3c))
* **test:** correct skip reason for test_mlp_surrogate_uncertainty_in_loop ([129325a](https://github.com/sequential-parameter-optimization/spotoptim/commit/129325a90f9328face512236b132ae662be65400))


### Performance Improvements

* **acquisition:** vectorise DE and tricands — 18× speedup for slow surrogates ([8870a51](https://github.com/sequential-parameter-optimization/spotoptim/commit/8870a5135a1e525f887e17a0462e79a5547e0423))

## [0.5.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.4.0...v0.5.0) (2026-03-14)


### Features

* class organisation ([0378c68](https://github.com/sequential-parameter-optimization/spotoptim/commit/0378c68fa106f5becb82be71f5090126a23ebb22))
* corrected mm ([#36](https://github.com/sequential-parameter-optimization/spotoptim/issues/36)) ([6888851](https://github.com/sequential-parameter-optimization/spotoptim/commit/6888851a346812b16db0da50567258fd542ae5e1))
* figsize for mm plots ([762f96a](https://github.com/sequential-parameter-optimization/spotoptim/commit/762f96a81b711e2b7610259ac236fb2fcd692985))
* init_surrogate() ([a5da126](https://github.com/sequential-parameter-optimization/spotoptim/commit/a5da12602f7961c6854eed5845aa84987d1d5996))
* mm-corrected ([0c22139](https://github.com/sequential-parameter-optimization/spotoptim/commit/0c22139d940ba03b03707bad897c61e77d1138b6))


### Bug Fixes

* **ci:** add UV_INDEX_STRATEGY for uv v7 index resolution ([adb6709](https://github.com/sequential-parameter-optimization/spotoptim/commit/adb670907f4108cbcbd5c1d6f25bcaf6e959044b))
* **ci:** configure PyTorch CPU index in pyproject.toml for uv v7 ([b4a0ae1](https://github.com/sequential-parameter-optimization/spotoptim/commit/b4a0ae12a881fd762841067b8ea5fbd731f8eb17))
* **ci:** increase test shard timeout to 45 minutes ([3a783aa](https://github.com/sequential-parameter-optimization/spotoptim/commit/3a783aa2ff71512e69eb1eb534c0d1a9e0dec8e6))
* **ci:** raise per-test timeout 300s→900s for slow optimization tests ([13486ce](https://github.com/sequential-parameter-optimization/spotoptim/commit/13486cea719820a79fb46f58a8fbd8abbc9a2859))
* **ci:** remove empty env block from release.yml ([da9fcbd](https://github.com/sequential-parameter-optimization/spotoptim/commit/da9fcbdcd0a5771d6f54a4c26a89c08e7df3feac))
* **ci:** resolve workflow conflicts with main, keep improved versions ([63ee649](https://github.com/sequential-parameter-optimization/spotoptim/commit/63ee6494d010e281159f3ae9f3780224960c90b3))
* release workflow ([67b0bce](https://github.com/sequential-parameter-optimization/spotoptim/commit/67b0bcee4fcf5c780aded7ef87a4d4df3a34e862))
* remove empty env block in release workflow ([ae853c2](https://github.com/sequential-parameter-optimization/spotoptim/commit/ae853c2d3f73aa0e3cbc30e30e9e6b07b350688e))
* transform_X bug ([2957e9a](https://github.com/sequential-parameter-optimization/spotoptim/commit/2957e9aa26ae359defd0b0d42a419aca9573b641))
* workflow ([3fa3f62](https://github.com/sequential-parameter-optimization/spotoptim/commit/3fa3f62dc4d62ef11e35128200bea42813dda0f4))

## [0.4.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.3.2...v0.4.0) (2026-03-06)


### Features

* spotoptim classes ([#31](https://github.com/sequential-parameter-optimization/spotoptim/issues/31)) ([9573929](https://github.com/sequential-parameter-optimization/spotoptim/commit/9573929795d7acbbeebc4b78784799ed41595783))

## [0.3.2](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.3.1...v0.3.2) (2026-03-05)


### Bug Fixes

* .gitignore ([9f392de](https://github.com/sequential-parameter-optimization/spotoptim/commit/9f392de665da6934310e0dd09b88c797f03ebd60))
* dcos ([26f5af5](https://github.com/sequential-parameter-optimization/spotoptim/commit/26f5af5cc4086eb4a73d478f5c52b66ceeb842d5))

## [0.3.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.3.0...v0.3.1) (2026-03-04)


### Bug Fixes

* doc refinement ([0570c31](https://github.com/sequential-parameter-optimization/spotoptim/commit/0570c315acac822ec72e421c61e4071ac9a437df))

## [0.3.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.6...v0.3.0) (2026-03-04)


### Features

* cleanup print_results_table and print_design_table ([47c4272](https://github.com/sequential-parameter-optimization/spotoptim/commit/47c4272913e1648dff0097e02108218c42770947))


### Bug Fixes

* **ci:** use Python 3.13 + uv, add linear history guard for PRs ([d3fb160](https://github.com/sequential-parameter-optimization/spotoptim/commit/d3fb160424b57d4561b3eeb59d924330f8628793))
* docs structure and tests updated ([5ad7197](https://github.com/sequential-parameter-optimization/spotoptim/commit/5ad7197d338b2e3e9c831012e4aa909fef29994b))
* spotOptim docs quarto ([2c997cc](https://github.com/sequential-parameter-optimization/spotoptim/commit/2c997cc0933143cbab6798e90447d6ee8b091494))
* **test:** use proportional tolerance for max_time CI assertion ([57b78d8](https://github.com/sequential-parameter-optimization/spotoptim/commit/57b78d886366f4c64cfd5f665857c133da2797ce))

## [0.2.6](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.5...v0.2.6) (2026-03-02)


### Bug Fixes

* core and SpotOptim splitted in doc ([bfe6123](https://github.com/sequential-parameter-optimization/spotoptim/commit/bfe6123242c4a52a63e7e432dde75f971cff01b7))


### Documentation

* switch docs toolchain from pip to uv ([438ebb4](https://github.com/sequential-parameter-optimization/spotoptim/commit/438ebb49265e8f0de065163e3b69022f6b8ca7b5))
* switch docs toolchain to uv, add _quarto.yml sync guide ([680a265](https://github.com/sequential-parameter-optimization/spotoptim/commit/680a265a58b6228cc23e86225c4116531a8de2ba))

## [0.2.5](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.4...v0.2.5) (2026-03-02)


### Bug Fixes

* var_trans ([21a7051](https://github.com/sequential-parameter-optimization/spotoptim/commit/21a70516eb1ec2ba897305d62a1d66d6f5f4c748))


### Documentation

* add CONTRIBUTING.md with dev setup, docs workflow, and pre-push hook ([ef51095](https://github.com/sequential-parameter-optimization/spotoptim/commit/ef510959aa6080c97ffaa96b85dfc3cb80268cb5))
* embed class method docs inline via quartodoc ([481933b](https://github.com/sequential-parameter-optimization/spotoptim/commit/481933bcede84f7b1f3dfd5d8d499f6d4c2e0f7d))

## [0.2.4](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.3...v0.2.4) (2026-03-02)


### Bug Fixes

* var_trans ([fddbefc](https://github.com/sequential-parameter-optimization/spotoptim/commit/fddbefc0b504fc17032bae634e89beeca20a2a4c))

## [0.2.3](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.2...v0.2.3) (2026-03-01)


### Bug Fixes

* **ci:** merge security fix for untrusted-checkout/critical to main ([a27ad10](https://github.com/sequential-parameter-optimization/spotoptim/commit/a27ad10f72772ffa05f7ca5f4f19665bd24e8d33))
* **ci:** resolve CodeQL untrusted-checkout/critical in workflow_run release job ([1dd089e](https://github.com/sequential-parameter-optimization/spotoptim/commit/1dd089e0bf5f578547b3411589d46b6a77c4db9b))
* create living examples ([32c55a5](https://github.com/sequential-parameter-optimization/spotoptim/commit/32c55a582b7bd8ed0d951d928a7a6a0eb451df3b))

## [0.2.2](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.1...v0.2.2) (2026-03-01)


### Bug Fixes

* doc tested ([c919dcc](https://github.com/sequential-parameter-optimization/spotoptim/commit/c919dccf6fbe1a60e0f3a577de07666203ac11ca))

## [0.2.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.2.0...v0.2.1) (2026-03-01)


### Bug Fixes

* uv.lock ([048b911](https://github.com/sequential-parameter-optimization/spotoptim/commit/048b911739b58250e8b26aa2798853ae8c3dbbb7))

## [0.2.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.1.1...v0.2.0) (2026-03-01)


### Features

* **ci:** add PyPI verification step to release workflow ([6ee1548](https://github.com/sequential-parameter-optimization/spotoptim/commit/6ee1548d415fc73818286355e55aa3654d5bd563))


### Bug Fixes

* **ci:** add twine to CI install, use pure OIDC for PyPI publish ([d0d6d25](https://github.com/sequential-parameter-optimization/spotoptim/commit/d0d6d25a769c2aca99bc32eb032f009196537977))

## [0.1.1](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.1.0...v0.1.1) (2026-03-01)


### Bug Fixes

* **ci:** add PYPI_API_TOKEN fallback for PyPI publish step ([ceffe7d](https://github.com/sequential-parameter-optimization/spotoptim/commit/ceffe7d6575567d1fc0e6e54ff0c2cfbb6459a19))

## [0.1.0](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.0.190...v0.1.0) (2026-03-01)


### Features

* get_internal_datasets_folder ([94e20fe](https://github.com/sequential-parameter-optimization/spotoptim/commit/94e20fe673e9fd01fccedae1b582445f5724883a))


### Bug Fixes

* **ci:** remove duplicate if and Semantic Release step in release workflow ([300ea20](https://github.com/sequential-parameter-optimization/spotoptim/commit/300ea201a2ccec958ca4e602606705132943a2bb))
* **ci:** remove tests from release workflow and restore semantic-release token ([e60dc46](https://github.com/sequential-parameter-optimization/spotoptim/commit/e60dc46dab388f19f30f22d7ddc651b94428ba5c))
* **ci:** remove tests from release workflow and restore semantic-release token ([9ab693e](https://github.com/sequential-parameter-optimization/spotoptim/commit/9ab693ed3f5e672d9d4d3eb848c1c2568434e581))
* colorama added ([c454601](https://github.com/sequential-parameter-optimization/spotoptim/commit/c454601654e83a3b5df4f106bdd2b89afddf47cf))
* dill added ([a7ee9e2](https://github.com/sequential-parameter-optimization/spotoptim/commit/a7ee9e2d7b8f74ecba76e0d0a0c056e0a3b38053))
* docstring living examples ([fef510b](https://github.com/sequential-parameter-optimization/spotoptim/commit/fef510b4d05b54c43d878af30b12e03dead5af7b))
* objects.json ([0caf311](https://github.com/sequential-parameter-optimization/spotoptim/commit/0caf3117cfdbe31cf540b7b1631e5e639813a78d))
* permissions ([6d4620b](https://github.com/sequential-parameter-optimization/spotoptim/commit/6d4620b6d91600a36ee9b0c1545370b587c4f25a))
* permissions ([eb541cd](https://github.com/sequential-parameter-optimization/spotoptim/commit/eb541cd1120b0e8d796d73637a6064e7dc0447ec))
* print verbose corrected ([dd88eb8](https://github.com/sequential-parameter-optimization/spotoptim/commit/dd88eb85b858fd99643227f6d205374e12d90dcb))
* pypi workflow ([075f757](https://github.com/sequential-parameter-optimization/spotoptim/commit/075f757c220167e213ffe9a602f41e93b097616d))
* pyproject.toml ([7b2a23e](https://github.com/sequential-parameter-optimization/spotoptim/commit/7b2a23e7f83143ea4e5405f8a6525a7480bb05a4))
* pyyaml added ([24e922e](https://github.com/sequential-parameter-optimization/spotoptim/commit/24e922e091afe755bb1b2d3e122e59abb6d0e57c))
* workflow, docs and tests ([a32648e](https://github.com/sequential-parameter-optimization/spotoptim/commit/a32648e48b4d8b4852f9c8642fc9e6e147c98e4c))


### Documentation

* _inv objects ([b6713fe](https://github.com/sequential-parameter-optimization/spotoptim/commit/b6713fea7ed155e846b71b1639e51d743612bf82))
* complete Quarto migration with frozen results and syntax fixes ([25baa82](https://github.com/sequential-parameter-optimization/spotoptim/commit/25baa8204524858ec45bd12b3d8f61a75ac1ae25))
* enable quartodoc cross-project interlinks ([92ac8ec](https://github.com/sequential-parameter-optimization/spotoptim/commit/92ac8ec6bf96566c7c1d7274602869dd85fb3ff1))
* implement quartodoc configuration and ensure REUSE compliance ([927891c](https://github.com/sequential-parameter-optimization/spotoptim/commit/927891c3924e18b640290ed2ef5d67f215a85d54))
* migrate from MkDocs to Quarto; add deep SpotOptim tests ([14b57e4](https://github.com/sequential-parameter-optimization/spotoptim/commit/14b57e43a6a619c34293c79e2033097aff1329f5))
* overhaul spotoptim api reference with full quartodoc structure ([450f2a3](https://github.com/sequential-parameter-optimization/spotoptim/commit/450f2a3b87a187715a5e71caa83192c457f8a07d))
* spdx updated ([acefbfe](https://github.com/sequential-parameter-optimization/spotoptim/commit/acefbfe1590eb2d3058035394346e7ce98e5fe15))
* trying to fix ([df2b33a](https://github.com/sequential-parameter-optimization/spotoptim/commit/df2b33afaca5511e2a91dcc700e6fb7c8e3f9156))
* updated (include also lower module levels) ([8269763](https://github.com/sequential-parameter-optimization/spotoptim/commit/82697632818f6ec43d200073cf0c09ef1354b398))

## [0.0.190](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.0.189...v0.0.190) (2026-02-11)


### Bug Fixes

* reuse ([9881677](https://github.com/sequential-parameter-optimization/spotoptim/commit/9881677b0517211cc4259b7484581e00d883dbfb))

## [0.0.189](https://github.com/sequential-parameter-optimization/spotoptim/compare/v0.0.188...v0.0.189) (2026-02-11)


### Performance Improvements

* pyrton >= 3.14 ([a92fd2e](https://github.com/sequential-parameter-optimization/spotoptim/commit/a92fd2ebff59c47fd7457a5fa77fb0c9d50488ce))


### Documentation

* update README and index ([a884d30](https://github.com/sequential-parameter-optimization/spotoptim/commit/a884d30451a4abbfd658d3fe4f3bd89ffc00b238))
* updated to 3.14 ([1c7e260](https://github.com/sequential-parameter-optimization/spotoptim/commit/1c7e26070318d73a32163bf0d04ff14636f9bd2e))
