Versioning
=======================

isQ follows [Semantics Versioning 2.0.0](https://semver.org/) for versioning. Specifically, the version is given in format `MAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCH+METADATA`.

The **main version** of a build `MAJOR.MINOR.PATCH` is defined as-is:

+ MAJOR version when you make incompatible API changes.
+ MINOR version when you add functionality in a backward compatible manner.
+ PATCH version when you make backward compatible bug fixes.

The metadata part might be empty, `+<REV>`, `+<DIRTY>`, or`+<REV>.<DIRTY>`, containing the git tree status of a given version:

+ The `<REV>` part contains the short version of git commit hash, if the commit is not tagged as a released version. 


+ The `<DIRTY>` part is `dirty` if the current git tree is dirty, or empty if the current git tree is clean.

The versioning of the current working tree can be obtained by the following command:

```bash
git describe --tags --dirty | sed -e 's/-[[:digit:]]\+-g/+/' -e 's/+\([a-zA-Z0-9]*\)-dirty/+\1.dirty/' -e 's/-dirty/+dirty/'
```


A few examples:

+ Version `0.1.0` is a clean build of released (tagged) version.
+ Version `0.1.0+dirty` is a dirty build of tagged version `0.1.0`, by for example building after modifying some code.
+ Version `0.1.0+8330d80` is a clean build of git commit `8330d80`; the closest tag to this commit is `0.1.0`. 
+ Version `0.1.0+8330d80.dirty` is a dirty build of git commit `8330d80`; the closest tag to this commit is `0.1.0`.

Usage of Versioning
----------------------

### Subproject metadata versioning

The main version of a build (e.g. `0.1.0`) is propagated to the version of package metadata of different projects, e.g. `Cargo.toml` for Rust subprojects and `package.yaml` for Haskell subprojects.

This versioning scheme makes sure we only need to update the main version (for all projects) only when we are ready for a release.

### Displayed version

The full version should be used in the project, e.g. when invoking `isqc --version`.

Due to the limitation of Nix, currently the revision of a dirty build cannot be passed to the Nix-built binary (See [Issue #4682 of Nix](https://github.com/NixOS/nix/issues/4682)) For example, the version `0.1.0+8330d80.dirty` can only show as `0.1.0+dirty`.



Roadmap
----------------------

Currently isQ is still in early stage of development (0.x.y) and everything may evolve. Informally:

+ MAJOR version is 0.
+ MINOR version is bumped when some important feature is added to isQ.
+ PATCH version is bumped when some patches or fixes are added to isQ.