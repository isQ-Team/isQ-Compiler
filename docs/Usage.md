Usage
=========================

We packed all tools into a tarball.


Quick start
-------------------------

```bash
isqc compile source.isq -o source.so
RUST_LOG=info isqc simulate ./source.so
```

Requirements
-------------------------

Note that since our tarball depends on [nix-user-chroot](https://github.com/nix-community/nix-user-chroot), you need to make sure that your kernel is configured with `CONFIG_USER_NS=y`.

To check: 

```bash
$ unshare --user --pid echo YES
YES
```


Using with Docker containers.
-------------------------

There are two approaches for using toolchain with Docker containers.

The first approach is to allow `unshare` in a container by granting `SYS_ADMIN` capability:

```bash
$ docker run --rm --cap-add SYS_ADMIN -v `pwd`:/isq -it ubuntu bash -c 'cd /isq && ./isqc --version'
isqc 0.1.0
```

The second approach is simpler: just throw the `nix` folder into root folder as `/nix`, and the commands under the folder followed by "`Tools directory:`" when running `run` (see Command Usage) will be available. In this case, the `run` wrapper, as well as `SYS_ADMIN` capability, is no longer required.

```bash
$ docker run --rm -v `pwd`/nix:/nix -it ubuntu /nix/store/*-isqv2/bin/isqc --version
isqc 0.1.0
```


