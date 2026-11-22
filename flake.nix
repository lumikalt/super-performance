{
  description = "super-performance: making algorithm challenges fast, one write-up at a time";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, rust-overlay }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          # The crates use edition 2024, so they need a recent stable toolchain.
          rust = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
              "clippy"
              "rustfmt"
            ];
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              rust
              # For reading the assembly of each variant.
              pkgs.cargo-show-asm
            ];
            env = {
              RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
            };
            # Use the tracked hooks, they require Rust to be formatted.
            shellHook = ''
              if git rev-parse --git-dir >/dev/null 2>&1; then
                git config core.hooksPath .githooks
              fi
            '';
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs }: pkgs.nixfmt);
    };
}
