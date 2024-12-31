{
  description = "Description for the project";
  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    devenv-root = {
      url = "file+file:///dev/null";
      flake = false;
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  outputs =
    inputs@{ flake-parts, devenv-root, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = [ "x86_64-linux" ];

      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          lib,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ ];
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };

          devenv.shells.default = {
            devenv.root =
              let
                devenvRootFileContent = builtins.readFile devenv-root.outPath;
              in
              pkgs.lib.mkIf (devenvRootFileContent != "") devenvRootFileContent;

            name = "mosaic-art";

            imports = [
              # This is just like the imports in devenv.nix.
              # See https://devenv.sh/guides/using-with-flake-parts/#import-a-devenv-module
              # ./devenv-foo.nix
            ];

            env =
              let
                cuda_home = pkgs.symlinkJoin {
                  name = "cuda-home";
                  paths = with pkgs.cudaPackages_12_1; [
                    cuda_nvcc
                    cudatoolkit
                    cuda_cudart.lib
                    libcublas
                    nccl
                    cudnn
                  ];
                };
              in
              {
                CUDA_HOME = cuda_home;
              };

            # https://devenv.sh/reference/options/
            packages = with pkgs; [
              bashInteractive
              glib
              libGL
              libz
            ];
            languages.python = {
              enable = true;
              venv.enable = true;
              uv = {
                enable = true;
                sync = {
                  enable = true;
                  arguments = [ "--frozen" ];
                };
              };
            };

            enterShell = ''
              export LD_LIBRARY_PATH="$(${
                lib.escapeShellArgs [
                  "${inputs.nix-gl-host.defaultPackage.${system}}/bin/nixglhost"
                  "--print-ld-library-path"
                ]
              })":''${LD_LIBRARY_PATH:-}
            '';

          };

        };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.

      };
    };
}
