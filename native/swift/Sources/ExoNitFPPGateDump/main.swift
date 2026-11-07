import ExoNitCore
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

// v3 FPP gate dump driver: see `FPPV3GateDump` in ExoNitCore.
exit(FPPV3GateDump.main(CommandLine.arguments))
